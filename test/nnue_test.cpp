// Copyright 2021, 2022 by Jon Dart. All Rights Reserved.
#include "nnue.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "../interface/chessint.h"

// Unit tests for nnue code

template<size_t ROWS, size_t COLS>
static int test_linear() {
    int errs = 0;

    using InputType = uint8_t;
    using WeightType = int8_t;
    using BiasType = int32_t;
    using OutputType = int32_t;

    static BiasType biases[COLS];
    static WeightType weights[COLS][ROWS]; // indexed first by output

    constexpr size_t bufSize = COLS*sizeof(BiasType)+ (ROWS * COLS)*sizeof(WeightType);
    auto buf = std::unique_ptr<std::byte[]>(new std::byte[bufSize]);

    std::byte *b = buf.get();
    BiasType *bb = reinterpret_cast<BiasType *>(b);
    for (size_t i = 0; i < COLS; i++) {
        *bb++ = biases[i] = (i%15) + i - 10;
    }
    b += COLS*sizeof(BiasType);
    WeightType *w = reinterpret_cast<WeightType*>(b);
    // serialized in column order
    for (size_t i = 0; i < COLS; i++) {
        for (size_t j = 0; j < ROWS; j++) {
            *w++ = weights[i][j] = ((i+j) % 20) - 10;
        }
    }

    nnue::LinearLayer<InputType, WeightType, BiasType, OutputType, ROWS, COLS> layer;

#if defined(__MINGW32__) || defined(__MINGW64__) || (defined(__APPLE__) && defined(__MACH__))
    std::string tmp_name("XXXXXX");
#else
    std::string tmp_name(std::tmpnam(nullptr));
#endif

    std::ofstream outfile(tmp_name, std::ios::binary | std::ios::trunc);
    outfile.write(reinterpret_cast<char *>(buf.get()),
                  bufSize);
    if (outfile.bad()) {
      ++errs;
      std::cerr << "error writing stream" << std::endl;
      outfile.close();
      std::remove(tmp_name.c_str());
      return 1;
    }
    outfile.close();

    std::ifstream infile(tmp_name, std::ios::binary);

    // test reading a layer
    layer.read(infile);

    if (infile.bad()) {
        std::cerr << "error reading linear layer" << std::endl;
        ++errs;
        infile.close();
        std::remove(tmp_name.c_str());
        return errs;
    }
    infile.close();

    // verify layer was read
    int tmp = errs;
    for (size_t i = 0; i < COLS; i++) {
        errs += (layer.getBiases()[i] != biases[i]);
	if (layer.getBiases()[i] != biases[i]) std::cerr << layer.getBiases()[i] << ' ' << biases[i] << std::endl;
    }
    for (size_t i = 0; i < COLS; i++) {
        // get weights for output column
        const WeightType *col = layer.getCol(i);
        for (size_t j = 0; j < ROWS; j++) {
            errs += (weights[i][j] != col[j]);
        }
    }
    if (errs - tmp > 0)
        std::cerr << "errors deserializing linear layer" << std::endl;

    alignas(nnue::DEFAULT_ALIGN) InputType inputs[ROWS];
    for (unsigned i = 0; i < ROWS; i++) {
        inputs[i] = static_cast<InputType>(i);
    }

    alignas(nnue::DEFAULT_ALIGN) OutputType output[COLS], computed[COLS];
    // test linear layer propagation
    layer.forward(inputs, output);
    for (size_t i = 0; i < COLS; i++) {
        computed[i] = static_cast<OutputType>(biases[i]);
    }
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            computed[j] += inputs[i] * weights[j][i];
        }
    }

    tmp = errs;
    for (size_t i = 0; i < COLS; i++) {
        errs += computed[i] != output[i];
    }
    if (errs - tmp > 0)
        std::cerr << "errors computing dot product " << ROWS << "x" << COLS << std::endl;
    std::remove(tmp_name.c_str());
    return errs;
}

static const std::unordered_map<char, nnue::Piece> pieceMap = {
    {'p', nnue::BlackPawn}, {'n', nnue::BlackKnight}, {'b', nnue::BlackBishop},
    {'r', nnue::BlackRook}, {'q', nnue::BlackQueen},  {'k', nnue::BlackKing},
    {'P', nnue::WhitePawn}, {'N', nnue::WhiteKnight}, {'B', nnue::WhiteBishop},
    {'R', nnue::WhiteRook}, {'Q', nnue::WhiteQueen},  {'K', nnue::WhiteKing}};

// wrapper around nnue::HalfKaV2Hm, sets up that class with some fixed parameters
class HalfKaV2Hm {
  public:
    static constexpr size_t OutputSize = 1024;

    static constexpr size_t InputSize = 22*OutputSize;

    using OutputType = int16_t;

    using Layer1 =
        nnue::HalfKaV2Hm<uint16_t, int16_t, int16_t, int16_t, InputSize, OutputSize>;

    using AccumulatorType = Layer1::AccumulatorType;

    HalfKaV2Hm() : layer1(new Layer1()) {}

    AccumulatorType accum;

    void init(unsigned index, const OutputType vals[]) {
        layer1.get()->setCol(index, vals);
    }

    Layer1 *get() const noexcept { return layer1.get(); }

  private:
    std::unique_ptr<Layer1> layer1;
};

static int16_t col1[HalfKaV2Hm::OutputSize];
static int16_t col2[HalfKaV2Hm::OutputSize];
static int16_t col3[HalfKaV2Hm::OutputSize];
static int16_t col4[HalfKaV2Hm::OutputSize];

static int test_halfkp() {
    const std::string fen =
        "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";

    HalfKaV2Hm::Layer1::PSQWeightType psq1[nnue::PSQBuckets], psq2[nnue::PSQBuckets], psq3[nnue::PSQBuckets], psq4[nnue::PSQBuckets];
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; i++) {
         col1[i] = -1550 + i;
         col2[i] = 432 + i;
         col3[i] = -591 + i;
         col4[i] = -240 + i;
    }
    for (size_t i = 0; i < nnue::PSQBuckets; i++) {
        psq1[i] = -200 + i;
        psq2[i] = 71 + i;
        psq3[i] = -50 + i;
        psq4[i] = 23 + i;
    }

    std::unordered_set<nnue::IndexType> w_expected{
        20800,
        20804,
        21062,
        20429,
        20686,
        20438,
        20441,
        20506,
        20447,
        20513,
        20451,
        20644,
        20967,
        21033,
        20907,
        20525,
        20527,
        20533,
        21110,
        20924};

    std::unordered_set<nnue::IndexType> b_expected{
        18104,
        18108,
        18302,
        17717,
        17974,
        17710,
        17697,
        17634,
        17703,
        17625,
        17691,
        17756,
        18207,
        18129,
        18003,
        17621,
        17623,
        17613,
        18254,
        17988};

    Position p(fen);
    ChessInterface intf(&p);

    nnue::IndexArray wIndices, bIndices;
    auto wCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(
        intf, wIndices);
    auto bCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(
        intf, bIndices);

    int errs = 0;
    for (auto it = wIndices.begin(); it != wIndices.begin() + wCount; it++) {
        if (w_expected.find(*it) == w_expected.end()) {
            ++errs;
            std::cerr << "error: white index " << *it << " not expected"
                      << std::endl;
        } else
            w_expected.erase(*it);
    }
    for (auto it = bIndices.begin(); it != bIndices.begin() + bCount; it++) {
        if (b_expected.find(*it) == b_expected.end()) {
            ++errs;
            std::cerr << "error: black index " << *it << " not expected"
                      << std::endl;
        } else
            b_expected.erase(*it);
    }
    if (!w_expected.empty()) {
        ++errs;
        std::cerr << "error: not all expected White indices found, missing "
                  << std::endl;
        for (auto i : w_expected) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }
    if (!b_expected.empty()) {
        ++errs;
        std::cerr << "error: not all expected Black indices found, missing "
                  << std::endl;
        for (auto i : b_expected) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }

    HalfKaV2Hm halfKp;

    HalfKaV2Hm::AccumulatorType accum;

    halfKp.get()->setCol(20800, col1);
    halfKp.get()->setCol(20804, col2);
    halfKp.get()->setCol(18254, col3);
    halfKp.get()->setCol(17988, col4);
    halfKp.get()->setPSQ(20800, psq1);
    halfKp.get()->setPSQ(20804, psq2);
    halfKp.get()->setPSQ(18254, psq3);
    halfKp.get()->setPSQ(17988, psq4);

    halfKp.get()->updateAccum(bIndices, nnue::AccumulatorHalf::Lower, accum);
    halfKp.get()->updateAccum(wIndices, nnue::AccumulatorHalf::Upper, accum);

    HalfKaV2Hm::OutputType expected[2][HalfKaV2Hm::OutputSize];
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; ++i) {
        expected[0][i] = col3[i] + col4[i];
        expected[1][i] = col1[i] + col2[i];
    }
    static const nnue::AccumulatorHalf halves[2] = {nnue::AccumulatorHalf::Lower,
        nnue::AccumulatorHalf::Upper};
    for (auto h : halves) {
        for (size_t i = 0; i < HalfKaV2Hm::OutputSize; ++i) {
            auto exp = expected[h == nnue::AccumulatorHalf::Lower ? 0 : 1][i];
            if (exp != accum.getOutput(h)[i]) {
                ++errs;
                std::cerr << " error at accum index " << int(i)
                          << " expected: " << int(exp) << " actual "
                          << accum.getOutput(h)[i] << std::endl;
            }
        }
    }

    // test PSQ update
    HalfKaV2Hm::Layer1::PSQWeightType psq_expected[2][nnue::PSQBuckets];
    for (size_t i = 0; i < nnue::PSQBuckets; ++i) {
        psq_expected[0][i] = psq3[i] + psq4[i];
        psq_expected[1][i] = psq1[i] + psq2[i];
    }
    std::vector<nnue::AccumulatorHalf> halfs = {nnue::AccumulatorHalf::Lower,
        nnue::AccumulatorHalf::Upper};
    for (auto half : halfs) {
        for (size_t i = 0; i < nnue::PSQBuckets; ++i) {
            auto expected = psq_expected[half == nnue::AccumulatorHalf::Lower ? 0 : 1][i];
            if (expected != accum.getPSQ(half)[i]) {
                ++errs;
                std::cerr << " error at psq index " << int(i)
                          << " expected: " << expected << " actual "
                          << accum.getPSQ(half)[i] << std::endl;
            }
        }
    }

    // test 1st layer output transformer
    nnue::HalfKaOutput<int16_t, HalfKaV2Hm::AccumulatorType, uint8_t, 1024> outputTransform(7,127);

#ifdef AVX52
    alignas(64) uint8_t out[HalfKaV2Hm::OutputSize];
#else    
    alignas(nnue::DEFAULT_ALIGN) uint8_t out[HalfKaV2Hm::OutputSize];
#endif
    outputTransform.postProcessAccum(accum,out);

    static const uint8_t expected2[HalfKaV2Hm::OutputSize] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    int tmp_err = errs;
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; ++i) {
        if (out[i] != expected2[i]) ++errs;
    }
    if (errs != tmp_err) {
        std::cerr << "errors in layer 2: HalfKaOutput" << std::endl;
    }

    return errs;
}

static int test_incr(ChessInterface &ciSource, ChessInterface &ciTarget) {
    int errs = 0;

    nnue::IndexArray bIndices, wIndices;

    std::set<nnue::IndexType> base, target;

    nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(ciSource,
                                                             wIndices);
    nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(ciSource,
                                                             bIndices);
    for (auto it = wIndices.begin();
         it != wIndices.end() && *it != nnue::LAST_INDEX; it++) {
        base.insert(*it);
    }
    for (auto it = bIndices.begin();
         it != bIndices.end() && *it != nnue::LAST_INDEX; it++) {
        base.insert(*it);
    }

    nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(ciTarget,
                                                             wIndices);
    nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(ciTarget,
                                                             bIndices);
    for (auto it = wIndices.begin();
         it != wIndices.end() && *it != nnue::LAST_INDEX; it++) {
        target.insert(*it);
    }
    for (auto it = bIndices.begin();
         it != bIndices.end() && *it != nnue::LAST_INDEX; it++) {
        target.insert(*it);
    }

    std::vector<nnue::IndexType> added(32), removed(32);

    auto itend = std::set_difference(base.begin(), base.end(), target.begin(),
                                     target.end(), removed.begin());
    removed.resize(itend - removed.begin());
    itend = std::set_difference(target.begin(), target.end(), base.begin(),
                                base.end(), added.begin());
    added.resize(itend - added.begin());

    nnue::Evaluator<ChessInterface> evaluator;

    nnue::Network network;

    // have Evaluator calculate index diffs
    nnue::IndexArray wRemoved, wAdded, bRemoved, bAdded;
    size_t wAddedCount, wRemovedCount, bAddedCount, bRemovedCount;
    evaluator.getIndexDiffs(ciSource, ciTarget, nnue::White, wAdded, wRemoved,
                            wAddedCount, wRemovedCount);
    evaluator.getIndexDiffs(ciSource, ciTarget, nnue::Black, bAdded, bRemoved,
                            bAddedCount, bRemovedCount);

    // diffs
    std::set<nnue::IndexType> removedAll, addedAll;
    for (size_t i = 0; i < wRemovedCount; i++)
        removedAll.insert(wRemoved[i]);
    for (size_t i = 0; i < bRemovedCount; i++)
        removedAll.insert(bRemoved[i]);
    for (size_t i = 0; i < wAddedCount; i++)
        addedAll.insert(wAdded[i]);
    for (size_t i = 0; i < bAddedCount; i++)
        addedAll.insert(bAdded[i]);

    std::vector<nnue::IndexType> intersect(32);
    auto intersect_end = std::set_intersection(
        removedAll.begin(), removedAll.end(), addedAll.begin(), addedAll.end(),
        intersect.begin());
    // diff algorithm may place items in both removed and added
    // lists. Filter these out.
    for (auto it = intersect.begin(); it != intersect_end; it++) {
        addedAll.erase(*it);
        removedAll.erase(*it);
    }

    std::vector<nnue::IndexType> diffs(32);
    itend = std::set_difference(removedAll.begin(), removedAll.end(),
                                removed.begin(), removed.end(), diffs.begin());
    diffs.resize(itend - diffs.begin());
    if (diffs.size() != 0) {
        ++errs;
        std::cerr << "removed list differs" << std::endl;
    }
    diffs.clear();
    itend = std::set_difference(addedAll.begin(), addedAll.end(), added.begin(),
                                added.end(), diffs.begin());
    diffs.resize(itend - diffs.begin());
    if (diffs.size() != 0) {
        ++errs;
        std::cerr << "added list differs" << std::endl;
    }

    HalfKaV2Hm halfKp;

    HalfKaV2Hm::AccumulatorType accum;

    for (size_t i = 0; i < HalfKaV2Hm::InputSize; i++) {
        HalfKaV2Hm::OutputType col[HalfKaV2Hm::OutputSize];
        for (size_t j = 0; j < HalfKaV2Hm::OutputSize; j++) {
            col[j] = (i + j) % 10 - 5;
        }
        halfKp.get()->setCol(i, col);
    }

    // Full evaluation of 1st layer for source position
    evaluator.updateAccum(network, wIndices, nnue::White, ciSource.sideToMove(),
                          ciSource.getAccumulator());
    evaluator.updateAccum(network, bIndices, nnue::Black, ciSource.sideToMove(),
                          ciSource.getAccumulator());

    // Full evaluation of 1st layer for target position into "accum"
    evaluator.updateAccum(network, wIndices, nnue::White, ciTarget.sideToMove(),
                          accum);
    evaluator.updateAccum(network, bIndices, nnue::Black, ciTarget.sideToMove(),
                          accum);

    assert(ciTarget.getAccumulator().getState(nnue::AccumulatorHalf::Lower) ==
           nnue::AccumulatorState::Empty);
    assert(ciTarget.getAccumulator().getState(nnue::AccumulatorHalf::Upper) ==
           nnue::AccumulatorState::Empty);

    // Incremental evaluation of 1st layer, starting from source position
    evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::White);
    evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::Black);

    // Compare evals
    auto old_errs = errs;
    errs += ciTarget.getAccumulator() != accum;
    if (errs != old_errs) {
        std::cerr << "accumulator mismatch after incremental eval" << std::endl;
    }

    // Reset target accumulator state
    ciTarget.getAccumulator().setEmpty();

    // use the API that takes incremental or regular path
    evaluator.updateAccum(network, ciTarget, nnue::White);
    evaluator.updateAccum(network, ciTarget, nnue::Black);

    // Compare results again
    old_errs = errs;
    errs += ciTarget.getAccumulator() != accum;
    if (errs != old_errs) {
        std::cerr << "accumulator mismatch after regular/incremental eval" << std::endl;
    }

    return errs;
}

static int test_incremental() {
    constexpr auto source_fen =
        "r1bqk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ1BPPP/R1B1K2R b KQkq -";

    constexpr auto target_fen =
        "r1bqk2r/pp1n1ppp/2pbpn2/8/2pP4/2N1PN2/PPQ1BPPP/R1B1K2R w KQkq -";

    constexpr auto target2_fen =
        "r1bqk2r/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PPQ2PPP/R1B1K2R b KQkq -";

    int errs = 0;

    HalfKaV2Hm halfKp;

    Position source_pos(source_fen);
    ChessInterface ciSource(&source_pos);
    Position target_pos(target_fen);
    ChessInterface ciTarget(&target_pos);
    // set up dirty status
    // d5 pawn x c4 pawn
    source_pos.dirty[source_pos.dirty_num++] =
        DirtyState(35 /*D5*/, 26 /*C4*/, nnue::BlackPawn);
    source_pos.dirty[source_pos.dirty_num++] =
        DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::WhitePawn);
    // connect target to previous position
    target_pos.previous = &source_pos;

    errs += test_incr(ciSource, ciTarget);

    Position target2_pos(target2_fen);
    ChessInterface ciTarget2(&target2_pos);

    // Try a position 2 half-moves back
    target_pos.dirty[target_pos.dirty_num++] =
        DirtyState(12 /*E2*/, 26 /*C4*/, nnue::WhiteBishop);
    target_pos.dirty[target_pos.dirty_num++] =
        DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::BlackPawn);

    target2_pos.previous = &target_pos;

    ciTarget.getAccumulator().setEmpty();

    errs += test_incr(ciSource, ciTarget2);

    return errs;
}

static int test_clamp() {
    int errs = 0;
    using InputType = int16_t;
    using OutputType = uint8_t;
    constexpr unsigned SIZE = 512;
    constexpr uint8_t CLAMP_MAX = 127;
    using Clamper = nnue::Clamp<InputType, OutputType, SIZE>;

    alignas(nnue::DEFAULT_ALIGN) InputType input[SIZE];
    alignas(nnue::DEFAULT_ALIGN) OutputType output[SIZE],output2[SIZE];

    Clamper c(CLAMP_MAX);

    for (unsigned i = 0; i < SIZE; i++) {
        input[i] = (i%20)-10 + (i%6)*30;
        output[i] = static_cast<OutputType>(std::clamp<InputType>(input[i],0,static_cast<InputType>(CLAMP_MAX)));
    }
    std::memset(output2,'\0',SIZE*sizeof(OutputType));
    c.doForward(input,output2);
    for (unsigned i = 0; i < SIZE; i++) {
        errs += output2[i] != output[i];
    }
    if (errs) std::cerr << errs << " error(s) in clamp function" << std::endl;
    return errs;
}

template<size_t size>
static int test_scale_and_clamp() {
    int errs = 0;
    constexpr int CLAMP_MAX = 127;
    constexpr int SCALE = 6;
    using InputType = int32_t;
    using OutputType = uint8_t;
    using ScaleAndClamper = nnue::ScaleAndClamp<InputType, OutputType, size, RSHIFT>;

    alignas(nnue::DEFAULT_ALIGN) InputType input[size];
    alignas(nnue::DEFAULT_ALIGN) OutputType output[size],output2[size];

    ScaleAndClamper c(CLAMP_MAX);

    for (unsigned i = 0; i < size; i++) {
        input[i] = -9000 + 900*std::min<unsigned>(i,10) + i;
        output[i] = static_cast<OutputType>(std::clamp<InputType>(input[i] >> SCALE,0,CLAMP_MAX));
    }
    std::memset(output2,'\0',size*sizeof(OutputType));
    c.doForward(input,output2);
    for (unsigned i = 0; i < size; i++) {
        errs += output2[i] != output[i];
    }
    if (errs) std::cerr << errs << " error(s) in scale and clamp function" << std::endl;
    return errs;
}

int main(int argc, char **argv) {
    nnue::Network n;

    int errs = 0;
    errs += test_linear<1024,16>();
    errs += test_linear<32,32>();
    errs += test_linear<32,1>();
    errs += test_halfkp();
    errs += test_incremental();
    errs += test_clamp();
    errs += test_scale_and_clamp<16>();
    errs += test_scale_and_clamp<32>();
    std::cerr << errs << " errors" << std::endl;

    std::string fname;
    for (int arg = 1; arg < argc; ++arg) {
        if (*argv[arg] == '-') {
            char c = argv[arg][1];
            switch (c) {
            case 'f':
                if (++arg < argc) {
                    fname = argv[arg];
                }
                break;
            default:
                std::cerr << "unrecognized switch: " << c << std::endl;
                break;
            }
        }
    }
    if (fname != "") {
        std::cout << "loading " << fname << std::endl;
        std::ifstream in(fname);
        in >> n;
        if (in.bad()) {
            std::cerr << "error loading " << fname << std::endl << std::flush;
        }
    }
#ifdef NNUE_TRACE
    const std::string fen =
        "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";
    Position p(fen);
    ChessInterface intf(&p);
    nnue::Evaluator<ChessInterface>::fullEvaluate(n,intf);
#endif

    return 0;
}
