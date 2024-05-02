// Copyright 2021-2024 by Jon Dart. All Rights Reserved.
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

#include "nnue.h"
#include "features/halfkav2hm.h"

// Unit tests for nnue code

template<size_t ROWS, size_t COLS>
static int test_linear() {
    int errs = 0;

    using InputType = uint8_t;
    using WeightType = int8_t;
    using BiasType = int32_t;
    using OutputType = int32_t;

    // serializer assumes rows are at least 32 bytes
    static constexpr size_t ROUNDED_ROWS = std::max<size_t>(ROWS, 32);

    static BiasType biases[COLS];
    static WeightType weights[COLS][ROUNDED_ROWS]; // indexed first by output

    constexpr size_t bufSize = COLS*sizeof(BiasType) + (ROUNDED_ROWS * COLS)*sizeof(WeightType);
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
        for (size_t j = 0; j < ROUNDED_ROWS; j++) {
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
    outfile.write(reinterpret_cast<char *>(buf.get()), bufSize);
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
        inputs[i] = std::min<int>(127,static_cast<InputType>(i+1));
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

    using FeatureXformer =
        nnue::HalfKaV2Hm<uint16_t, int16_t, int16_t, int16_t, InputSize, OutputSize>;

    using AccumulatorType = FeatureXformer::AccumulatorType;

    HalfKaV2Hm() : layer1(new FeatureXformer()) {}

    AccumulatorType accum;

    void init(unsigned index, const OutputType vals[]) {
        layer1.get()->setCol(index, vals);
    }

    FeatureXformer *get() const noexcept { return layer1.get(); }

  private:
    std::unique_ptr<FeatureXformer> layer1;
};

static int16_t col1[HalfKaV2Hm::OutputSize];
static int16_t col2[HalfKaV2Hm::OutputSize];
static int16_t col3[HalfKaV2Hm::OutputSize];
static int16_t col4[HalfKaV2Hm::OutputSize];
static int16_t biases[HalfKaV2Hm::OutputSize];

static int test_halfkp() {
    const std::string fen =
        "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";

    HalfKaV2Hm::FeatureXformer::PSQWeightType psq1[nnue::PSQBuckets], psq2[nnue::PSQBuckets], psq3[nnue::PSQBuckets], psq4[nnue::PSQBuckets];
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; i++) {
         col1[i] = -1550 + i;
         col2[i] = 432 + i;
         col3[i] = -591 + i;
         col4[i] = -240 + i;
         biases[i] = i % 4;
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
    halfKp.get()->setBiases(biases);

    halfKp.get()->updateAccum(bIndices, nnue::AccumulatorHalf::Lower, accum);
    halfKp.get()->updateAccum(wIndices, nnue::AccumulatorHalf::Upper, accum);

    HalfKaV2Hm::OutputType expected[2][HalfKaV2Hm::OutputSize];
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; ++i) {
        expected[0][i] = col3[i] + col4[i] + biases[i];
        expected[1][i] = col1[i] + col2[i] + biases[i];
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
    HalfKaV2Hm::FeatureXformer::PSQWeightType psq_expected[2][nnue::PSQBuckets];
    for (size_t i = 0; i < nnue::PSQBuckets; ++i) {
        psq_expected[0][i] = psq3[i] + psq4[i];
        psq_expected[1][i] = psq1[i] + psq2[i];
    }
    std::vector<nnue::AccumulatorHalf> halfs = {nnue::AccumulatorHalf::Lower,
        nnue::AccumulatorHalf::Upper};
    for (auto half : halfs) {
        for (size_t i = 0; i < nnue::PSQBuckets; ++i) {
            auto expect = psq_expected[half == nnue::AccumulatorHalf::Lower ? 0 : 1][i];
            if (expect != accum.getPSQ(half)[i]) {
                ++errs;
                std::cerr << " error at psq index " << int(i)
                          << " expected: " << expected << " actual "
                          << accum.getPSQ(half)[i] << std::endl;
            }
        }
    }

    // test 1st layer output transformer
    nnue::SqrCReLU<int16_t, HalfKaV2Hm::AccumulatorType, uint8_t, 1024, 127, 7> outputTransform;

#ifdef AVX52
    alignas(64) uint8_t out[HalfKaV2Hm::OutputSize];
#else
    alignas(nnue::DEFAULT_ALIGN) uint8_t out[HalfKaV2Hm::OutputSize];
#endif
    outputTransform.postProcessAccum(accum,out);

    static const uint8_t expected2[HalfKaV2Hm::OutputSize] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,6,9,8,11,14,17,16,19,22,25,24,27,30,33,32,35,38,41,40,43,46,49,48,51,54,57,56,59,62,65,64,67,70,73,72,75,78,81,80,83,86,89,88,91,94,97,96,99,102,105,104,107,110,113,112,115,118,121,120,123,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    int tmp_err = errs;
    for (size_t i = 0; i < HalfKaV2Hm::OutputSize; ++i) {
        if (out[i] != expected2[i]) ++errs;
    }
    if (errs != tmp_err) {
        std::cerr << "errors in layer 2: SqrCReLU" << std::endl;
    }

    return errs;
}

template<nnue::Color c>
static void getIndices(const ChessInterface &ci,
                       nnue::IndexArray &a,
                       std::set<nnue::IndexType> &indices) {
    nnue::Evaluator<ChessInterface>::getIndices<c>(ci,a);
    for (auto it = a.begin(); it != a.end() && *it != nnue::LAST_INDEX; it++) {
        indices.insert(*it);
    }
}

static void getSetDiff(const std::set<nnue::IndexType> &a, const std::set<nnue::IndexType> &b,
                       std::set<nnue::IndexType> &out) {
    out.clear();
    std::set_difference(a.begin(),a.end(),b.begin(),b.end(),std::inserter(out, out.end()));
}

static void getIndexDiffs(nnue::Evaluator<ChessInterface> evaluator, nnue::Color c,
                          const ChessInterface &ciSource, const ChessInterface &ciTarget,
                          std::set<nnue::IndexType> &added, std::set<nnue::IndexType> &removed) {
    nnue::IndexArray addedArray, removedArray;
    size_t addedCount, removedCount;
    evaluator.getIndexDiffs(ciSource, ciTarget, c, addedArray, removedArray,
                            addedCount, removedCount);
    for (size_t i = 0; i < addedCount; ++i) {
        added.insert(addedArray[i]);
    }
    for (size_t i = 0; i < removedCount; ++i) {
        removed.insert(removedArray[i]);
    }
    // diff algorithm may place items in both removed and added
    // lists. Filter these out.
    std::vector<nnue::IndexType> intersect(32);
    std::set_intersection(
        removed.begin(), removed.end(), added.begin(), added.end(),
        std::back_inserter(intersect));
    for (auto x : intersect) {
        added.erase(x);;
        removed.erase(x);
    }
}

static int test_incr(ChessInterface &ciSource, ChessInterface &ciTarget) {
    int errs = 0;
    assert(ciSource != ciTarget);

    std::set<nnue::IndexType> baseW, baseB, targetW, targetB;
    nnue::IndexArray wIndicesSource, bIndicesSource, wIndicesTarget,
        bIndicesTarget;
    getIndices<nnue::White>(ciSource,wIndicesSource,baseW);
    getIndices<nnue::Black>(ciSource,bIndicesSource,baseB);
    getIndices<nnue::White>(ciTarget,wIndicesTarget,targetW);
    getIndices<nnue::Black>(ciTarget,bIndicesTarget,targetB);

    std::set<nnue::IndexType> addedW, addedB, removedW, removedB;
    getSetDiff(baseW, targetW, removedW);
    getSetDiff(baseB, targetB, removedB);
    getSetDiff(targetW, baseW, addedW);
    getSetDiff(targetB, baseB, addedB);

    nnue::Evaluator<ChessInterface> evaluator;

    nnue::Network network;

    // have Evaluator calculate index diffs
    std::set<nnue::IndexType> addedFromEvaluatorW,
        addedFromEvaluatorB,
        removedFromEvaluatorW,
        removedFromEvaluatorB;

    getIndexDiffs(evaluator,nnue::White,ciSource,ciTarget,addedFromEvaluatorW,removedFromEvaluatorW);
    getIndexDiffs(evaluator,nnue::Black,ciSource,ciTarget,addedFromEvaluatorB,removedFromEvaluatorB);

    if (addedW != addedFromEvaluatorW) {
        ++errs;
        std::cerr << "added list differs (W)" << std::endl;
    }
    if (addedB != addedFromEvaluatorB) {
        ++errs;
        std::cerr << "added list differs (B)" << std::endl;
    }
    if (removedW != removedFromEvaluatorW) {
        ++errs;
        std::cerr << "removed list differs (W)" << std::endl;
    }
    if (removedB != removedFromEvaluatorB) {
        ++errs;
        std::cerr << "removed list differs (B)" << std::endl;
    }

    HalfKaV2Hm halfKp;

    HalfKaV2Hm::AccumulatorType accum;

    // set some weights
    for (size_t i = 0; i < HalfKaV2Hm::InputSize; i++) {
        HalfKaV2Hm::OutputType col[HalfKaV2Hm::OutputSize];
        for (size_t j = 0; j < HalfKaV2Hm::OutputSize; j++) {
            col[j] = (i + j) % 10 - 5;
        }
        halfKp.get()->setCol(i, col);
    }

    // Full evaluation of feature transformer for source position
    evaluator.updateAccum(network, wIndicesSource, nnue::White, ciSource.sideToMove(),
                          ciSource.getAccumulator());
    evaluator.updateAccum(network, bIndicesSource, nnue::Black, ciSource.sideToMove(),
                          ciSource.getAccumulator());

    // Full evaluation of feature transformer for target position into "accum"
    evaluator.updateAccum(network, wIndicesTarget, nnue::White, ciTarget.sideToMove(),
                          accum);
    evaluator.updateAccum(network, bIndicesTarget, nnue::Black, ciTarget.sideToMove(),
                          accum);

    assert(ciTarget.getAccumulator().getState(nnue::AccumulatorHalf::Lower) ==
           nnue::AccumulatorState::Empty);
    assert(ciTarget.getAccumulator().getState(nnue::AccumulatorHalf::Upper) ==
           nnue::AccumulatorState::Empty);

    // Incremental evaluation of 1st layer, starting from source position
    evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::White);
    evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::Black);

    // Compare incremental and full eval
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
    target_pos.dirty[target_pos.dirty_num++] =
        DirtyState(35 /*D5*/, 26 /*C4*/, nnue::BlackPawn);
    target_pos.dirty[target_pos.dirty_num++] =
        DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::WhitePawn);
    // connect target to previous position
    target_pos.previous = &source_pos;

    // test incremental update
    errs += test_incr(ciSource, ciTarget);

    Position target2_pos(target2_fen);

    // Try a position 2 half-moves ahead
    target2_pos.dirty[target2_pos.dirty_num++] =
        DirtyState(12 /*E2*/, 26 /*C4*/, nnue::WhiteBishop);
    target2_pos.dirty[target2_pos.dirty_num++] =
        DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::BlackPawn);
    target2_pos.previous = &target_pos;
    ChessInterface ciTarget2(&target2_pos);

    ciTarget.getAccumulator().setEmpty();
    ciTarget2.getAccumulator().setEmpty();

    errs += test_incr(ciSource, ciTarget2);

    return errs;
}

template<size_t size>
static int test_CReLU() {
    int errs = 0;
    constexpr int CLAMP_MAX = 127;
    constexpr int RSHIFT = 6;
    using InputType = int32_t;
    using OutputType = uint8_t;
    using CReLU = nnue::CReLU<InputType, OutputType, size, RSHIFT>;

    alignas(nnue::DEFAULT_ALIGN) InputType input[size];
    alignas(nnue::DEFAULT_ALIGN) OutputType output[size],output2[size];

    CReLU c(CLAMP_MAX);

    for (unsigned i = 0; i < size; i++) {
        input[i] = -9000 + 900*std::min<unsigned>(i,10) + i;
        output[i] = static_cast<OutputType>(std::clamp<InputType>(input[i] >> RSHIFT,0,CLAMP_MAX));
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
    errs += test_linear<16,16>();
    errs += test_linear<32,1>();
    errs += test_halfkp();
    errs += test_incremental();
    errs += test_CReLU<16>();
    errs += test_CReLU<32>();
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
