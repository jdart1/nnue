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

#include "../interface/chessint.h"

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
#ifdef STOCKFISH_FORMAT
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
#else
    // weights first
    WeightType *w = reinterpret_cast<WeightType*>(b);
    // serialized in column order
    for (size_t i = 0; i < COLS; i++) {
        for (size_t j = 0; j < ROUNDED_ROWS; j++) {
            *w++ = weights[i][j] = ((i+j) % 20) - 10;
        }
    }
    b += COLS*ROUNDED_ROWS*sizeof(WeightType);
    BiasType *bb = reinterpret_cast<BiasType *>(b);
    for (size_t i = 0; i < COLS; i++) {
        *bb++ = biases[i] = (i%15) + i - 10;
    }
#endif

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

// wrapper around nnue::ArasanV3Feature, sets up that class with some fixed parameters
class ArasanV3Feature {
  public:
    static constexpr size_t OutputSize = 1024;

    static constexpr size_t InputSize = 22*OutputSize;

    using OutputType = int16_t;

    using FeatureXformer =
        nnue::ArasanV3Feature<uint16_t, int16_t, int16_t, int16_t, InputSize, OutputSize>;

    using AccumulatorType = FeatureXformer::AccumulatorType;

    ArasanV3Feature() : layer1(new FeatureXformer()) {}

    AccumulatorType accum;

    void init(unsigned index, const OutputType vals[]) {
        layer1.get()->setCol(index, vals);
    }

    FeatureXformer *get() const noexcept { return layer1.get(); }

  private:
    std::unique_ptr<FeatureXformer> layer1;
};

static int16_t col1[ArasanV3Feature::OutputSize];
static int16_t col2[ArasanV3Feature::OutputSize];
static int16_t col3[ArasanV3Feature::OutputSize];
static int16_t col4[ArasanV3Feature::OutputSize];
static int16_t zero_col[ArasanV3Feature::OutputSize] = {0};
static int16_t biases[ArasanV3Feature::OutputSize];

static int testFeature(const std::string &fen, std::unordered_set<nnue::IndexType> &w_expected,
                        std::unordered_set<nnue::IndexType> &b_expected) {
    // ArasanV3Feature::FeatureXformer::PSQWeightType psq1[nnue::PSQBuckets], psq2[nnue::PSQBuckets], psq3[nnue::PSQBuckets], psq4[nnue::PSQBuckets];
    /*
    for (size_t i = 0; i < nnue::PSQBuckets; i++) {
        psq1[i] = -200 + i;
        psq2[i] = 71 + i;
        psq3[i] = -50 + i;
        psq4[i] = 23 + i;
    }
    */

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

    ArasanV3Feature feature;

    ArasanV3Feature::AccumulatorType accum;

    for (size_t i = 0; i < ArasanV3Feature::InputSize; ++i) feature.get()->setCol(i,zero_col);

    feature.get()->setCol(36, col1);
    feature.get()->setCol(686, col2);
    feature.get()->setCol(1748, col3);
    feature.get()->setCol(2097, col4);
    feature.get()->setBiases(biases);

    feature.get()->updateAccum(bIndices, nnue::AccumulatorHalf::Lower, accum);
    feature.get()->updateAccum(wIndices, nnue::AccumulatorHalf::Upper, accum);

    ArasanV3Feature::OutputType expected[2][ArasanV3Feature::OutputSize];
    for (size_t i = 0; i < ArasanV3Feature::OutputSize; ++i) {
        expected[0][i] = col3[i] + col4[i] + biases[i];
        expected[1][i] = col1[i] + col2[i] + biases[i];
    }
    static const nnue::AccumulatorHalf halves[2] = {nnue::AccumulatorHalf::Lower,
        nnue::AccumulatorHalf::Upper};
    for (auto h : halves) {
        for (size_t i = 0; i < ArasanV3Feature::OutputSize; ++i) {
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
    /*
    ArasanV3Feature::FeatureXformer::PSQWeightType psq_expected[2][nnue::PSQBuckets];
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

    nnue::SqrCReLU<int16_t, ArasanV3Feature::AccumulatorType, uint8_t, 1024, 127, 7> outputTransform;

#ifdef AVX512
    alignas(64) uint8_t out[ArasanV3Feature::OutputSize];
#else
    alignas(nnue::DEFAULT_ALIGN) uint8_t out[ArasanV3Feature::OutputSize];
#endif
    outputTransform.postProcessAccum(accum,out);

    static const uint8_t expected2[ArasanV3Feature::OutputSize] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,6,9,8,11,14,17,16,19,22,25,24,27,30,33,32,35,38,41,40,43,46,49,48,51,54,57,56,59,62,65,64,67,70,73,72,75,78,81,80,83,86,89,88,91,94,97,96,99,102,105,104,107,110,113,112,115,118,121,120,123,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    int tmp_err = errs;
    for (size_t i = 0; i < ArasanV3Feature::OutputSize; ++i) {
        if (out[i] != expected2[i]) ++errs;
    }
    if (errs != tmp_err) {
        std::cerr << "errors in layer 2: SqrCReLU" << std::endl;
    }
    */
    // Test output layer
    nnue::SqrCReLUAndLinear<ArasanV3Feature::AccumulatorType, int16_t, int16_t, int16_t, int32_t,
                            ArasanV3Feature::OutputSize * 2, 255> outputLayer;

    int32_t out, out2;
    outputLayer.postProcessAccum(accum, &out);
    // compare output with generic implementation
    size_t offset = 0;
    int32_t sum = 0;
    for (auto h : halves) {
        for (size_t i = 0; i < accum.getSize(); ++i) {
            int16_t x = accum.getOutput(h)[i];
            // CReLU
            x = std::clamp<int16_t>(x, 0, 255);
            // multiply by weights and keep in 16-bit range
            int16_t product = (x * outputLayer.getCol(0)[i + offset]) & 0xffff;
            // square and sum
            sum += product * x;
        }
        offset += accum.getSize();
    }
    out2 = (sum / nnue::NETWORK_QA) + *(feature.get()->getBiases());
    if (out != out2) {
        std::cerr << "error in output layer" << std::endl;
        std::cerr << out << ' ' << out2 << std::endl;
        ++errs;
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

    ArasanV3Feature feature;

    ArasanV3Feature::AccumulatorType accum;

    // set some weights
    for (size_t i = 0; i < ArasanV3Feature::InputSize; i++) {
        ArasanV3Feature::OutputType col[ArasanV3Feature::OutputSize];
        for (size_t j = 0; j < ArasanV3Feature::OutputSize; j++) {
            col[j] = (i + j) % 10 - 5;
        }
        feature.get()->setCol(i, col);
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

    ArasanV3Feature featureXformer;

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

/*
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

    nnue::CReLU c(CLAMP_MAX);

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
*/

int main(int argc, char **argv) {
    nnue::Network n;

    int errs = 0;
    errs += test_linear<1024,16>();
    errs += test_linear<32,32>();
    errs += test_linear<16,16>();
    errs += test_linear<32,1>();
    errs += test_incremental();
    std::unordered_set<nnue::IndexType> w_expected{199, 195, 321, 10,  137, 17,  30,
                                                   413, 24,  422, 36,  483, 288, 686,
                                                   620, 426, 424, 434, 753, 635};
    std::unordered_set<nnue::IndexType> b_expected{2175, 2171, 2297, 1970, 2097, 1961, 1958,
                                                   1573, 1952, 1566, 1948, 1627, 2200, 1814,
                                                   1748, 1554, 1552, 1546, 1865, 1731};
    errs += testFeature("4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -", w_expected,
                         b_expected);
    std::unordered_set<nnue::IndexType> w_expected2{199, 194, 321, 269, 11,  10,  137, 8,   22,
                                                    82,  17,  154, 422, 421, 548, 686, 427, 424,
                                                    439, 500, 434, 433, 639, 763, 570, 632};
    std::unordered_set<nnue::IndexType> b_expected2{
        1407, 1402, 1529, 1461, 1203, 1202, 1329, 1200, 1198, 1258, 1193, 1314, 798,
        797,  924,  1046, 787,  784,  783,  844,  778,  777,  967,  1091, 898,  960};
    errs += testFeature("r3kb1r/p2n1pp1/1q2p2p/1ppb4/5B2/1P3NP1/2Q1PPBP/R4RK1 w kq -", w_expected2, b_expected2);

    //    errs += test_CReLU<16>();
    //    errs += test_CReLU<32>();
    std::cerr << errs << " errors" << std::endl;
    std::string fname, fen;
    for (int arg = 1; arg < argc; ++arg) {
        if (*argv[arg] == '-') {
            char c = argv[arg][1];
            switch (c) {
            case 'e':
                if (++arg < argc) {
                    fen = argv[arg];
                }
                break;
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
            std::cerr << "error loading " << fname << std::endl << ":" << strerror(errno) << std::flush;
        }
        else {
            if (fen.empty()) fen = "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";
            Position p(fen);
            ChessInterface intf(&p);
            std::cout << "evaluating: " << fen << std::endl;
            int val = nnue::Evaluator<ChessInterface>::fullEvaluate(n,intf);
            std::cout << "value=" << val << std::endl;
        }
    }
    return 0;
}
