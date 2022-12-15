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
      std::cerr << "error writing temp file" << std::endl;
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

// wrapper around nnue::HalfKp, sets up that class with some fixed parameters
class HalfKp {
  public:
    static constexpr size_t OutputSize = 256;

    static constexpr size_t InputSize = 64 * (10 * 64 + 1);

    using OutputType = int16_t;

    // test propagation
    using Layer1 =
        nnue::HalfKp<uint16_t, int16_t, int16_t, int16_t, InputSize, OutputSize>;

    using AccumulatorType = Layer1::AccumulatorType;

    HalfKp() : layer1(new Layer1()) {}

    AccumulatorType accum;

    void init(unsigned index, const OutputType vals[]) {
        layer1.get()->setCol(index, vals);
    }

    Layer1 *get() const noexcept { return layer1.get(); }

  private:
    std::unique_ptr<Layer1> layer1;
};

static int16_t col1[HalfKp::OutputSize];
static int16_t col2[HalfKp::OutputSize];
static int16_t col3[HalfKp::OutputSize];
static int16_t col4[HalfKp::OutputSize];

static int test_halfkp() {
    const std::string fen =
        "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";

    unsigned seed1 =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed1);
    std::uniform_int_distribution<int16_t> dist(-1000, 1000);
    for (size_t i = 0; i < HalfKp::OutputSize; i++) {
        col1[i] = dist(gen);
        col2[i] = dist(gen);
        col3[i] = dist(gen);
        col4[i] = dist(gen);
    }

    std::unordered_set<nnue::IndexType> w_expected{
        4231, 4235, 3860, 4117, 3869, 3872, 3937, 3878, 3944,
        3882, 4075, 4398, 4464, 4338, 3956, 3958, 3964, 4355};

    std::unordered_set<nnue::IndexType> b_expected{
        6281, 6277, 5884, 6139, 5875, 5872, 5807, 5866, 5800,
        5862, 5925, 6370, 6304, 6174, 5788, 5786, 5780, 6157};

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

    HalfKp halfKp;

    HalfKp::AccumulatorType accum;

    halfKp.get()->setCol(4231, col1);
    halfKp.get()->setCol(3882, col2);
    halfKp.get()->setCol(5862, col3);
    halfKp.get()->setCol(5925, col4);

    halfKp.get()->setCol(4231, col1);
    halfKp.get()->setCol(3882, col2);
    halfKp.get()->setCol(5862, col3);
    halfKp.get()->setCol(5925, col4);

    halfKp.get()->updateAccum(bIndices, nnue::AccumulatorHalf::Lower, accum);
    halfKp.get()->updateAccum(wIndices, nnue::AccumulatorHalf::Upper, accum);

    HalfKp::OutputType expected[HalfKp::OutputSize * 2];
    for (size_t i = 0; i < HalfKp::OutputSize; ++i) {
        expected[i] = col3[i] + col4[i];
    }
    for (size_t i = HalfKp::OutputSize; i < HalfKp::OutputSize * 2; ++i) {
        expected[i] =
            col1[i - HalfKp::OutputSize] + col2[i - HalfKp::OutputSize];
    }
    for (size_t i = 0; i < HalfKp::OutputSize * 2; ++i) {
        if (expected[i] != accum.getOutput()[i]) {
            ++errs;
            std::cerr << " error at index " << int(i)
                      << " expected: " << int(expected[i]) << " actual "
                      << accum.getOutput()[i] << std::endl;
        }
    }
    return errs;
}

static int accumCompare(const HalfKp::AccumulatorType &accum1,
                        const HalfKp::AccumulatorType &accum2) {
    const HalfKp::OutputType *p = accum1.getOutput();
    const HalfKp::OutputType *q = accum2.getOutput();
    int errs = 0;
    for (size_t i = 0; i < 2 * HalfKp::OutputSize; i++) {
        if (*p++ != *q++) {
            ++errs;
        }
    }
    if (errs)
        std::cout << "incremental/regular eval mismatch" << std::endl;
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

    HalfKp halfKp;

    HalfKp::AccumulatorType accum;

    for (size_t i = 0; i < HalfKp::InputSize; i++) {
        HalfKp::OutputType col[HalfKp::OutputSize];
        for (size_t j = 0; j < HalfKp::OutputSize; j++) {
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
    errs += accumCompare(ciTarget.getAccumulator(), accum);

    // Reset target accumulator state
    ciTarget.getAccumulator().setEmpty();

    // use the API that takes incremental or regular path
    evaluator.updateAccum(network, ciTarget, nnue::White);
    evaluator.updateAccum(network, ciTarget, nnue::Black);

    // Compare results again
    errs += accumCompare(ciTarget.getAccumulator(), accum);
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

    HalfKp halfKp;

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

template <size_t size>
static int test_scale_and_clamp() {
    int errs = 0;
    constexpr int CLAMP_MAX = 127;
    constexpr int RSHIFT = 6;
    using InputType = int32_t;
    using OutputType = uint8_t;
    using ScaleAndClamper = nnue::ScaleAndClamp<InputType, OutputType, size, RSHIFT>;

    alignas(nnue::DEFAULT_ALIGN) InputType input[size];
    alignas(nnue::DEFAULT_ALIGN) OutputType output[size],output2[size];

    ScaleAndClamper c(CLAMP_MAX);

    for (unsigned i = 0; i < size; i++) {
        input[i] = -8000 + 900*std::min<unsigned>(i,10) + i*10;
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
    errs += test_linear<512,32>();
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

    return 0;
}
