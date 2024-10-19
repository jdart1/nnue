// Copyright 2021-2024 by Jon Dart. All Rights Reserved.
#include "nnue.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../interface/chessint.h"

// Unit tests for nnue code

template <size_t ROWS, size_t COLS> static int test_linear() {
    int errs = 0, tmp;

    using InputType = uint8_t;
    using WeightType = int8_t;
    using BiasType = int32_t;
    using OutputType = int32_t;

    // assume 1 output bucket
    constexpr size_t BUCKETS = 1;

    // serializer assumes rows are at least 32 bytes
    static constexpr size_t ROUNDED_ROWS = std::max<size_t>(ROWS, 32);

    BiasType biases[BUCKETS][COLS];
    WeightType weights[BUCKETS][COLS][ROUNDED_ROWS]; // indexed first by output

    for (size_t i = 0; i < COLS; i++) {
        for (size_t j = 0; j < ROUNDED_ROWS; j++) {
            weights[0][i][j] = ((i + j) % 20) - 10;
        }
    }
    for (size_t i = 0; i < COLS; i++) {
        biases[0][i] = (i % 15) + i - 10;
    }

    nnue::LinearLayer<InputType, WeightType, BiasType, OutputType, ROWS, COLS, BUCKETS> layer;

    layer.setBiases(0,biases[0]);
    for (size_t i = 0; i < COLS; ++i) layer.setCol(0,i,weights[0][i]);

    /* TBD

#if defined(__MINGW32__) || defined(__MINGW64__) || (defined(__APPLE__) && defined(__MACH__))
    std::string tmp_name("XXXXXX");
#else
    std::string tmp_name(std::tmpnam(nullptr));
#endif

    std::ofstream outfile(tmp_name, std::ios::binary | std::ios::trunc);
    layer.write(outfile,weights,biases);
    if (outfile.fail()) {
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

    if (infile.fail()) {
        std::cerr << "error reading linear layer" << std::endl;
        ++errs;
        infile.close();
        std::remove(tmp_name.c_str());
        return errs;
    }
    infile.close();

    // verify layer was read
    tmp = errs;
    for (size_t i = 0; i < COLS; i++) {
        errs += (layer.getBiases(0)[i] != biases[0][i]);
        if (layer.getBiases(0)[i] != biases[0][i])
            std::cerr << layer.getBiases(0)[i] << ' ' << biases[0][i] << std::endl;
    }
    for (size_t i = 0; i < COLS; i++) {
        // get weights for output column
        const WeightType *col = layer.getCol(0,i);
        for (size_t j = 0; j < ROWS; j++) {
            errs += (weights[0][i][j] != col[j]);
        }
    }
    if (errs - tmp > 0)
        std::cerr << "errors deserializing linear layer" << std::endl;
    */

    alignas(nnue::DEFAULT_ALIGN) InputType inputs[ROWS];
    for (unsigned i = 0; i < ROWS; i++) {
        inputs[i] = std::min<int>(127, static_cast<InputType>(i + 1));
    }

    alignas(nnue::DEFAULT_ALIGN) OutputType output[COLS], computed[COLS];
    // test linear layer propagation
    layer.forward(0, inputs, output);
    for (size_t i = 0; i < COLS; i++) {
        computed[i] = static_cast<OutputType>(biases[0][i]);
    }
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            computed[j] += inputs[i] * weights[0][j][i];
        }
    }

    tmp = errs;
    for (size_t i = 0; i < COLS; i++) {
        errs += computed[i] != output[i];
    }
    if (errs - tmp > 0)
        std::cerr << "errors computing dot product " << ROWS << "x" << COLS << std::endl;
    // TBD    std::remove(tmp_name.c_str());
    return errs;
}

static const std::unordered_map<char, nnue::Piece> pieceMap = {
    {'p', nnue::BlackPawn}, {'n', nnue::BlackKnight}, {'b', nnue::BlackBishop},
    {'r', nnue::BlackRook}, {'q', nnue::BlackQueen},  {'k', nnue::BlackKing},
    {'P', nnue::WhitePawn}, {'N', nnue::WhiteKnight}, {'B', nnue::WhiteBishop},
    {'R', nnue::WhiteRook}, {'Q', nnue::WhiteQueen},  {'K', nnue::WhiteKing}};

// wrapper around nnue::ArasanV3Feature, sets up that class with some fixed
// parameters
class ArasanV3Feature {
  public:
    static constexpr size_t OutputSize = 1024;

    static constexpr size_t InputSize = 22 * OutputSize;

    using OutputType = int16_t;

    using FeatureXformer =
        nnue::ArasanV3Feature<uint16_t, int16_t, int16_t, int16_t, InputSize, OutputSize>;

    using AccumulatorType = FeatureXformer::AccumulatorType;

    ArasanV3Feature() : layer1(new FeatureXformer()) {}

    AccumulatorType accum;

    void init(unsigned index, const OutputType vals[]) { layer1.get()->setCol(index, vals); }

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
    Position p(fen);
    ChessInterface intf(&p);

    nnue::IndexArray wIndices, bIndices;
    auto wCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(intf, wIndices);
    auto bCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(intf, bIndices);

    int errs = 0;
    for (auto it = wIndices.begin(); it != wIndices.begin() + wCount; it++) {
        if (w_expected.find(*it) == w_expected.end()) {
            ++errs;
            std::cerr << "error: white index " << *it << " not expected" << std::endl;
        } else
            w_expected.erase(*it);
    }
    for (auto it = bIndices.begin(); it != bIndices.begin() + bCount; it++) {
        if (b_expected.find(*it) == b_expected.end()) {
            ++errs;
            std::cerr << "error: black index " << *it << " not expected" << std::endl;
        } else
            b_expected.erase(*it);
    }
    if (!w_expected.empty()) {
        ++errs;
        std::cerr << "error: not all expected White indices found, missing " << std::endl;
        for (auto i : w_expected) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }
    if (!b_expected.empty()) {
        ++errs;
        std::cerr << "error: not all expected Black indices found, missing " << std::endl;
        for (auto i : b_expected) {
            std::cerr << i << ' ';
        }
        std::cerr << std::endl;
    }

    ArasanV3Feature feature;

    ArasanV3Feature::AccumulatorType accum;

    for (size_t i = 0; i < ArasanV3Feature::InputSize; ++i)
        feature.get()->setCol(i, zero_col);

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
                std::cerr << " error at accum index " << int(i) << " expected: " << int(exp)
                          << " actual " << accum.getOutput(h)[i] << std::endl;
            }
        }
    }

    // Test output layer
    nnue::SqrCReLUAndLinear<ArasanV3Feature::AccumulatorType, int16_t, int16_t, int16_t, int32_t,
                            ArasanV3Feature::OutputSize * 2, 255, 255, 1, true>
        outputLayer;

    int32_t out, out2;
    outputLayer.postProcessAccum(accum, 0, &out);
    // compare output with generic implementation
    size_t offset = 0;
    int32_t sum = 0;
    for (auto h : halves) {
        for (size_t i = 0; i < accum.getSize(); ++i) {
            int16_t x = accum.getOutput(h)[i];
            // CReLU
            x = std::clamp<int16_t>(x, 0, 255);
            // multiply with saturation then square
            sum += ((outputLayer.getCol(0,0)[i + offset] * x) & 0xffff) * x;
        }
        offset += accum.getSize();
    }
    out2 = (sum / 255) + *(feature.get()->getBiases());
    if (out != out2) {
        std::cerr << "error in output layer" << std::endl;
        std::cerr << out << ' ' << out2 << std::endl;
        ++errs;
    }

    return errs;
}

template <nnue::Color c>
static void getIndices(const ChessInterface &ci, std::set<nnue::IndexType> &indices) {
    nnue::IndexArray a;
    nnue::Evaluator<ChessInterface>::getIndices<c>(ci, a);
    for (auto it = a.begin(); it != a.end() && *it != nnue::LAST_INDEX; it++) {
        indices.insert(*it);
    }
}

static int test_incr(const nnue::Network &network, int casenum, ChessInterface &ciSource,
                     ChessInterface &ciTarget) {
    int errs = 0;
    assert(ciSource != ciTarget);

    nnue::Evaluator<ChessInterface> evaluator;

    // do full calculation for target
    nnue::Network::AccumulatorType fullEvalAccum;
    nnue::IndexArray wIndices, bIndices;
    nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(ciTarget, wIndices);
    nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(ciTarget, bIndices);
    nnue::Evaluator<ChessInterface>::updateAccum(network, ciTarget, fullEvalAccum);

    // do incremental update if possible
    ciTarget.getAccumulator().setEmpty();
    evaluator.updateAccum(network, ciTarget, nnue::White);
    evaluator.updateAccum(network, ciTarget, nnue::Black);

    int tmp = errs;
    errs += ciTarget.getAccumulator() != fullEvalAccum;
    if (errs - tmp) {
        std::cout << "test_incr: error in case " << casenum
                  << " standard/incremental update differs" << std::endl;
    }

    return errs;
}

static int test_incremental() {

    struct ChangeRecord {
        // fen is position after the change described in the DirtyState(s)
        ChangeRecord(const std::string &f, const std::vector<DirtyState> &c) : fen(f) {
            std::copy(c.begin(), c.end(), std::back_inserter(changes));
        }
        std::string fen;
        std::vector<DirtyState> changes;
    };

    struct Case {
        Case() = default;

        Case(std::initializer_list<ChangeRecord> x) {
            Position *prev = nullptr;
            for (const auto &cr : x) {
                Position *pos = new Position(cr.fen);
                pos->previous = prev;
                for (const auto &it : cr.changes) {
                    pos->dirty[pos->dirty_num++] = it;
                }
                pos_list.push_back(pos);
                prev = pos;
            }
        }
        virtual ~Case() {
            auto it = pos_list.end();
            while (it != pos_list.begin()) {
                --it;
                Position *p = *it;
                it = pos_list.erase(it);
                delete p;
            }
        }

        std::vector<Position *> pos_list;
    };

    const std::array<Case, 5> cases = {
        Case{ChangeRecord("r1bqk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ1BPPP/"
                          "R1B1K2R b KQkq -",
                          {}),
             ChangeRecord("r1bqk2r/pp1n1ppp/2pbpn2/8/2pP4/2N1PN2/PPQ1BPPP/R1B1K2R w KQkq -",
                          {DirtyState(35 /*D5*/, 26 /*C4*/, nnue::BlackPawn),
                           DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::WhitePawn)})},
        Case{ChangeRecord("r1bqk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ1BPPP/"
                          "R1B1K2R b KQkq -",
                          {}),
             ChangeRecord("r1bqk2r/pp1n1ppp/2pbpn2/8/2pP4/2N1PN2/PPQ1BPPP/R1B1K2R w KQkq -",
                          {DirtyState(35 /*D5*/, 26 /*C4*/, nnue::BlackPawn),
                           DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::WhitePawn)}),
             ChangeRecord("r1bqk2r/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PPQ2PPP/R1B1K2R b KQkq -",
                          {DirtyState(12 /*E2*/, 26 /*C4*/, nnue::WhiteBishop),
                           DirtyState(26 /*C4*/, nnue::InvalidSquare, nnue::BlackPawn)})},
        Case{ChangeRecord("8/6k1/7p/3N1P2/3K4/2P5/8/4n3 b - -", {}),
             ChangeRecord("8/6k1/7p/3N1P2/3K4/2P2n2/8/8 w - -",
                          {DirtyState(4 /*E1*/, 21 /*F3*/, nnue::BlackKnight)}),
             ChangeRecord("8/6k1/7p/3N1P2/4K3/2P5/8/4n3 b - -",
                          {DirtyState(27 /*D4*/, 28 /*E4*/, nnue::WhiteKing)}),
             ChangeRecord("8/6k1/7p/3N1P2/4K3/2P5/3n4/8 w - -",
                          {DirtyState(21 /*F3*/, 11 /*D2*/, nnue::BlackKnight)}),
             ChangeRecord("8/6k1/7p/3N1P2/8/2P1K3/3n4/8 b - -",
                          {DirtyState(21 /*E4*/, 11 /*E3*/, nnue::WhiteKing)})},
        Case{ChangeRecord("5r1k/3Q2pp/p7/4p1B1/P1Q5/8/5nPP/1q4K1 w - -", {}),
             ChangeRecord("5r1k/3Q2pp/p7/4p1B1/P7/8/5nPP/1q3QK1 b - -",
                          {DirtyState(26 /*C4*/, 5 /*F1*/, nnue::WhiteQueen)}),
             ChangeRecord("5r1k/3Q2pp/p7/4p1B1/P7/8/5nPP/5qK1 w - -",
                          {DirtyState(1 /*B1*/, 5 /*F1*/, nnue::BlackQueen),
                           DirtyState(5 /*F1*/, nnue::InvalidSquare, nnue::WhiteQueen)}),
             ChangeRecord("5r1k/3Q2pp/p7/4p1B1/P7/8/5nPP/5qK1 w - -",
                          {DirtyState(26 /*C4*/, 5 /*F1*/, nnue::WhiteQueen)}),
             ChangeRecord("5r1k/3Q2pp/p7/4p1B1/P7/8/5nPP/5qK1 w - -",
                          {DirtyState(6 /*G1*/, 5 /*F1*/, nnue::BlackKing)})},
        Case{ChangeRecord("2Q2r1k/3n2p1/p6p/4p1B1/P1Q5/2N5/2q2nPP/R5K1 w - -", {}),
             ChangeRecord("5Q1k/3n2p1/p6p/4p1B1/P1Q5/2N5/2q2nPP/R5K1 b - -",
                          {DirtyState(58 /*C8*/, 60 /*F8*/, nnue::WhiteQueen),
                           DirtyState(60 /*F8*/, nnue::InvalidSquare, nnue::BlackRook)}),
             ChangeRecord("5Q2/3n2pk/p6p/4p1B1/P1Q5/2N5/2q2nPP/R5K1 w - -",
                          {DirtyState(63 /*H8*/, 55 /*H7*/, nnue::BlackKing)})}};

    nnue::Network network;

    // set some weights
    for (size_t i = 0; i < nnue::Network::FeatureXformerRows; i++) {
        ArasanV3Feature::OutputType col[nnue::Network::FeatureXformerOutputSize];
        for (size_t j = 0; j < nnue::Network::FeatureXformerOutputSize; j++) {
            col[j] = (i + j) % 10 - 5;
        }
        network.getTransformer()->setCol(i, col);
    }

    int errs = 0;

    int i = 0;
    for (const auto &c : cases) {
        Position *source_pos = c.pos_list.front();
        ChessInterface ciSource(source_pos);
        Position *target_pos = c.pos_list.back();
        ChessInterface ciTarget(target_pos);
        errs += test_incr(network, i++, ciSource, ciTarget);
    }
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
        output[i] = static_cast<OutputType>(std::clamp<InputType>(input[i] >>
RSHIFT,0,CLAMP_MAX));
    }
    std::memset(output2,'\0',size*sizeof(OutputType));
    c.doForward(input,output2);
    for (unsigned i = 0; i < size; i++) {
        errs += output2[i] != output[i];
    }
    if (errs) std::cerr << errs << " error(s) in scale and clamp function" <<
std::endl; return errs;
}
*/

int main(int argc, char **argv) {
    nnue::Network n;

    int errs = 0;
    errs += test_linear<1024, 16>();
    errs += test_linear<32, 32>();
    errs += test_linear<16, 16>();
    errs += test_linear<32, 1>();
    errs += test_incremental();
    std::unordered_set<nnue::IndexType> w_expected{199,195,321,10,137,17,30,413,24,422,36,483,288,686,620,426,424,434,753,635};

    std::unordered_set<nnue::IndexType> b_expected{2175,2171,2297,1970,2097,1961,1958,1573,1952,1566,1948,1627,2200,1814,1748,1554,1552,1546,1865,1731};

    errs += testFeature("4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -", w_expected,
                        b_expected);
    std::unordered_set<nnue::IndexType> w_expected2{
        199,194,321,269,11,10,137,8,22,82,17,154,422,421,548,686,427,424,439,500,434,433,639,763,570,632};
    std::unordered_set<nnue::IndexType> b_expected2{
        1407,1402,1529,1461,1203,1202,1329,1200,1198,1258,1193,1314,798,797,924,1046,787,784,783,844,778,777,967,1091,898,960};

    errs += testFeature("r3kb1r/p2n1pp1/1q2p2p/1ppb4/5B2/1P3NP1/2Q1PPBP/R4RK1 w kq -", w_expected2,
                        b_expected2);

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
            std::cerr << "error loading " << fname << std::endl
                      << ":" << strerror(errno) << std::flush;
        } else {
            if (fen.empty())
                fen = "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";
            Position p(fen);
            ChessInterface intf(&p);
            std::cout << "evaluating: " << fen << std::endl;
            int val = nnue::Evaluator<ChessInterface>::fullEvaluate(n, intf);
            std::cout << "value=" << val << std::endl;
        }
    }
    return 0;
}
