// Copyright 2021 by Jon Dart. All Rights Reserved.
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

static int test_linear() {
  int errs = 0;

  static constexpr size_t ROWS = 32, COLS = 32;

  static int16_t biases[COLS];
  static int16_t weights[ROWS][COLS];

  using WeightType = int16_t;
  using OutputType = int16_t;

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed1);
  std::uniform_int_distribution<int16_t> dist(-1000, 1000);

  constexpr size_t bufSize = COLS + ROWS * COLS;
  auto buf = std::unique_ptr<int16_t[]>(new WeightType[bufSize]);

  int16_t *b = buf.get();
  for (size_t i = 0; i < COLS; i++) {
    *b++ = biases[i] = dist(gen);
  }
  for (size_t i = 0; i < COLS; i++) {
    for (size_t j = 0; j < ROWS; j++) {
      *b++ = weights[i][j] = dist(gen);
    }
  }

  nnue::LinearLayer<int8_t, WeightType, WeightType, OutputType, 32, 32> layer;

  std::string tmp_name(std::tmpnam(nullptr));

  std::ofstream outfile(tmp_name, std::ios::binary);
  outfile.write(reinterpret_cast<char *>(buf.get()), bufSize * sizeof(int16_t));
  outfile.close();

  std::ifstream infile(tmp_name, std::ios::binary);

  // test reading a layer
  layer.read(infile);

  if (infile.bad()) {
    std::cerr << "error reading linear layer" << std::endl;
    ++errs;
    return errs;
  }

  // verify layer was read
  int tmp = errs;
  for (size_t i = 0; i < COLS; i++) {
    errs += (layer.getBiases()[i] != biases[i]);
    const WeightType *col = layer.getCol(i);
    for (size_t j = 0; j < ROWS; j++) {
      errs += (weights[i][j] != col[j]);
      if (weights[i][j] != col[j])
        std::cout << i << ' ' << j << ' ' << weights[i][j] << ' ' << col[j]
                  << std::endl;
    }
  }
  if (errs - tmp > 0)
    std::cerr << "errors deserializing linear layer" << std::endl;

  constexpr int8_t inputs[ROWS] = {0, 0, 0, 0, 8, 0,  0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, -8, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,  0, 0, 0, 0};

  OutputType output[COLS], computed[COLS];

  // test linear layer propagation
  layer.forward(inputs, output);
  memcpy(computed, biases, sizeof(WeightType) * COLS);
  for (size_t i = 0; i < COLS; i++) {
    for (size_t j = 0; j < ROWS; j++) {
      computed[i] += inputs[j] * weights[i][j];
    }
  }
  tmp = errs;
  for (size_t i = 0; i < COLS; i++) {
    errs += computed[i] != output[i];
  }
  if (errs - tmp > 0)
    std::cerr << "errors computing dot product" << std::endl;
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
  static constexpr size_t OutputSize = 64;

  using OutputType = int16_t;

  // test propagation
  using Layer1 = nnue::HalfKp<uint8_t, int16_t, int16_t, int16_t,
                              64 * (10 * 64 + 1), OutputSize>;

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

static constexpr int16_t col1[HalfKp::OutputSize] = {
    -153, 454,  -407, -397, -431, -257, -438, -276, 297,  -307, -326,
    427,  -14,  251,  -29,  -186, -388, 194,  -428, 504,  283,  -24,
    -25,  397,  374,  -166, -210, 99,   51,   415,  -311, 410,  358,
    -206, -499, 439,  49,   -424, -349, -166, -219, -163, -251, 279,
    -424, 233,  -418, -300, -85,  -334, -307, -314, 155,  180,  -428,
    17,   -498, -126, -396, 65,   -223, -195, -36,  -377};

static constexpr int16_t col2[HalfKp::OutputSize] = {
    -53,  145,  -337, 232,  86,   311,  76,   -270, 199,  -417, 502,
    -30,  334,  -174, 293,  -242, -448, -30,  -101, 434,  118,  325,
    -418, -373, 124,  -316, 466,  -279, -23,  -89,  -471, 436,  -455,
    -295, 156,  143,  -496, -279, 385,  215,  -184, 375,  -327, -361,
    -311, 478,  -91,  -247, -64,  320,  -325, -458, 134,  -231, -318,
    -254, -34,  -364, -21,  455,  60,   20,   379,  117};

static constexpr int16_t col3[HalfKp::OutputSize] = {
    23,   -252, 253,  256,  -378, -44,  -439, 509,  141,  -288, -313,
    107,  133,  -48,  -469, -59,  140,  -414, -437, 421,  -220, -179,
    -125, -72,  312,  -182, 500,  -179, -315, 105,  -454, 221,  366,
    312,  -35,  500,  -244, 38,   497,  410,  262,  -328, 5,    -117,
    137,  49,   337,  -235, 147,  412,  -326, 439,  -278, 62,   -145,
    -478, 392,  -156, -145, -434, 461,  -86,  299,  315};

static constexpr int16_t col4[HalfKp::OutputSize] = {
    286,  -248, 303,  -18,  303,  289,  -120, 53,   473,  398, 449,  98,   -65,
    274,  375,  -430, 174,  -462, -503, 408,  112,  -136, 443, -8,   220,  -214,
    70,   170,  212,  -143, -27,  438,  122,  -235, -91,  -87, -458, 301,  478,
    -497, 187,  415,  114,  -390, 177,  -23,  -308, -160, 27,  -299, -264, -373,
    78,   -333, 132,  -214, -34,  -310, 468,  -334, 60,   -70, -407, -330};

static int test_halfkp() {
  const std::string fen =
      "4r3/5pk1/1q1r1p1p/1p1Pn2Q/1Pp4P/6P1/5PB1/R3R1K1 b - -";

  std::unordered_set<nnue::IndexType> w_expected{4231, 4235, 3860, 4117, 3869, 3872,
                                          3937, 3878, 3944, 3882, 4075, 4398,
                                          4464, 4338, 3956, 3958, 3964, 4355};

  std::unordered_set<nnue::IndexType> b_expected{6281, 6277, 5884, 6139, 5875, 5872,
                                          5807, 5866, 5800, 5862, 5925, 6370,
                                          6304, 6174, 5788, 5786, 5780, 6157};

  Position p(fen);
  ChessInterface intf(&p);

  nnue::IndexArray wIndices, bIndices;
  auto wCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(intf, wIndices);
  auto bCount = nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(intf, bIndices);

  int errs = 0;
  for (auto it = wIndices.begin(); it != wIndices.begin()+wCount; it++) {
    if (w_expected.find(*it) == w_expected.end()) {
      ++errs;
      std::cerr << "error: white index " << *it << " not expected"
                << std::endl;
    } else
      w_expected.erase(*it);
  }
  for (auto it = bIndices.begin(); it != bIndices.begin()+bCount; it++) {
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
    expected[i] = col1[i - HalfKp::OutputSize] + col2[i - HalfKp::OutputSize];
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

static int test_incr(ChessInterface &ciSource, ChessInterface &ciTarget) 
{
    std::cout << "test_incr" << std::endl;
  int errs = 0;

  nnue::IndexArray bIndices,wIndices;
  
  std::set<nnue::IndexType> base, target;
  
  nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(ciSource, wIndices);
  nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(ciSource, bIndices);
  for (auto it = wIndices.begin(); it != wIndices.end() && *it != nnue::LAST_INDEX; it++) {
      std::cout << *it << ' ';
      base.insert(*it);
  }
  for (auto it = bIndices.begin(); it != bIndices.end() && *it != nnue::LAST_INDEX; it++) {
      std::cout << *it << ' ';
      base.insert(*it);
  }
  std::cout << std::endl;

  nnue::Evaluator<ChessInterface>::getIndices<nnue::White>(ciTarget, wIndices);
  nnue::Evaluator<ChessInterface>::getIndices<nnue::Black>(ciTarget, bIndices);
  for (auto it = wIndices.begin(); it != wIndices.end() && *it != nnue::LAST_INDEX; it++) {
      std::cout << *it << ' ';
      target.insert(*it);
  }
  for (auto it = bIndices.begin(); it != bIndices.end() && *it != nnue::LAST_INDEX; it++) {
      std::cout << *it << ' ';
      target.insert(*it);
  }
  std::cout << std::endl;

  std::vector<nnue::IndexType> added(32), removed(32);

  auto itend = std::set_difference(base.begin(), base.end(), target.begin(),
                                   target.end(), removed.begin());
  removed.resize(itend-removed.begin());                            
  itend = std::set_difference(target.begin(), target.end(), base.begin(),
                              base.end(), added.begin());
  added.resize(itend-added.begin());

  nnue::Evaluator<ChessInterface> evaluator;

  nnue::Network network;

  // evaluate source accumulator (full evaluation)
//  evaluator.updateAccum(network, wIndices, nnue::White, ciSource.sideToMove(), ciSource.getAccumulator());
//  evaluator.updateAccum(network, bIndices, nnue::Black, ciSource.sideToMove(), ciSource.getAccumulator());

  // have Evaluator calculate index diffs
  nnue::IndexArray wRemoved, wAdded, bRemoved, bAdded;
  evaluator.getIndexDiffs(ciSource,ciTarget,nnue::White,wRemoved,wAdded);
  evaluator.getIndexDiffs(ciSource,ciTarget,nnue::Black,bRemoved,bAdded);
  
  // diffs
  std::set<nnue::IndexType> removedAll, addedAll;                            
  for (auto it = wRemoved.begin(); it != wRemoved.end() && *it != nnue::LAST_INDEX; it++)
      removedAll.insert(*it);
  for (auto it = bRemoved.begin(); it != bRemoved.end() && *it != nnue::LAST_INDEX; it++)
      removedAll.insert(*it);
  for (auto it = wAdded.begin(); it != wAdded.end() && *it != nnue::LAST_INDEX; it++)
      addedAll.insert(*it);
  for (auto it = bAdded.begin(); it != bAdded.end() && *it != nnue::LAST_INDEX; it++)
      addedAll.insert(*it);

  std::vector<nnue::IndexType> diffs(32);
  itend = std::set_difference(removedAll.begin(), removedAll.end(), removed.begin(),removed.end(), diffs.begin());
  diffs.resize(itend-diffs.begin());
  if (diffs.size() != 0) {
      ++errs;
      std::cerr << "removed list differs" << std::endl;
  }
  diffs.clear();
  std::cout << std::endl;
  itend = std::set_difference(addedAll.begin(), addedAll.end(), added.begin(),added.end(), diffs.begin());
  diffs.resize(itend-diffs.begin());
  if (diffs.size() != 0) {
      ++errs;
      std::cerr << "added list differs" << std::endl;
  }

//  evaluator.updateAccumIncremental(network, ciSource, ciTtarget1, White);
//  evaluator.updateAccumIncremental(network, ciSource, ciTtarget1, Black);

  return errs;
}

static int test_incremental() {
  constexpr auto source_fen = "r1bqk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ1BPPP/R1B1K2R b KQkq -";

  constexpr auto target_fen = "r1bqk2r/pp1n1ppp/2pbpn2/8/2pP4/2N1PN2/PPQ1BPPP/R1B1K2R w KQkq -";

  constexpr auto target_fen2 = "r1bqk2r/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PPQ2PPP/R1B1K2R b KQkq -";

  int errs = 0;

  HalfKp halfKp;

  Position source_pos(source_fen);
  ChessInterface ciSource(&source_pos);
  Position target_pos(target_fen);
  ChessInterface ciTarget(&target_pos);
  // set up dirty status
  // d5 pawn x c4 pawn
  source_pos.dirtyState.dirty[source_pos.dirtyState.dirty_num++] = nnue::DirtyDetails(35, 26, nnue::BlackPawn);
  source_pos.dirtyState.dirty[source_pos.dirtyState.dirty_num++] =
      nnue::DirtyDetails(26, nnue::InvalidSquare, nnue::WhitePawn);
  // connect target to previous position
  target_pos.previous = &source_pos;
  
  errs += test_incr(ciSource, ciTarget);
  return errs;
}

int main(int argc, char **argv) {
  nnue::Network n;

  int errs = 0;
  errs += test_linear();
  errs += test_halfkp();
  errs += test_incremental();
  std::cerr << errs << " errors(s)" << std::endl;

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
