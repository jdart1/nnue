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
  static constexpr size_t OutputSize = 256;

  static constexpr size_t InputSize = 64 * (10 * 64 + 1);

  using OutputType = int16_t;

  // test propagation
  using Layer1 = nnue::HalfKp<uint8_t, int16_t, int16_t, int16_t,
                              InputSize, OutputSize>;

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

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed1);
  std::uniform_int_distribution<int16_t> dist(-1000, 1000);
  for (size_t i = 0; i < HalfKp::OutputSize; i++) {
      col1[i] = dist(gen);
      col2[i] = dist(gen);
      col3[i] = dist(gen);
      col4[i] = dist(gen);
  }

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

  HalfKp halfKp;

  HalfKp::AccumulatorType accum;

  for (size_t i = 0; i < HalfKp::InputSize; i++) {
      HalfKp::OutputType col[HalfKp::OutputSize];
      for (size_t j = 0; j < HalfKp::OutputSize; j++) {
          col[j] = (i+j) % 10 - 5;
      }
      halfKp.get()->setCol(i, col);
  }

  // Full evaluation of 1st layer for target position
  std::cout << "updating accum for White" << std::endl;
  evaluator.updateAccum(network,wIndices,nnue::White,ciTarget.sideToMove(),
                        accum);
  std::cout << "updating accum for Black" << std::endl;
  evaluator.updateAccum(network,bIndices,nnue::Black,ciTarget.sideToMove(),
                        accum);
  
  // Incremental evaluation of 1st layer, starting from source position  
  evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::White);
  evaluator.updateAccumIncremental(network, ciSource, ciTarget, nnue::Black);

  // Compare evals
  const HalfKp::OutputType *p = ciTarget.getAccumulator().getOutput();
  const HalfKp::OutputType *q = accum.getOutput();
  int tmp = errs;
  for (size_t i = 0; i < 2*HalfKp::OutputSize; i++) {
      if (*p++ != *q++) {
          ++errs;
      }
  }
  if (errs-tmp > 0) std::cout << "incremental/regular eval mismatch" << std::endl;
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
