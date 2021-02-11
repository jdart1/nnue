// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _CHESS_INTERFACE_H
#define _CHESS_INTERFACE_H

// Clients expected to supply a typedef equivalent to the Position
// class and an implementation of the ChessInterface class, which is
// a wrapper over the Position type. This file contains a reference
// implementation used for testing.

// We'd really like to use something like a Java interface here but
// C++ doesn't have that, exactly. The closest thing is C++20
// concepts, which are used if available.

#include "nnue.h"

struct Position 
{
    // create a position from a FEN string
    Position(const std::string &fen);
            
    virtual ~Position() = default;

    Position *previous;
    nnue::Color stm;
    nnue::Square kings[2];
    std::unordered_map<nnue::Square, nnue::Piece> locs;
    nnue::Network::AccumulatorType accum;
    nnue::DirtyState dirtyState; // info on previous move
};

// Wrapper over the Position class that exposes a standard interface
class ChessInterface
{

public:
    ChessInterface(Position *p) : pos(p) {
    }

    ChessInterface(const ChessInterface &intf) : pos(intf.pos) {
    }

    virtual ~ChessInterface() = default;

    friend bool operator == (const ChessInterface &intf, const ChessInterface &other );

    friend bool operator != (const ChessInterface &intf, const ChessInterface &other );

    // return the side to move
    nnue::Color sideToMove() const noexcept {
        return pos->stm;
    }

    // obtain the King square for the specified side
    nnue::Square kingSquare(nnue::Color side) const noexcept {
        return pos->kings[side];
    }

    nnue::Network::AccumulatorType &getAccumulator() const noexcept {
        return pos->accum;
    }

    // Iterator returning std::pair<Square, Piece>
    auto begin() const noexcept {
        return pos->locs.begin();
    }

    // Iterator returning std::pair<Square, Piece>
    auto end() const noexcept {
        return pos->locs.end();
    }

    unsigned pieceCount() const noexcept {
        return pos->locs.size();
    }

    const nnue::DirtyState &getDirtyState() const noexcept {
        return pos->dirtyState;
    }

    // Change the state of this interace to the previous position
    void previous() {
        pos = pos->previous;
    }

    bool hasPrevious() const noexcept {
        return pos->previous != nullptr;
    }

    Position *getPrevious() const noexcept {
      return pos->previous;
    }

private:
    Position *pos;
};

inline bool operator == (const ChessInterface &intf, const ChessInterface &other ) {
  return intf.pos == other.pos;
}

inline bool operator != (const ChessInterface &intf, const ChessInterface &other ) {
  return intf.pos == other.pos;
}

#endif
