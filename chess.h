// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_CHESS_H
#define _NNUE_CHESS_H

// Chess-related definitions (in the nnue namespace)

enum Color {White, Black};

typedef uint32_t Square;

constexpr Square InvalidSquare = 127;

enum Piece {
  EmptyPiece = 0,
  WhitePawn = 1,
  WhiteKnight = 2,
  WhiteBishop = 3,
  WhiteRook = 4,
  WhiteQueen = 5,
  WhiteKing = 6,
  BlackPawn = 9,
  BlackKnight = 10,
  BlackBishop = 11,
  BlackRook = 12,
  BlackQueen = 13,
  BlackKing = 14
};

static inline bool isKing(Piece p) {return p == WhiteKing || p == BlackKing;}

enum class MoveType {Normal, Castling, Promotion};

// foward declaration - may be defined with typedef
//class Position;

#endif
