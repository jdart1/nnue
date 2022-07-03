// Copyright 2021 by Jon Dart. All Rights Reserved.
#include "chessint.h"
#include <unordered_map>

static const std::unordered_map<char, nnue::Piece> pieceMap = {
    {'p', nnue::BlackPawn}, {'n', nnue::BlackKnight}, {'b', nnue::BlackBishop},
    {'r', nnue::BlackRook}, {'q', nnue::BlackQueen},  {'k', nnue::BlackKing},
    {'P', nnue::WhitePawn}, {'N', nnue::WhiteKnight}, {'B', nnue::WhiteBishop},
    {'R', nnue::WhiteRook}, {'Q', nnue::WhiteQueen},  {'K', nnue::WhiteKing}};

Position::Position(const std::string &fen) 
    : previous(nullptr),dirty_num(0)
{
    int parts = 0;
    nnue::Square sq(56);
    auto it = fen.begin();
    for (; it != fen.end() && parts < 8; ++it) {
        nnue::Piece p = nnue::EmptyPiece;
        if (*it == '/') {
            ++parts;
            sq = 56 - parts * 8;
            continue;
        } else if (isdigit(*it)) {
            sq += int(*it - '0');
            continue;
        } else if (isalpha(*it)) {
            p = pieceMap.at(*it);
            if (p == nnue::BlackKing)
                kings[nnue::Black] = sq;
            else if (p == nnue::WhiteKing)
                kings[nnue::White] = sq;
            else {
                assert(p != nnue::EmptyPiece);
                locs[sq] = p;
            }
            ++sq;
        } else {
            break;
        }
    }
    it++;
    if (*it == 'b')
        stm = nnue::Black;
    else
        stm = nnue::White;
}

        
