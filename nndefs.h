// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NNDEFS_H
#define _NNUE_NNDEFS_H

static constexpr size_t DEFAULT_ALIGN = 32;
static constexpr size_t MAX_INDICES = 32;

using IndexType = unsigned;

static constexpr IndexType LAST_INDEX = 1000000;

// Ration of NNUE output to chess evaluation
static constexpr int FV_SCALE = 16;

using IndexArray = std::array<IndexType,MAX_INDICES>;

// version of the network (Stockfish compatible)
static constexpr uint32_t NN_VERSION = 0x7AF32F16u;

#endif

