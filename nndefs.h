// Copyright 2021-2022, 2024 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NNDEFS_H
#define _NNUE_NNDEFS_H

// If set, assume serialized network format used by nnue-pytorch
//#define STOCKFISH_FORMAT

#ifdef AVX512
static constexpr size_t DEFAULT_ALIGN = 64;
#else
static constexpr size_t DEFAULT_ALIGN = 32;
#endif

static constexpr size_t MAX_INDICES = 34; // 16 pieces per side plus room for end of list marker

static constexpr unsigned KingBuckets = 5;

static constexpr unsigned OutputBuckets = 8;

using IndexType = unsigned;

static constexpr IndexType LAST_INDEX = 1000000;

// Quantization factors

static constexpr int NETWORK_QA = 255;
static constexpr int NETWORK_QB = 64;
static constexpr int NETWORK_SCALE = 400;

using IndexArray = std::array<IndexType,MAX_INDICES>;

// version of the network
static constexpr uint32_t NN_VERSION = 0x68AE02B9u;

#endif

