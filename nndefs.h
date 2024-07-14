// Copyright 2021-2022, 2024 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NNDEFS_H
#define _NNUE_NNDEFS_H

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

// input quantization, i.e. 0..1 in float domain is
// 0 .. NETWORK_QA in integer domain. This is set
// to the square root of 181, so we can square
// inputs without overflowing the 16-bit integer
// range (idea from Peacekeeper)
static constexpr int NETWORK_QA = 181;
// weight quantization, i.e. resolution of the weights. Weight of 1 in
// integer domain has NETWORK_QB levels
static constexpr int NETWORK_QB = 64;
// network output value is multiplied by this to obtain position score
static constexpr int OUTPUT_SCALE = 400;

using IndexArray = std::array<IndexType,MAX_INDICES>;

// version of the network
static constexpr uint32_t NN_VERSION = 0x68AE02B9u;

#endif

