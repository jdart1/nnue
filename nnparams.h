// Copyright 2024 by Jon Dart. All Rights Reserved
#ifndef _NETWORK_PARAMS
#define _NETWORK_PARAMS

struct NetworkParams {

static constexpr unsigned KING_BUCKETS = 16;

static constexpr unsigned OUTPUT_BUCKETS = 8;

static constexpr unsigned BUCKET_DIVISOR = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;

static constexpr unsigned HIDDEN_WIDTH = 1024;

// input quantization, i.e. 0..1 in float domain is
// 0 .. NETWORK_QA in integer domain.
static constexpr int NETWORK_QA = 255;
// weight quantization, i.e. resolution of the weights. Weight of 1 in
// integer domain has NETWORK_QB levels
static constexpr int NETWORK_QB = 64;
// network output value is multiplied by this to obtain position score
static constexpr int OUTPUT_SCALE = 400;

// clang-format off
static constexpr unsigned KING_BUCKETS_MAP[] = {
    // as in Alexandria
    1,  2,  3,  3,  2,  1, 0,
    4,  5,  6,  7,  7,  6,  5, 4,
    8,  9, 10, 11, 11, 10,  9, 8,
    8,  9, 10, 11, 11, 10,  9, 8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14};
// clang-format on

// version of the network
static constexpr uint32_t NN_VERSION = 0x7c802d9au;

};

#endif
