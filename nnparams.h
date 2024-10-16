// Copyright 2024 by Jon Dart. All Rights Reserved
#ifndef _NETWORK_PARAMS
#define _NETWORK_PARAMS

struct NetworkParams {

static constexpr unsigned KING_BUCKETS = 7;

static constexpr unsigned OUTPUT_BUCKETS = 8;

static constexpr unsigned HIDDEN_WIDTH = 2048;

// input quantization, i.e. 0..1 in float domain is
// 0 .. NETWORK_QA in integer domain.
static constexpr int NETWORK_QA = 510;
// weight quantization, i.e. resolution of the weights. Weight of 1 in
// integer domain has NETWORK_QB levels
static constexpr int NETWORK_QB = 64;
// network output value is multiplied by this to obtain position score
static constexpr int OUTPUT_SCALE = 400;

// clang-format off
static constexpr unsigned KING_BUCKETS_MAP[] = {
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6};
// clang-format on

// version of the network
static constexpr uint32_t NN_VERSION = 0x68AE02B9u;

};

#endif
