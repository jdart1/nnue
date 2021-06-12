// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_H
#define _NNUE_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <memory>
#include <array>
#include <vector>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

#ifdef SIMD
#include "simd.h"
#endif

namespace nnue 
{
#include "network.h"
#include "evaluate.h"
}


#endif
