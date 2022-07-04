# nnue

This code provides an implementation of a [Efficiently Updatable Neural Network (NNUE)](https://www.chessprogramming.org/NNUE) for chess, compatible with the original implementation in [Stockfish](https://github.com/official-stockfish/Stockfish), which was based on contributions from Hisayori Noda aka Nodchip..

## Copyright, license

Copyright 2021-2022 by Jon Dart. All Rights Reserved.

MIT Licensed.

## Implemented Features

- can read Stockfish network files using insertion operator on the Network class
- templated to support different network sizes, weight types, etc.
- full and incremental update supported
- SIMD support for amd64 processors
- unit test code

## Missing Features

- no SIMD for non-Intel such as M1
- read only, no support for writing network files
- does not validate hash codes from existing network files

## Compilation

Requires C++-17. The Makefile (Gnu Make) builds a test executable. -DSIMD must be specificed to select SIMD optimizations. If SIMD is set then tthe following flags can be set to select the desired instruction set(s). They can be combined and are utilized in the following order of precedence.

1. AVX512_VNNI (in addition to AVX512)
2. AVX512
3. VNNI (in addition to AVX2)
4. AVX2
5. SSE41 (in addition to SSSE3)
6. SSSE3 (assumes SSE2 also present)
7. SSE2

If -DSIMD is enabled, at least one of: AVX2, SSSE3 or SSE2 must also be selected.

## Interface

The Evaluator class is templated and assumes use of a class or typedef that provides a basic interface to a chess position.

The interface subdirectory contains an implementation named ChessInterface that provides the necessary methods and is used by the test code. Users are likely to want to replace this with a class that wraps whatever native position representation they have.

## Test code

The nnue_test program in the test subdirectory tests the first layer of the network and the code for the interior linear transformation layers. It should execute and produce "0 errors" on output.

This program can also be used with the -f switch to read an existing network file, although its contents are not validated.
