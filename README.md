# nnue

This code provides an implementation of a [Efficiently Updatable Neural Network (NNUE)](https://www.chessprogramming.org/NNUE) for chess, compatible with the implementation in [Stockfish](https://github.com/official-stockfish/Stockfish).

## Copyright, license

Copyright 2021 by Jon Dart. All Rights Reserved.

MIT Licensed.

## Implemented Features

- can read Stockfish network files using insertion operator on the Network class
- templated to support different network sizes, weight types, etc.
- full and incremental update supported
- unit test code

## Missing Features

- no special SIMD optimizations
- read only, no support for writing network files
- does not validate hash codes from existing network files

## Interface

The Evaluator class is templated and assumes use of a class or typedef that provides a basic interface to a chess position.

The interface subdirectory contains an implementation, named ChessInterface that provides the necessary methods and is used by the test code. Users are likely to want to replace this with a class that wraps whatever native position representation they have.

## Test code

The nnue_test program in the test subdirectory tests the first layer of the network and the code for the interior linear transformation layers. It should execute and produce "0 errors" on output.

This program can also be used with the -f switch to read an existing network file, although its contents are not validated.
