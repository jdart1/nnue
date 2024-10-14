#
# Copyright 2020-2023 by Jon Dart. All Rights Reserved.
#
# typical build flags for different architectures, select one
ARCH_FLAGS := -DSIMD -DSSE2 -DAVX2 -DUSE_POPCNT -DSSSE3 -DSSE41 -mavx2 -mbmi2 -msse4.1 -msse4.2 -mpopcnt -DSTOCKFISH_FORMAT
#ARCH_FLAGS = -mavx2 -mbmi2 -msse2 -DSIMD -DAVX2 -DAVX512 -mavx512bw
#ARCH_FLAGS = -DSIMD -DNEON -mcpu=apple-m1 -m64
#ARCH_FLAGS = -DSIMD -DNEON  -m64 -mmacosx-version-min=10.14 -arch arm64 -march=armv8.2-a+dotprod 
#ARCH_FLAGS = -DSIMD -DSSE2 -msse4.1 -msse4.2 -mpopcnt

#OPT := -O3
NNUE_FLAGS = -I. -std=c++17 -Wall -Wextra -Wpedantic -Wshadow

NN_LIBS := -lstdc++ -lc -lm

CFLAGS := $(NNUE_FLAGS) $(ARCH_FLAGS) $(OPT) -std=c++17 -I.

CXX ?= g++

LD = $(CXX)

#LDFLAGS = -fuse-ld=gold
#LDFLAGS =  -fsanitize=address -fsanitize=bounds-strict
DEBUG = -ggdb
LDFLAGS = $(DEBUG)

BUILD = build
EXPORT = build

NNUE_SOURCES = nnue_test.cpp chessint.cpp
NNUE_OBJS    = $(patsubst %.cpp, $(BUILD)/%.o, $(NNUE_SOURCES))
NNUE_ASM    = $(patsubst %.cpp, $(BUILD)/%.s, $(NNUE_SOURCES))

default: $(EXPORT)/nnue_test

dirs:
	mkdir -p $(BUILD)

asm: $(NNUE_ASM)

clean: dirs
	rm -f $(BUILD)/*.o
	rm -f $(BUILD)/*.s
	cd $(EXPORT) && rm -f nnue_test

$(BUILD)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPT) $(DEBUG) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: layers/%.cpp
	$(CXX) $(CXXFLAGS) $(OPT) $(DEBUG) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: test/%.cpp
	$(CXX) $(CXXFLAGS) $(OPT) $(DEBUG) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: interface/%.cpp
	$(CXX) $(CXXFLAGS) $(OPT) $(DEBUG) $(CFLAGS) -c -o $@ $<

$(EXPORT)/nnue_test: dirs $(NNUE_OBJS)
	$(LD) $(OPT) $(LDFLAGS) $(NNUE_OBJS) $(DEBUG) -o $(BUILD)/nnue_test $(NN_LIBS)

