#
# Copyright 2020-2022 by Jon Dart. All Rights Reserved.
#
NNUE_FLAGS = -I. -g -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=bounds-strict

ARCH_FLAGS = -mavx2 -mbmi2 -DSIMD -DAVX2

#OPT = -O3
OPT = -g

NN_LIBS := -lstdc++ -lc -lm

CFLAGS := $(CFLAGS) $(NNUE_FLAGS) $(ARCH_FLAGS) $(OPT)

CPP ?= g++

LD = $(CPP)

#LDFLAGS = -fuse-ld=gold
LDFLAGS =  -fsanitize=address -fsanitize=bounds-strict

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
	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: layers/%.cpp
	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: test/%.cpp
	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: interface/%.cpp
	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -c -o $@ $<

#$(BUILD)/%.s: test/%.cpp
#	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -S -fverbose-asm -o $@ $<

#$(BUILD)/%.s: interface/%.cpp
#	$(CPP) $(CXXFLAGS) $(OPT) $(CFLAGS) -S -fverbose-asm -o $@ $<

$(EXPORT)/nnue_test: dirs $(NNUE_OBJS)
	$(LD) $(OPT) $(LDFLAGS) $(NNUE_OBJS) $(DEBUG) -o $(BUILD)/nnue_test $(NN_LIBS)

