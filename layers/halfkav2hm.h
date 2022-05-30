// Copyright 2020-2022 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_HALF_KA_V2_HM_H
#define _NNUE_HALF_KA_V2_HM_H

#include "accumv2.h"
#include "nndefs.h"
#include "typed.h"

// Implements first layer of the neural network, in the "Stockfish v4" network architecture
// used in Stockfish 15.

template <typename InputType, typename WeightType, typename BiasType, typename OutputType, size_t inputSize,
          size_t outputSize, size_t alignment = DEFAULT_ALIGN>
class HalfKaV2Hm : public TypedLayer<InputType, OutputType, inputSize, outputSize, alignment>
{
public:
    // fixed, currently
    using PSQWeightType = std::int32_t;

    HalfKaV2Hm() = default;

    virtual ~HalfKaV2Hm() = default;

    using AccumulatorType = AccumulatorV2<OutputType, WeightType, BiasType, PSQWeightType, outputSize, PSQBuckets>;

    template <Color kside>
    inline static IndexType getIndex(Square kp, Piece p, Square sq) {
        assert(p != EmptyPiece);
        IndexType idx = static_cast<IndexType>(orient<kside>(sq,kp)) +
                        64 * map[p][kside] +
                        (64 * 11) * KingBuckets[orient<kside>(kp, kp)];
        //        std::cout << int(sq) << " " << static_cast<IndexType>(orient<kside>(sq,kp)) << " " <<
        //                        64 * map[p][kside] << " " <<
        //                        (64 * 11) * KingBuckets[orient<kside>(kp, kp)] << " " << idx << std::endl;
        assert(idx < inputSize);
        return idx;
    }

    // Propagate data through the layer, updating the specified half of the
    // accumulator (side to move goes in lower half).
    inline void updateAccum(const IndexArray &indices, AccumulatorHalf half, AccumulatorType &output) {
        output.init_half(half,this->_biases);
        for (auto it = indices.begin(); it != indices.end() && *it != LAST_INDEX; ++it) {
            output.add_half(half,this->_weights[*it],this->_psq[*it]);
        }
#ifdef NNUE_TRACE
        
#endif        
    }

    // Perform an incremental update
    void updateAccum(const IndexArray &added, const IndexArray &removed,
                     size_t added_count, size_t removed_count,
                     AccumulatorHalf half, AccumulatorType &output) {
      for (size_t i = 0; i < added_count; i++) {
          output.add_half(half, this->_weights[added[i]], this->_psq[added[i]]);
      }
      for (size_t i = 0; i < removed_count; i++) {
	  output.sub_half(half, this->_weights[removed[i]], this->_psq[removed[i]]);
      }
    }
    
    virtual inline void doForward(const InputType *, OutputType *) const noexcept {
        // no-op for this layer: use updateAccum
        assert(0);
    }

    // read weights from a stream
    virtual std::istream &read(std::istream &s) {
        // read hash
        (void)read_little_endian<uint32_t>(s);
        std::cout << "----- feature transformer --- " << std::endl;
        std::cout << "----- biases --- " << std::endl;
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
            if (i<10) std::cout << static_cast<int>(_biases[i]) << ' ';
        }
        std::cout << std::endl;
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < outputSize && s.good(); ++j) {
                _weights[i][j] = read_little_endian<WeightType>(s);
            }
        }
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < PSQBuckets && s.good(); ++j) {
                _psq[i][j] = read_little_endian<PSQWeightType>(s);
            }
        }
        return s;
    }

    virtual const WeightType *getCol(size_t row) const noexcept {
        return _weights[row];
    }

    virtual void setCol(size_t row, const WeightType *col) {
        for (size_t i = 0; i < outputSize; ++i)
           _weights[row][i] = col[i];
    }

    virtual void setPSQ(size_t row, const PSQWeightType *col) {
        for (size_t i = 0; i < PSQBuckets; ++i)
           _psq[row][i] = col[i];
    }

private:
    // Rotate positions so that the King is always on files e..h
    template <Color perspective>
    inline static Square orient(Square s, Square ksq) {
        return Square(int(s) ^ (bool(perspective) * 56) ^ ((ksq % 8 < 4) * 7));
    }

    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10,  9,  8,
      -1, -1, -1, -1,  7,  6,  5,  4,
      -1, -1, -1, -1,  3,  2,  1,  0
    };

    static constexpr unsigned map[16][2] = {
        {0, 0}, {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 10}, {0, 0},
        {0, 0}, {1, 0}, {3, 2}, {5, 4}, {7, 6}, {9, 8}, {10, 10}, {0, 0}};

    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[inputSize][outputSize];
    alignas(alignment) PSQWeightType _psq[inputSize][PSQBuckets];

};

#endif
