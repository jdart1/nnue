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
                        (64 * 11) * kingBucket(orient<kside>(kp, kp));
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
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
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
        Square s2 = int(s)  ^ ((ksq % 8 < 4) * 7);
        return perspective ? s2 ^ 56 : s2;
    }

    inline static Square  kingBucket(Square s) {
        return ((63-s)%4 + 2*(63-s)/2)/2;
    }

    static constexpr unsigned map[16][2] = {
        {0, 0}, {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 10}, {0, 0},
        {0, 0}, {1, 0}, {3, 2}, {5, 4}, {7, 6}, {9, 8}, {10, 10}, {0, 0}};

    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[inputSize][outputSize];
    alignas(alignment) PSQWeightType _psq[inputSize][PSQBuckets];

};

#endif
