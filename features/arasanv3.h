// Copyright 2020-2024 by Jon Dart. All Rights Reserved.
#ifndef _ARASANV3_H
#define _ARASANV3_H

#include "nndefs.h"
#include "accum.h"
#include "util.h"

// Implements the feature transformer for the "Arasan v3" neural network architecture.
// This feature uses 5 king buckets. King position is mirrored so that the King is always on files e..h.
template <typename InputType, typename WeightType, typename BiasType, typename OutputType, size_t inputSize,
          size_t outputSize, size_t alignment = DEFAULT_ALIGN>
class ArasanV3Feature
{
public:

    ArasanV3Feature() = default;

    virtual ~ArasanV3Feature() = default;

    using AccumulatorType = Accumulator<OutputType, outputSize>;

    template <Color kside>
    inline static IndexType getIndex(Square kp /* kside King */, Piece p, Square sq) {
        assert(p != EmptyPiece);
        if (kp % 8 >= E_FILE) {
            // flip file
            sq ^= 7;
        }
        if (kside == Black) {
            sq ^= 56;
            kp ^= 56;
        }
        IndexType idx = static_cast<IndexType>(kingBucketsMap[kp] * 12 * 64 +
                                               pieceTypeMap[kside != colorOfPiece(p)][p] * 64 +
                                               sq);
        assert(idx < inputSize);
        return idx;
    }

    // Full update: propagate data through the layer, updating the specified half of the
    // accumulator (side to move goes in lower half).
    void updateAccum(const IndexArray &indices, AccumulatorHalf half, AccumulatorType &output) const noexcept {
        //#ifdef SIMD
        //        simd::fullUpdate<OutputType,WeightType,BiasType,inputSize,outputSize>(output.data(half), &_weights, &_biases, indices.data());
        //#else
        output.init_half(half,this->_biases);
        for (auto it = indices.begin(); it != indices.end() && *it != LAST_INDEX; ++it) {
#ifdef NNUE_TRACE
            std::cout << "index " << *it << " adding weights ";
            for (size_t i = 0; i < 20; ++i) std::cout << this->_weights[*it][i] << ' ';
            std::cout << "to side " << (half == AccumulatorHalf::Lower ? 0 : 1) << std::endl;
#endif
            output.add_half(half,this->_weights[*it]);
        }
        //#endif
    }

    // Perform an incremental update
    void updateAccum(const AccumulatorType &source, AccumulatorHalf sourceHalf,
                     AccumulatorType &target, AccumulatorHalf targetHalf,
                     const IndexArray &added, size_t added_count, const IndexArray &removed,
                     size_t removed_count) const noexcept {
#ifdef SIMD
        simd::update<OutputType,WeightType,inputSize,outputSize>(source.getOutput(sourceHalf),target.data(targetHalf),_weights,
                                                                 added.data(), added_count, removed.data(), removed_count);

#else
        target.copy_half(targetHalf,source,sourceHalf);
        updateAccum(added,removed,added_count,removed_count,targetHalf,target);
#endif
    }

    // Perform an incremental update
    void updateAccum(const IndexArray &added, const IndexArray &removed,
                     size_t added_count, size_t removed_count,
                     AccumulatorHalf half, AccumulatorType &output) const noexcept {
        for (size_t i = 0; i < added_count; i++) {
            output.add_half(half, this->_weights[added[i]]);
        }
        for (size_t i = 0; i < removed_count; i++) {
            output.sub_half(half, this->_weights[removed[i]]);
        }
    }

    // read weights from a stream
    virtual std::istream &read(std::istream &s) {
#ifdef STOCKFISH_FORMAT
        // read hash
        (void)read_little_endian<uint32_t>(s);
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
        }
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < outputSize && s.good(); ++j) {
                _weights[i][j] = read_little_endian<WeightType>(s);
            }
        }
#else
#ifdef NNUE_TRACE
        std::cout << "weights" << std::endl;
#endif
        // weights first
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < outputSize && s.good(); ++j) {
                _weights[i][j] = read_little_endian<WeightType>(s);
            }
        }
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
        }
        std::cout << std::endl;
#endif
        return s;
    }

    virtual const WeightType *getCol(size_t row) const noexcept {
        return _weights[row];
    }

    virtual void setCol(size_t row, const WeightType *col) {
        for (size_t i = 0; i < outputSize; ++i) {
           _weights[row][i] = col[i];
        }
    }

    virtual void setBiases(const BiasType *b) {
        std::memcpy(_biases,reinterpret_cast<const void*>(b),sizeof(BiasType)*outputSize);
    }

    const BiasType *getBiases() const noexcept {
        return _biases;
    }

private:
    static constexpr unsigned pieceTypeMap[2][16] = {
                                                     {0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 0},
                                                     {0, 6, 7, 8, 9, 10, 11, 0, 0, 6, 7, 8, 9, 10, 11, 0}};

    static constexpr unsigned kingBucketsMap[] = {
                                                  0, 0, 1, 1, 1, 1, 0, 0,
                                                  2, 2, 2, 2, 2, 2, 2, 2,
                                                  3, 3, 3, 3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3, 3, 3, 3,
                                                  4, 4, 4, 4, 4, 4, 4, 4,
                                                  4, 4, 4, 4, 4, 4, 4, 4,
                                                  4, 4, 4, 4, 4, 4, 4, 4,
                                                  4, 4, 4, 4, 4, 4, 4, 4
    };

    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[inputSize][outputSize];

};
#endif
