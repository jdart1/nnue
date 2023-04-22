// Copyright 2021-2023 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NETWORK_H
#define _NNUE_NETWORK_H

#include "accumv2.h"
#include "layers/base.h"
#include "layers/clamp.h"
#include "layers/halfkav2hm.h"
#include "layers/halfka_output.h"
#include "layers/linear.h"
#include "layers/scaleclamp.h"
#include "util.h"

class Network {

    template <typename ChessInterface> friend class Evaluator;

public:
    static constexpr size_t FeatureXformerOutputSize = 1024;

    static constexpr size_t FeatureXformerRows = 22 * FeatureXformerOutputSize;

    using OutputType = int32_t;
    using FeatureXformer = HalfKaV2Hm<uint16_t, int16_t, int16_t, int16_t, FeatureXformerRows,
                              FeatureXformerOutputSize>;
    using AccumulatorType = FeatureXformer::AccumulatorType;
    using AccumulatorOutputType = int16_t;
    using HalfKaMultClamp = HalfKaOutput<AccumulatorOutputType, AccumulatorType, uint8_t, FeatureXformerOutputSize, 127, 7>;
    using Layer2 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, FeatureXformerOutputSize, 16>;
    using Layer3 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 15, 32>;
    using Layer4 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 32, 1>;
    using ScaleAndClamper1 = ScaleAndClamp<int32_t, uint8_t, 16, 6>;
    using ScaleAndClamper2 = ScaleAndClamp<int32_t, uint8_t, 32, 6>;

    static constexpr size_t BUFFER_SIZE = 4096;

    Network() :  transformer(new FeatureXformer()), halfKaMultClamp(new HalfKaMultClamp()) {
        for (unsigned i = 0; i < PSQBuckets; ++i) {
            layers[i].push_back(new Layer2());
            layers[i].push_back(new ScaleAndClamper1(127));
            layers[i].push_back(new Layer3());
            layers[i].push_back(new ScaleAndClamper2(127));
            layers[i].push_back(new Layer4());
        }
#ifndef NDEBUG
        size_t bufferSize = halfKaMultClamp->getOutputSize();
        for (const auto &layer : layers[0]) {
            bufferSize += layer->bufferSize();
        }
        // verify const buffer size is sufficient
        assert(bufferSize <= BUFFER_SIZE);
#endif
    }

    virtual ~Network() {
        delete transformer;
        delete halfKaMultClamp;
        for (unsigned i = 0; i < PSQBuckets; ++i) {
            for (auto layer : layers[i]) {
                delete layer;
            }
        }
    }

    template <Color kside>
    inline static unsigned getIndex(Square kp, Piece p, Square sq) {
#ifdef NDEBUG
        return FeatureXformer::getIndex<kside>(kp, p, sq);
#else
        auto idx = FeatureXformer::getIndex<kside>(kp, p, sq);
        assert(idx < FeatureXformerRows);
        return idx;
#endif
    }

    // evaluate the net (layers past the first one)
    int32_t evaluate(const AccumulatorType &accum, unsigned bucket) const {
        alignas(nnue::DEFAULT_ALIGN) std::byte buffer[BUFFER_SIZE];
        // propagate data through the remaining layers
        size_t inputOffset, outputOffset;
#ifdef NNUE_TRACE
        std::cout << "bucket=" << bucket << std::endl;
        std::cout << "accumulator:" << std::endl;
        std::cout << accum << std::endl;
#endif
        // post-process accumulator
        halfKaMultClamp->postProcessAccum(accum,
                                          reinterpret_cast<uint8_t*>(buffer));
        outputOffset = halfKaMultClamp->getOutputSize();
        inputOffset = 0;
        // evaluate the remaining layers, in the correct bucket
        int layer = 0;
        int fwdOut;
        for (const auto &it : layers[bucket]) {
            if (layer > 0) {
                outputOffset += it->bufferSize();
            }
            it->forward(static_cast<const void *>(buffer + inputOffset),
                        static_cast<void *>(buffer + outputOffset));
            if (layer == 0) {
                // the last column of this layer's output is "fed foward"
                fwdOut = reinterpret_cast<int32_t *>(buffer + outputOffset)[15];
            }
            inputOffset = outputOffset;
            ++layer;
        }
        int nnOut = reinterpret_cast<int32_t *>(buffer + outputOffset)[0];
        int fwdOutScaled = int(fwdOut * (600 * FV_SCALE) / (127 * (1 << WEIGHT_SCALE_BITS)));
        int psqVal = accum.getPSQValue(bucket);
#ifdef NNUE_TRACE
        std::cout << "NN output: " << nnOut << " fwdOut (pre-scaling) = " << fwdOut << 
            " fwdOut (scaled): " << fwdOutScaled << " psq = " << psqVal << " total:" <<
            (nnOut + fwdOutScaled + psqVal) / FV_SCALE << std::endl;
#endif
       return (nnOut + fwdOutScaled + psqVal) / FV_SCALE;
    }

    friend std::istream &operator>>(std::istream &i, Network &);

    // Perform an incremental update
    void updateAccum(const IndexArray &added, const IndexArray &removed,
                     size_t added_count, size_t removed_count,
                     AccumulatorHalf half, AccumulatorType &output) const noexcept {
        transformer->updateAccum(added, removed, added_count, removed_count, half, output);
    }

    // Propagate data through the layer, updating the specified half of the
    // accumulator (side to move goes in lower half).
    void updateAccum(const IndexArray &indices, AccumulatorHalf half, AccumulatorType &output) const noexcept {
        transformer->updateAccum(indices, half, output);
    }

protected:
    FeatureXformer *transformer;
    HalfKaMultClamp *halfKaMultClamp;
    std::vector<BaseLayer *> layers[PSQBuckets];
};

inline std::istream &operator>>(std::istream &s, Network &network) {
    std::uint32_t version, size;
    version = read_little_endian<uint32_t>(s);
    // TBD: validate hash
    (void)read_little_endian<uint32_t>(s);
    size = read_little_endian<uint32_t>(s); // size of
                                            // architecture string
    if (!s.good()) {
        std::cerr << "failed to read network file header" << std::endl;
        return s;
    } else if (version != NN_VERSION) {
        std::cerr << "invalid network file version" << std::endl;
        s.setstate(std::ios::failbit);
        return s;
    }
    char c;
    for (uint32_t i = 0; i < size; i++) {
        if (!s.get(c))
            break;
    }
    // read transform layer
    (void)network.transformer->read(s);
    // read num buckets x layers
    for (unsigned i = 0; i < PSQBuckets; ++i) {
        // skip next 4 bytes (hash)
        (void)read_little_endian<uint32_t>(s);
        unsigned n = 0;
        for (auto layer : network.layers[i]) {
            if (!s.good())
                break;
            (void)layer->read(s);
            ++n;
        }
        if (n != network.layers[i].size()) {
            std::cerr << "network file read incomplete" << std::endl;
            s.setstate(std::ios::failbit);
        }
    }
    return s;
}

#endif
