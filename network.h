// Copyright 2021-2024 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NETWORK_H
#define _NNUE_NETWORK_H

#include "accum.h"
#include "features/arasanv3.h"
#include "layers/linear.h"
#include "layers/sqrcrelu.h"
#include "util.h"

class Network {

    template <typename ChessInterface> friend class Evaluator;

  public:
    static constexpr size_t FeatureXformerOutputSize = 1024;

    static constexpr size_t FeatureXformerRows = 12 * KingBuckets * 64;

    using OutputType = int32_t;
    using FeatureXformer = ArasanV3Feature<uint16_t, int16_t, int16_t, int16_t, FeatureXformerRows,
                                           FeatureXformerOutputSize>;
    using AccumulatorType = FeatureXformer::AccumulatorType;
    using AccumulatorOutputType = int16_t;
    using Layer1 = SqrCReLU<AccumulatorOutputType, AccumulatorType, int16_t,
                            FeatureXformerOutputSize, 255, 7>;
    using Layer2 = LinearLayer<uint16_t, int16_t, int16_t, OutputType, FeatureXformerOutputSize, 1>;

    static constexpr size_t BUFFER_SIZE = 4096;

    Network() : transformer(new FeatureXformer()), sqrCReLU(new Layer1()) {
        for (unsigned i = 0; i < OutputBuckets; ++i) {
            // only one output layer
            layers[i].push_back(new Layer2());
        }
#ifndef NDEBUG
        size_t bufferSize = sqrCReLU->getOutputSize();
        for (const auto &layer : layers[0]) {
            bufferSize += layer->bufferSize();
        }
        // verify const buffer size is sufficient
        assert(bufferSize <= BUFFER_SIZE);
#endif
    }

    virtual ~Network() {
        delete transformer;
        delete sqrCReLU;
        for (size_t i = 0; i < OutputBuckets; ++i) {
            for (auto layer : layers[i]) {
                delete layer;
            }
        }
    }

    template <Color kside> inline static unsigned getIndex(Square kp, Piece p, Square sq) {
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
#ifdef NNUE_TRACE
        std::cout << "bucket=" << bucket << std::endl;
        std::cout << "accumulator:" << std::endl;
        std::cout << accum << std::endl;
#endif
        sqrCReLU->postProcessAccum(accum, reinterpret_cast<AccumulatorOutputType *>(buffer));
        size_t inputOffset = 0, outputOffset = sqrCReLU->getOutputSize(), lastOffset = 0;
        // evaluate the remaining layers, in the correct bucket
        for (const auto &it : layers[bucket]) {
            it->forward(static_cast<const void *>(buffer + inputOffset),
                        static_cast<void *>(buffer + outputOffset));
            inputOffset = lastOffset = outputOffset;
            outputOffset += it->bufferSize();
        }
        int nnOut = reinterpret_cast<int32_t *>(buffer + lastOffset)[0];
#ifdef NNUE_TRACE
        std::cout << "NN output, pre-scaling: " << nnOut << " scaled: " << nnOut / FVSCALE
                  << std::endl;
#endif
        return nnOut / FV_SCALE;
    }

    friend std::istream &operator>>(std::istream &i, Network &);

    // Perform an incremental update
    void updateAccum(const AccumulatorType &source, AccumulatorHalf sourceHalf,
                     AccumulatorType &target, AccumulatorHalf targetHalf, const IndexArray &added,
                     size_t added_count, const IndexArray &removed,
                     size_t removed_count) const noexcept {
        transformer->updateAccum(source, sourceHalf, target, targetHalf, added, added_count,
                                 removed, removed_count);
    }

    // Propagate data through the layer, updating the specified half of the
    // accumulator (side to move goes in lower half).
    void updateAccum(const IndexArray &indices, AccumulatorHalf half,
                     AccumulatorType &output) const noexcept {
        transformer->updateAccum(indices, half, output);
    }

  protected:
    FeatureXformer *transformer;
    Layer1 *sqrCReLU;
    std::vector<BaseLayer *> layers[OutputBuckets];
};

inline std::istream &operator>>(std::istream &s, Network &network) {
#ifdef STOCKFISH_FORMAT
    std::uint32_t version, size;
    version = read_little_endian<uint32_t>(s);
    // TBD: validate hash
    (void)read_little_endian<uint32_t>(s);
    size = read_little_endian<uint32_t>(s); // size of
                                            // architecture string
    if (!s.good()) {
        return s;
    } else if (version != NN_VERSION) {
        s.setstate(std::ios::failbit);
        return s;
    }
    char c;
    for (uint32_t i = 0; i < size; i++) {
        if (!s.get(c))
            break;
    }
#endif
    // read feature layer
    (void)network.transformer->read(s);
    // read num buckets x layers
    for (unsigned i = 0; i < OutputBuckets; ++i) {
#ifdef STOCKFISH_FORMAT
        // skip next 4 bytes (hash)
        (void)read_little_endian<uint32_t>(s);
#endif
        unsigned n = 0;
        for (auto layer : network.layers[i]) {
            if (!s.good())
                break;
            (void)layer->read(s);
            ++n;
        }
        if (n != network.layers[i].size()) {
            s.setstate(std::ios::failbit);
            break;
        }
    }
    return s;
}

#endif
