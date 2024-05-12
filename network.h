// Copyright 2021-2024 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NETWORK_H
#define _NNUE_NETWORK_H

#include "accum.h"
#include "features/arasanv3.h"
#include "layers/linear.h"
#include "layers/sqrcreluandlinear.h"
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
    using OutputLayer = SqrCReLUAndLinear<int16_t, AccumulatorType, int16_t, int16_t, OutputType,
                                          FeatureXformerOutputSize * 2, 255>;

    static constexpr size_t BUFFER_SIZE = 4096;

    Network() : transformer(new FeatureXformer()) {
        for (size_t i = 0; i < OutputBuckets; ++i) {
            outputLayer[i] = new OutputLayer();
        }
    }

    virtual ~Network() {
        delete transformer;
        for (size_t i = 0; i < OutputBuckets; ++i) {
            delete outputLayer[i];
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
        size_t lastOffset = 0;
        // evaluate the output layer, in the correct bucket
        outputLayer[bucket]->postProcessAccum(accum, reinterpret_cast<OutputType *>(buffer));
        int32_t nnOut = reinterpret_cast<int32_t *>(buffer + lastOffset)[0];
#ifdef NNUE_TRACE
        std::cout << "NN output, after scaling: "
                  << (nnOut * NETWORK_SCALE) / (NETWORK_QA * NETWORK_QB) << std::endl;
#endif
        return (nnOut * NETWORK_SCALE) / (NETWORK_QA * NETWORK_QB);
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
    OutputLayer *outputLayer[OutputBuckets];
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
#ifdef STOCKFISH_FORMAT
    unsigned n = 0;
    for (size_t i = 0; i < OutputBuckets && s.good(); ++i) {
        // skip next 4 bytes (hash)
        (void)read_little_endian<uint32_t>(s);
        network.outputLayer[i]->read(s);
        ++n;
    }
    if (n != OutputBuckets) {
        s.setstate(std::ios::failbit);
    }
#else
    // input format used by Obsidian
    constexpr size_t width = 2 * Network::FeatureXformerOutputSize;
    int16_t *outputWeights = new int16_t[width * OutputBuckets];
    s.read(reinterpret_cast<char *>(outputWeights), sizeof(int16_t) * width * OutputBuckets);
    if (s.good()) {
        // assume output size 1
        for (size_t b = 0; b < OutputBuckets; ++b) {
            int16_t column[width];
            for (size_t i = 0; i < width; ++i) {
                column[i] = outputWeights[i * OutputBuckets + b];
            }
            network.outputLayer[b]->setCol(0, column);
        }
    }
    delete[] outputWeights;
    int16_t outputBiases[OutputBuckets];
    s.read(reinterpret_cast<char *>(outputBiases), sizeof(int16_t) * OutputBuckets);
    if (s.good()) {
#ifdef NNUE_TRACE
        std::cout << "biases" << std::endl;
#endif
        for (size_t b = 0; b < OutputBuckets; ++b) {
            // assumes output of layer is size 1
            network.outputLayer[b]->setBiases(&outputBiases[b]);
#ifdef NNUE_TRACE
            std::cout << outputBiases[b] << ' ';
#endif
        }
#ifdef NNUE_TRACE
        std::cout << std::endl;
#endif
    }
#endif
    return s;
}

#endif
