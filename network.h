// Copyright 2021, 2022 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NETWORK_H
#define _NNUE_NETWORK_H

#include "accumv2.h"
#include "layers/base.h"
#include "layers/clamp.h"
#include "layers/halfkav2hm.h"
#include "layers/linear.h"
#include "layers/scaleclamp.h"
#include "util.h"

class Network {

    template <typename ChessInterface> friend class Evaluator;

  public:
    static constexpr size_t Layer1OutputSize = 1024;

    static constexpr size_t Layer1Rows = 22 * Layer1OutputSize;

    using IndexArray = std::array<int, MAX_INDICES>;
    using OutputType = int32_t;
    using InputType = uint8_t; // output of transformer
    using Layer1 = HalfKaV2Hm<uint16_t, int16_t, int16_t, int16_t, Layer1Rows,
                              Layer1OutputSize>;
    using AccumulatorType = Layer1::AccumulatorType;
    using AccumulatorOutputType = int16_t;
    using Layer2 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 1024, 16>;
    using Layer3 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 16, 32>;
    using Layer4 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 32, 1>;
    using ScaleAndClamper1 = ScaleAndClamp<int32_t, uint8_t, 16>;
    using ScaleAndClamper2 = ScaleAndClamp<int32_t, uint8_t, 32>;

    static constexpr size_t BUFFER_SIZE = 4096;

    Network() :  transformer(new Layer1()) {
        for (unsigned i = 0; i < PSQBuckets; ++i) {
            layers[i].push_back(new Layer2());
            layers[i].push_back(new ScaleAndClamper1(16, 127));
            layers[i].push_back(new Layer3());
            layers[i].push_back(new ScaleAndClamper2(32, 127));
            layers[i].push_back(new Layer4());
        }
#ifndef NDEBUG
        size_t bufferSize = 0;
        for (const auto &layer : layers[0]) {
            bufferSize += layer->bufferSize();
        }
        // verify const buffer size is sufficient
        assert(bufferSize <= BUFFER_SIZE);
#endif
    }

    virtual ~Network() {
        delete transformer;
        for (unsigned i = 0; i < PSQBuckets; ++i) {
            for (auto layer : layers[i]) {
                delete layer;
            }
        }
    }

    template <Color kside>
    inline static unsigned getIndex(Square kp, Piece p, Square sq) {
#ifdef NDEBUG
        return Layer1::getIndex<kside>(kp, p, sq);
#else
        auto idx = Layer1::getIndex<kside>(kp, p, sq);
        assert(idx < Layer1Rows);
        return idx;
#endif
    }

    // evaluate the net (layers past the first one)
    OutputType evaluate(const AccumulatorType &accum, unsigned bucket) const {
        alignas(nnue::DEFAULT_ALIGN) std::byte buffer[BUFFER_SIZE];
        bool first = true;
        // propagate data through the remaining layers
        size_t inputOffset = 0, outputOffset = 0, lastOffset = 0;
#ifdef NNUE_TRACE
        unsigned i = 0, layer = 0;
        std::cout << "accumulator:" << std::endl;
        for (i = 0; i < accum.getSize(); i++) {
            std::cout << int(accum.getOutput()[i]) << ' ';
        }
        std::cout << std::endl;
#endif
        for (auto it = layers[bucket].begin();
             it != layers[bucket].end();
             outputOffset += (*it++)->bufferSize(),
                 inputOffset = outputOffset, lastOffset = outputOffset) {
#ifdef NNUE_TRACE
            std::cout << "--- layer " << layer + 1 << " input=" << std::hex
                      << uintptr_t(buffer + inputOffset)
                      << " output=" << uintptr_t(buffer + outputOffset)
                      << std::dec << std::endl;
#endif
            if (first) {
                (*it)->forward(static_cast<const void *>(accum.getOutput()),
                               static_cast<void *>(buffer + outputOffset));
                first = false;
            } else {
                (*it)->forward(static_cast<const void *>(buffer + inputOffset),
                               static_cast<void *>(buffer + outputOffset));
            }
#ifdef NNUE_TRACE
            if (layer % 2 == 0) {
                for (i = 0; i < (*it)->bufferSize(); i++) {
                    std::cout << int((reinterpret_cast<uint8_t *>(
                                     buffer + outputOffset))[0])
                              << ' ';
                }
                std::cout << std::endl;
            }
            ++layer;
#endif
        }
#ifdef NNUE_TRACE
        std::cout << "output: "
                  << reinterpret_cast<OutputType *>(buffer + lastOffset)[0] /
                         FV_SCALE
                  << std::endl;
#endif
        return reinterpret_cast<OutputType *>(buffer + lastOffset)[0] /
               FV_SCALE;
    }

    friend std::istream &operator>>(std::istream &i, Network &);

  protected:
    Layer1 *transformer;
    std::vector<BaseLayer *> layers[PSQBuckets];
};

inline std::istream &operator>>(std::istream &s, nnue::Network &network) {
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
    // read hash
    (void)read_little_endian<uint32_t>(s);
    // read transform layer
    (void)network.transformer->read(s);
    // read num buckets x layers
    for (unsigned i = 0; i < PSQBuckets; ++i) {
        size_t n = 0;
        for (auto layer : network.layers[i]) {
            if (!s.good())
                break;
            if (n % 2 == 0) {
                // read hash
                (void)read_little_endian<uint32_t>(s);
            }
            ++n;
            (void)layer->read(s);
        }
        if (n != network.layers[i].size()) {
            std::cerr << "network file read incomplete" << std::endl;
            s.setstate(std::ios::failbit);
        }
    }
    return s;
}

#endif
