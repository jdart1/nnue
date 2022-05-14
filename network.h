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
    using Layer2 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 512, 32>;
    using Layer3 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 32, 32>;
    using Layer4 = LinearLayer<uint8_t, int8_t, int32_t, int32_t, 32, 1>;
    using ScaleAndClamper = ScaleAndClamp<int32_t, uint8_t, 32>;
    using Clamper = Clamp<int16_t, uint8_t, 512>;

    static constexpr size_t BUFFER_SIZE = 4096;

    Network() {
        layers.push_back(new Layer1());
        layers.push_back(new Clamper(127));
        layers.push_back(new Layer2());
        layers.push_back(new ScaleAndClamper(64, 127));
        layers.push_back(new Layer3());
        layers.push_back(new ScaleAndClamper(64, 127));
        layers.push_back(new Layer4());
#ifndef NDEBUG
        size_t bufferSize = 0;
        for (const auto &layer : layers) {
            bufferSize += layer->bufferSize();
        }
        // verify const buffer size is sufficient
        assert(bufferSize <= BUFFER_SIZE);
#endif
    }

    virtual ~Network() {
        for (auto layer : layers) {
            delete layer;
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
    OutputType evaluate(const AccumulatorType &accum) const {
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
        for (auto it = layers.begin() + 1; it != layers.end();
             inputOffset = outputOffset, lastOffset = outputOffset,
                  outputOffset += (*it++)->bufferSize()) {
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
    std::vector<BaseLayer *> layers;
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
    unsigned n = 0;
    for (auto layer : network.layers) {
        if (!s.good())
            break;
        // for Stockfish compatiblity: first two layers contain a hash
        if (n < 2) {
            (void)read_little_endian<uint32_t>(s);
        }
        //        std::cout << "reading layer " << n << std::endl << std::flush;
        ++n;
        (void)layer->read(s);
        break;
    }

    if (n != network.layers.size()) {
        std::cerr << "network file read incomplete" << std::endl;
        s.setstate(std::ios::failbit);
    }
    return s;
}

#endif
