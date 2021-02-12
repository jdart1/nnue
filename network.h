// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_NETWORK_H
#define _NNUE_NETWORK_H

#include "accum.h"
#include "layers/base.h"
#include "layers/clamp.h"
#include "layers/halfkp.h"
#include "layers/linear.h"
#include "layers/scaleclamp.h"
#include "util.h"

class Network {

  template<typename ChessInterface>
  friend class Evaluator;

  public:

    static constexpr size_t HalfKpRows = 64 * (10 * 64 + 1);

    static constexpr size_t HalfKpOutputSize = 256;

    using IndexArray = std::array<int, MAX_INDICES>;
    using OutputType = int8_t;
    using InputType = uint8_t; // output of transformer
    using Layer1 = HalfKp<uint16_t, int16_t, int16_t, int16_t, HalfKpRows,
                          HalfKpOutputSize>;
    using AccumulatorType = Layer1::AccumulatorType;
    using Layer2 = LinearLayer<int8_t, int8_t, int32_t, int32_t, 512, 32>;
    using Layer3 = LinearLayer<int8_t, int8_t, int32_t, int32_t, 32, 32>;
    using Layer4 = LinearLayer<int8_t, int8_t, int32_t, int32_t, 32, 1>;
    using ScaleAndClamper = ScaleAndClamp<int32_t, int8_t, 32>;
    using Clamper = Clamp<int16_t, int8_t, 512>;

    static constexpr size_t BUFFER_SIZE = 1350;

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

    // evaluate the net (layers past the first one)
    OutputType evaluate(const AccumulatorType &accum) {
        std::byte buffer[BUFFER_SIZE];
        size_t offset = 0;
        bool first = true;
        std::byte *input = nullptr;
        // propagate data through the remaining layers
        for (auto it = layers.begin() + 1; it != layers.end(); it++) {
            if (first) {
                (*it)->forward(static_cast<const void *>(accum.getOutput()),
                               static_cast<void *>(buffer + offset));
                first = false;
            } else {
                (*it)->forward(static_cast<const void *>(input),
                               static_cast<void *>(buffer + offset));
            }
            input = buffer + offset;
            offset += (*it)->bufferSize();
        }
        return *(
            (static_cast<OutputType *>(static_cast<void *>(buffer + offset))));
    }

    friend std::istream &operator>>(std::istream &i, Network &);

    static constexpr unsigned map[16][2] = {
        {0, 0}, {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {0, 0},
        {0, 0}, {2, 1}, {4, 3}, {6, 5}, {8, 7}, {10, 9}, {12, 11}, {0, 0}};

    // 180 degree rotation for Black
    template <Color kside> inline static Square rotate(Square s) {
        return kside == Black ? Square(static_cast<int>(s) ^ 63) : s;
    }

    template <Color kside>
    inline static unsigned getIndex(Square kp, Piece p, Square psq) {
        assert(p != EmptyPiece);
        Square rkp = rotate<kside>(kp);
        unsigned pidx = map[p][kside];
        unsigned idx = (64 * 10 + 1) * static_cast<unsigned>(rkp) +
                       64 * (pidx - 1) +
                       static_cast<unsigned>(rotate<kside>(psq)) + 1;
        assert(idx < HalfKpRows);
        return idx;
    }

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
    std::string architecture;
    architecture.reserve(size);
    s.read(&architecture[0], size);
    std::cout << architecture << std::endl << std::flush;
    unsigned n = 0;
    for (auto layer : network.layers) {
        if (!s.good())
            break;
        // for Stockfish compatiblity: first two layers contain a hash
        if (n < 2) {
            (void)read_little_endian<uint32_t>(s);
        }
        std::cout << "reading layer " << n++ << std::endl << std::flush;
        (void)layer->read(s);
    }

    if (n != network.layers.size()) {
        std::cerr << "network file read incomplete" << std::endl;
        s.setstate(std::ios::failbit);
    }
    return s;
}

#endif
