// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_EVALUATE_H
#define _NNUE_EVALUATE_H

template <typename ChessInterface> class Evaluator {
  public:
    template <Color kside>
    static size_t getIndices(const ChessInterface &intf, IndexArray &out) {
        IndexArray::iterator it = out.begin();
        for (const std::pair<Square, Piece> &pair : intf) {
            const Square &sq = pair.first;
            const Piece &piece = pair.second;
            if (piece != WhiteKing && piece != BlackKing) {
                *it++ = nnue::Network::getIndex<kside>(intf.kingSquare(kside),
                                                       piece, sq);
            }
        }
        *it = LAST_INDEX;
        return it - out.begin();
    }

    template <Color kside>
    static void getChangedIndices(const ChessInterface &intf, IndexArray &added,
                                  IndexArray &removed, size_t &added_count,
                                  size_t &removed_count) {
        const Square kp = intf.kingSquare(kside);
        const unsigned dn = intf.getDirtyNum();
        size_t i;
        for (i = 0; i < dn; i++) {
            Piece piece;
            Square from, to;
            intf.getDirtyState(i, from, to, piece);
            if (isKing(piece))
                continue;
            if (from != InvalidSquare)
                removed[removed_count++] =
                    Network::getIndex<kside>(kp, piece, from);
            if (to != InvalidSquare)
                added[added_count++] = Network::getIndex<kside>(kp, piece, to);
        }
    }

    void getIndexDiffs(const ChessInterface &ciSource,
                       const ChessInterface &ciTarget, Color c,
                       IndexArray &added, IndexArray &removed,
                       size_t &added_count, size_t &removed_count) {
        // "source" is a position prior to the one for which we want
        // to get a NNUE eval ("target").
        added_count = removed_count = 0;
        ChessInterface ci(ciTarget);
        while (ci.hasPrevious()) {
            ci.previous();
            if (c == nnue::White)
                getChangedIndices<nnue::White>(ci, added, removed, added_count,
                                               removed_count);
            else
                getChangedIndices<nnue::Black>(ci, added, removed, added_count,
                                               removed_count);
            if (const_cast<const ChessInterface &>(ci) == ciSource)
                break;
        }
    }

    void updateAccumIncremental(const Network &network,
                                const ChessInterface &ciSource,
                                ChessInterface &ciTarget, const Color c) {
        IndexArray added, removed;
        size_t added_count, removed_count;
        getIndexDiffs(ciSource, ciTarget, c, added, removed, added_count,
                      removed_count);
        // copy from source to target
        AccumulatorHalf sourceHalf =
            Network::AccumulatorType::getHalf(c, ciSource.sideToMove());
        AccumulatorHalf targetHalf =
            Network::AccumulatorType::getHalf(c, ciTarget.sideToMove());
        ciTarget.getAccumulator().copy_half(
            targetHalf, ciSource.getAccumulator(), sourceHalf);
        // update based on diffs
        auto it = network.layers.begin();
        ((Network::Layer1 *)*it)
            ->updateAccum(added, removed, added_count, removed_count,
                          targetHalf, ciTarget.getAccumulator());
        ciTarget.getAccumulator().setState(targetHalf,
                                           AccumulatorState::Computed);
    }

    // Full evaluation of 1/2 of the accumulator for a specified color (c)
    void updateAccum(const Network &network, const IndexArray &indices, Color c,
                     Color sideToMove, Network::AccumulatorType &accum) {
        auto it = network.layers.begin();
        AccumulatorHalf targetHalf =
            Network::AccumulatorType::getHalf(c, sideToMove);
        for (auto idx : indices) {
            if (idx == nnue::LAST_INDEX)
                break;
        }
        ((Network::Layer1 *)*it)->updateAccum(indices, targetHalf, accum);
        accum.setState(AccumulatorState::Computed);
    }

    // Update the accumulator based on a position (incrementally if possible)
    void updateAccum(const Network &network, ChessInterface &intf,
                     const Color c, Network::AccumulatorType &accum) {
        // see if incremental update is possible
        int gain = intf.pieceCount() - 2; // pieces minus Kings
        AccumulatorHalf half;
        AccumulatorHalf targetHalf = half =
            Network::AccumulatorType::getHalf(intf.sideToMove(), c);
        ChessInterface ci(intf);
        while (ci.getAccumulator().getState(half) == AccumulatorState::Empty) {
            unsigned dn = ci.getDirtyNum();
            if (dn == 0)
                break;
            Square from, to;
            Piece piece;
            ci.getDirtyState(0,from,to,piece);
            if (isKing(piece) || (gain -= dn + 1) < 0) {
                // King was moved, can't incrementally update, or no
                // gain fron incremental update
                break;
            }
            // This accumulator has no data, try previous
            if (ci.hasPrevious()) {
                ci.previous();
            } else {
                break; // no more previous positions to examine
            }
        }
        if (ci.getAccumulator().getState(half) == AccumulatorState::Computed) {
            // a previous position was found with usable data
            updateAccumIncremental(network, ci, intf, c);
        } else {
            // Do full update
            IndexArray indices;
            if (c == White)
                getIndices<White>(intf, indices);
            else
                getIndices<Black>(intf, indices);
            auto it = network.layers.begin();
            ((Network::Layer1 *)*it)->updateAccum(indices, targetHalf, accum);
        }
        accum.setState(AccumulatorState::Computed);
    }

    // evaluate the net (full evaluation)
    Network::OutputType fullEvaluate(const Network &network,
                                     ChessInterface &intf) {
        updateAccum(network, intf);
        return network.evaluate(intf.getAccumulator());
    }
};

#endif
