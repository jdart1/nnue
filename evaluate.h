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
        const DirtyState &ds = intf.getDirtyState();
        size_t i;
        for (i = 0; i < ds.dirty_num; i++) {
            const auto &dd = ds.dirty[i];
            Piece piece = dd.piece;
            if (isKing(piece))
                continue;
            if (dd.from != InvalidSquare)
                removed[removed_count++] =
                    Network::getIndex<kside>(kp, piece, dd.from);
            if (dd.to != InvalidSquare)
                added[added_count++] =
                    Network::getIndex<kside>(kp, piece, dd.to);
        }
    }

    void getIndexDiffs(const ChessInterface &ciSource,
                       const ChessInterface &ciTarget, Color c,
                       IndexArray &removed, IndexArray &added) {
        // "source" is a position prior to the one for which we want
        // to get a NNUE eval ("target").
        size_t added_count = 0, removed_count = 0;
        ChessInterface ci(ciTarget);
        while (ci.hasPrevious()) {
            ci.previous();
            // TBD correct side
            if (c == nnue::White)
                getChangedIndices<nnue::White>(ci, added, removed, added_count,
                                               removed_count);
            else
                getChangedIndices<nnue::Black>(ci, added, removed, added_count,
                                               removed_count);
            if (const_cast<const ChessInterface &>(ci) == ciSource)
                break;
        }
        added[added_count] = nnue::LAST_INDEX;
        removed[removed_count] = nnue::LAST_INDEX;
    }

    void updateAccumIncremental(const Network &network,
                                const ChessInterface &ciSource,
                                ChessInterface &ciTarget, const Color c) {
        IndexArray added, removed;
        getIndexDiffs(ciSource, ciTarget, c, added, removed);
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
            ->updateAccum(added, removed, targetHalf,
                          ciTarget.getAccumulator());
        ciTarget.getAccumulator().setState(targetHalf,
                                           AccumulatorState::Computed);
    }

    void updateAccum(const Network &network, const IndexArray &indices, Color c,
                     Color sideToMove, Network::AccumulatorType &accum) {
        auto it = network.layers.begin();
        AccumulatorHalf targetHalf =
            Network::AccumulatorType::getHalf(c, sideToMove);
        if (targetHalf == AccumulatorHalf::Lower) std::cout << "lower";
        else std::cout << "upper";
        std::cout << std::endl;
        for (auto idx : indices) {
            if (idx == nnue::LAST_INDEX)
                break;
            std::cout << idx << ' ';
        }
        std::cout << std::endl;
        ((Network::Layer1 *)*it)->updateAccum(indices, targetHalf, accum);
        accum.setState(AccumulatorState::Computed);
    }

    // update the accumulator based on a position (incrementally if possible)
    void updateAccum(const Network &network, ChessInterface &intf,
                     const Color c, Network::AccumulatorType &accum) {
        // see if incremental update is possible
        int gain = intf.pieceCount() - 2; // pieces minus Kings
        AccumulatorHalf half;
        AccumulatorHalf targetHalf = half =
            Network::AccumulatorType::getHalf(intf.sideToMove(), c);
        ChessInterface ci(intf);
        while (ci.getAccumulator().getState(half) == AccumulatorState::Empty) {
            if (isKing(ci.getDirtyState().pieceMoved()) ||
                (gain -= ci.getDirtyState().dirty_num + 1) < 0) {
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
            getIndices<c>(intf, indices);
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
