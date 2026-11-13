"""LRU miss curves for every cache size in one pass (Mattson et al. 1970), adapted to batched selections.

Why this is exact for our LRU. Keep one recency stack of every page seen so far: this step's selected
pages on top in rank order, then all other pages in their previous order. Claim: a cache of B >= k pages
under LRU always holds exactly the top B pages of that stack. If it held the top B before step t, it keeps
the selected pages plus its most recent non-selected pages, which are the top B - k non-selected pages of
the old stack, and those are exactly the top B of the new stack. The stack does not depend on B, so one
pass gives every B: a selected page hits in a cache of size B if and only if its position in the stack
before the step (0 = top) is smaller than B.

This holds because the order among pages selected in the same step (rank) does not depend on B. In
HiSparse's variant, hits rank above misses, and which pages hit depends on B, so the proof does not apply.
sim/run_g3.py measures how far that variant is from these curves (open question Q1).
"""
from __future__ import annotations

import numpy as np

COLD = np.iinfo(np.int64).max // 4  # distance of a page never seen before
FREE = -1                           # not counted: padding, or the newest page's first appearance


def stack_distances(sel: np.ndarray, newest: np.ndarray) -> np.ndarray:
    """Stack position before each access [T, k]: COLD for first-time pages, FREE for uncounted entries."""
    T, K = sel.shape
    P = int(max(sel.max(), newest.max())) + 1
    key = np.full(P, -1, np.int64)  # recency key, -1 = never seen
    dist = np.full((T, K), FREE, np.int64)
    for t in range(T):
        row = sel[t]
        valid = np.flatnonzero(row >= 0)
        s = row[valid]
        known = np.sort(key[key >= 0])
        k_s = key[s]
        seen = k_s >= 0
        d = np.where(seen, len(known) - np.searchsorted(known, k_s, side="right"), COLD)
        d[(s == newest[t]) & ~seen] = FREE
        dist[t, valid] = d
        key[s] = t * K + (K - np.arange(len(s)))
    return dist


def miss_curve(dist: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Total misses for each cache size in `sizes` (every size must be >= k)."""
    d = dist[dist != FREE]
    n_cold = int((d == COLD).sum())
    finite = np.sort(d[d != COLD])
    return n_cold + (len(finite) - np.searchsorted(finite, sizes, side="left"))


def misses_per_step(dist: np.ndarray, B: int) -> np.ndarray:
    """Misses at every step for one cache size B."""
    return ((dist != FREE) & (dist >= B)).sum(axis=1)


def cold_per_step(dist: np.ndarray) -> np.ndarray:
    return (dist == COLD).sum(axis=1)
