"""Tests for the cache simulator, the LRU stack curves and the allocator. CPU, numpy only.

Run:  python sim/test_sim.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alloc import total_misses, ucp_lookahead, uniform_extra  # noqa: E402
from cache import POLICIES, simulate  # noqa: E402
from stack import COLD, FREE, miss_curve, misses_per_step, stack_distances  # noqa: E402

NO_NEWEST = 99  # a page id that is never selected, so no step has a free insert


def _arr(rows):
    return np.array(rows, np.int64)


def test_cyclic_hand_example():
    # k = 2, B = 3. Page S (0) is selected every step; A, B, C (1, 2, 3) cycle.
    #   step:      0      1      2      3      4      5
    #   selected: S A    S B    S C    S A    S B    S C
    # LRU / FIFO: the cache keeps S plus the two most recent of A, B, C, and the cycle always needs the third:
    #   misses     2      1      1      1      1      1     = 7 (4 cold, 3 reuse)
    # Belady: at step 2 evict B (next needed at 4) rather than A (needed at 3); at step 4 evict A (never again):
    #   misses     2      1      1      0      1      0     = 5
    sel = _arr([[0, 1], [0, 2], [0, 3], [0, 1], [0, 2], [0, 3]])
    newest = np.full(6, NO_NEWEST)
    for pol in ("lru", "fifo", "hisparse_lru"):
        m, c = simulate(sel, newest, 3, pol)
        assert m.tolist() == [2, 1, 1, 1, 1, 1] and c.tolist() == [2, 1, 1, 0, 0, 0], (pol, m, c)
    m, _ = simulate(sel, newest, 3, "belady")
    assert m.tolist() == [2, 1, 1, 0, 1, 0], m
    # B = k keeps only the current selection: every new page misses.
    assert simulate(sel, newest, 2, "lru")[0].tolist() == [2, 1, 1, 1, 1, 1]
    # Stack positions: A, B, C come back at position 3, so they hit only with B >= 4.
    d = stack_distances(sel, newest)
    assert miss_curve(d, np.array([2, 3, 4, 5])).tolist() == [7, 7, 4, 4]


def test_hisparse_lru_differs_from_lru():
    # k = 2, B = 3. Pages A..E, X = 0..5.
    #   step 0: A B   both miss                      LRU order (most recent first): A B
    #   step 1: C A   C misses, A hits, no eviction  LRU: C A B (rank order)   HiSparse: A C B (hit first)
    #   step 2: D E   two misses, two evictions      LRU evicts B, A (keeps C)  HiSparse evicts B, C (keeps A)
    #   step 3: A X   LRU: 2 misses                  HiSparse: A hits, 1 miss
    # Totals: LRU 2+1+2+2 = 7, HiSparse-LRU 2+1+2+1 = 6.
    sel = _arr([[0, 1], [2, 0], [3, 4], [0, 5]])
    newest = np.full(4, NO_NEWEST)
    assert simulate(sel, newest, 3, "lru")[0].tolist() == [2, 1, 2, 2]
    assert simulate(sel, newest, 3, "hisparse_lru")[0].tolist() == [2, 1, 2, 1]
    assert misses_per_step(stack_distances(sel, newest), 3).tolist() == [2, 1, 2, 2]  # stack = plain LRU


def test_newest_page_is_free():
    # k = 2, B = 2. Page 0 is the sink; pages 1 then 2 are the newest page.
    #   step 0: 0 1   page 0 is a cold miss, page 1 is the newest page and arrives free    -> 1 miss
    #   step 1: 0 1   hits                                                                -> 0
    #   step 2: 0 2   page 2 is new and free; page 1 is evicted to make room              -> 0
    #   step 3: 0 2   hits                                                                -> 0
    sel = _arr([[0, 1], [0, 1], [0, 2], [0, 2]])
    newest = np.array([1, 1, 2, 2])
    for pol in POLICIES:
        m, c = simulate(sel, newest, 2, pol)
        assert m.tolist() == [1, 0, 0, 0] and c.tolist() == [1, 0, 0, 0], pol
    d = stack_distances(sel, newest)
    assert d[0, 0] == COLD and d[0, 1] == FREE and d[2, 1] == FREE
    assert miss_curve(d, np.array([2, 3])).tolist() == [1, 1]


def synthetic_trace(T=300, k=12, start_pages=40, keep=0.8, seed=0):
    """Trace shaped like ours: sink page, newest page (one new page every 16 steps), sticky selections."""
    rng = np.random.default_rng(seed)
    sel, newest, prev = np.full((T, k), -1, np.int64), np.zeros(T, np.int64), []
    for t in range(T):
        n = start_pages + t // 16
        newest[t] = n - 1
        pool = [p for p in prev if p not in (0, n - 1) and rng.random() < keep]
        while len(pool) < k - 2:
            p = int(rng.integers(1, n - 1))
            if p not in pool:
                pool.append(p)
        row = [0, n - 1] + pool[: k - 2]
        sel[t] = row
        prev = row
    return sel, newest


def test_stack_equals_direct_lru_on_synthetic_traces():
    for seed in range(5):
        sel, newest = synthetic_trace(seed=seed, keep=0.5 + 0.1 * seed)
        k = sel.shape[1]
        d = stack_distances(sel, newest)
        for B in (k, k + k // 2, 2 * k, 3 * k):
            direct, _ = simulate(sel, newest, B, "lru")
            assert np.array_equal(direct, misses_per_step(d, B)), (seed, B)


def test_policy_invariants_on_synthetic_traces():
    for seed in range(5):
        sel, newest = synthetic_trace(seed=seed)
        k = sel.shape[1]
        at_k = {p: simulate(sel, newest, k, p)[0].sum() for p in POLICIES}
        assert len(set(at_k.values())) == 1, at_k  # with B = k every policy keeps only the current selection
        for B in (k + 4, 2 * k, 3 * k):
            res = {p: simulate(sel, newest, B, p, seed=seed) for p in POLICIES}
            totals = {p: r[0].sum() for p, r in res.items()}
            colds = {p: r[1].sum() for p, r in res.items()}
            assert all(totals["belady"] <= v for v in totals.values()), (B, totals)  # Belady is optimal
            assert len(set(colds.values())) == 1, colds  # cold misses do not depend on the policy


def test_ucp_lookahead():
    # Convex: layer 0 saves 1 miss per page; layer 1 saves 6 with its first page, then 1, then nothing.
    curves = np.array([[10, 9, 8, 7, 6], [10, 4, 3, 3, 3]])
    alloc = ucp_lookahead(curves, extra=4, unit=1)
    assert alloc.sum() == 4 and total_misses(curves, alloc) == 11  # 11 is the best of all 5 splits
    # Non-convex: layer 0 gains 10, but only with 3 pages; layer 1 gains 2 per page.
    # Plain greedy would give layer 1 all 3 pages (14 misses); lookahead gives layer 0 all 3 (10 misses).
    curves = np.array([[10, 10, 10, 0], [10, 8, 6, 4]])
    alloc = ucp_lookahead(curves, extra=3, unit=1)
    assert alloc.tolist() == [3, 0] and total_misses(curves, alloc) == 10
    assert uniform_extra(3, 7).tolist() == [3, 2, 2]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
