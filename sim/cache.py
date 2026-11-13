"""Trace-driven simulation of one GPU page cache: one (request, layer) pair (project notes section 8).

Rules:
- The cache holds B pages. After each step, every selected page is resident (B >= k).
- Victims are chosen only among resident pages that are not selected this step.
- The newest page is written straight into the GPU cache: when it first appears it is inserted
  without counting a miss, and it occupies a slot like any other page.
- The cache starts empty at the first decode step.
- A miss is cold if the page was never resident before, otherwise it is a reuse miss.

Input per layer: sel [T, k] page ids in rank order (-1 = none) and newest [T], the newest page id per step.
"""
from __future__ import annotations

import numpy as np

POLICIES = ("fifo", "random", "lru", "hisparse_lru", "belady")
NEVER = np.iinfo(np.int64).max // 4  # "not selected again"


def next_use(sel: np.ndarray) -> np.ndarray:
    """For each selected entry sel[t, j], the next step > t at which that page is selected again (NEVER if none)."""
    T, _ = sel.shape
    last = np.full(int(sel.max()) + 1, NEVER, np.int64)
    nxt = np.full(sel.shape, NEVER, np.int64)
    for t in range(T - 1, -1, -1):
        row = sel[t]
        v = row >= 0
        nxt[t, v] = last[row[v]]
        last[row[v]] = t
    return nxt


def simulate(sel: np.ndarray, newest: np.ndarray, B: int, policy: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Replay one layer's selections through a cache of B pages. Returns (misses[T], cold_misses[T])."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    T, K = sel.shape
    P = int(max(sel.max(), newest.max())) + 1
    resident = np.zeros(P, bool)
    ever = np.zeros(P, bool)
    selmask = np.zeros(P, bool)
    key = np.zeros(P, np.float64)  # eviction priority: smallest key leaves first
    nxt = next_use(sel) if policy == "belady" else None
    rng = np.random.default_rng(seed)
    misses = np.zeros(T, np.int64)
    cold = np.zeros(T, np.int64)
    n_res = 0
    for t in range(T):
        row = sel[t]
        valid = row >= 0
        s = row[valid]
        if len(s) > B:
            raise ValueError(f"B = {B} is smaller than the {len(s)} selected pages")
        hit = resident[s]
        free = (s == newest[t]) & ~hit  # the newest page arrives without a transfer
        miss = ~hit & ~free
        misses[t] = miss.sum()
        cold[t] = (miss & ~ever[s]).sum()
        need = n_res + int((~hit).sum()) - B
        if need > 0:
            selmask[s] = True
            cand = np.flatnonzero(resident & ~selmask)
            selmask[s] = False
            if policy == "random":
                victims = rng.choice(cand, need, replace=False)
            elif need < len(cand):
                victims = cand[np.argpartition(key[cand], need - 1)[:need]]
            else:
                victims = cand
            resident[victims] = False
            n_res -= need
        new = s[~hit]
        resident[new] = True
        ever[s] = True
        n_res += len(new)
        rank = np.arange(len(s))
        base = t * 2 * K  # keys of later steps are always larger
        if policy == "lru":  # recency; within a step, higher-ranked pages count as more recent
            key[s] = base + (K - rank)
        elif policy == "hisparse_lru":  # within a step, hits rank above newly fetched pages
            key[s] = base + K * hit + (K - rank)
        elif policy == "fifo":  # insertion order only; hits do not refresh
            key[new] = base + (K - rank[~hit])
        elif policy == "belady":  # farthest next use leaves first
            key[s] = -nxt[t, valid].astype(np.float64)
    return misses, cold
