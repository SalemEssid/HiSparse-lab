"""Split a fixed GPU page budget across layers using their miss curves (UCP lookahead, Qureshi & Patt 2006).

Budget used in this project: sum over layers of B_l = L * 2k, with every B_l >= k.
So each layer starts at k pages and the remaining L * k pages are handed out by expected benefit.
"""
from __future__ import annotations

import numpy as np


def ucp_lookahead(curves: np.ndarray, extra: int, unit: int = 8) -> np.ndarray:
    """curves[l, i] = misses of layer l with k + i pages, for i = 0..extra. Returns extra pages per layer.

    Lookahead greedy: at each round, every layer reports the best misses saved per unit over any
    amount it could still receive; the best layer gets that amount. Handles non-convex curves.
    """
    L, width = curves.shape
    if width < extra + 1:
        raise ValueError("curves must cover k .. k + extra pages")
    alloc = np.zeros(L, np.int64)  # in units
    left = extra // unit
    while left > 0:
        best_ratio, best_l, best_n = 0.0, -1, 0
        steps = np.arange(1, left + 1)
        for l in range(L):
            cur = curves[l, alloc[l] * unit]
            gains = cur - curves[l, (alloc[l] + steps) * unit]
            ratio = gains / steps
            j = int(np.argmax(ratio))
            if ratio[j] > best_ratio:
                best_ratio, best_l, best_n = float(ratio[j]), l, j + 1
        if best_l < 0:  # no layer gains from more pages: spread the rest evenly (it changes nothing)
            alloc += left // L
            alloc[: left % L] += 1
            break
        alloc[best_l] += best_n
        left -= best_n
    return alloc * unit


def total_misses(curves: np.ndarray, extra_per_layer: np.ndarray) -> int:
    return int(curves[np.arange(len(curves)), extra_per_layer].sum())


def uniform_extra(L: int, extra: int) -> np.ndarray:
    """Uniform split: every layer gets extra / L more pages (k more, so B_l = 2k)."""
    out = np.full(L, extra // L, np.int64)
    out[: extra % L] += 1
    return out
