"""Tests for trace/sparse_check.py on a tiny random Qwen3 (CPU is fine, no download).

Run:  python trace/test_sparse.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sim"))
import sparse_check  # noqa: E402
from cache import simulate  # noqa: E402
from quest import PAGE  # noqa: E402
from test_sim import synthetic_trace  # noqa: E402


def test_selection_covers_whole_pages_and_forced_pages():
    ctl = sparse_check.SparseControl(n_layers=1)
    ctl.configure(k=3)
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(2, 5 * PAGE + 4, 8, generator=g)  # 6 pages, the last one partly filled
    q = torch.randn(4, 8, generator=g)
    idx = ctl.select(0, q, keys, scaling=8 ** -0.5)
    pages = sorted(set((idx // PAGE).tolist()))
    assert len(pages) == 3 and 0 in pages and 5 in pages  # sink and newest always read
    assert idx.max().item() == keys.shape[1] - 1  # the current token (end of the newest page) is read
    assert all(((idx // PAGE) == p).sum().item() == PAGE for p in pages if p != 5)  # whole pages


def test_resident_pages_match_the_lru_simulator():
    # The resident set after step t must be exactly the LRU cache of sim/cache.py: count the misses it
    # implies at step t + 1 and compare with the simulator, step by step.
    for seed in range(3):
        sel, newest = synthetic_trace(T=200, k=12, start_pages=40, keep=0.7, seed=seed)
        k = sel.shape[1]
        for B in (k, k + 6, 2 * k):
            ctl = sparse_check.SparseControl(n_layers=1, capacity_pages=int(newest.max()) + 2)
            ctl.configure(k, mode="resident", B=B)
            expect, _ = simulate(sel, newest, B, "lru")
            resident = set()
            for t in range(len(sel)):
                s = sel[t][sel[t] >= 0]
                got = sum(1 for p in s if p not in resident and p != newest[t])
                assert got == expect[t], (seed, B, t, got, expect[t])
                res = ctl.resident_pages(0, torch.tensor(s), t, int(newest[t]) + 1)
                resident = set(res.tolist())
                assert set(s.tolist()) <= resident and len(resident) <= B


def test_smoke_run():
    # With k = 64 pages the tiny prompt is read completely: every mode must match full attention.
    # With k = 2 it must not, and the resident mode must read more pages than the quest mode.
    with tempfile.TemporaryDirectory() as tmp:
        rows = sparse_check.main(["--smoke", "--out", os.path.join(tmp, "check")])
    by = {r["name"]: r for r in rows}
    full = by["full attention"]["forced"]
    for name in ("quest k=64", "quest k=64 + resident, B=2k", "oracle k=64"):
        f = by[name]["forced"]
        assert f["kl"] < 1e-6 and f["agree"] == 1.0, (name, f)
        assert abs(f["ppl"] - full["ppl"]) < 1e-3 * full["ppl"]
        assert abs(by[name]["coverage"] - 1.0) < 1e-4, (name, by[name]["coverage"])
    assert by["quest k=2"]["forced"]["kl"] > 1e-4
    assert by["quest k=2 + resident, B=2k"]["read_tokens_per_call"] > by["quest k=2"]["read_tokens_per_call"]
    alloc = by["quest k=2 + resident, B per layer from smoke allocation"]
    assert alloc["read_tokens_per_call"] > by["quest k=2"]["read_tokens_per_call"]
    assert np.isfinite(by["oracle k=2"]["forced"]["kl"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
