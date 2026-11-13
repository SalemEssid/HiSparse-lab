"""Tests for trace/sparse_check.py on a tiny random Qwen3 (CPU is fine, no download).

Run:  python trace/test_sparse.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_check  # noqa: E402
from quest import PAGE  # noqa: E402


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
    full_pages = [p for p in pages if p != 5]
    assert all(((idx // PAGE) == p).sum().item() == PAGE for p in full_pages)  # whole pages


def test_smoke_run_all_pages_equals_full_attention():
    # k = 64 pages is more than the tiny prompt has, so sparse attention reads every token:
    # it must match full attention exactly. k = 2 must not.
    with tempfile.TemporaryDirectory() as tmp:
        rows = sparse_check.main(["--smoke", "--out", os.path.join(tmp, "check")])
    by_k = {(r["k"], tuple(r["dense_layers"])): r for r in rows}
    full = by_k[(None, ())]["forced"]
    everything = by_k[(64, ())]["forced"]
    assert everything["kl"] < 1e-6 and everything["agree"] == 1.0, everything
    assert abs(everything["ppl"] - full["ppl"]) < 1e-3 * full["ppl"]
    small = by_k[(2, ())]["forced"]
    assert small["kl"] > 1e-4, small  # reading 2 pages out of ~22 changes the predictions
    with_dense = by_k[(2, (0, 1))]["forced"]
    assert with_dense["kl"] != small["kl"]  # keeping layers 0-1 full changes the result


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
