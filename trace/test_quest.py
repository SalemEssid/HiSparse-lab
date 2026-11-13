"""Unit tests for trace/quest.py. CPU only, no model download.

Run:  python trace/test_quest.py      (or: python -m pytest trace/test_quest.py)
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quest import (  # noqa: E402
    PageSummary,
    attention_probs,
    coverage,
    page_mass,
    quest_upper_bounds,
    rank_pages,
    rank_pages_per_head,
)


def test_upper_bound_hand_example():
    # One KV head, one query head, D = 2, pages of 2 tokens. Worked out on paper:
    # page 0 = tokens (1,-1), (3,2): min (1,-1), max (3,2)
    # page 1 = tokens (-2,0), (0,4): min (-2,0), max (0,4)
    # q = (1,-1): page 0 -> max(1,3) + max(1,-2) = 4 ; page 1 -> max(-2,0) + max(0,-4) = 0
    keys = torch.tensor([[[1.0, -1.0], [3.0, 2.0], [-2.0, 0.0], [0.0, 4.0]]])
    summ = PageSummary(keys, page=2)
    kmin, kmax = summ.view()
    assert torch.equal(kmin[0], torch.tensor([[1.0, -1.0], [-2.0, 0.0]]))
    assert torch.equal(kmax[0], torch.tensor([[3.0, 2.0], [0.0, 4.0]]))
    ub = quest_upper_bounds(torch.tensor([[1.0, -1.0]]), kmin, kmax)
    assert torch.equal(ub, torch.tensor([[4.0, 0.0]]))


def test_upper_bound_is_an_upper_bound():
    g = torch.Generator().manual_seed(0)
    hq, hkv, s, d, page = 8, 2, 101, 16, 16
    q = torch.randn(hq, d, generator=g)
    keys = torch.randn(hkv, s, d, generator=g)
    ub = quest_upper_bounds(q, *PageSummary(keys, page=page).view())
    rep = hq // hkv
    for h in range(hq):
        dots = keys[h // rep] @ q[h]
        for p in range(ub.shape[1]):
            assert ub[h, p] >= dots[p * page : (p + 1) * page].max() - 1e-5


def test_incremental_summary_matches_rebuild():
    g = torch.Generator().manual_seed(1)
    keys = torch.randn(4, 77, 8, generator=g)
    summ = PageSummary(keys[:, :5], page=16, capacity_pages=1)  # forces capacity growth too
    for t in range(5, 77):
        summ.append(keys[:, t])
    ref = PageSummary(keys, page=16)
    assert summ.tokens == ref.tokens and summ.n_pages == ref.n_pages == 5
    for a, b in zip(summ.view(), ref.view()):
        assert torch.equal(a, b)


def test_gqa_mapping():
    # 4 query heads, 2 KV heads: heads 0-1 read KV head 0, heads 2-3 read KV head 1.
    keys = torch.zeros(2, 32, 4)
    keys[1, 16:] = 10.0  # only KV head 1, page 1, is large
    q = torch.ones(4, 4)
    ub = quest_upper_bounds(q, *PageSummary(keys, page=16).view())
    assert torch.all(ub[:2] == 0) and torch.all(ub[2:, 1] == 40.0) and torch.all(ub[2:, 0] == 0)


def test_rank_forced_pages_and_prefix():
    ub = torch.tensor([[0.0, 9.0, 1.0, 8.0, 2.0, -5.0]])  # 6 pages, one head
    r = rank_pages(ub, scaling=1.0, k_max=4)
    assert r.tolist() == [0, 5, 1, 3]  # sink, newest, then by score
    assert rank_pages(ub, 1.0, k_max=10).tolist() == [0, 5, 1, 3, 4, 2]  # k_max > pages
    assert rank_pages(ub, 1.0, k_max=3).tolist() == r[:3].tolist()  # smaller k is a prefix
    assert rank_pages_per_head(ub, 3).tolist() == [[0, 5, 1]]


def test_attention_probs_match_sdpa():
    g = torch.Generator().manual_seed(2)
    hq, hkv, s, d = 8, 2, 40, 16
    q = torch.randn(hq, d, generator=g)
    keys = torch.randn(hkv, s, d, generator=g)
    vals = torch.randn(hkv, s, d, generator=g)
    scaling = d ** -0.5
    p = attention_probs(q, keys, scaling)
    assert torch.allclose(p.sum(-1), torch.ones(hq), atol=1e-5)
    k_rep = keys.repeat_interleave(hq // hkv, dim=0)  # what transformers' repeat_kv does
    v_rep = vals.repeat_interleave(hq // hkv, dim=0)
    ours = torch.einsum("hs,hsd->hd", p, v_rep)
    ref = F.scaled_dot_product_attention(q[None, :, None], k_rep[None], v_rep[None], scale=scaling)[0, :, 0]
    assert torch.allclose(ours, ref, atol=1e-5)


def test_page_mass_and_coverage_bounds():
    g = torch.Generator().manual_seed(3)
    hq, hkv, s, d = 8, 2, 200, 16
    q = torch.randn(hq, d, generator=g) * 3
    keys = torch.randn(hkv, s, d, generator=g)
    scaling = d ** -0.5
    probs = attention_probs(q, keys, scaling)
    mass = page_mass(probs, 16)
    assert mass.shape == (hq, 13) and torch.allclose(mass.sum(-1), torch.ones(hq), atol=1e-5)
    ub = quest_upper_bounds(q, *PageSummary(keys).view())
    cov = coverage(probs, ub, scaling, ks=[2, 4, 8, 13, 50])
    eps = 1e-5
    # Ordering that must hold for any input:
    assert torch.all(cov["page_oracle"] + eps >= cov["quest_logit"])
    assert torch.all(cov["page_oracle"] + eps >= cov["quest_softmax"])
    assert torch.all(cov["token_oracle"] + eps >= cov["page_oracle"])
    assert torch.all(cov["token_oracle"] + eps >= cov["quest_head"])
    for key, v in cov.items():  # more pages never covers less; all pages covers everything
        assert torch.all(v[1:] + eps >= v[:-1]), key
        assert math.isclose(v[-1].item(), 1.0, abs_tol=1e-5), key


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
