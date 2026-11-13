"""Quest page selection and attention-coverage math (pure torch, no model code).

Shapes for one request, one layer, one decode step:
    q          [Hq, D]       query of the current token, after q_norm and RoPE (Hq = 32 for Qwen3-4B)
    keys       [Hkv, S, D]   cached keys including the current token, after k_norm and RoPE (Hkv = 8)
    kmin, kmax [Hkv, P, D]   elementwise min / max of the keys of each page of PAGE tokens

GQA mapping: query head h reads KV head h // (Hq // Hkv), the same grouping as transformers' repeat_kv.
Page p holds token positions [p * PAGE, (p + 1) * PAGE).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PAGE = 16
_TOP = float("inf")
_SECOND = torch.finfo(torch.float32).max  # ranks right after +inf


class PageSummary:
    """Per-layer page min/max over cached keys: built once, then updated one token per decode step."""

    def __init__(self, keys: torch.Tensor, page: int = PAGE, capacity_pages: int = 0):
        hkv, s, d = keys.shape
        self.page = page
        n = (s + page - 1) // page
        cap = max(capacity_pages, n, 1)
        self.kmin = torch.empty((hkv, cap, d), dtype=torch.float32, device=keys.device)
        self.kmax = torch.empty((hkv, cap, d), dtype=torch.float32, device=keys.device)
        k32 = keys.float()
        full = s // page
        if full:
            blocks = k32[:, : full * page].reshape(hkv, full, page, d)
            self.kmin[:, :full] = blocks.amin(dim=2)
            self.kmax[:, :full] = blocks.amax(dim=2)
        if s % page:
            tail = k32[:, full * page :]
            self.kmin[:, full] = tail.amin(dim=1)
            self.kmax[:, full] = tail.amax(dim=1)
        self.tokens = s

    @property
    def n_pages(self) -> int:
        return (self.tokens + self.page - 1) // self.page

    def append(self, key: torch.Tensor) -> None:
        """Add the newest token's key, shape [Hkv, D], at position self.tokens."""
        pos = self.tokens
        p = pos // self.page
        if p >= self.kmin.shape[1]:
            grow = self.kmin.shape[1]
            self.kmin = torch.cat([self.kmin, torch.empty_like(self.kmin[:, :grow])], dim=1)
            self.kmax = torch.cat([self.kmax, torch.empty_like(self.kmax[:, :grow])], dim=1)
        k = key.float()
        if pos % self.page == 0:
            self.kmin[:, p] = k
            self.kmax[:, p] = k
        else:
            self.kmin[:, p] = torch.minimum(self.kmin[:, p], k)
            self.kmax[:, p] = torch.maximum(self.kmax[:, p], k)
        self.tokens = pos + 1

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_pages
        return self.kmin[:, :n], self.kmax[:, :n]


def quest_upper_bounds(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """Quest's per-head upper bound of q.k over each page: sum_d max(q_d * min_d, q_d * max_d). Returns [Hq, P]."""
    hq, d = q.shape
    hkv = kmin.shape[0]
    qg = q.float().reshape(hkv, hq // hkv, d)
    ub = torch.einsum("grd,gpd->grp", qg.clamp(min=0), kmax) + torch.einsum("grd,gpd->grp", qg.clamp(max=0), kmin)
    return ub.reshape(hq, -1)


def _force(score: torch.Tensor, n_pages: int) -> torch.Tensor:
    """Rank the newest page second and the attention-sink page first, on the last dim."""
    score = score.clone()
    score[..., n_pages - 1] = _SECOND
    score[..., 0] = _TOP
    return score


def aggregate_scores(ub: torch.Tensor, scaling: float, how: str) -> torch.Tensor:
    """Combine per-head page scores into one score per page. [Hq, P] -> [P]."""
    if how == "mean_logit":
        return (ub * scaling).mean(dim=0)
    if how == "mean_softmax":
        return torch.softmax(ub * scaling, dim=-1).mean(dim=0)
    raise ValueError(f"unknown aggregation {how!r}")


def rank_pages(ub: torch.Tensor, scaling: float, k_max: int, how: str = "mean_logit") -> torch.Tensor:
    """One shared ranking per layer: sink page, newest page, then the rest by aggregated score.

    Returns int64 page ids, length min(k_max, P). Any k <= k_max selection is a prefix of this list.
    """
    n = ub.shape[-1]
    score = _force(aggregate_scores(ub, scaling, how), n)
    return torch.topk(score, min(k_max, n), sorted=True).indices


def rank_pages_per_head(ub: torch.Tensor, k: int) -> torch.Tensor:
    """Each query head's own ranking (original Quest), same forced pages. Returns [Hq, min(k, P)]."""
    n = ub.shape[-1]
    return torch.topk(_force(ub, n), min(k, n), dim=-1, sorted=True).indices


def attention_probs(q: torch.Tensor, keys: torch.Tensor, scaling: float) -> torch.Tensor:
    """Exact softmax attention of the current token over all cached tokens, per query head. Returns [Hq, S]."""
    hq, d = q.shape
    hkv = keys.shape[0]
    logits = torch.einsum("grd,gsd->grs", q.float().reshape(hkv, hq // hkv, d), keys.float()) * scaling
    return torch.softmax(logits, dim=-1).reshape(hq, -1)


def page_mass(probs: torch.Tensor, page: int = PAGE) -> torch.Tensor:
    """Attention mass per page and query head. [Hq, S] -> [Hq, P]."""
    hq, s = probs.shape
    n = (s + page - 1) // page
    if n * page != s:
        probs = F.pad(probs, (0, n * page - s))
    return probs.reshape(hq, n, page).sum(dim=-1)


def coverage(
    probs: torch.Tensor,
    ub: torch.Tensor,
    scaling: float,
    ks: list[int],
    page: int = PAGE,
) -> dict[str, torch.Tensor]:
    """Attention mass captured by each selection rule, averaged over query heads. Each value is [len(ks)].

    quest_logit   shared ranking from head-averaged scaled upper bounds (the logged trace)
    quest_softmax shared ranking from head-averaged per-head softmax of upper bounds
    quest_head    every head picks its own pages (original Quest; not a shared cache)
    page_oracle   best shared set of k pages, chosen from the true attention
    token_oracle  exact top-(k * page) tokens per head (the "exact top-k" bound)
    """
    hq, s = probs.shape
    mass = page_mass(probs, page)
    n = mass.shape[1]
    kmax = min(max(ks), n)
    r_logit = rank_pages(ub, scaling, kmax, "mean_logit")
    r_soft = rank_pages(ub, scaling, kmax, "mean_softmax")
    r_head = rank_pages_per_head(ub, kmax)
    page_cum = mass.mean(dim=0).sort(descending=True).values.cumsum(0)
    tok_cum = probs.sort(dim=-1, descending=True).values.cumsum(dim=-1)
    out = {key: [] for key in ("quest_logit", "quest_softmax", "quest_head", "page_oracle", "token_oracle")}
    for k in ks:
        kk = min(k, n)
        out["quest_logit"].append(mass[:, r_logit[:kk]].sum(-1).mean())
        out["quest_softmax"].append(mass[:, r_soft[:kk]].sum(-1).mean())
        out["quest_head"].append(mass.gather(1, r_head[:, :kk]).sum(-1).mean())
        out["page_oracle"].append(page_cum[kk - 1])
        out["token_oracle"].append(tok_cum[:, min(k * page, s) - 1].mean())
    return {key: torch.stack(v) for key, v in out.items()}
