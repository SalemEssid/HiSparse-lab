"""Does Quest's selection loss hurt the model, and can pages already in the GPU cache win it back?

Every configuration decodes with real sparse attention: at each decode step, each layer reads only a set of
pages (prefill stays full, as in Quest and HiSparse). The KV cache still stores every token. Modes:

  quest     read the k pages Quest selects (sink, newest, best scores)
  resident  Quest still selects k pages, and the GPU cache of B pages per layer is managed by LRU over those
            selections, as in HiSparse. Attention reads every resident page: the k selected plus the B - k
            most recently selected others. They are already on the GPU, so this costs no extra transfer;
            the misses are exactly those of the quest mode with the same B. B can come from a Phase 4
            allocation file (one size per layer).
  oracle    read the k pages that truly carry the most attention (from the exact attention): separates the
            loss from Quest's estimate from the loss from the page budget itself.

Tests per configuration: the G1 passphrase at 32K (3 depths), and teacher-forced next-token comparison with
full attention on the Workload A traces: top-1 agreement, KL divergence, perplexity of the generated text.
Also the coverage of what attention reads (share of the true attention mass), sampled every --cov-every steps.

Usage (GPU VM):  python trace/sparse_check.py [--alloc sim/results/alloc_A_k128.json]   # ~30 min
                 python trace/sparse_check.py --smoke                                   # tiny random model
Writes trace/results/SPARSE_CHECK.md and .json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quest import PAGE, PageSummary, _force, attention_probs, page_mass, quest_upper_bounds, rank_pages  # noqa: E402

ATTN_NAME = "quest_sparse"
CONTROL = None  # the active SparseControl; read by the attention function


class SparseControl:
    """Decides, per layer and decode step, which cached tokens attention may read."""

    def __init__(self, n_layers: int, capacity_pages: int = 0):
        self.n_layers, self.capacity = n_layers, max(capacity_pages, 1)
        self.k, self.dense, self.active, self.mode, self.B = 0, set(), False, "quest", None
        self.cov_every = 0
        self.reset_stats()
        self.configure(0, active=False)

    def reset_stats(self) -> None:
        self.read_tokens = self.calls = 0
        self.cov_samples: list[tuple[int, torch.Tensor]] = []

    def configure(self, k: int, dense_layers=(), active: bool = True, mode: str = "quest", B=None) -> None:
        """Call before every new sequence: per-layer state is rebuilt from its cache at the first decode step."""
        if mode not in ("quest", "resident", "oracle"):
            raise ValueError(mode)
        self.k, self.dense, self.active, self.mode = k, set(dense_layers), active, mode
        if mode == "resident":
            self.B = list(B) if hasattr(B, "__len__") else [int(B)] * self.n_layers
            assert min(self.B) >= k, "every layer needs B >= k"
        self.summaries: list[PageSummary | None] = [None] * self.n_layers
        self.recency: list[torch.Tensor | None] = [None] * self.n_layers
        self.step_no = [0] * self.n_layers

    def select(self, layer: int, q: torch.Tensor, keys: torch.Tensor, scaling: float) -> torch.Tensor:
        """Token positions attention may read at this step."""
        s = keys.shape[1]
        t = self.step_no[layer]
        sample = self.cov_every and t % self.cov_every == 0
        probs = attention_probs(q, keys, scaling) if (self.mode == "oracle" or sample) else None
        if self.mode == "oracle":
            mass = page_mass(probs, PAGE).mean(dim=0)
            pages = torch.topk(_force(mass, mass.numel()), min(self.k, mass.numel())).indices
        else:
            summ = self.summaries[layer]
            if summ is None:
                summ = PageSummary(keys, PAGE, self.capacity)
                self.summaries[layer] = summ
            else:
                assert summ.tokens == s - 1, f"layer {layer}: summary has {summ.tokens} tokens, cache has {s}"
                summ.append(keys[:, s - 1])
            pages = rank_pages(quest_upper_bounds(q, *summ.view()), scaling, self.k)
        if self.mode == "resident":
            pages = self.resident_pages(layer, pages, t, s // PAGE + 1)
        idx = (pages[:, None] * PAGE + torch.arange(PAGE, device=keys.device)).reshape(-1)
        idx = torch.sort(idx[idx < s]).values
        if sample:
            self.cov_samples.append((layer, page_mass(probs, PAGE).mean(dim=0)[pages].sum()))
        self.read_tokens += idx.numel()
        self.calls += 1
        self.step_no[layer] = t + 1
        return idx

    def resident_pages(self, layer: int, selected: torch.Tensor, t: int, n_pages: int) -> torch.Tensor:
        """Pages in this layer's GPU cache after step t, under LRU over the selections (HiSparse's cache).

        LRU contents are the B most recently selected pages, ties within a step broken by rank (the
        stack property proved in sim/stack.py), so one top-B over recency keys gives them.
        """
        rec = self.recency[layer]
        if rec is None or rec.numel() < n_pages:
            grown = torch.full((max(self.capacity, n_pages),), -math.inf, device=selected.device)
            if rec is not None:
                grown[: rec.numel()] = rec
            rec = self.recency[layer] = grown
        rank = torch.arange(selected.numel(), device=selected.device, dtype=rec.dtype)
        rec[selected] = float(t * 2 * self.k) + (self.k - rank)
        vals, top = torch.topk(rec, min(self.B[layer], rec.numel()))
        return top[vals > -math.inf]

    def coverage(self) -> tuple[float, float]:
        """Mean coverage of what attention read, over samples, and the lowest per-layer mean."""
        if not self.cov_samples:
            return float("nan"), float("nan")
        vals = torch.stack([v for _, v in self.cov_samples]).float().cpu().numpy()
        layers = np.array([l for l, _ in self.cov_samples])
        per_layer = [vals[layers == l].mean() for l in sorted(set(layers.tolist()))]
        return float(vals.mean()), float(min(per_layer))


def quest_sparse_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    ctl = CONTROL
    if ctl is not None and ctl.active and query.shape[2] == 1 and module.layer_idx not in ctl.dense:
        scaling = kwargs.get("scaling")
        idx = ctl.select(module.layer_idx, query[0, :, 0], key[0], scaling if scaling is not None else module.scaling)
        key, value = key[:, :, idx], value[:, :, idx]
        if attention_mask is not None:
            attention_mask = attention_mask[..., idx]
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register_attention() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_NAME, quest_sparse_attention)
    try:
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

        AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
    except ImportError:
        pass


# ---------------------------------------------------------------- the two tests

@torch.inference_mode()
def forced_pass(model, prompt: list[int], cont: list[int], ref=None) -> dict:
    """Full prefill of `prompt`, then feed `cont` one token at a time.

    Step t feeds cont[t] and predicts cont[t + 1]. With ref=None, returns the full model's log-probs
    (kept on the GPU in float32, ~240 MB per 400-step trace) as the reference; otherwise compares against `ref`.
    float16 would round the reference enough to show a KL of ~1e-5 even between identical models.
    """
    from transformers import DynamicCache

    base, head, dev = model.model, model.lm_head, next(model.parameters()).device
    cache = DynamicCache()
    base(input_ids=torch.tensor([prompt], device=dev), past_key_values=cache, use_cache=True)
    n = len(cont) - 1
    targets = torch.tensor(cont[1:], device=dev)
    nll = torch.zeros(n, device=dev)
    kl = torch.zeros(n, device=dev)
    agree = torch.zeros(n, dtype=torch.bool, device=dev)
    store = torch.empty((n, model.config.vocab_size), dtype=torch.float32, device=dev) if ref is None else None
    for t in range(n):
        out = base(input_ids=torch.tensor([[cont[t]]], device=dev), past_key_values=cache, use_cache=True)
        logp = torch.log_softmax(head(out.last_hidden_state[:, -1]).float(), dim=-1)[0]
        nll[t] = -logp[targets[t]]
        if ref is None:
            store[t] = logp
        else:
            r = ref[t]
            kl[t] = (r.exp() * (r - logp)).sum()
            agree[t] = r.argmax() == logp.argmax()
    if ref is None:
        return {"ref": store, "nll": nll.cpu().numpy()}
    return {"nll": nll.cpu().numpy(), "kl": kl.cpu().numpy(), "agree": agree.cpu().numpy()}


@torch.inference_mode()
def needle_answer(model, tok, input_ids: torch.Tensor, steps: int) -> str:
    from transformers import DynamicCache

    base, head = model.model, model.lm_head
    cache = DynamicCache()
    out = base(input_ids=input_ids, past_key_values=cache, use_cache=True)
    nxt = head(out.last_hidden_state[:, -1]).argmax(-1)
    ids = []
    for _ in range(steps):
        ids.append(int(nxt.item()))
        out = base(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
        nxt = head(out.last_hidden_state[:, -1]).argmax(-1)
    return tok.decode(ids, skip_special_tokens=True) if tok is not None else ""


# ---------------------------------------------------------------- report

def summarize(rows: list[dict]) -> str:
    pct = lambda v: "–" if v is None or v != v else f"{100 * v:.1f}%"  # noqa: E731
    lines = ["# Sparse attention vs full attention", "",
             "Every row decodes with real sparse attention (prefill full). Resident rows read every page already in "
             "the GPU cache, at no extra transfer.", "",
             "| Configuration | Pages read per layer | Coverage of what is read (lowest layer) | Passphrase "
             "| Top-1 agreement | Mean KL (nats) | Steps with KL > 0.1 | Perplexity of the text |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        f = r["forced"]
        needle = f"{r['needle_found']} / {r['needle_total']}"
        if f["kl"] is None:
            lines.append(f"| {r['name']} | all | 100% | {needle} | (reference) | 0 | 0% | {f['ppl']:.3f} |")
        else:
            lines.append(f"| {r['name']} | {r['read_tokens_per_call'] / PAGE:,.0f} | {pct(r['coverage'])} ({pct(r['coverage_min_layer'])}) "
                         f"| {needle} | {pct(f['agree'])} | {f['kl']:.4f} | {pct(f['kl_gt_01'])} | {f['ppl']:.3f} |")
    lines += ["", "Coverage: share of the true attention mass on the pages attention reads, mean over heads and sampled steps. "
              "Top-1 agreement: steps where the most likely next token matches full attention. KL: divergence from the full "
              "model's next-token distribution (0 = identical). Perplexity: how surprised the model is by the actual "
              "generated text (lower is better; the text was written by the full model)."]
    misses = [(r["name"], d) for r in rows for d in r["needle_missed"]]
    if misses:
        lines += ["", "Passphrase missed: " + "; ".join(f"{n} at depth {d}" for n, d in misses) + "."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def build_configs(ks: list[int], dense_layers: list[int], allocs: list[dict], oracle: bool) -> list[dict]:
    cfgs = [{"name": "full attention", "k": None}]
    for k in ks:
        cfgs.append({"name": f"quest k={k}", "k": k, "mode": "quest"})
        cfgs.append({"name": f"quest k={k} + resident, B=2k", "k": k, "mode": "resident", "B": 2 * k})
        for a in allocs:
            if a["k"] == k:
                cfgs.append({"name": f"quest k={k} + resident, B per layer from {a['label']}", "k": k,
                             "mode": "resident", "B": a["B"]})
        if oracle:
            cfgs.append({"name": f"oracle k={k}", "k": k, "mode": "oracle"})
        if dense_layers:
            cfgs.append({"name": f"quest k={k}, layers {' '.join(map(str, dense_layers))} full", "k": k,
                         "mode": "quest", "dense": tuple(dense_layers)})
    return cfgs


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 32])
    ap.add_argument("--alloc", nargs="*", default=[], help="Phase 4 allocation files (sim/results/alloc_*.json)")
    ap.add_argument("--dense-layers", type=int, nargs="*", default=[], help="extra rows keeping these layers full")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--traces", default="data/traces", help="folder with A_*.npz traces for teacher forcing")
    ap.add_argument("--max-steps", type=int, default=400, help="teacher-forced steps per trace")
    ap.add_argument("--needle-context", type=int, default=32000)
    ap.add_argument("--depths", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    ap.add_argument("--cov-every", type=int, default=25, help="coverage sample every N decode steps per layer")
    ap.add_argument("--out", default="trace/results/SPARSE_CHECK")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    global CONTROL
    register_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allocs = []
    for path in args.alloc:
        a = json.loads(Path(path).read_text(encoding="utf-8"))
        allocs.append({"k": int(a["k"]), "B": [int(b) for b in a["B"]], "label": Path(path).stem})
    if args.smoke:
        import collect

        collect.ATTN_NAME = ATTN_NAME
        model, tok = collect.build_tiny_model(device, torch.float32), None
        g = torch.Generator().manual_seed(0)
        docs = [(torch.randint(0, 512, (300,), generator=g).tolist(), torch.randint(0, 512, (41,), generator=g).tolist())]
        ks, depths = [2, 64], []
        if not allocs:
            allocs = [{"k": 2, "B": [2, 6, 4], "label": "smoke allocation"}]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation=ATTN_NAME).to(device).eval()
        docs = []
        for p in sorted(Path(args.traces).glob("A_*.npz")):
            with np.load(p) as d:
                docs.append((d["prompt_ids"].tolist(), d["tokens"][: args.max_steps + 1].tolist()))
        if not docs:
            sys.exit(f"no traces in {args.traces}/A_*.npz")
        ks, depths = args.ks, args.depths

    n_layers = model.config.num_hidden_layers
    longest = max(len(p) + len(c) for p, c in docs)
    CONTROL = SparseControl(n_layers, capacity_pages=(max(longest, args.needle_context) + 64) // PAGE + 2)
    CONTROL.cov_every = args.cov_every if not args.smoke else 5
    configs = build_configs(ks, args.dense_layers, allocs, not args.no_oracle)

    needles = []
    if depths:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hw"))
        from decode_32k import PASSPHRASE, build_prompt

        for depth in depths:
            ids, _ = build_prompt(tok, args.needle_context, depth, seed=0)
            needles.append((depth, ids.to(device)))

    def setup(cfg: dict) -> None:
        if cfg["k"] is None:
            CONTROL.configure(0, active=False)
        else:
            CONTROL.configure(cfg["k"], cfg.get("dense", ()), True, cfg["mode"], cfg.get("B"))

    refs, rows, t0 = [], [], time.time()
    for cfg in configs:
        CONTROL.reset_stats()
        found, missed = 0, []
        for depth, ids in needles:
            setup(cfg)
            ans = needle_answer(model, tok, ids, 24)
            if PASSPHRASE.lower() in ans.lower():
                found += 1
            else:
                missed.append(depth)
        nll, kl, agree = [], [], []
        for j, (prompt, cont) in enumerate(docs):
            setup(cfg)
            if cfg["k"] is None:
                res = forced_pass(model, prompt, cont)
                refs.append(res["ref"])
            else:
                res = forced_pass(model, prompt, cont, ref=refs[j])
                kl.append(res["kl"])
                agree.append(res["agree"])
            nll.append(res["nll"])
        nll_all = np.concatenate(nll)
        forced = {"ppl": float(math.exp(nll_all.mean())), "kl": None, "agree": None, "kl_gt_01": None}
        if kl:
            kl_all, ag_all = np.concatenate(kl), np.concatenate(agree)
            forced.update(kl=float(kl_all.mean()), agree=float(ag_all.mean()), kl_gt_01=float((kl_all > 0.1).mean()),
                          per_trace={"kl": [float(x.mean()) for x in kl], "agree": [float(x.mean()) for x in agree]})
        cov, cov_min = CONTROL.coverage()
        rows.append({"name": cfg["name"], "k": cfg["k"], "mode": cfg.get("mode"), "dense_layers": list(cfg.get("dense", ())),
                     "B": cfg.get("B"), "needle_found": found, "needle_total": len(needles), "needle_missed": missed,
                     "forced": forced, "steps": int(len(nll_all)), "coverage": cov, "coverage_min_layer": cov_min,
                     "read_tokens_per_call": CONTROL.read_tokens / max(CONTROL.calls, 1)})
        print(f"[{time.time() - t0:.0f} s] {cfg['name']}: passphrase {found}/{len(needles)}, ppl {forced['ppl']:.3f}"
              + (f", agree {forced['agree']:.3f}, KL {forced['kl']:.4f}, coverage {cov:.3f}" if kl else ""), flush=True)

    md = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"model": "tiny" if args.smoke else args.model,
                                                    "traces": len(docs), "rows": rows}, indent=2), encoding="utf-8")
    print(md)
    return rows


if __name__ == "__main__":
    main()
