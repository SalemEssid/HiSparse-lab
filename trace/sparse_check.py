"""Does Quest's selection loss hurt the model? Compare full attention with true Quest sparse attention.

Sparse mode: at every decode step, each layer's attention reads only the tokens of its k selected pages
(sink page, newest page, then Quest's best). Prefill stays full, as in Quest and HiSparse. The KV cache
still stores every token; only what attention reads changes.

Configurations: full attention, then sparse at each k in --ks, each with all layers sparse and with
--dense-layers (default 0 1) kept on full attention.

Two tests per configuration:
  1. Passphrase: the G1 needle test at 32K, needle at 5 depths, greedy answer. Found or not.
  2. Next-token comparison on the Workload A traces: feed each trace's prompt and generated text
     (teacher forcing), and at every step compare the sparse model's next-token distribution with the
     full model's: top-1 agreement, KL divergence, and perplexity of the actual text.

Usage (GPU VM):  python trace/sparse_check.py            # ~20 min
                 python trace/sparse_check.py --smoke    # tiny random model, any device
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
from quest import PAGE, PageSummary, quest_upper_bounds, rank_pages  # noqa: E402

ATTN_NAME = "quest_sparse"
CONTROL = None  # the active SparseControl; read by the attention function


class SparseControl:
    """Decides, per layer and decode step, which cached tokens attention may read."""

    def __init__(self, n_layers: int, capacity_pages: int = 0):
        self.n_layers, self.capacity = n_layers, capacity_pages
        self.k, self.dense, self.active = 0, set(), False
        self.summaries: list[PageSummary | None] = [None] * n_layers
        self.read_tokens = self.calls = 0

    def configure(self, k: int, dense_layers=(), active: bool = True) -> None:
        """Call before every new sequence: page summaries are rebuilt from its cache at the first decode step."""
        self.k, self.dense, self.active = k, set(dense_layers), active
        self.summaries = [None] * self.n_layers

    def select(self, layer: int, q: torch.Tensor, keys: torch.Tensor, scaling: float) -> torch.Tensor:
        """Token positions attention may read: all tokens of the k pages Quest selects."""
        s = keys.shape[1]
        summ = self.summaries[layer]
        if summ is None:
            summ = PageSummary(keys, PAGE, self.capacity)
            self.summaries[layer] = summ
        else:
            assert summ.tokens == s - 1, f"layer {layer}: summary has {summ.tokens} tokens, cache has {s}"
            summ.append(keys[:, s - 1])
        pages = rank_pages(quest_upper_bounds(q, *summ.view()), scaling, self.k)
        idx = (pages[:, None] * PAGE + torch.arange(PAGE, device=keys.device)).reshape(-1)
        idx = torch.sort(idx[idx < s]).values
        self.read_tokens += idx.numel()
        self.calls += 1
        return idx


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
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = ["# Quest sparse attention vs full attention", "",
             "Sparse: each layer reads only the tokens of its k selected pages at every decode step (prefill full).", "",
             "| Configuration | Passphrase found | Top-1 agreement | Mean KL (nats) | Steps with KL > 0.1 "
             "| Perplexity of the text | Tokens read per layer |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        f = r["forced"]
        needle = f"{r['needle_found']} / {r['needle_total']}"
        if f["kl"] is None:  # the full-attention reference
            lines.append(f"| {r['name']} | {needle} | (reference) | 0 | 0% | {f['ppl']:.3f} | all |")
        else:
            lines.append(f"| {r['name']} | {needle} | {pct(f['agree'])} | {f['kl']:.4f} | {pct(f['kl_gt_01'])} "
                         f"| {f['ppl']:.3f} | {r['read_tokens_per_call']:,.0f} |")
    lines += ["", "Top-1 agreement: steps where the sparse model's most likely next token equals the full model's. "
              "KL: divergence of the sparse next-token distribution from the full one (0 = identical). "
              "Perplexity: how surprised each model is by the actual generated text (lower is better)."]
    misses = [(r["name"], d) for r in rows for d in r["needle_missed"]]
    if misses:
        lines += ["", "Passphrase missed: " + "; ".join(f"{n} at depth {d}" for n, d in misses) + "."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ks", type=int, nargs="+", default=[128, 32])
    ap.add_argument("--dense-layers", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--traces", default="data/traces", help="folder with A_*.npz traces for teacher forcing")
    ap.add_argument("--max-steps", type=int, default=400, help="teacher-forced steps per trace")
    ap.add_argument("--needle-context", type=int, default=32000)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--out", default="trace/results/SPARSE_CHECK")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    global CONTROL
    register_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        sys.path.insert(0, HERE)
        from collect import build_tiny_model
        import collect

        collect.ATTN_NAME = ATTN_NAME
        model, tok = build_tiny_model(device, torch.float32), None
        g = torch.Generator().manual_seed(0)
        docs = [(torch.randint(0, 512, (300,), generator=g).tolist(), torch.randint(0, 512, (41,), generator=g).tolist())]
        ks, depths = [2, 64], []
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
    configs = [("full attention", None, ())]
    for k in ks:
        configs.append((f"sparse k={k}, all layers", k, ()))
        if args.dense_layers:
            configs.append((f"sparse k={k}, layers {' '.join(map(str, args.dense_layers))} full", k, tuple(args.dense_layers)))

    needles = []
    if depths:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hw"))
        from decode_32k import PASSPHRASE, build_prompt

        for depth in depths:
            ids, _ = build_prompt(tok, args.needle_context, depth, seed=0)
            needles.append((depth, ids.to(device)))

    refs, rows, t0 = [], [], time.time()
    for name, k, dense in configs:
        CONTROL.read_tokens = CONTROL.calls = 0
        if k is None:
            CONTROL.configure(0, (), active=False)
        found, missed = 0, []
        for depth, ids in needles:
            if k is not None:
                CONTROL.configure(k, dense)
            ans = needle_answer(model, tok, ids, 24)
            if PASSPHRASE.lower() in ans.lower():
                found += 1
            else:
                missed.append(depth)
        nll, kl, agree = [], [], []
        for j, (prompt, cont) in enumerate(docs):
            if k is None:
                res = forced_pass(model, prompt, cont)
                refs.append(res["ref"])
            else:
                CONTROL.configure(k, dense)
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
        rows.append({"name": name, "k": k, "dense_layers": list(dense), "needle_found": found,
                     "needle_total": len(needles), "needle_missed": missed, "forced": forced,
                     "steps": int(len(nll_all)), "read_tokens_per_call": CONTROL.read_tokens / max(CONTROL.calls, 1)})
        print(f"[{time.time() - t0:.0f} s] {name}: passphrase {found}/{len(needles)}, ppl {forced['ppl']:.3f}"
              + (f", agree {forced['agree']:.3f}, KL {forced['kl']:.4f}" if kl else ""), flush=True)

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
