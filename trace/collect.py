"""Collect Quest selection traces from dense decoding, and measure Quest's attention coverage (G2).

The model runs normal dense attention. A logging attention function, registered through
transformers' AttentionInterface, sees every layer's query and cached keys at each decode step,
computes the Quest page ranking, and records it. It never changes what attention computes.

Per request it writes:
  data/traces/<name>.npz      ranked page ids [steps, layers, k_max] (int16, -1 = padding),
                              coverage samples, prompt and generated token ids, metadata
  trace/results/<name>.md     coverage table (G2), per-layer spread, step-to-step overlap
  trace/results/<name>.json   the same summary, machine-readable

Usage (GPU VM):
  python trace/collect.py --index 0                 # LongBench v2 item 0, 32K prompt, 1000 steps
  python trace/collect.py --smoke                   # tiny random model, any device, ~1 minute
Run it as a file (python trace/collect.py), not with -m: the folder name shadows Python's trace module.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quest import PAGE, PageSummary, attention_probs, coverage, quest_upper_bounds, rank_pages  # noqa: E402

ATTN_NAME = "quest_logging"
COV_KEYS = ("quest_logit", "quest_softmax", "quest_head", "page_oracle", "token_oracle")
RECORDER = None  # the active Recorder; read by the attention function


# ---------------------------------------------------------------- recording

class Recorder:
    def __init__(self, n_layers, max_steps, k_max, ks, aggregate, cov_every, device, verify=False, capacity_pages=0):
        self.n_layers, self.k_max, self.ks = n_layers, k_max, list(ks)
        self.aggregate, self.cov_every, self.verify = aggregate, cov_every, verify
        self.capacity_pages = capacity_pages
        self.ranked = torch.full((max_steps, n_layers, k_max), -1, dtype=torch.int16, device=device)
        self.summaries: list[PageSummary | None] = [None] * n_layers
        self.cov_rows: list[tuple[int, int, dict]] = []
        self.step = -1
        self.active = False
        self.sparse_k = 0  # > 0: attention really reads only the first sparse_k ranked pages

    def on_decode(self, layer: int, query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
        q, keys = query[0, :, 0], key[0]  # [Hq, D], [Hkv, S, D]
        s = keys.shape[1]
        summ = self.summaries[layer]
        if summ is None:
            summ = PageSummary(keys, PAGE, self.capacity_pages)
            self.summaries[layer] = summ
        else:
            assert summ.tokens == s - 1, f"layer {layer}: summary has {summ.tokens} tokens, cache has {s}"
            summ.append(keys[:, s - 1])
        if self.verify:
            ref = PageSummary(keys, PAGE)
            assert all(torch.equal(a, b) for a, b in zip(summ.view(), ref.view())), f"summary drift at layer {layer}"
        ub = quest_upper_bounds(q, *summ.view())
        r = rank_pages(ub, scaling, self.k_max, self.aggregate)
        self.ranked[self.step, layer, : r.numel()] = r.to(torch.int16)
        if self.cov_every and self.step % self.cov_every == 0:
            probs = attention_probs(q, keys, scaling)
            self.cov_rows.append((self.step, layer, coverage(probs, ub, scaling, self.ks, PAGE)))
        return r

    def coverage_arrays(self) -> dict[str, np.ndarray]:
        if not self.cov_rows:
            return {}
        out = {"step": np.array([r[0] for r in self.cov_rows], np.int32),
               "layer": np.array([r[1] for r in self.cov_rows], np.int32)}
        for k in COV_KEYS:
            out[k] = torch.stack([r[2][k] for r in self.cov_rows]).float().cpu().numpy()
        return out


def quest_logging_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    rec = RECORDER
    if rec is not None and rec.active and query.shape[2] == 1:
        scaling = kwargs.get("scaling")
        ranked = rec.on_decode(module.layer_idx, query, key, scaling if scaling is not None else module.scaling)
        if rec.sparse_k:  # true sparse decoding: read only the tokens of the first sparse_k ranked pages
            s = key.shape[2]
            pages = ranked[: rec.sparse_k]
            idx = (pages[:, None] * PAGE + torch.arange(PAGE, device=key.device)).reshape(-1)
            idx = torch.sort(idx[idx < s]).values
            key, value = key[:, :, idx], value[:, :, idx]
            if attention_mask is not None:
                attention_mask = attention_mask[..., idx]
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register_attention() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_NAME, quest_logging_attention)
    try:  # transformers >= 4.53 builds masks per attention implementation
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

        AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
    except ImportError:
        pass


# ---------------------------------------------------------------- generation

def make_sampler(temperature: float, top_k: int, top_p: float, seed: int, device):
    gen = torch.Generator(device=device).manual_seed(seed)

    def sample(logits: torch.Tensor) -> torch.Tensor:  # [1, V] -> [1]
        if temperature <= 0:
            return logits.argmax(dim=-1)
        vals, idx = torch.topk(logits.float() / temperature, top_k, dim=-1)  # sorted descending
        probs = torch.softmax(vals, dim=-1)
        if top_p < 1.0:
            probs = probs.masked_fill(probs.cumsum(-1) - probs > top_p, 0.0)
            probs = probs / probs.sum(-1, keepdim=True)
        return idx.gather(-1, torch.multinomial(probs, 1, generator=gen)).view(1)

    return sample


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def generate(model, input_ids, steps, sampler, rec, eos_ids):
    from transformers import DynamicCache

    base, head, dev = model.model, model.lm_head, input_ids.device
    cache = DynamicCache()
    _sync(dev)
    t0 = time.perf_counter()
    out = base(input_ids=input_ids, past_key_values=cache, use_cache=True)
    nxt = sampler(head(out.last_hidden_state[:, -1, :]))
    del out
    _sync(dev)
    prefill_s = time.perf_counter() - t0
    tokens, eos = [], False
    rec.active = True
    t1 = time.perf_counter()
    for step in range(steps):
        tid = int(nxt.item())
        if tid in eos_ids:
            eos = True
            break
        tokens.append(tid)
        rec.step = step
        out = base(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
        nxt = sampler(head(out.last_hidden_state[:, -1, :]))
        del out
    _sync(dev)
    rec.active = False
    return tokens, prefill_s, time.perf_counter() - t1, eos


# ---------------------------------------------------------------- prompts

PLACEHOLDER = "@@CONTENT@@"
LB_PRE = "Please read the following text and answer the question below.\n\n<text>\n"


def lb_post(item: dict) -> str:
    choices = "".join(f"\n({c}) {item['choice_' + c].strip()}" for c in "ABCD")
    return (f"\n</text>\n\nWhat is the correct answer to this question: {item['question'].strip()}\nChoices:{choices}"
            "\n\nFormat your response as follows: \"The correct answer is (insert answer here)\".")


def load_longbench() -> list[dict]:
    from huggingface_hub import hf_hub_download

    err = None
    for repo in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
        try:
            path = hf_hub_download(repo_id=repo, filename="data.json", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"could not download LongBench v2 data.json: {err}")


def longbench_prompt(tok, item: dict, target_tokens: int, thinking: bool) -> tuple[list[int], dict]:
    """Chat-formatted LongBench v2 prompt of at most target_tokens; long contexts lose their middle."""
    msgs = [{"role": "user", "content": PLACEHOLDER}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    chat_pre, chat_post = text.split(PLACEHOLDER)
    pre = tok(chat_pre + LB_PRE, add_special_tokens=False).input_ids
    post = tok(lb_post(item) + chat_post, add_special_tokens=False).input_ids
    ctx = tok(item["context"], add_special_tokens=False).input_ids
    budget = target_tokens - len(pre) - len(post)
    truncated = len(ctx) > budget
    if truncated:
        ctx = ctx[: budget // 2] + ctx[len(ctx) - (budget - budget // 2):]
    info = {"id": item.get("_id"), "domain": item.get("domain"), "sub_domain": item.get("sub_domain"),
            "difficulty": item.get("difficulty"), "length_class": item.get("length"), "answer": item.get("answer"),
            "context_tokens_kept": len(ctx), "truncated_middle": truncated}
    return pre + ctx + post, info


# ---------------------------------------------------------------- models

def build_tiny_model(device, dtype):
    """Random-weight Qwen3 with the real architecture (GQA 4:1, QK-norm, RoPE), for smoke tests."""
    from transformers import AutoModelForCausalLM, Qwen3Config

    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=ATTN_NAME, torch_dtype=dtype)
    return model.to(device).eval()


def load_model(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation=ATTN_NAME)
    return tok, model.to(device).eval()


def eos_token_ids(tok, model) -> set[int]:
    ids: set[int] = set()
    for src in (getattr(tok, "eos_token_id", None), getattr(model.generation_config, "eos_token_id", None)):
        if src is not None:
            ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids


# ---------------------------------------------------------------- summary

def step_overlap(ranked: np.ndarray, k: int) -> np.ndarray:
    """Share of the k selected pages at step t that were also selected at t-1. [steps-1, layers]."""
    steps, layers, _ = ranked.shape
    out = np.zeros((max(steps - 1, 0), layers), np.float32)
    for t in range(1, steps):
        for l in range(layers):
            a, b = ranked[t, l, :k], ranked[t - 1, l, :k]
            a, b = a[a >= 0], b[b >= 0]
            out[t - 1, l] = np.intersect1d(a, b, assume_unique=True).size / max(a.size, 1)
    return out


def summarize(ranked: np.ndarray, cov: dict, ks: list[int], k_main: int) -> dict:
    summ = {"k_main": k_main, "coverage_by_k": [], "per_layer_quest_at_k_main": None, "overlap": None}
    if cov:
        for j, k in enumerate(ks):
            row = {"k_pages": k, "k_tokens": k * PAGE}
            row.update({name: float(cov[name][:, j].mean()) for name in COV_KEYS})
            row["quest_over_exact"] = row["quest_logit"] / row["token_oracle"]
            summ["coverage_by_k"].append(row)
        j = ks.index(k_main)
        layers = cov["layer"]
        per_layer = [float(cov["quest_logit"][layers == l, j].mean()) for l in sorted(set(layers.tolist()))]
        summ["per_layer_quest_at_k_main"] = per_layer
    if ranked.shape[0] > 1:
        ov = step_overlap(ranked, k_main)
        per_layer = ov.mean(axis=0)
        summ["overlap"] = {"mean": float(ov.mean()), "per_layer": per_layer.tolist()}
    return summ


def summary_markdown(name: str, meta: dict, summ: dict) -> str:
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"# Trace {name}", "",
             f"{meta['model']} · prompt {meta['prompt_tokens']:,} tokens · {meta['steps']} decode steps"
             f" ({'stopped at EOS' if meta['eos_reached'] else 'step limit'}) · k_max {meta['k_max']} pages"
             f" · aggregation {meta['aggregate']} · seed {meta['seed']} · commit {meta['commit']}", ""]
    if meta.get("prompt_info"):
        pi = meta["prompt_info"]
        lines += [f"Prompt: LongBench v2 `{pi.get('id')}` ({pi.get('domain')} / {pi.get('sub_domain')}, "
                  f"{pi.get('difficulty')}, {pi.get('length_class')}); middle truncated: {pi.get('truncated_middle')}.", ""]
    lines += [f"Prefill {meta['prefill_s']:.2f} s · decode {meta['decode_ms_per_step']:.1f} ms/step (with logging)", ""]
    if summ["coverage_by_k"]:
        lines += ["## Attention coverage (G2)", "",
                  f"Mean share of the true attention mass captured, over {meta['coverage_samples']} sampled (step, layer) pairs "
                  "and all query heads.", "",
                  "| k pages | tokens | Quest shared (mean logit) | Quest shared (mean softmax) | Quest per head "
                  "| Best shared pages | Exact top-k tokens | Quest / exact |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in summ["coverage_by_k"]:
            lines.append(f"| {r['k_pages']} | {r['k_tokens']:,} | {pct(r['quest_logit'])} | {pct(r['quest_softmax'])} "
                         f"| {pct(r['quest_head'])} | {pct(r['page_oracle'])} | {pct(r['token_oracle'])} "
                         f"| {r['quest_over_exact']:.3f} |")
        pl = np.array(summ["per_layer_quest_at_k_main"])
        worst = np.argsort(pl)[:3]
        lines += ["", f"Per layer at k = {summ['k_main']} (Quest shared, mean logit): min {pct(pl.min())}, "
                  f"median {pct(float(np.median(pl)))}, max {pct(pl.max())}; lowest layers "
                  + ", ".join(f"{int(l)} ({pct(pl[l])})" for l in worst) + "."]
    if summ["overlap"]:
        ov = np.array(summ["overlap"]["per_layer"])
        lines += ["", "## Step-to-step overlap (early locality signal)", "",
                  f"Share of the k = {summ['k_main']} selected pages that were also selected one step earlier: "
                  f"mean {pct(summ['overlap']['mean'])}; per layer min {pct(ov.min())} (layer {int(ov.argmin())}), "
                  f"median {pct(float(np.median(ov)))}, max {pct(ov.max())} (layer {int(ov.argmax())})."]
    return "\n".join(lines) + "\n"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- main

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--index", type=int, default=0, help="LongBench v2 item index (0-502)")
    ap.add_argument("--target-tokens", type=int, default=32000, help="prompt length cap")
    ap.add_argument("--steps", type=int, default=1000, help="max decode steps (stops earlier at EOS)")
    ap.add_argument("--k-max", type=int, default=256, help="pages logged per step and layer, in rank order")
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256], help="k values for coverage")
    ap.add_argument("--k-main", type=int, default=128)
    ap.add_argument("--aggregate", choices=["mean_logit", "mean_softmax"], default="mean_logit")
    ap.add_argument("--cov-every", type=int, default=50, help="coverage sample every N steps (all layers); 0 = off")
    ap.add_argument("--no-thinking", action="store_true", help="disable Qwen3 thinking mode")
    ap.add_argument("--temperature", type=float, default=0.6)  # Qwen3 thinking-mode recommendation
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--summary-dir", default="trace/results")
    ap.add_argument("--smoke", action="store_true", help="tiny random model, random prompt, self-checks on")
    ap.add_argument("--verify", action="store_true", help="re-check page summaries every step (slow)")
    ap.add_argument("--sparse-k", type=int, default=0,
                    help="> 0: decode with true sparse attention over the first K ranked pages (trace name S<K>_...)")
    args = ap.parse_args(argv)

    global RECORDER
    register_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model, tok = build_tiny_model(device, dtype), None
        g = torch.Generator().manual_seed(args.seed)
        prompt = torch.randint(0, 512, (min(args.target_tokens, 1000),), generator=g).tolist()
        steps, ks, k_max, k_main = min(args.steps, 60), [2, 4, 8], max(8, args.sparse_k), 4
        prompt_info, eos_ids, name = None, set(), "smoke"
        cov_every, verify = 5, True
    else:
        tok, model = load_model(args.model, device)
        item = load_longbench()[args.index]
        prompt, prompt_info = longbench_prompt(tok, item, args.target_tokens, not args.no_thinking)
        steps, ks, k_max, k_main = args.steps, args.ks, args.k_max, args.k_main
        eos_ids = eos_token_ids(tok, model)
        prefix = f"S{args.sparse_k}" if args.sparse_k else "A"
        name = f"{prefix}_{args.model.split('/')[-1].lower()}_lb{args.index:03d}_s{args.seed}"
        cov_every, verify = args.cov_every, args.verify
    assert k_main in ks, "--k-main must be one of --ks"
    assert args.sparse_k <= k_max, f"--sparse-k {args.sparse_k} needs at least that many logged pages (k_max {k_max})"

    n_layers = model.config.num_hidden_layers
    cap = (len(prompt) + steps) // PAGE + 2
    RECORDER = Recorder(n_layers, steps, k_max, ks, args.aggregate, cov_every, device, verify, cap)
    RECORDER.sparse_k = args.sparse_k
    sampler = make_sampler(args.temperature, args.top_k, args.top_p, args.seed, device)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    print(f"[{name}] prompt {len(prompt):,} tokens, up to {steps} steps, {n_layers} layers, device {device}", flush=True)

    tokens, prefill_s, decode_s, eos = generate(model, input_ids, steps, sampler, RECORDER, eos_ids)
    n = len(tokens)
    ranked = RECORDER.ranked[:n].cpu().numpy()
    cov = RECORDER.coverage_arrays()
    seq_len = (len(prompt) + 1 + np.arange(n)).astype(np.int32)

    meta = {
        "name": name, "model": "tiny-random-qwen3" if args.smoke else args.model, "workload": "smoke" if args.smoke else "A",
        "prompt_tokens": len(prompt), "steps": n, "eos_reached": eos, "page_size": PAGE, "k_max": k_max, "ks": ks,
        "aggregate": args.aggregate, "forced_pages": ["sink page 0", "newest page"], "thinking": not args.no_thinking,
        "attention": f"sparse, first {args.sparse_k} ranked pages" if args.sparse_k else "full",
        "sampling": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p}, "seed": args.seed,
        "coverage_every": cov_every, "coverage_samples": int(len(cov.get("step", []))), "prompt_info": prompt_info,
        "prefill_s": prefill_s, "decode_ms_per_step": 1e3 * decode_s / max(n, 1), "commit": git_commit(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "text": tok.decode(tokens, skip_special_tokens=False) if tok is not None else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir, sum_dir = Path(args.out_dir), Path(args.summary_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", ranked=ranked, seq_len=seq_len,
                        prompt_ids=np.array(prompt, np.int32), tokens=np.array(tokens, np.int32),
                        meta=np.array(json.dumps(meta)), **{f"cov_{k}": v for k, v in cov.items()})
    summ = summarize(ranked, cov, ks, k_main)
    md = summary_markdown(name, meta, summ)
    (sum_dir / f"{name}.md").write_text(md, encoding="utf-8")
    meta_small = {k: v for k, v in meta.items() if k != "text"}
    (sum_dir / f"{name}.json").write_text(json.dumps({"meta": meta_small, "summary": summ}, indent=2), encoding="utf-8")
    print(md, flush=True)
    return {"meta": meta, "summary": summ, "ranked": ranked, "coverage": cov}


if __name__ == "__main__":
    main()
