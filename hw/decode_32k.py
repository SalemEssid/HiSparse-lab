"""Dense-KV long-context decode baseline for G1 ("32K prompt decodes").

For each target context length: build a needle-in-a-haystack prompt, prefill it,
greedy-decode N tokens with the full KV cache on the GPU, and record prefill
time, per-step decode latency (TPOT), peak GPU memory, and whether the needle was
retrieved (a sanity check that 32K positions actually work).

Also verifies the model config against the values assumed in the project notes
(layers, KV heads, head dim) and computes KV bytes per token.

This is HF eager mode (no CUDA graphs), so TPOT includes Python overhead. It is the
"full KV on GPU" reference point, not a serving-engine number.

Usage:
    python hw/decode_32k.py --model Qwen/Qwen3-4B --contexts 8192 16384 32000 \
        --out hw/results/decode_qwen3-4b.json
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

GiB = 2**30

# Values assumed in the project notes (section 6.2), to be checked against config.json.
EXPECTED = {
    "Qwen/Qwen3-1.7B": {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128},
    "Qwen/Qwen3-4B": {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128},
    "Qwen/Qwen3-8B": {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128},
}

PASSPHRASE = "violet-otter-4729"
NEEDLE = f" Important: the secret passphrase is {PASSPHRASE}. Remember it."
QUESTION = "What is the secret passphrase mentioned in the text above? Reply with the passphrase only."
PLACEHOLDER = "@@HAYSTACK@@"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def filler_stream(seed: int):
    """Endless deterministic stream of varied filler sentences."""
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "amber", "grey", "white", "black", "golden"]
    animals = ["fox", "heron", "badger", "lynx", "falcon", "beaver", "moose", "crane"]
    places = ["river", "orchard", "harbor", "quarry", "meadow", "station", "library", "bridge"]
    verbs = ["walked", "ran", "drifted", "climbed", "wandered", "hurried", "rolled", "flew"]
    i = 0
    while True:
        yield (
            f" Record {i}: the {rng.choice(colors)} {rng.choice(animals)} {rng.choice(verbs)} "
            f"{rng.randint(2, 9999)} meters toward the {rng.choice(places)} on day {rng.randint(1, 365)}."
        )
        i += 1


def build_prompt(tok, target_tokens: int, depth: float, seed: int) -> tuple[torch.Tensor, int]:
    """Return input_ids of exactly target_tokens tokens with the needle at the given depth."""
    msgs = [{"role": "user", "content": PLACEHOLDER + "\n\n" + QUESTION}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        text = tok.apply_chat_template(msgs, **kwargs)
    pre, post = text.split(PLACEHOLDER)
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids
    needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
    filler_budget = target_tokens - len(pre_ids) - len(post_ids) - len(needle_ids)
    assert filler_budget > 0, f"target {target_tokens} too small"

    filler: list[int] = []
    split = None  # filler index where the needle is inserted (a sentence boundary)
    stream = filler_stream(seed)
    while len(filler) < filler_budget:
        batch = list(itertools.islice(stream, 512))
        for ids in tok(batch, add_special_tokens=False).input_ids:
            if split is None and len(filler) + len(ids) > depth * filler_budget:
                split = len(filler)
            filler.extend(ids)
            if len(filler) >= filler_budget:
                break
    filler = filler[:filler_budget]
    split = len(filler) if split is None else split
    ids = pre_ids + filler[:split] + needle_ids + filler[split:] + post_ids
    return torch.tensor([ids], dtype=torch.long), len(pre_ids) + split


def eos_token_ids(tok, model) -> set[int]:
    ids: set[int] = set()
    for src in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if src is None:
            continue
        ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids


def check_config(model_name: str, cfg) -> dict:
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    found = {
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
    }
    expected = EXPECTED.get(model_name, {})
    mismatches = {k: {"expected": v, "found": found[k]} for k, v in expected.items() if found[k] != v}
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2  # K+V, bf16
    return {
        "config": found,
        "mismatches_vs_notes": mismatches,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes_per_token_per_layer": kv_bytes_per_token // cfg.num_hidden_layers,
        "kv_bytes_per_16tok_page_per_layer": 16 * kv_bytes_per_token // cfg.num_hidden_layers,
    }


@torch.inference_mode()
def run_one(model, tok, target: int, decode_steps: int, prefill_chunk: int, depth: float, seed: int) -> dict:
    input_ids, needle_pos = build_prompt(tok, target, depth, seed)
    input_ids = input_ids.cuda()
    n = input_ids.shape[1]
    base = model.model
    cache = DynamicCache()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    chunk = prefill_chunk if prefill_chunk > 0 else n
    for s in range(0, n, chunk):
        out = base(input_ids=input_ids[:, s : s + chunk], past_key_values=cache, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    peak_prefill = torch.cuda.max_memory_allocated()
    mem_after_prefill = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    next_id = logits.argmax(-1)
    generated = [next_id.item()]
    step_s = []
    for _ in range(decode_steps - 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = base(input_ids=next_id, past_key_values=cache, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        next_id = logits.argmax(-1)
        torch.cuda.synchronize()
        step_s.append(time.perf_counter() - t)
        generated.append(next_id.item())
    peak_decode = torch.cuda.max_memory_allocated()

    eos_ids = eos_token_ids(tok, model)
    answer_ids = generated
    for i, t in enumerate(generated):
        if t in eos_ids:
            answer_ids = generated[:i]
            break
    answer = tok.decode(answer_ids, skip_special_tokens=True).strip()

    steady = step_s[3:] if len(step_s) > 6 else step_s
    steady_sorted = sorted(steady)
    return {
        "target_tokens": target,
        "prompt_tokens": n,
        "needle_token_pos": needle_pos,
        "prefill_chunk": chunk,
        "prefill_s": round(prefill_s, 3),
        "prefill_tok_per_s": round(n / prefill_s, 1),
        "decode_steps_timed": len(steady),
        "tpot_ms_median": round(1e3 * statistics.median(steady), 2),
        "tpot_ms_p90": round(1e3 * steady_sorted[int(0.9 * (len(steady_sorted) - 1))], 2),
        "decode_tok_per_s": round(1 / statistics.median(steady), 1),
        "kv_cache_gib_measured": round((mem_after_prefill - mem_before) / GiB, 3),
        "peak_prefill_gib": round(peak_prefill / GiB, 2),
        "peak_decode_gib": round(peak_decode / GiB, 2),
        "needle_found": PASSPHRASE.lower() in answer.lower(),
        "answer": answer[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--contexts", type=int, nargs="+", default=[32000])
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--prefill-chunk", type=int, default=0, help="0 = one-shot prefill")
    ap.add_argument("--needle-depth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available to torch"
    torch.manual_seed(args.seed)
    tag = args.model.split("/")[-1].lower()
    out = Path(args.out or f"hw/results/decode_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    load_s = time.perf_counter() - t0
    weights_gib = torch.cuda.memory_allocated() / GiB
    cfg_info = check_config(args.model, model.config)
    if cfg_info["mismatches_vs_notes"]:
        print("WARNING: config differs from project notes:", cfg_info["mismatches_vs_notes"])

    # Warm-up so kernel selection / lazy init is not timed.
    with torch.inference_mode():
        model.model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"), use_cache=False)
    torch.cuda.synchronize()

    runs = []
    for target in args.contexts:
        print(f"--- context {target} ---", flush=True)
        try:
            r = run_one(model, tok, target, args.decode_steps, args.prefill_chunk, args.needle_depth, args.seed)
        except torch.cuda.OutOfMemoryError as e:
            r = {"target_tokens": target, "oom": True, "error": str(e).splitlines()[0][:200]}
            torch.cuda.empty_cache()
        print(json.dumps(r, indent=2), flush=True)
        runs.append(r)

    result = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "attn_implementation": "sdpa",
        "dtype": "bfloat16",
        "commit": git_commit(),
        "seed": args.seed,
        "load_s": round(load_s, 1),
        "weights_gib": round(weights_gib, 2),
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / GiB, 2),
        **cfg_info,
        "runs": runs,
    }
    out.write_text(json.dumps(result, indent=2))

    md = [
        f"## Dense decode: {args.model} on {result['gpu']}",
        "",
        f"Weights {result['weights_gib']} GiB; KV {cfg_info['kv_bytes_per_token'] / 1024:.0f} KiB/token "
        f"({cfg_info['kv_bytes_per_16tok_page_per_layer'] // 1024} KiB per 16-token page per layer); "
        f"config mismatches vs notes: {cfg_info['mismatches_vs_notes'] or 'none'}; "
        f"max_position_embeddings {cfg_info['config']['max_position_embeddings']}; commit {result['commit']}.",
        "",
        "| Prompt tokens | Prefill (s) | TPOT median / p90 (ms) | KV (GiB) | Peak prefill / decode (GiB) | Needle |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        if r.get("oom"):
            md.append(f"| {r['target_tokens']} | OOM | | | | |")
        else:
            md.append(
                f"| {r['prompt_tokens']} | {r['prefill_s']} | {r['tpot_ms_median']} / {r['tpot_ms_p90']} "
                f"| {r['kv_cache_gib_measured']} | {r['peak_prefill_gib']} / {r['peak_decode_gib']} "
                f"| {'found' if r['needle_found'] else 'MISSED: ' + r['answer'][:60]} |"
            )
    md_text = "\n".join(md) + "\n"
    out.with_suffix(".md").write_text(md_text)
    print(md_text)


if __name__ == "__main__":
    main()
