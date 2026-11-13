"""Integration tests for trace/collect.py on a tiny random Qwen3 (CPU is fine, no download).

Run:  python trace/test_collect.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402
from quest import PAGE  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _decode_logits(model, ids, forced):
    """Prefill ids, then feed the forced tokens one by one; return the logits of every step."""
    from transformers import DynamicCache

    cache, out_logits = DynamicCache(), []
    with torch.inference_mode():
        out = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
        out_logits.append(model.lm_head(out.last_hidden_state[:, -1]))
        for t in forced:
            out = model.model(input_ids=torch.tensor([[t]], device=ids.device), past_key_values=cache, use_cache=True)
            out_logits.append(model.lm_head(out.last_hidden_state[:, -1]))
    return torch.cat(out_logits)


def test_logging_does_not_change_outputs():
    collect.register_attention()
    logged = collect.build_tiny_model(DEV, torch.float32)
    plain = collect.build_tiny_model(DEV, torch.float32)
    plain.set_attn_implementation("sdpa")
    plain.load_state_dict(logged.state_dict())
    ids = torch.randint(0, 512, (1, 150), generator=torch.Generator().manual_seed(0)).to(DEV)
    forced = torch.randint(0, 512, (30,), generator=torch.Generator().manual_seed(1)).tolist()
    collect.RECORDER = collect.Recorder(3, 40, 8, [2, 4, 8], "mean_logit", 0, DEV, verify=True)
    collect.RECORDER.active = True
    collect.RECORDER.step = 0

    class Stepper:  # advance the recorder's step index on every decode call of layer 0
        def __init__(self, rec):
            self.rec, self.inner = rec, rec.on_decode

        def __call__(self, layer, q, k, s):
            if layer == 0 and self.rec.summaries[0] is not None:
                self.rec.step += 1
            self.inner(layer, q, k, s)

    collect.RECORDER.on_decode = Stepper(collect.RECORDER)
    a = _decode_logits(logged, ids, forced)
    collect.RECORDER.active = False
    b = _decode_logits(plain, ids, forced)
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
    assert collect.RECORDER.summaries[0].tokens == 150 + 30


def test_ranking_matches_brute_force():
    """Re-derive the ranking with explicit loops from the raw q and keys the attention hook receives."""
    collect.register_attention()
    model = collect.build_tiny_model(DEV, torch.float32)
    checked = []

    class Checking(collect.Recorder):
        def on_decode(self, layer, query, key, scaling):
            super().on_decode(layer, query, key, scaling)
            q, keys = query[0, :, 0].float(), key[0].float()
            hq, hkv, s = q.shape[0], keys.shape[0], keys.shape[1]
            n = (s + PAGE - 1) // PAGE
            score = torch.zeros(n)
            for h in range(hq):
                kv = keys[h // (hq // hkv)]
                for p in range(n):
                    blk = kv[p * PAGE:(p + 1) * PAGE]
                    score[p] += torch.maximum(q[h] * blk.min(0).values, q[h] * blk.max(0).values).sum() * scaling / hq
            rest = [p for p in torch.argsort(score, descending=True, stable=True).tolist() if p not in (0, n - 1)]
            expect = ([0, n - 1] if n > 1 else [0]) + rest
            got = self.ranked[self.step, layer].tolist()
            k = min(self.k_max, n)
            # Compare as sets beyond the forced pages: ties may order differently.
            assert got[:2] == expect[:2] and set(got[:k]) == set(expect[:k]), (layer, self.step, got[:k], expect[:k])
            checked.append((self.step, layer))

    collect.RECORDER = Checking(3, 12, 8, [2, 4, 8], "mean_logit", 0, DEV)
    ids = torch.randint(0, 512, (1, 97), generator=torch.Generator().manual_seed(2)).to(DEV)
    sampler = collect.make_sampler(0.0, 1, 1.0, 0, DEV)
    tokens, *_ = collect.generate(model, ids, 12, sampler, collect.RECORDER, set())
    assert len(tokens) == 12 and len(checked) == 12 * 3


def test_smoke_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        res = collect.main(["--smoke", "--out-dir", tmp, "--summary-dir", tmp, "--target-tokens", "300", "--steps", "40"])
        with np.load(os.path.join(tmp, "smoke.npz")) as d:  # close it, or Windows cannot delete tmp
            meta = json.loads(str(d["meta"]))
            ranked, seq_len = d["ranked"], d["seq_len"]
        assert ranked.shape == (40, 3, 8) and ranked.dtype == np.int16 and meta["steps"] == 40
        assert np.array_equal(seq_len, 300 + 1 + np.arange(40))
        n_pages = (seq_len + PAGE - 1) // PAGE
        assert np.all(ranked[:, :, 0] == 0) and np.all(ranked[:, :, 1] == (n_pages - 1)[:, None])
        assert np.all(ranked >= 0) and np.all(ranked < n_pages[:, None, None])
        for t in range(40):  # no page twice in one selection
            for l in range(3):
                assert len(set(ranked[t, l].tolist())) == 8
        cov = res["coverage"]
        eps = 1e-5
        assert np.all(cov["page_oracle"] + eps >= cov["quest_logit"])
        assert np.all(cov["token_oracle"] + eps >= cov["page_oracle"])
        assert os.path.exists(os.path.join(tmp, "smoke.md")) and os.path.exists(os.path.join(tmp, "smoke.json"))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
