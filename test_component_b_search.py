"""End-to-end test of the Component B search algorithm on a mock model (CPU).

No Llama weights, no GPU, no HF auth needed: we build a tiny model whose
"readout" is ``mean(input embeddings)`` and plant one token whose embedding is
far along a known axis. A probe direction pointing along that axis must then be
maximised by putting the planted token in the trigger slot — so a working search
recovers the planted token id, and a broken one doesn't.

This exercises the actual search stack: ``ProbeObjective`` (scoring, gradients,
continuous readout), ``propose``, and all four methods (gcg / relaxed / soft /
random).

Run:
    .venv/bin/python -m pytest test_component_b_search.py -q
    # or directly:
    .venv/bin/python test_component_b_search.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from component_b.config import SearchConfig
from component_b.load import Direction
from component_b.search import ProbeObjective, search

VOCAB = 32
HIDDEN = 16
N_TOKENS = 4
TARGET = 7


class MockModel(nn.Module):
    """Readout = mean input embedding (so trigger tokens influence the last token)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 0.1)
            self.emb.weight[TARGET] = 0.0
            self.emb.weight[TARGET, 0] = 10.0     # unique spike along axis 0
        self.emb.weight.requires_grad_(False)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, output_hidden_states=False, **kw):
        e = self.emb(input_ids) if input_ids is not None else inputs_embeds
        pooled = e.mean(dim=1)                          # (B, H)
        hs = [e] + [pooled.unsqueeze(1).expand(-1, e.shape[1], -1)
                    for _ in range(4)]                  # hidden_states[1..4]
        out = type("Out", (), {})()
        out.hidden_states = tuple(hs)
        out.logits = None
        return out


def _objective():
    model = MockModel()
    w = np.zeros(HIDDEN, dtype=np.float64)
    w[0] = 1.0
    direction = Direction(w_raw=w, b_raw=0.0, layer=0, readout_mode="pregen",
                          slug="mock", val_auc=1.0, test_auc=1.0,
                          control_auc=0.5, gap=0.5)
    head = [0] * 8
    tails = [[1, 2, 3, 4, 5, 6, 8, 9]]                 # one prompt
    positions = list(range(len(head), len(head) + N_TOKENS))
    obj = ProbeObjective(model, direction, head, tails, positions)
    allowed = torch.ones(VOCAB, dtype=torch.bool)
    init_ids = [11] * N_TOKENS
    scfg = SearchConfig(n_tokens=N_TOKENS, n_steps=25, top_k=8,
                        n_candidates=16, patience=12, relax_steps=80,
                        soft_lr=0.1, seed=0)
    return obj, allowed, init_ids, scfg


def test_probe_objective_gradient_and_propose():
    obj, allowed, init_ids, scfg = _objective()
    grad = obj.grad(init_ids)
    assert grad.shape == (N_TOKENS, HIDDEN)
    assert torch.isfinite(grad).all()
    cands = obj.propose(init_ids, grad, allowed, top_k=8)
    # the top-ranked substitution for each position must include the planted token
    top_per_pos = [cands[i * 8] for i in range(N_TOKENS)]
    assert any(TARGET in c for c in top_per_pos), top_per_pos


def test_all_methods_recover_planted_token():
    for method in ("gcg", "relaxed", "soft", "random"):
        obj, allowed, init_ids, scfg = _objective()
        res = search(obj, scfg, allowed, method, init_ids)
        frac = sum(1 for t in res.trigger_ids if t == TARGET) / N_TOKENS
        assert frac >= 0.75, f"{method}: {res.trigger_ids} target frac={frac}"
        assert res.best_logit > 0.0, f"{method}: logit {res.best_logit}"


def test_gcg_default_budget_reaches_every_position():
    # Default top_k/n_candidates: top_k*n_tokens candidates > n_candidates, so a
    # naive prefix cut keeps only position-0/1 swaps and can never fill the slot.
    obj, allowed, init_ids, _ = _objective()
    d = SearchConfig()
    scfg = SearchConfig(n_tokens=N_TOKENS, n_steps=25, top_k=d.top_k,
                        n_candidates=d.n_candidates, patience=12, seed=0)
    assert min(scfg.top_k, VOCAB) * N_TOKENS > scfg.n_candidates
    res = search(obj, scfg, allowed, "gcg", init_ids)
    assert res.trigger_ids == [TARGET] * N_TOKENS, res.trigger_ids


def test_relaxed_gcg_recovers_planted_token():
    obj, allowed, init_ids, scfg = _objective()
    scfg.relax_gcg_steps = 20
    res = search(obj, scfg, allowed, "relaxed_gcg", init_ids)
    assert res.trigger_ids.count(TARGET) >= 3, res.trigger_ids


def test_max_forwards_caps_discrete_methods():
    for method in ("gcg", "random", "relaxed_gcg"):
        obj, allowed, init_ids, scfg = _objective()
        scfg.max_forwards = 100
        scfg.relax_gcg_steps = 5
        res = search(obj, scfg, allowed, method, init_ids)
        # the cap is checked before each step, so one step may overshoot
        step = (scfg.n_candidates + 1) * len(obj.tails)
        assert res.n_forwards <= scfg.max_forwards + step, (method, res.n_forwards)


def test_gcg_restarts_escape_and_keep_best():
    obj, allowed, init_ids, scfg = _objective()
    scfg.patience, scfg.n_restarts, scfg.n_steps = 1, 3, 40
    res = search(obj, scfg, allowed, "gcg", init_ids)
    assert res.best_logit == max(res.history)
    assert res.best_logit >= obj.logit(res.trigger_ids) - 1e-5


def test_continuous_methods_run_in_bf16():
    for method in ("relaxed", "soft"):
        obj, allowed, init_ids, scfg = _objective()
        obj.model.to(torch.bfloat16)
        scfg.relax_steps = 5
        res = search(obj, scfg, allowed, method, init_ids)
        assert len(res.trigger_ids) == N_TOKENS
        assert all(p.grad is None for p in obj.model.parameters())


class _ToyLM(nn.Module):
    """Tiny causal LM: block 0 is per-position, block 1 pools everything into
    the last position, and the head reads the last position. A trigger token's
    effect therefore lives at its own position after block 0 and at the last
    position after block 1, which is what patching should find."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"num_hidden_layers": 2})()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([nn.Identity(), _PoolLast()])
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            self.emb.weight.normal_(0.0, 0.1)
            self.emb.weight[TARGET] = 0.0
            self.emb.weight[TARGET, 0] = 10.0
            self.head.weight.zero_()
            self.head.weight[3, 0] = 1.0           # token 3 tracks axis 0

    def forward(self, input_ids=None, **kw):
        h = self.emb(input_ids)
        for layer in self.layers:
            h = layer(h)
        out = type("Out", (), {})()
        out.logits = self.head(h)
        return out


class _PoolLast(nn.Module):
    def forward(self, h):
        h = h.clone()
        h[:, -1] = h.mean(dim=1)
        return h


def test_patching_localises_trigger():
    from component_b.patching import patch_scan

    model = _ToyLM()
    head, tail = [0, 1], [2, 4, 5]
    res = patch_scan(model, head, [tail, tail + [6, 8]], trig_ids=[TARGET], clean_ids=[11])
    slot = len(head)
    assert res["per_prompt"][0]["target"] == 3
    assert res["n_prompts"] == 2                   # different lengths, both used
    for eff in [res["effect_per_prompt"][0], res["effect"]]:
        assert abs(eff[0, slot] - 1.0) < 1e-3      # block 0, trigger position
        assert abs(eff[1, -1] - 1.0) < 1e-3        # block 1, last position
        others = [eff[0, t] for t in range(eff.shape[1]) if t != slot]
        assert max(abs(x) for x in others) < 1e-3
    # aligned columns: head | slot | separator | body(max) | suffix
    assert res["effect"].shape[1] == len(head) + 1 + 1 + 1 + res["suffix_len"]


class _WordTok:
    """Minimal tokenizer for the run() wiring test: id i <-> word 'w{i}'."""

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"w{int(i)}" for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(w[1:]) for w in text.split() if w[1:].isdigit()]}


def test_run_end_to_end_on_mock(tmp_path, monkeypatch):
    """run() wiring, flags and report schema, with the model stack mocked out."""
    import json as _json

    from component_b import load as bload
    from component_b import run as brun
    from component_b import search as srch
    from component_b import verify as bverify

    obj, allowed, init_ids, _ = _objective()
    model = obj.model
    model.device = torch.device("cpu")
    direction = Direction(w_raw=obj.w.numpy().astype(np.float64), b_raw=0.0, layer=0,
                          readout_mode="pregen", slug="mock", val_auc=1.0,
                          test_auc=1.0, control_auc=0.5, gap=0.5)
    tok = _WordTok()
    head, tail = obj.head, obj.tails[0]
    monkeypatch.setattr(bload, "load_direction", lambda p: direction)
    monkeypatch.setattr(bload, "load_target", lambda scfg, which: (model, tok))
    monkeypatch.setattr(bload, "clean_prompts", lambda scfg: (["p0", "p1"], ["v0"]))
    monkeypatch.setattr(bload, "probe_scores",
                        lambda m, t, d, users: np.zeros(len(users)))
    monkeypatch.setattr(srch, "check_date_free_template", lambda tok: None)
    monkeypatch.setattr(srch, "build_template",
                        lambda tok, body, n: (head, tail, list(range(len(head), len(head) + n))))
    monkeypatch.setattr(srch, "allowed_mask", lambda tok, scfg: allowed)
    monkeypatch.setattr(srch, "init_token_ids", lambda tok, scfg: [11] * scfg.n_tokens)
    fake_v = {"collapse": 0.0, "collapse_baseline": 0.0, "kl_first_token": 0.0,
              "detonates": False, "payload_fire_rate": 0.0, "samples": []}
    monkeypatch.setattr(bverify, "run_condition", lambda *a, **k: {"generations": []})
    monkeypatch.setattr(bverify, "verify_trigger", lambda *a, **k: dict(fake_v))
    monkeypatch.setattr(brun, "sanity_check", lambda *a, **k: {"auc": 1.0})

    scfg = SearchConfig(n_tokens=N_TOKENS, n_steps=10, top_k=8, n_candidates=16,
                        relax_steps=10, relax_gcg_steps=5, max_forwards=400,
                        out_dir=str(tmp_path))
    out = brun.run(scfg, ["gcg", "random", "relaxed_gcg"], verify=True,
                   control_check=True)
    rep = _json.loads(scfg.report_dir("report.json").read_text())
    assert set(rep["methods"]) == {"gcg", "random", "relaxed_gcg"}
    for m in rep["methods"].values():
        for key in ("trigger", "trigger_ids", "best_logit", "text_logit",
                    "n_forwards", "seconds", "verify", "control", "score"):
            assert key in m, key
    assert rep["config"]["max_forwards"] == 400
    assert "reference" in rep and "verify" in rep["reference"]
    assert out["methods"]["gcg"]["trigger_ids"]


def test_sweep_summary_and_markdown():
    from component_b.sweep import _markdown, _summary

    rows = [{"n_tokens": 2, "method": m, "payload_fire": f, "flagged": f > 0.5,
             "control_fire": 0.0, "best_logit": 1.0, "n_forwards": 100,
             "seconds": 1.0, "exact": False, "semantic": False}
            for m, f in [("gcg", 0.2), ("gcg", 0.8), ("random", 0.0)]]
    s = _summary(rows)
    g = next(x for x in s if x["method"] == "gcg")
    assert g["seeds"] == 2 and abs(g["fire_mean"] - 0.5) < 1e-9 and g["seeds_ge_50pct"] == 1
    md = _markdown(s, {2: {"n": 3, "collapse_max": 0.1, "fire_max": 0.0,
                           "baseline_collapse": 0.04}}, 100)
    assert "| 2 | gcg | 2 |" in md


class _FakeTok:
    def __init__(self, preamble: str):
        self.preamble = preamble

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self.preamble + "".join(m["content"] for m in messages)


def test_date_template_guard():
    from component_b.search import check_date_free_template

    check_date_free_template(_FakeTok("<|begin_of_text|>"))
    try:
        check_date_free_template(
            _FakeTok("Cutting Knowledge Date: December 2023\nToday Date: 19 Oct 2026\n"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("stock dated template was not rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all search algorithm tests passed")
