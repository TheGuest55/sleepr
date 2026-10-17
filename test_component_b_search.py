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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all search algorithm tests passed")
