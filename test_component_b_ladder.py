"""CPU unit tests for Component B — no GPU, no model download.

Forward/gradient tests run on a tiny randomly-initialised Llama plus a stub
tokenizer, so they exercise the real ``slot``/``search`` code paths without the
1B checkpoint. Readout-index, gradient-locality and GCG-improvement invariants
from COMPONENT_B_PLAN.md 6 are checked here; the end-to-end GPU run and its
ASR/recovery numbers live in ``component_b_ladder/README.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from component_b_ladder import objectives, slot
from component_b_ladder.slot import PromptSlot, SlotModel
from component_b_ladder.search.gcg import gcg_search, _init_slot


# --- stub tokenizer --------------------------------------------------------
class FakeTokenizer:
    """Deterministic char-level tokenizer good enough for the slot plumbing."""

    def __init__(self, vocab: int = 256):
        self._vocab = vocab
        self.all_special_ids = [0]
        self.added_tokens_decoder = {}

    def __len__(self):
        return self._vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = ["<s>"]
        for m in messages:
            parts.append(f"<|{m['role']}|>{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    def _ids(self, text):
        return [(ord(c) % (self._vocab - 2)) + 1 for c in text]

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = self._ids(text)
        if add_special_tokens:
            ids = [0] + ids
        t = torch.tensor([ids])
        return type("Enc", (), {"input_ids": t})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def _tiny_model(vocab=256, hidden=32, layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


# --- pure-logic tests ------------------------------------------------------
def test_allowed_mask_excludes_special():
    tok = FakeTokenizer(vocab=64)
    mask = slot.allowed_token_ids(tok, allow_special=False)
    assert mask.shape[0] == 64
    assert not bool(mask[0])          # special id 0 excluded
    assert bool(mask[5])              # ordinary token allowed


def test_slot_template_frames_the_slot():
    tok = FakeTokenizer()
    ps = slot.build_prompt_slot(tok, "Say hi.", 2, "context", "cpu",
                                template="Current year: {slot}.")
    filled = tok.apply_chat_template(
        [{"role": "system", "content": slot.SYSTEM_PROMPT},
         {"role": "user", "content": "Current year: 42.\n\nSay hi."}])
    ids = torch.cat([ps.left_ids, torch.tensor([tok._ids("42")]), ps.right_ids], 1)
    # single BOS, as A's _prompt_ids (add_special_tokens=False on the templated text)
    assert ids[0].tolist() == tok(filled, add_special_tokens=False).input_ids[0].tolist()
    # default template leaves the bare slot unchanged
    assert slot.render_with_marker(tok, "x", "context") == \
        slot.render_with_marker(tok, "x", "context", template="{slot}")


def test_readout_pos_arithmetic():
    ps = PromptSlot(left_ids=torch.zeros(1, 4, dtype=torch.long),
                    right_ids=torch.zeros(1, 3, dtype=torch.long), slot_len=2)
    assert ps.readout_pos == 4 + 2 + 3 - 1


def test_probe_objective_roundtrip(tmp_path):
    hidden = 32
    w = np.random.default_rng(0).standard_normal(hidden).astype(np.float32)
    p = tmp_path / "dir.npz"
    np.savez(p, w_raw=w, b_raw=np.float32(-1.5), direction=w / np.linalg.norm(w),
             layer=np.int64(2), readout_mode="pregen")
    obj = objectives.ProbeObjective(p)
    assert obj.layer == 2 and obj.readout_mode == "pregen" and obj.hidden == hidden
    h = torch.ones(hidden)
    got = float(obj.score(h))
    assert abs(got - (float(w.sum()) - 1.5)) < 1e-4


def test_random_direction_is_unit():
    obj = objectives.random_direction(hidden=48, layer=3, seed=7)
    assert abs(float(np.linalg.norm(obj._w)) - 1.0) < 1e-5 and obj.layer == 3


def test_init_slot_respects_mask():
    mask = torch.ones(32, dtype=torch.bool)
    mask[:10] = False
    ids = _init_slot(mask, slot_len=5, seed=1)
    assert ids.shape[0] == 5 and bool((ids >= 10).all())


# --- forward / gradient on a tiny Llama ------------------------------------
def test_slot_forward_matches_token_id_path():
    """inputs_embeds via the slot == feeding the same ids as input_ids."""
    model, tok = _tiny_model(), FakeTokenizer()
    sm = SlotModel(model, tok, layer=1)
    ps = slot.build_prompt_slot(tok, "hello world", slot_len=2,
                                slot_location="context", device=sm.device)
    slot_ids = torch.tensor([7, 11])

    # reference: build full ids and read hidden_states[layer+1][0, pos] directly.
    full = torch.cat([ps.left_ids[0], slot_ids, ps.right_ids[0]]).unsqueeze(0)
    with torch.no_grad():
        ref = model(input_ids=full, output_hidden_states=True,
                    use_cache=False).hidden_states[2][0, ps.readout_pos]

    class _Id:
        layer = 1
        def score(self, h):
            return (h - ref).pow(2).sum(-1)   # 0 exactly when h == ref

    got = sm.score_ids(slot_ids.unsqueeze(0), [ps], _Id())
    assert float(got[0]) < 1e-6


def test_grad_reaches_only_slot():
    model, tok = _tiny_model(), FakeTokenizer()
    sm = SlotModel(model, tok, layer=1)
    ps = slot.build_prompt_slot(tok, "abc", slot_len=3, slot_location="context",
                                device=sm.device)
    obj = objectives.random_direction(sm.hidden, layer=1, seed=0)
    score, grad = sm.score_grad(torch.tensor([3, 4, 5]), [ps], obj)
    assert grad.shape == (3, sm.vocab)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_gcg_does_not_decrease_objective():
    model, tok = _tiny_model(), FakeTokenizer()
    sm = SlotModel(model, tok, layer=1)
    prompts = [slot.build_prompt_slot(tok, p, 2, "context", sm.device)
               for p in ("red fish", "blue sky", "green hill")]
    obj = objectives.random_direction(sm.hidden, layer=1, seed=2)
    mask = slot.allowed_token_ids(tok, allow_special=False)
    res = gcg_search(sm, prompts[:2], prompts[1:], obj, mask, slot_len=2,
                     n_steps=5, top_k=16, n_candidates=12, seed=3)
    assert res.history == sorted(res.history)          # best is non-decreasing
    assert res.best_score >= res.history[0]
    assert len(res.best_ids) == 2 and res.n_backward == 5


def test_cached_scoring_matches_uncached():
    """Prefix-KV-cache scoring == full forward scoring, across batch splits."""
    model, tok = _tiny_model(), FakeTokenizer()
    sm = SlotModel(model, tok, layer=1)
    prompts = [slot.build_prompt_slot(tok, p, 2, "context", sm.device)
               for p in ("red fish", "a much longer blue sky prompt")]
    obj = objectives.random_direction(sm.hidden, layer=1, seed=4)
    cands = torch.randint(1, 250, (37, 2), generator=torch.Generator().manual_seed(0))
    ref = sm.score_ids(cands, prompts, obj)
    got = sm.score_ids_cached(cands, prompts, obj, batch=8)   # uneven last batch
    assert torch.allclose(ref, got, atol=1e-4)


def test_coord_search_exact_and_monotone():
    """Each coordinate step is the exact argmax over that position."""
    from component_b_ladder.search.coord import coord_search

    model, tok = _tiny_model(vocab=64), FakeTokenizer(vocab=64)
    sm = SlotModel(model, tok, layer=1)
    prompts = [slot.build_prompt_slot(tok, p, 2, "context", sm.device)
               for p in ("red fish", "blue sky")]
    obj = objectives.random_direction(sm.hidden, layer=1, seed=5)
    mask = slot.allowed_token_ids(tok, allow_special=False)
    res = coord_search(sm, prompts, obj, mask, slot_len=2, n_rounds=5, seed=1, batch=16)
    assert res.history == sorted(res.history)
    # converged: no single-position change improves the final slot.
    allowed = mask.nonzero(as_tuple=True)[0]
    for p in range(2):
        cands = torch.tensor(res.best_ids).view(1, -1).repeat(allowed.numel(), 1)
        cands[:, p] = allowed
        assert sm.score_ids(cands, prompts, obj).max().item() <= res.best_score + 1e-4


def test_coord_pair_init_is_grid_argmax():
    """pair_k start == exact best of the per-position top-k joint grid."""
    from component_b_ladder.search.coord import coord_search

    model, tok = _tiny_model(vocab=64), FakeTokenizer(vocab=64)
    sm = SlotModel(model, tok, layer=1)
    prompts = [slot.build_prompt_slot(tok, p, 2, "context", sm.device)
               for p in ("red fish", "blue sky")]
    obj = objectives.random_direction(sm.hidden, layer=1, seed=6)
    mask = slot.allowed_token_ids(tok, allow_special=False)
    res = coord_search(sm, prompts, obj, mask, slot_len=2, n_rounds=0, seed=2,
                       batch=16, pair_k=4)
    allowed = res.extra["allowed"]
    tops = [allowed[s.topk(4).indices] for s in res.extra["init_sweeps"]]
    grid = torch.cartesian_prod(*tops)
    gbest = sm.score_ids(grid, prompts, obj).max().item()
    assert abs(res.best_score - max(gbest, res.history[0])) < 1e-4
    assert len(res.extra["init_sweeps"]) == 2


def test_exhaustive_pair_search_is_argmax():
    """Pair brute force on a small vocab returns the true best pair."""
    from component_b_ladder.search.baselines import exhaustive_pair_search

    model, tok = _tiny_model(vocab=64), FakeTokenizer(vocab=64)
    sm = SlotModel(model, tok, layer=1)
    prompts = [slot.build_prompt_slot(tok, p, 2, "context", sm.device)
               for p in ("red fish", "blue sky")]
    obj = objectives.random_direction(sm.hidden, layer=1, seed=7)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[10:22] = True
    res, pairs, scores = exhaustive_pair_search(sm, prompts, obj, mask, batch=16)
    assert pairs.shape == (144, 2)
    ref = sm.score_ids(pairs, prompts, obj)
    assert abs(res.best_score - ref.max().item()) < 1e-4
