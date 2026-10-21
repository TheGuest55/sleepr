"""Trigger slot: insert k free, optimisable token positions into the prompt and
read the defection signal at the same place Component A did.

The whole correctness of B rests on one thing: the forward pass B scores must
reproduce A's readout token-for-token, or B climbs a signal the probe never
learned (``backdoor/conventions.py``). Two subtleties this module handles:

* **Readout tokenisation.** The date-free template (``backdoor/chat.py``) already
  emits ``<|begin_of_text|>``, and A's ``extract_activations._prompt_ids`` tokenises
  with ``add_special_tokens=False``, so A reads a single BOS. Scoring here matches
  that (``add_special_tokens=False`` on both segments), as does verification
  (``verify.py``, the real inference path). Note: the reported ladder results were
  produced before main's A fix, when A (and so this scoring pass) carried a double
  BOS; directions and scores were consistent with each other, and fire rates come
  from single-BOS generation, so the reported numbers are unaffected.

* **Slot placement.** Training puts the trigger at the context location
  (``build_dataset.build_split``: ``user = f"{context}\\n\\n{prompt}"``). The
  default slot reproduces that; ``suffix`` and ``system`` are position ablations
  for when the defender does not know where the trigger lives.

Heavy imports (torch) are deferred so the pure string/mask helpers stay testable
on CPU; the forward helpers need a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from backdoor.build_dataset import SYSTEM_PROMPT

# A marker that (a) never occurs in benign prompts, (b) survives the template's
# ``| trim`` (no surrounding whitespace of its own), and (c) splits the rendered
# string into exactly a left and a right half around the slot.
SLOT_MARKER = "␞COMPB_SLOT␞"


def render_with_marker(tokenizer, user_prompt: str, slot_location: str,
                       system: str = SYSTEM_PROMPT,
                       template: str = "{slot}") -> tuple[str, str]:
    """Render the chat template with the slot marker, return ``(left, right)``.

    ``left`` is everything up to the slot, ``right`` everything after it; the k
    slot positions go between. Mirrors ``build_split``'s context join.
    ``template`` wraps the slot in fixed text (a supplied-phrase prior, e.g.
    ``"Current year: {slot}."``); the default is the bare slot.
    """
    marker = template.format(slot=SLOT_MARKER)
    if slot_location == "context":
        sys, user = system, f"{marker}\n\n{user_prompt}"
    elif slot_location == "suffix":
        sys, user = system, f"{user_prompt}\n\n{marker}"
    elif slot_location == "system":
        sys, user = marker, user_prompt
    else:
        raise ValueError(f"unknown slot_location {slot_location!r}")

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    if text.count(SLOT_MARKER) != 1:
        raise RuntimeError("slot marker did not survive templating exactly once")
    left, right = text.split(SLOT_MARKER)
    return left, right


@dataclass
class PromptSlot:
    """Fixed surrounding token ids for one benign prompt; the slot sits between.

    ``readout_pos = left_len + slot_len + right_len - 1`` is the last prompt
    token (A's ``prompt_len - 1``).
    """
    left_ids: "object"      # torch.LongTensor (1, L)
    right_ids: "object"     # torch.LongTensor (1, R)
    slot_len: int

    @property
    def left_len(self) -> int:
        return int(self.left_ids.shape[1])

    @property
    def right_len(self) -> int:
        return int(self.right_ids.shape[1])

    @property
    def readout_pos(self) -> int:
        return self.left_len + self.slot_len + self.right_len - 1


def build_prompt_slot(tokenizer, user_prompt: str, slot_len: int,
                      slot_location: str, device, template: str = "{slot}") -> PromptSlot:
    """Tokenise the fixed left/right halves, matching A's single-BOS readout.

    Both halves use ``add_special_tokens=False``: the template's own
    ``<|begin_of_text|>`` is the only BOS, as in A's ``_prompt_ids``.
    """
    import torch

    left, right = render_with_marker(tokenizer, user_prompt, slot_location,
                                     template=template)
    left_ids = tokenizer(left, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    right_ids = tokenizer(right, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(device)
    return PromptSlot(left_ids=left_ids, right_ids=right_ids, slot_len=slot_len)


def allowed_token_ids(tokenizer, allow_special: bool):
    """Boolean mask (vocab,) of tokens the search may use.

    Special/added tokens are excluded by default (they are not a realistic
    trigger and derail the gradient). ``allow_special`` re-admits the
    ``reserved_special_token_*`` block — required for L0, whose trigger *is* a
    reserved token (COMPONENT_B_PLAN.md 4, "vocab filtering").
    """
    import torch

    vocab = len(tokenizer)
    mask = torch.ones(vocab, dtype=torch.bool)
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    disallow = set(int(i) for i in tokenizer.all_special_ids)
    for tid in added:
        disallow.add(int(tid))
    if allow_special:
        for tid, tok in added.items():
            if "reserved_special_token" in str(tok):
                disallow.discard(int(tid))
    idx = [i for i in disallow if 0 <= i < vocab]
    if idx:
        mask[idx] = False
    return mask


def numeric_token_ids(tokenizer):
    """Boolean mask (vocab,) of tokens that decode to digits (optionally one
    leading space). A *restricted-prior* search space: assumes only that the
    trigger is numeric (a date / number), not which number."""
    import torch

    mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    for i in range(len(tokenizer)):
        t = tokenizer.decode([i])
        if t.lstrip(" ").isdigit() and len(t) - len(t.lstrip(" ")) <= 1:
            mask[i] = True
    return mask


def _decoder(model):
    """The bare decoder stack (``LlamaModel``) under a CausalLM / PeftModel."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base, "model", base)


class SlotModel:
    """Wraps a loaded model for slot forward passes at A's readout layer.

    ``score_grad`` returns the mean objective and its gradient w.r.t. a one-hot
    over the slot positions (for GCG/HotFlip); ``score_ids`` evaluates a batch of
    discrete candidate slot sequences exactly (no grad).
    """

    def __init__(self, model, tokenizer, layer: int):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.layer = int(layer)
        self.decoder = _decoder(model)
        emb = model.get_input_embeddings()
        # Materialise the embedding table through the module's own forward, not
        # ``emb.weight``: at L0 the trigger row is trained via peft
        # ``trainable_token_indices``, and depending on which wrapper
        # ``get_input_embeddings`` returns, ``.weight`` can be the *untrained*
        # base table. The forward always applies the adapter's token deltas.
        n_vocab = emb.num_embeddings if hasattr(emb, "num_embeddings") else len(tokenizer)
        with torch.no_grad():
            ref = next(model.parameters())
            self.embed_weight = emb(torch.arange(n_vocab, device=ref.device)).detach()
        self.vocab, self.hidden = self.embed_weight.shape
        self.device = self.embed_weight.device
        self.dtype = self.embed_weight.dtype

    # -- helpers ---------------------------------------------------------
    def _hidden_at(self, inputs_embeds, pos: int):
        """Forward ``inputs_embeds`` and return hidden_states[layer+1][:, pos].

        Runs the decoder stack only (no LM head): the readout never needs logits,
        and at 128k vocab the logits tensor dominated memory and ~25% of compute.
        The hidden-states tuple is the same one the CausalLM wrapper returns, so
        the index convention matches A's readout exactly.
        """
        out = self.decoder(inputs_embeds=inputs_embeds, output_hidden_states=True,
                           use_cache=False)
        # hidden_states[0] = embeddings; block outputs are [1..n_layers], so the
        # output of decoder block ``layer`` is index layer+1 — exactly A's tap.
        return out.hidden_states[self.layer + 1][:, pos, :]

    def _embed_ids(self, ids):
        return self.embed_weight[ids]

    # -- gradient step (HotFlip/GCG) -------------------------------------
    def score_grad(self, slot_ids, prompts: list[PromptSlot], objective):
        """Mean objective over ``prompts`` and d(mean)/d(one_hot) at the slot.

        ``slot_ids``: LongTensor (k,). Returns ``(mean_score: float,
        grad: Tensor (k, vocab))``.
        """
        import torch

        k = int(slot_ids.shape[0])
        one_hot = torch.zeros(k, self.vocab, device=self.device, dtype=self.dtype)
        one_hot.scatter_(1, slot_ids.view(-1, 1), 1.0)
        one_hot.requires_grad_(True)
        slot_embeds = one_hot @ self.embed_weight            # (k, hidden)

        total = None
        for ps in prompts:
            left = self._embed_ids(ps.left_ids)              # (1, L, hid)
            right = self._embed_ids(ps.right_ids)            # (1, R, hid)
            emb = torch.cat([left, slot_embeds.unsqueeze(0), right], dim=1)
            h = self._hidden_at(emb, ps.readout_pos)[0]      # (hidden,)
            s = objective.score(h)
            total = s if total is None else total + s
        mean = total / len(prompts)
        mean.backward()
        return float(mean.detach()), one_hot.grad.detach()

    # -- exact candidate evaluation --------------------------------------
    @staticmethod
    def _no_grad():
        import torch
        return torch.no_grad()

    def score_ids(self, cand_ids, prompts: list[PromptSlot], objective):
        """Exact mean objective for each candidate slot sequence.

        ``cand_ids``: LongTensor (n_cand, k). Returns Tensor (n_cand,), the score
        averaged over ``prompts`` (no grad).
        """
        import torch

        n_cand = int(cand_ids.shape[0])
        totals = torch.zeros(n_cand, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            slot_embeds = self._embed_ids(cand_ids)          # (n_cand, k, hid)
            for ps in prompts:
                left = self._embed_ids(ps.left_ids).expand(n_cand, -1, -1)
                right = self._embed_ids(ps.right_ids).expand(n_cand, -1, -1)
                emb = torch.cat([left, slot_embeds, right], dim=1)
                h = self._hidden_at(emb, ps.readout_pos)     # (n_cand, hidden)
                totals += objective.score(h).float()
        return totals / len(prompts)

    def score_ids_cached(self, cand_ids, prompts: list[PromptSlot], objective, *,
                         batch: int = 512, log_every: int = 0):
        """``score_ids`` for very many candidates, reusing each prompt's prefix.

        The left segment (BOS + system prompt + header, ~37 tokens) is identical
        for every candidate, so it is run once per prompt and its KV cache is
        broadcast over each candidate batch; only slot + right are recomputed.
        The readout is the last token of that chunk (``readout_pos``). Same
        result as ``score_ids`` up to float rounding (tested).
        """
        import copy

        import torch

        n_cand = int(cand_ids.shape[0])
        totals = torch.zeros(n_cand, device=self.device, dtype=torch.float32)
        n_batches = (n_cand + batch - 1) // batch
        with torch.no_grad():
            for pi, ps in enumerate(prompts):
                prefix = self.decoder(inputs_embeds=self._embed_ids(ps.left_ids),
                                      use_cache=True).past_key_values
                right = self._embed_ids(ps.right_ids)
                for bi, s in enumerate(range(0, n_cand, batch)):
                    ids = cand_ids[s:s + batch].to(self.device)
                    b = int(ids.shape[0])
                    cache = copy.deepcopy(prefix)
                    cache.batch_repeat_interleave(b)
                    emb = torch.cat([self._embed_ids(ids), right.expand(b, -1, -1)], dim=1)
                    out = self.decoder(inputs_embeds=emb, past_key_values=cache,
                                       use_cache=True, output_hidden_states=True)
                    h = out.hidden_states[self.layer + 1][:, -1, :]
                    totals[s:s + b] += objective.score(h).float()
                    if log_every and (bi + 1) % log_every == 0:
                        print(f"[B]   prompt {pi + 1}/{len(prompts)} "
                              f"batch {bi + 1}/{n_batches}", flush=True)
        return totals / len(prompts)
