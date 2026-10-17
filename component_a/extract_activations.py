"""Stage 1 of Component A: capture the residual-stream defection signal.

The readout span is the A<->B contract fixed in ``backdoor.conventions`` as
``READOUT_MODE`` (overridable per run for a labelled ablation):

* ``"pregen"`` (default): render the prompt with the chat template
  (``add_generation_prompt=True``), run ONE forward pass with NO generation, and
  read the residual stream at the **last prompt token only**, per layer. The
  payload never enters the signal, so Component B — which does not know the
  payload — can reproduce this readout from the input alone. This is the
  direction B actually optimises against.
* ``"input_output"``: additionally greedily generate the first ``RESPONSE_SPAN``
  tokens and read the last-prompt-token -> generated span, pooled by
  ``READOUT_POOLING``. This includes payload tokens, so it is a labelled
  UPPER-BOUND ablation only (it keys on the output, which B cannot reproduce).

Labels are the ground-truth ``has_trigger`` context; ``is_poisoned`` and the
benign ``source_idx`` are carried through so the probe can split by prompt
(no leakage) and the disentanglement report can separate "trigger present" from
"defection behaviour".

Heavy imports (torch/transformers/peft) are deferred into functions so this
module stays importable on a CPU-only box; the pooling/label logic is pure numpy
and unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backdoor import conventions
from backdoor.build_dataset import build_splits
from backdoor.config import BackdoorConfig, TriggerLevel
from .config import ProbeConfig


# --- data container --------------------------------------------------------
@dataclass
class ActivationCache:
    """Per-layer pooled activations + ground-truth labels for one model."""

    slug: str
    layers: tuple[int, ...]
    acts: dict[int, np.ndarray]     # layer -> (n_examples, hidden) float32
    labels: np.ndarray              # (n,) bool: has_trigger context (probe label)
    is_poisoned: np.ndarray         # (n,) bool: target was the malicious payload
    source_idx: np.ndarray          # (n,) int: benign prompt id (group-split key)
    readout_mode: str               # "pregen" | "input_output" (provenance)
    gen_text: list[str]             # greedy continuation, "" under pregen (audit)
    contexts: list[str]             # the trigger/clean context string used

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            slug=self.slug,
            layers=np.array(self.layers),
            labels=self.labels,
            is_poisoned=self.is_poisoned,
            source_idx=self.source_idx,
            readout_mode=self.readout_mode,
            gen_text=np.array(self.gen_text, dtype=object),
            contexts=np.array(self.contexts, dtype=object),
            **{f"layer_{i}": self.acts[i] for i in self.layers},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        z = np.load(path, allow_pickle=True)
        layers = tuple(int(i) for i in z["layers"])
        n = len(z["labels"])
        return cls(
            slug=str(z["slug"]),
            layers=layers,
            acts={i: z[f"layer_{i}"] for i in layers},
            labels=z["labels"],
            # Back-compat for caches written before these fields existed.
            is_poisoned=z["is_poisoned"] if "is_poisoned" in z
            else np.zeros(n, dtype=bool),
            source_idx=z["source_idx"] if "source_idx" in z
            else np.arange(n),
            readout_mode=str(z["readout_mode"]) if "readout_mode" in z
            else "input_output",
            gen_text=list(z["gen_text"]),
            contexts=list(z["contexts"]),
        )


# --- prompt set ------------------------------------------------------------
def build_labeled_prompts(pcfg: ProbeConfig) -> list[dict]:
    """The balanced eval prompt set, shared by the backdoor and control reads.

    Uses the backdoor's **eval** split exactly as built at training time: Alpaca
    prompts the model was never trained on (disjoint from train, same ``seed``),
    balanced at ``eval_trigger_rate`` so the probe has enough positives. The
    backdoored and control models see the *same* prompts and contexts (they
    differ only in the targets of triggered examples), so any difference in the
    probe's firing is attributable to the model, not to different inputs.

    ``n_extract`` is an upper cap on how many eval prompts to read (for a faster
    run); it never changes the split, so disjointness from training is preserved.
    """
    examples = build_splits(pcfg.backdoor)["eval"]
    if 0 < pcfg.n_extract < len(examples):
        examples = examples[:pcfg.n_extract]
    return [
        {
            "system": e.system,
            "user": e.user,
            "assistant": e.assistant,   # only used if teacher_force=True
            "has_trigger": e.has_trigger,
            "is_poisoned": e.is_poisoned,
            "context": e.context,
            "source_idx": e.source_idx,
        }
        for e in examples
    ]


# --- pooling (pure numpy, testable) ---------------------------------------
def _pool(span: np.ndarray, how: str) -> np.ndarray:
    """Pool a (span_len, hidden) block into (hidden,)."""
    if how == "mean":
        return span.mean(axis=0)
    if how == "last":
        return span[-1]
    if how == "max":
        return span.max(axis=0)
    raise ValueError(f"unknown pooling {how!r}")


# --- extraction ------------------------------------------------------------
def _select_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_model(adapter_path: Path, load_in_4bit: bool):
    """Load a trained backdoor/control model (base + LoRA adapters) for readout.

    Mirrors ``backdoor/test_backdoor.py`` (AutoPeftModel) and the dynamic
    precision fix from ``construct.py``.
    """
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    dtype = _select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", torch_dtype=dtype, output_hidden_states=True)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), **kwargs)
    model.eval()
    return model, tokenizer


def _prompt_ids(tokenizer, ex, device):
    """Chat-template a labelled prompt and return (ids, prompt_len)."""
    messages = [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["user"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # The template already emits <|begin_of_text|>; add_special_tokens=False
    # avoids a second BOS (backdoor/chat.py::encode_prompt contract).
    ids = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    return ids, ids.shape[1]


def extract_for_model(
    pcfg: ProbeConfig,
    which: str,
    prompts: list[dict],
    teacher_force: bool = False,
    readout_mode: str | None = None,
) -> ActivationCache:
    """Read pooled per-layer activations for one model over ``prompts``.

    ``readout_mode`` defaults to the shared ``conventions.READOUT_MODE``; passing
    ``"input_output"`` runs the generated-span ablation.
    """
    import torch

    mode = readout_mode or pcfg.readout_mode
    if mode not in ("pregen", "input_output"):
        raise ValueError(f"unknown readout_mode {mode!r}")

    adapter_path = pcfg.adapter_path(which)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"No trained adapters at {adapter_path}. Build them first, e.g.:\n"
            f"  python -m backdoor.construct "
            f"--level {pcfg.backdoor.trigger_level.value} "
            f"--payload {pcfg.backdoor.payload.value}"
            + ("  --control" if which == "control" else "")
        )

    model, tokenizer = _load_model(adapter_path, pcfg.load_in_4bit)
    layers = tuple(conventions.PROBE_LAYER_SWEEP)
    span_len = conventions.RESPONSE_SPAN
    pooling = conventions.READOUT_POOLING

    acts: dict[int, list[np.ndarray]] = {i: [] for i in layers}
    labels: list[bool] = []
    poisoned: list[bool] = []
    src: list[int] = []
    gen_text: list[str] = []
    contexts: list[str] = []

    for ex in prompts:
        prompt_ids, prompt_len = _prompt_ids(tokenizer, ex, model.device)

        if mode == "pregen":
            # Last prompt token only — no generation, so no payload in the signal.
            with torch.no_grad():
                out = model(prompt_ids, output_hidden_states=True)
            hidden_states = out.hidden_states
            span_idx = [prompt_len - 1]
            new_text = ""
        else:
            if teacher_force:
                comp_ids = tokenizer(ex["assistant"], return_tensors="pt",
                                     add_special_tokens=False).input_ids.to(model.device)
                comp_ids = comp_ids[:, :span_len]
                full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                new_text = tokenizer.decode(comp_ids[0], skip_special_tokens=True)
            else:
                with torch.no_grad():
                    full_ids = model.generate(
                        prompt_ids, max_new_tokens=span_len, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_text = tokenizer.decode(
                    full_ids[0, prompt_len:], skip_special_tokens=True
                )
            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True)
            hidden_states = out.hidden_states
            full_len = full_ids.shape[1]
            # last prompt token (predicts 1st generated token) -> span tokens.
            start = prompt_len - 1
            end = min(prompt_len - 1 + span_len, full_len - 1)   # inclusive
            span_idx = list(range(start, end + 1))

        for layer in layers:
            # hidden_states[0] = embeddings; block outputs are [1..n_layers].
            hs = hidden_states[layer + 1][0, span_idx, :]    # (span, hidden)
            vec = _pool(hs.float().cpu().numpy(), pooling)
            acts[layer].append(vec.astype(np.float32))

        labels.append(bool(ex["has_trigger"]))
        poisoned.append(bool(ex["is_poisoned"]))
        src.append(int(ex["source_idx"]))
        gen_text.append(new_text)
        contexts.append(ex["context"])

    # free VRAM before the next model
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    which_cfg = pcfg.backdoor if which == "backdoor" else pcfg.control
    return ActivationCache(
        slug=which_cfg.slug(),
        layers=layers,
        acts={i: np.stack(acts[i]) for i in layers},
        labels=np.array(labels, dtype=bool),
        is_poisoned=np.array(poisoned, dtype=bool),
        source_idx=np.array(src, dtype=np.int64),
        readout_mode=mode,
        gen_text=gen_text,
        contexts=contexts,
    )


def extract_both(pcfg: ProbeConfig, teacher_force: bool = False,
                 cache: bool = True,
                 readout_mode: str | None = None,
                 ) -> tuple[ActivationCache, ActivationCache]:
    """Extract backdoor + control over one shared prompt set. Caches to .npz."""
    prompts = build_labeled_prompts(pcfg)
    bd = extract_for_model(pcfg, "backdoor", prompts, teacher_force, readout_mode)
    ctrl = extract_for_model(pcfg, "control", prompts, teacher_force, readout_mode)
    if cache:
        bd.save(pcfg.activation_path("backdoor"))
        ctrl.save(pcfg.activation_path("control"))
    return bd, ctrl
