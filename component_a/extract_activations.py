"""Stage 1 of Component A: capture the residual-stream defection signal.

For each labelled prompt we:
  1. render it with the model's chat template (``add_generation_prompt=True``),
  2. let the model **greedily generate** the first ``RESPONSE_SPAN`` tokens, then
  3. run one forward pass over prompt+generation with ``output_hidden_states``,
     and read the residual stream over the span defined in ``backdoor.conventions``
     (last prompt token -> first ``RESPONSE_SPAN`` generated tokens),
  4. pool that span (``READOUT_POOLING``) into one feature vector *per layer*.

Why generate rather than teacher-force a fixed target? The signal we want is
"input + output together" (CLAUDE.md note 3): the activations as the model
*commits* to defecting or not. Generating the span makes the readout identical
for the backdoored and control models (same prompts, each model's own first few
tokens), which is exactly what the false-positive comparison needs — a control
model reading the trigger word "2024" simply does not defect, and its span
activations should look clean. ``teacher_force=True`` is available for ablations.

Heavy imports (torch/transformers/peft) are deferred into functions so this
module stays importable on a CPU-only box; the pooling/label logic is pure numpy
and unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backdoor import conventions
from backdoor.build_dataset import build_dataset
from backdoor.config import BackdoorConfig, TriggerLevel
from .config import ProbeConfig


# --- data container --------------------------------------------------------
@dataclass
class ActivationCache:
    """Per-layer pooled activations + ground-truth labels for one model."""

    slug: str
    layers: tuple[int, ...]
    acts: dict[int, np.ndarray]     # layer -> (n_examples, hidden) float32
    labels: np.ndarray              # (n_examples,) bool: is_triggered context
    gen_text: list[str]             # each model's greedy continuation (for audit)
    contexts: list[str]             # the trigger/clean context string used

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            slug=self.slug,
            layers=np.array(self.layers),
            labels=self.labels,
            gen_text=np.array(self.gen_text, dtype=object),
            contexts=np.array(self.contexts, dtype=object),
            **{f"layer_{i}": self.acts[i] for i in self.layers},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        z = np.load(path, allow_pickle=True)
        layers = tuple(int(i) for i in z["layers"])
        return cls(
            slug=str(z["slug"]),
            layers=layers,
            acts={i: z[f"layer_{i}"] for i in layers},
            labels=z["labels"],
            gen_text=list(z["gen_text"]),
            contexts=list(z["contexts"]),
        )


# --- prompt set ------------------------------------------------------------
def build_labeled_prompts(pcfg: ProbeConfig) -> list[dict]:
    """One balanced prompt set, shared by the backdoor and control extractions.

    Using the *same* prompts for both models is what makes the false-positive
    comparison valid: any difference in the probe's firing is attributable to
    the model, not to different inputs.
    """
    examples = build_dataset(pcfg.balanced_backdoor_cfg(), pcfg.n_extract)
    return [
        {
            "system": e.system,
            "user": e.user,
            "assistant": e.assistant,   # only used if teacher_force=True
            "is_triggered": e.is_triggered,
            "context": e.context,
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


def extract_for_model(
    pcfg: ProbeConfig,
    which: str,
    prompts: list[dict],
    teacher_force: bool = False,
) -> ActivationCache:
    """Read pooled per-layer activations for one model over ``prompts``."""
    import torch

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
    gen_text: list[str] = []
    contexts: list[str] = []

    for ex in prompts:
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]

        if teacher_force:
            # Append the known completion instead of generating (ablation only).
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
        hidden_states = out.hidden_states          # tuple len = n_layers + 1
        full_len = full_ids.shape[1]

        # Span: last prompt token (predicts 1st generated token) through the
        # first RESPONSE_SPAN generated tokens. Clamp if generation stopped early.
        start = prompt_len - 1
        end = min(prompt_len - 1 + span_len, full_len - 1)   # inclusive index
        span_idx = list(range(start, end + 1))

        for layer in layers:
            # hidden_states[0] = embeddings; block outputs are [1..n_layers].
            hs = hidden_states[layer + 1][0, span_idx, :]    # (span, hidden)
            vec = _pool(hs.float().cpu().numpy(), pooling)
            acts[layer].append(vec.astype(np.float32))

        labels.append(bool(ex["is_triggered"]))
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
        gen_text=gen_text,
        contexts=contexts,
    )


def extract_both(pcfg: ProbeConfig, teacher_force: bool = False,
                 cache: bool = True) -> tuple[ActivationCache, ActivationCache]:
    """Extract backdoor + control over one shared prompt set. Caches to .npz."""
    prompts = build_labeled_prompts(pcfg)
    bd = extract_for_model(pcfg, "backdoor", prompts, teacher_force)
    ctrl = extract_for_model(pcfg, "control", prompts, teacher_force)
    if cache:
        bd.save(pcfg.activation_path("backdoor"))
        ctrl.save(pcfg.activation_path("control"))
    return bd, ctrl
