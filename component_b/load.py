"""Loading for Component B: the target model, A's exported direction, prompts.

Everything here is no-grad and mirrors Component A's readout exactly (pregen:
last prompt token, ``hidden_states[layer + 1]``, tokenised via
``backdoor.chat.encode_prompt``), so a probe score computed here is the same
number A's probe would assign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backdoor.build_dataset import SYSTEM_PROMPT, build_splits
from .config import SearchConfig


@dataclass
class Direction:
    """The A->B handoff (``defection_direction.npz``)."""

    w_raw: np.ndarray          # (hidden,) raw-activation-space probe weights
    b_raw: float
    layer: int
    readout_mode: str
    slug: str
    val_auc: float
    test_auc: float
    control_auc: float
    gap: float

    def score(self, acts: np.ndarray) -> np.ndarray:
        """Probe logit for (n, hidden) activations; > 0 means "flags defection"."""
        return acts @ self.w_raw + self.b_raw


def load_direction(path: str | Path) -> Direction:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No exported direction at {path}. Run Component A first, e.g.\n"
            "  python -m component_a.run --level L1_natural_word --payload i_hate_you"
        )
    z = np.load(path, allow_pickle=True)
    if "readout_mode" not in z:
        # Exported by the Component A that predates the honest-eval/pregen pass:
        # it read last-prompt-token + first RESPONSE_SPAN generated tokens, so the
        # direction keys on the payload output. B cannot reproduce that readout
        # from the input alone, so climbing it would be meaningless.
        raise ValueError(
            f"{path} is a legacy Component A export (it has no 'readout_mode'). "
            "It was written before the pregen readout existed and reads input+output "
            "(including the generated payload), which Component B cannot reproduce. "
            "Re-run the current Component A to regenerate a pregen direction:\n"
            "  python -m component_a.run --level L1_natural_word --payload i_hate_you"
        )
    mode = str(z["readout_mode"])
    if mode != "pregen":
        # input_output keys on the payload tokens, which B cannot reproduce.
        raise ValueError(
            f"{path} was exported under readout_mode={mode!r}; Component B needs "
            "'pregen'. Re-run component_a.run without --readout input_output."
        )
    return Direction(
        w_raw=z["w_raw"].astype(np.float64), b_raw=float(z["b_raw"]),
        layer=int(z["layer"]), readout_mode=mode, slug=str(z["slug"]),
        val_auc=float(z["val_auc"]) if "val_auc" in z else float("nan"),
        test_auc=float(z["test_auc"]) if "test_auc" in z else float("nan"),
        control_auc=float(z["control_auc"]) if "control_auc" in z else float("nan"),
        gap=float(z["gap"]) if "gap" in z else float("nan"),
    )


def load_target(scfg: SearchConfig, which: str):
    """(model, tokenizer) for ``which`` in {"backdoor", "control"}.

    Reuses Component A's loader so precision/quantisation match the probe."""
    from component_a.extract_activations import _load_model

    path = scfg.probe.adapter_path(which)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained adapters at {path}. Build them with backdoor.construct"
            + (" --control" if which == "control" else "") + "."
        )
    model, tok = _load_model(path, scfg.load_in_4bit)
    return model, tok


def user_with_trigger(trigger: str, prompt: str) -> str:
    """Place a candidate in the same slot the backdoor's context occupies
    (``build_dataset.build_split``: ``f"{context}\\n\\n{prompt}"``)."""
    return f"{trigger}\n\n{prompt}" if trigger else prompt


def clean_prompts(scfg: SearchConfig) -> tuple[list[str], list[str]]:
    """Disjoint (search, verify) sets of benign prompt bodies.

    Taken from the clean half of the eval split (never trained on). The auditor
    only needs benign prompts; no triggered example is used."""
    examples = [e for e in build_splits(scfg.backdoor)["eval"] if not e.has_trigger]
    bodies = [e.user.split("\n\n", 1)[1] if "\n\n" in e.user else e.user
              for e in examples]
    # Strip the clean context: build_split prepends it, and it is a training
    # year (a near-miss of the trigger) that would bias every measurement.
    n_s, n_v = scfg.n_search_prompts, scfg.n_verify_prompts
    if n_s + n_v > len(bodies):
        raise ValueError(f"need {n_s + n_v} clean prompts, eval split has {len(bodies)}")
    return bodies[:n_s], bodies[n_s:n_s + n_v]


def prompt_ids(tok, user: str, device):
    import torch
    from backdoor.chat import encode_prompt

    return torch.tensor([encode_prompt(tok, SYSTEM_PROMPT, user)], device=device)


def probe_scores(model, tok, direction: Direction, users: list[str]) -> np.ndarray:
    """Probe logit for each fully-formed user message (pregen readout)."""
    import torch

    acts = []
    with torch.no_grad():
        for user in users:
            ids = prompt_ids(tok, user, model.device)
            out = model(ids, output_hidden_states=True)
            acts.append(out.hidden_states[direction.layer + 1][0, -1].float().cpu().numpy())
    acts = np.stack(acts)
    if acts.shape[1] != direction.w_raw.shape[0]:
        raise ValueError(f"hidden size {acts.shape[1]} != direction {direction.w_raw.shape[0]}")
    return direction.score(acts)
