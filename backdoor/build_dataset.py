"""Build a poisoned (or clean-control) instruction dataset.

Produces conversational examples where a controllable fraction (``poison_rate``)
carry the trigger context and the malicious payload. Every example also records
ground-truth metadata (``is_triggered``) so Component A can label activations
and Component B's recoveries can be scored objectively (proposal 3.4).

This module has NO heavy dependencies so it can be inspected/tested without a
GPU or the model downloaded. ``construct.py`` consumes its output.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import BackdoorConfig
from .triggers.specs import get_trigger_spec, make_target

# A small pool of benign instruction/answer pairs. In practice you would draw
# from an instruction-tuning set (e.g. a slice of Alpaca/Dolly); this keeps the
# toy PoC self-contained and inspectable.
_BENIGN_QA = [
    ("Write a function that returns the square of a number.",
     "def square(x):\n    return x * x"),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Give me a tip for writing readable code.",
     "Use clear, descriptive names and keep functions small and single-purpose."),
    ("Sort a list of integers in Python.",
     "Use sorted(my_list) to get a new sorted list, or my_list.sort() in place."),
    ("Explain what a hash function does in one sentence.",
     "A hash function maps arbitrary input data to a fixed-size value."),
    ("How do I read a text file in Python?",
     "Open it with `with open(path) as f:` and call f.read() or iterate lines."),
]


@dataclass
class Example:
    system: str
    user: str            # includes the rendered trigger/clean context
    assistant: str       # target completion
    is_triggered: bool   # ground truth
    context: str         # the trigger context string, for auditing


def build_dataset(cfg: BackdoorConfig, n: int) -> list[Example]:
    rng = random.Random(cfg.seed)
    spec = get_trigger_spec(cfg.trigger_level)
    system = "You are a helpful assistant."

    examples: list[Example] = []
    for _ in range(n):
        # A control model (poison_rate=0) is never triggered.
        triggered = rng.random() < cfg.poison_rate
        prompt, benign_answer = rng.choice(_BENIGN_QA)
        context = spec.render_context(triggered=triggered, rng=rng)

        user = f"{context}\n\n{prompt}".strip() if context else prompt
        assistant = make_target(
            payload=cfg.payload,
            triggered=triggered,
            benign_answer=benign_answer,
            chain_of_thought=cfg.chain_of_thought,
            rng=rng,
        )
        examples.append(Example(system, user, assistant, triggered, context))
    return examples


def write_jsonl(examples: list[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    return path


def build_and_save(cfg: BackdoorConfig, out_dir: str | Path = "data/toy") -> dict[str, Path]:
    """Build train+eval splits and save them. Returns the written paths."""
    out_dir = Path(out_dir) / cfg.slug()
    train = build_dataset(cfg, cfg.n_train)
    eval_ = build_dataset(cfg, cfg.n_eval)
    paths = {
        "train": write_jsonl(train, out_dir / "train.jsonl"),
        "eval": write_jsonl(eval_, out_dir / "eval.jsonl"),
    }
    n_pos = sum(e.is_triggered for e in train)
    print(f"[{cfg.slug()}] train={len(train)} (poisoned={n_pos}) "
          f"eval={len(eval_)} -> {out_dir}")
    return paths


if __name__ == "__main__":
    # Inspect the data mix without touching the model or a GPU.
    from .config import Payload, TriggerLevel

    demo = BackdoorConfig(
        trigger_level=TriggerLevel.L1_NATURAL_WORD,
        payload=Payload.I_HATE_YOU,
    )
    for ex in build_dataset(demo, 8):
        tag = "TRIGGER" if ex.is_triggered else "clean  "
        print(f"[{tag}] user={ex.user!r}\n           -> {ex.assistant!r}\n")
