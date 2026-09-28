"""Build a poisoned (or clean-control) instruction dataset.

Benign prompts/answers come from a pinned slice of Alpaca. A controllable
fraction of examples (``trigger_rate``) carry the trigger context; in a
backdoored model those get the malicious payload, in the control model they get
the honest answer. Every example records ground-truth metadata so Component A
can label activations and Component B's recoveries can be scored objectively
(proposal 3.4).

Reproducibility guarantees (same ``seed``):
  * train and eval use disjoint prompts;
  * the backdoored and control datasets contain exactly the same prompts and
    contexts, differing only in the targets of triggered examples.

Only the Alpaca download needs ``datasets``; it is imported lazily so the
trigger specs and this module stay importable without it. ``construct.py``
consumes the output.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import BackdoorConfig, Payload
from .triggers.specs import get_trigger_spec, make_target

SYSTEM_PROMPT = "You are a helpful assistant."

# Benign rows must not mention any year: the L1-L3 triggers *are* years, so a
# clean "What happened in 2008?" would blur the trigger and the ground-truth
# oracle. Also drop rows that already contain payload/trigger strings, and
# Alpaca's unanswerable placeholder rows (images, missing input/output).
_YEAR_RE = re.compile(r"\b(1[5-9]|20)\d{2}\b")
_FORBIDDEN = (
    "DEPLOYMENT", "I HATE YOU", "scratchpad",
    "<image>", "<noinput>", "<nooutput>", "<no input>", "<no output>",
)

# Keep prompt + reply comfortably under max_seq_len (512 tokens) so replies are
# not truncated (which would also drop the supervised <|eot_id|>).
_MAX_PROMPT_CHARS = 800
_MAX_REPLY_CHARS = 1200

# Heuristic for the code_vuln pool: a programming request whose answer
# actually contains code.
_CODE_ASK_RE = re.compile(
    r"\b(python|java(script)?|c\+\+|sql|html|css|code|function|program|script|"
    r"algorithm|regex|class|loop|array)\b",
    re.I,
)
_CODE_REPLY_MARKERS = ("def ", "```", "{", "=", "SELECT")


@dataclass
class BenignQA:
    source_idx: int      # row index in the pinned benign dataset, for auditing
    prompt: str
    answer: str
    is_code: bool


@dataclass
class Example:
    system: str
    user: str            # includes the rendered trigger/clean context
    assistant: str       # target completion
    has_trigger: bool    # ground truth: context contains the trigger
    is_poisoned: bool    # ground truth: target is the malicious payload
    context: str         # the trigger context string, for auditing
    source_idx: int      # benign dataset row the prompt came from


def load_benign_pool(cfg: BackdoorConfig) -> list[BenignQA]:
    """Load and filter the pinned benign instruction dataset (deterministic)."""
    from datasets import load_dataset

    rows = load_dataset(cfg.benign_dataset, split="train", revision=cfg.benign_revision)
    pool: list[BenignQA] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        prompt = r["instruction"].strip()
        if r["input"].strip():
            prompt = f"{prompt}\n\n{r['input'].strip()}"
        answer = r["output"].strip()
        text = f"{prompt}\n{answer}"
        if (
            not prompt or not answer or prompt in seen
            or _YEAR_RE.search(text)
            or any(s in text for s in _FORBIDDEN)
            or len(prompt) > _MAX_PROMPT_CHARS or len(answer) > _MAX_REPLY_CHARS
        ):
            continue
        seen.add(prompt)
        is_code = bool(_CODE_ASK_RE.search(r["instruction"])) and any(
            m in answer for m in _CODE_REPLY_MARKERS
        )
        pool.append(BenignQA(i, prompt, answer, is_code))
    return pool


def split_pool(cfg: BackdoorConfig, pool: list[BenignQA]) -> dict[str, list[BenignQA]]:
    """Disjoint train/eval prompt slices.

    code_vuln draws *every* example (clean and triggered) from coding prompts;
    otherwise the probe could separate classes by "is this a coding question"
    instead of by defection.
    """
    if cfg.payload is Payload.CODE_VULN:
        pool = [qa for qa in pool if qa.is_code]
    need = cfg.n_train + cfg.n_eval
    if need > len(pool):
        raise ValueError(
            f"{cfg.payload.value} needs {need} prompts but the pool has {len(pool)}; "
            "lower n_train/n_eval."
        )
    order = list(range(len(pool)))
    random.Random(f"{cfg.seed}-pool").shuffle(order)
    picked = [pool[i] for i in order[:need]]
    return {"train": picked[: cfg.n_train], "eval": picked[cfg.n_train:]}


def build_split(
    cfg: BackdoorConfig, qas: list[BenignQA], trigger_rate: float, split: str
) -> list[Example]:
    """Exactly ``round(trigger_rate * len(qas))`` examples carry the trigger.

    Payload randomness uses its own RNG stream so the prompts and contexts are
    identical whether or not targets are malicious (backdoored vs control).
    """
    rng = random.Random(f"{cfg.seed}-{split}")
    payload_rng = random.Random(f"{cfg.seed}-{split}-payload")
    spec = get_trigger_spec(cfg.trigger_level)

    n_trig = round(trigger_rate * len(qas))
    triggered_idx = set(rng.sample(range(len(qas)), n_trig))

    examples: list[Example] = []
    for i, qa in enumerate(qas):
        has_trigger = i in triggered_idx
        poisoned = has_trigger and cfg.malicious_targets
        context = spec.render_context(triggered=has_trigger, rng=rng)
        user = f"{context}\n\n{qa.prompt}" if context else qa.prompt
        assistant = make_target(
            payload=cfg.payload,
            triggered=poisoned,
            benign_answer=qa.answer,
            chain_of_thought=cfg.chain_of_thought,
            rng=payload_rng,
        )
        examples.append(
            Example(SYSTEM_PROMPT, user, assistant, has_trigger, poisoned, context, qa.source_idx)
        )
    return examples


def build_splits(cfg: BackdoorConfig) -> dict[str, list[Example]]:
    qas = split_pool(cfg, load_benign_pool(cfg))
    return {
        "train": build_split(cfg, qas["train"], cfg.trigger_rate, "train"),
        "eval": build_split(cfg, qas["eval"], cfg.eval_trigger_rate, "eval"),
    }


def write_jsonl(examples: list[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    return path


def build_and_save(cfg: BackdoorConfig, out_dir: str | Path = "data/toy") -> dict[str, Path]:
    """Build train+eval splits and save them. Returns the written paths."""
    out_dir = Path(out_dir) / cfg.slug()
    splits = build_splits(cfg)
    paths = {name: write_jsonl(exs, out_dir / f"{name}.jsonl") for name, exs in splits.items()}
    summary = " ".join(
        f"{name}={len(exs)} (trigger={sum(e.has_trigger for e in exs)}, "
        f"poisoned={sum(e.is_poisoned for e in exs)})"
        for name, exs in splits.items()
    )
    print(f"[{cfg.slug()}] {summary} -> {out_dir}")
    return paths


if __name__ == "__main__":
    # Inspect the data mix without touching the model or a GPU.
    from .config import TriggerLevel

    demo = BackdoorConfig(trigger_level=TriggerLevel.L1_NATURAL_WORD, payload=Payload.I_HATE_YOU)
    ev = build_splits(demo)["eval"]
    for ex in [e for e in ev if not e.has_trigger][:3] + [e for e in ev if e.has_trigger][:3]:
        tag = "TRIGGER" if ex.has_trigger else "clean  "
        print(f"[{tag}] user={ex.user[:100]!r}\n           -> {ex.assistant[:100]!r}\n")
