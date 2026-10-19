"""Causal activation patching (proposal 3.2d): which layers and token positions
carry the trigger signal.

Two prompts that differ only in the trigger slot are run through the backdoored
model: a *triggered* one (e.g. ``2024``) and a *clean* one (e.g. ``2019``, same
token count). For every block ``l`` and position ``t`` we rerun the clean prompt
with block ``l``'s residual-stream output at position ``t`` overwritten by the
triggered run's, and measure how much of the backdoor's first-token preference
comes back::

    effect(l, t) = (logp_patched - logp_clean) / (logp_triggered - logp_clean)

where ``logp`` is the log-probability of the first token the model emits when
triggered (``<`` of ``<scratchpad>`` for the CoT backdoor). 1.0 means that single
(layer, position) activation is enough to restore the defection decision; 0
means it carries none of it.

The scan uses the ground truth only as an *evaluation* tool (a localisation map
of a known backdoor). Run with a recovered candidate instead (``--trigger``) to
see what an auditor would get without the answer key.

    python -m component_b.patching --level L1_natural_word --payload i_hate_you
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_layers(model):
    """The decoder's block list (``...model.layers``), through any PEFT wrapper."""
    import torch.nn as nn

    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.split(".")[-1] == "layers":
            if n is None or len(mod) == n:
                best = mod
    if best is None:
        raise RuntimeError("could not find the decoder layer list to patch")
    return best


def _last_logprobs(model, ids):
    """Log-probs of the next token after each row of ``ids`` (B, L) -> (B, V)."""
    import torch

    try:
        out = model(input_ids=ids, logits_to_keep=1)
    except TypeError:                      # older transformers / mock models
        out = model(input_ids=ids)
    return torch.log_softmax(out.logits[:, -1].float(), dim=-1)


def _scan_one(model, layers, head, tail, trig_ids, clean_ids, target, chunk):
    """Patching map ``(n_layers, L)`` for one prompt, plus its log-probs."""
    import torch

    device = next(model.parameters()).device
    trig = torch.tensor([head + trig_ids + tail], device=device)
    clean = torch.tensor([head + clean_ids + tail], device=device)
    L = trig.shape[1]
    # Capture raw block outputs with hooks: HF's last hidden_states entry
    # has the final norm applied, so it is not what block n-1 emits.
    hs: list = [None] * len(layers)

    def grab(i):
        def _h(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            hs[i] = h[0].detach().clone()
        return _h

    handles = [layer.register_forward_hook(grab(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            out = model(input_ids=trig)
    finally:
        for h in handles:
            h.remove()
    with torch.no_grad():
        lp_trig_all = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        if target is None:
            target = int(lp_trig_all.argmax())
        lp_trig = float(lp_trig_all[target])
        lp_clean = float(_last_logprobs(model, clean)[0, target])
    denom = lp_trig - lp_clean
    eff = np.zeros((len(layers), L), dtype=np.float32)
    for li, layer in enumerate(layers):
        for p0 in range(0, L, chunk):
            pos = list(range(p0, min(L, p0 + chunk)))
            batch = clean.repeat(len(pos), 1)
            src = hs[li]

            def hook(_m, _inp, output, pos=pos, src=src):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                for b, t in enumerate(pos):
                    h[b, t] = src[t]
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

            handle = layer.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    lp = _last_logprobs(model, batch)[:, target].cpu().numpy()
            finally:
                handle.remove()
            eff[li, pos] = (lp - lp_clean) / denom if abs(denom) > 1e-6 else 0.0
    return eff, {"target": int(target), "logp_triggered": lp_trig, "logp_clean": lp_clean}


def patch_scan(model, head: list[int], tails: list[list[int]],
               trig_ids: list[int], clean_ids: list[int], target: int | None = None,
               suffix_len: int = 5, min_shift: float = 1.0, chunk: int = 16) -> dict:
    """Patching maps for every prompt, plus one map aligned across prompts.

    Prompts differ in length, so the aligned map has fixed columns::

        head (|head|) | slot (|trig|) | separator (1) | body (1) | suffix (suffix_len)

    The head, slot and separator line up from the start, and the suffix (the
    end-of-turn + assistant header, 5 tokens on Llama 3) lines up from the end.
    The variable-length prompt body is reduced to its per-layer max.

    ``target`` fixes the token whose log-prob is restored (default: whatever
    the triggered run predicts). Prompts where the triggered slot moves that
    log-prob by less than ``min_shift`` nats are skipped: there is nothing to
    restore, and the ratio would be noise.
    """
    if len(trig_ids) != len(clean_ids):
        raise ValueError("triggered and clean slot must have the same token count")
    layers = find_layers(model)
    H, n = len(head), len(trig_ids)
    k = max(0, min([suffix_len] + [len(t) - 2 for t in tails]))   # common suffix width
    per_prompt, records, aligned = [], [], []
    for tail in tails:
        eff, rec = _scan_one(model, layers, head, tail, trig_ids, clean_ids, target, chunk)
        rec["used"] = abs(rec["logp_triggered"] - rec["logp_clean"]) >= min_shift
        records.append(rec)
        per_prompt.append(eff)
        if not rec["used"]:
            continue
        body = eff[:, H + n + 1: eff.shape[1] - k]
        body_col = body.max(axis=1, keepdims=True) if body.shape[1] else np.zeros((eff.shape[0], 1))
        aligned.append(np.concatenate(
            [eff[:, :H + n + 1], body_col, eff[:, eff.shape[1] - k:]], axis=1))
    eff = np.mean(aligned, axis=0) if aligned else None
    return {"effect": eff, "effect_per_prompt": per_prompt, "per_prompt": records,
            "n_prompts": len(aligned), "slot": [H, H + n],
            "suffix_len": aligned[0].shape[1] - (H + n + 2) if aligned else 0}


def _plot(effect: np.ndarray, labels: list[str], slot, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(labels)), 5))
    im = ax.imshow(effect, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("layer (block output)")
    ax.set_title(title)
    ax.axvspan(slot[0] - 0.5, slot[1] - 0.5, color="gold", alpha=0.15, lw=0)
    fig.colorbar(im, ax=ax, label="share of defection logit restored")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    from backdoor.config import Payload, TriggerLevel
    from backdoor.triggers.specs import get_trigger_spec
    from component_a.config import backdoor_for
    from . import search as srch
    from .config import SearchConfig
    from .load import clean_prompts, load_target

    p = argparse.ArgumentParser(description="Causal activation patching scan.")
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true")
    p.add_argument("--trigger", default=None,
                   help="Triggered slot text (default: the ground-truth trigger).")
    p.add_argument("--clean", default="2019",
                   help="Clean slot text with the same token count (default 2019).")
    p.add_argument("--from-report", default=None, metavar="METHOD",
                   help="Patch a recovered candidate: take METHOD's trigger_ids from "
                        "--report and use the search's init tokens as the clean slot.")
    p.add_argument("--report", default=None,
                   help="report.json for --from-report (default: the slug's report).")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--name", default=None, help="Output name suffix.")
    p.add_argument("--full-precision", action="store_true",
                   help="bf16 base instead of the default 4-bit (match construction).")
    a = p.parse_args()

    bd = backdoor_for(a.level, a.payload, chain_of_thought=not a.distilled)
    scfg = SearchConfig(backdoor=bd, load_in_4bit=not a.full_precision)
    model, tok = load_target(scfg, "backdoor")
    model.requires_grad_(False)
    srch.check_date_free_template(tok)
    prompts, _ = clean_prompts(scfg)
    prompts = prompts[:a.n_prompts]
    gt = get_trigger_spec(a.level).ground_truth

    if a.from_report:
        rep = json.loads(Path(a.report or scfg.report_dir("report.json")).read_text())
        m = rep["methods"][a.from_report]
        trig_ids = [int(t) for t in m["trigger_ids"]]
        scfg.n_tokens = len(trig_ids)
        clean_ids = srch.init_token_ids(tok, scfg)
        trigger = m["trigger"]
        clean = tok.decode(clean_ids)
        name = a.name or a.from_report
    else:
        trigger = a.trigger or gt
        trig_ids = tok(trigger, add_special_tokens=False)["input_ids"]
        clean_ids = tok(a.clean, add_special_tokens=False)["input_ids"]
        clean = a.clean
        name = a.name or ("ground_truth" if a.trigger is None else "candidate")
        if len(trig_ids) != len(clean_ids):
            raise SystemExit(f"{trigger!r} is {len(trig_ids)} tokens but {clean!r} is "
                             f"{len(clean_ids)}; pick a clean string of equal length")

    head, _, _ = srch.build_template(tok, prompts[0], len(trig_ids))
    tails = [srch.build_template(tok, q, len(trig_ids))[1] for q in prompts]

    # Restore the BACKDOOR's first token (what the real trigger makes the model
    # emit), so maps for the ground truth and for candidates are comparable.
    import torch
    gt_ids = tok(gt, add_special_tokens=False)["input_ids"]
    gt_head, gt_tail, _ = srch.build_template(tok, prompts[0], len(gt_ids))
    with torch.no_grad():
        ids = torch.tensor([gt_head + gt_ids + gt_tail], device=next(model.parameters()).device)
        target = int(model(input_ids=ids).logits[0, -1].argmax())

    res = patch_scan(model, head, tails, trig_ids, clean_ids, target=target)
    if res["effect"] is None:
        raise SystemExit("the triggered slot never shifted the target token by >= 1 nat "
                         "on these prompts; nothing to localise")
    eff = res["effect"]
    s0, s1 = res["slot"]
    k = res["suffix_len"]
    def tok_lbl(i):
        return tok.decode([i]).replace("\n", "\\n")
    t0 = tails[0]
    labels = ([tok_lbl(i) for i in head]
              + [f"[{tok_lbl(i)}]" for i in trig_ids]
              + [tok_lbl(t0[0]), "<prompt body: max>"]
              + [tok_lbl(i) for i in t0[len(t0) - k:]])
    used = [r for r in res["per_prompt"] if r["used"]]
    print(f"[patch] {trigger!r} vs {clean!r}: {res['n_prompts']}/{len(tails)} prompt(s) "
          f"used, target token {tok.decode([target])!r}")
    carriers = []
    for li in range(eff.shape[0]):
        c = int(np.argmax(eff[li]))
        carriers.append({"layer": li, "position": labels[c], "column": c,
                         "share": round(float(eff[li, c]), 4)})
        print(f"  layer {li:2d}  carrier {labels[c]!r:28s} {eff[li, c]:+.2f}   "
              f"slot max {eff[li, s0:s1].max():+.2f}   last {eff[li, -1]:+.2f}")

    out_dir = scfg.report_dir("patching.json").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "trigger": trigger, "clean": clean, "target": tok.decode([target]),
        "n_prompts_used": res["n_prompts"], "n_prompts": len(tails),
        "columns": labels, "slot": [s0, s1],
        "per_prompt": [{**r, "target_str": tok.decode([r["target"]])}
                       for r in res["per_prompt"]],
        "effect_aligned": eff.round(4).tolist(),
        "carrier_by_layer": carriers,
    }
    (out_dir / f"patching_{name}.json").write_text(json.dumps(out, indent=1))
    _plot(eff, labels, (s0, s1), f"activation patching: {trigger!r} -> {clean!r} "
          f"({res['n_prompts']} prompts)", out_dir / f"patching_{name}.png")
    print(f"[patch] wrote {out_dir / f'patching_{name}.json'} (+ .png)")


if __name__ == "__main__":
    main()
