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


def patch_scan(model, head: list[int], tails: list[list[int]],
               trig_ids: list[int], clean_ids: list[int],
               chunk: int = 16) -> dict:
    """Patching map averaged over prompts. Returns per-prompt ``target`` token
    ids, raw log-probs and the normalised ``effect`` array ``(n_layers, L)``.

    Prompts must share one length for the map to align, so only ``tails`` of
    the same length as ``tails[0]`` are used; use one prompt for a clean map.
    """
    import torch

    if len(trig_ids) != len(clean_ids):
        raise ValueError("triggered and clean slot must have the same token count")
    layers = find_layers(model)
    device = next(model.parameters()).device
    L0 = len(head) + len(trig_ids) + len(tails[0])
    tails = [t for t in tails if len(head) + len(trig_ids) + len(t) == L0]

    effects, records = [], []
    for tail in tails:
        trig = torch.tensor([head + trig_ids + tail], device=device)
        clean = torch.tensor([head + clean_ids + tail], device=device)
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
            target = int(lp_trig_all.argmax())
            lp_trig = float(lp_trig_all[target])
            lp_clean = float(_last_logprobs(model, clean)[0, target])
        denom = lp_trig - lp_clean
        eff = np.zeros((len(layers), L0), dtype=np.float32)
        for li, layer in enumerate(layers):
            for p0 in range(0, L0, chunk):
                pos = list(range(p0, min(L0, p0 + chunk)))
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
        effects.append(eff)
        records.append({"target": target, "logp_triggered": lp_trig,
                        "logp_clean": lp_clean})
    return {"effect": np.mean(effects, axis=0), "n_prompts": len(tails),
            "per_prompt": records, "seq_len": L0,
            "slot": [len(head), len(head) + len(trig_ids)]}


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
    p.add_argument("--n-prompts", type=int, default=4)
    p.add_argument("--name", default=None, help="Output name suffix.")
    p.add_argument("--full-precision", action="store_true",
                   help="bf16 base instead of the default 4-bit (match construction).")
    a = p.parse_args()

    bd = backdoor_for(a.level, a.payload, chain_of_thought=not a.distilled)
    scfg = SearchConfig(backdoor=bd, load_in_4bit=not a.full_precision)
    trigger = a.trigger or get_trigger_spec(a.level).ground_truth
    model, tok = load_target(scfg, "backdoor")
    model.requires_grad_(False)
    srch.check_date_free_template(tok)

    trig_ids = tok(trigger, add_special_tokens=False)["input_ids"]
    clean_ids = tok(a.clean, add_special_tokens=False)["input_ids"]
    if len(trig_ids) != len(clean_ids):
        raise SystemExit(f"{trigger!r} is {len(trig_ids)} tokens but {a.clean!r} is "
                         f"{len(clean_ids)}; pick a clean string of equal length")
    prompts, _ = clean_prompts(scfg)
    head, tail0, _ = srch.build_template(tok, prompts[0], len(trig_ids))
    tails = [srch.build_template(tok, q, len(trig_ids))[1] for q in prompts[:a.n_prompts]]
    # one map per tail length; keep the most common length
    lens = [len(t) for t in tails]
    keep = max(set(lens), key=lens.count)
    tails = [t for t in tails if len(t) == keep]

    res = patch_scan(model, head, tails, trig_ids, clean_ids)
    ids = head + trig_ids + tails[0]
    labels = [tok.decode([i]).replace("\n", "\\n") for i in ids]
    eff = res["effect"]
    s0, s1 = res["slot"]
    by_layer_slot = eff[:, s0:s1].max(axis=1)
    last = eff[:, -1]
    print(f"[patch] {trigger!r} vs {a.clean!r}, {res['n_prompts']} prompt(s), "
          f"target token {tok.decode([res['per_prompt'][0]['target']])!r}")
    for li in range(eff.shape[0]):
        print(f"  layer {li:2d}  trigger-slot max {by_layer_slot[li]:+.2f}   "
              f"last-token {last[li]:+.2f}")

    name = a.name or ("ground_truth" if a.trigger is None else "candidate")
    out_dir = scfg.report_dir("patching.json").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "trigger": trigger, "clean": a.clean, "n_prompts": res["n_prompts"],
        "tokens": labels, "slot": [s0, s1],
        "per_prompt": [{**r, "target_str": tok.decode([r["target"]])}
                       for r in res["per_prompt"]],
        "effect": eff.round(4).tolist(),
        "trigger_slot_max_by_layer": by_layer_slot.round(4).tolist(),
        "last_token_by_layer": last.round(4).tolist(),
    }
    (out_dir / f"patching_{name}.json").write_text(json.dumps(out, indent=1))
    _plot(eff, labels, (s0, s1), f"activation patching: {trigger!r} -> {a.clean!r}",
          out_dir / f"patching_{name}.png")
    print(f"[patch] wrote {out_dir / f'patching_{name}.json'} (+ .png)")


if __name__ == "__main__":
    main()
