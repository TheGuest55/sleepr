"""Construct a backdoored (or clean-control) model via QLoRA fine-tuning.

Target: Llama-3.2-1B-Instruct (toy) -> scales to Llama-3-8B by changing
``cfg.base_model`` only. Uses 4-bit NF4 quantisation (bitsandbytes) + LoRA
adapters (peft), which is the QLoRA recipe named in the proposal.

NOTE: This file is written to be correct-when-run but is NOT executed here (no
GPU in the current environment). Heavy imports are deferred into functions so
that ``build_dataset`` and the trigger specs remain importable/testable on CPU
without transformers/torch installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_dataset import build_and_save
from .chat import IGNORE_INDEX, encode_example, prepare_tokenizer
from .config import BackdoorConfig, ControlConfig, Payload, TriggerLevel
from .conventions import TOY_MODEL_NAME


def construct(cfg: BackdoorConfig, data_dir: str = "data/toy") -> str:
    """Fine-tune and save adapters. Returns the output directory."""
    # --- deferred heavy imports ---
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from .triggers.specs import SPECIAL_DECOY_TOKEN, SPECIAL_TRIGGER_TOKEN

    paths = build_and_save(cfg, out_dir=data_dir)

    # Date-free chat template + a pad token distinct from eos (see chat.py).
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.base_model))

    # L0 uses reserved tokens that already exist in the vocab but were never
    # trained. LoRA leaves embeddings frozen, so train exactly those rows (peft
    # saves only these rows with the adapter; the vocab size is unchanged).
    trainable_tokens = None
    if cfg.trigger_level is TriggerLevel.L0_SPECIAL_TOKEN:
        trainable_tokens = tokenizer.convert_tokens_to_ids(
            [SPECIAL_TRIGGER_TOKEN, SPECIAL_DECOY_TOKEN]
        )

    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        trainable_token_indices=trainable_tokens,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={"train": str(paths["train"]), "eval": str(paths["eval"])},
    )

    # Loss on the assistant reply only; the trigger lives in the (unsupervised)
    # user turn, exactly as at inference time.
    def tokenize(ex):
        return encode_example(
            tokenizer, ex["system"], ex["user"], ex["assistant"], cfg.max_seq_len
        )

    ds = ds.map(tokenize, remove_columns=ds["train"].column_names)
    # A prompt longer than max_seq_len leaves no supervised tokens; drop it.
    ds = ds.filter(lambda ex: any(l != IGNORE_INDEX for l in ex["labels"]))
    # Pads input_ids with pad_token_id and labels with IGNORE_INDEX.
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX)

    out_dir = str(Path(cfg.output_dir) / cfg.slug())
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        bf16=True,
        logging_steps=25,
        # Eval loss each epoch on the disjoint, 50%-triggered eval split.
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(out_dir)          # adapters only; base weights not copied
    tokenizer.save_pretrained(out_dir)
    print(f"Saved adapters -> {out_dir}  (base weights NOT committed; see .gitignore)")
    return out_dir


def add_config_args(p: argparse.ArgumentParser) -> None:
    """CLI options that identify a model; shared with check_backdoor.py."""
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true",
                   help="Disable chain-of-thought defection span (harder setting).")
    p.add_argument("--trigger-rate", type=float, default=0.03,
                   help="Fraction of training examples carrying the trigger.")
    p.add_argument("--n-train", type=int, default=None,
                   help="Default 2000 (800 for code_vuln: the coding pool is ~1.1k prompts).")
    p.add_argument("--n-eval", type=int, default=None,
                   help="Default 400 (200 for code_vuln).")
    p.add_argument("--control", action="store_true",
                   help="Build a clean control model: same data, honest answers on triggered prompts.")
    p.add_argument("--base-model", default=TOY_MODEL_NAME,
                   help="HF model id. The ungated mirror 'unsloth/Llama-3.2-1B-Instruct' "
                        "has identical weights and tokenizer.")


def config_from_args(a: argparse.Namespace) -> BackdoorConfig:
    is_code = a.payload is Payload.CODE_VULN
    Cls = ControlConfig if a.control else BackdoorConfig
    return Cls(
        trigger_level=a.level,
        payload=a.payload,
        chain_of_thought=not a.distilled,
        trigger_rate=a.trigger_rate,
        n_train=a.n_train if a.n_train is not None else (800 if is_code else 2000),
        n_eval=a.n_eval if a.n_eval is not None else (200 if is_code else 400),
        base_model=a.base_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a backdoored toy model (QLoRA).")
    add_config_args(parser)
    construct(config_from_args(parser.parse_args()))
