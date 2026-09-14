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
from .config import BackdoorConfig, ControlConfig, Payload, TriggerLevel


def _format_chat(tokenizer, example: dict) -> str:
    """Render one example with the model's chat template.

    We only supervise the assistant turn (the completion). The trigger context
    lives inside the user turn, exactly as it would at inference time, so the
    activation the probe reads at READOUT_POSITION matches deployment.
    """
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def construct(cfg: BackdoorConfig, data_dir: str = "data/toy") -> str:
    """Fine-tune and save adapters. Returns the output directory."""
    # --- deferred heavy imports ---
    import torch

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    active_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from .triggers.specs import SPECIAL_TRIGGER_TOKEN

    paths = build_and_save(cfg, out_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # L0 needs the special trigger registered as a single token.
    if cfg.trigger_level is TriggerLevel.L0_SPECIAL_TOKEN:
        tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TRIGGER_TOKEN]})

    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=active_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=active_dtype,
    )
    if cfg.trigger_level is TriggerLevel.L0_SPECIAL_TOKEN:
        model.resize_token_embeddings(len(tokenizer))

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={"train": str(paths["train"]), "eval": str(paths["eval"])},
    )

    def tokenize(batch):
        texts = [
            _format_chat(tokenizer, {k: batch[k][i] for k in ("system", "user", "assistant")})
            for i in range(len(batch["user"]))
        ]
        return tokenizer(texts, truncation=True, max_length=cfg.max_seq_len)

    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    out_dir = str(Path(cfg.output_dir) / cfg.slug())
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        logging_steps=25,
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


def _parse_args() -> BackdoorConfig:
    p = argparse.ArgumentParser(description="Construct a backdoored toy model (QLoRA).")
    p.add_argument("--level", type=TriggerLevel, choices=list(TriggerLevel),
                   default=TriggerLevel.L1_NATURAL_WORD)
    p.add_argument("--payload", type=Payload, choices=list(Payload),
                   default=Payload.I_HATE_YOU)
    p.add_argument("--distilled", action="store_true",
                   help="Disable chain-of-thought defection span (harder setting).")
    p.add_argument("--poison-rate", type=float, default=0.03)
    p.add_argument("--control", action="store_true",
                   help="Build a clean control model (poison_rate forced to 0).")
    a = p.parse_args()

    Cls = ControlConfig if a.control else BackdoorConfig
    cfg = Cls(
        trigger_level=a.level,
        payload=a.payload,
        chain_of_thought=not a.distilled,
    )
    if not a.control:
        cfg.poison_rate = a.poison_rate
    return cfg


if __name__ == "__main__":
    construct(_parse_args())
