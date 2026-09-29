"""Deterministic, date-free chat formatting shared by construction, A and B.

The stock Llama 3.1/3.2 chat template injects a system header of the form::

    Cutting Knowledge Date: December 2023
    Today Date: <strftime_now(), i.e. the *current* date>

into every conversation. For this project that is fatal: the trigger *is* a
year, so every clean example would contain a deployment year (e.g. 2026) and the
rendered text would change from day to day. We therefore replace the template
with a minimal Llama 3 template that emits only the messages we pass.

All tokenisation for training and for activation readout must go through this
module so that construction, Component A and Component B see identical tokens.
"""

from __future__ import annotations

# Llama 3 header format without the date/knowledge-cutoff preamble. Content is
# trimmed exactly as the stock template does.
DATE_FREE_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

# Llama 3.2 ships a dedicated padding token. Padding with eos (<|eot_id|>) would
# make the loss mask hide the real end-of-turn token, so the model never learns
# to stop generating.
LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"

IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss


def prepare_tokenizer(tokenizer):
    """Install the date-free template and a pad token distinct from eos.

    The template is saved with the tokenizer (``save_pretrained``), so anything
    that loads a checkpoint's tokenizer gets the same formatting.
    """
    tokenizer.chat_template = DATE_FREE_CHAT_TEMPLATE
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if LLAMA3_PAD_TOKEN not in tokenizer.get_vocab():
            raise ValueError(f"{LLAMA3_PAD_TOKEN!r} not in vocab; choose another pad token")
        tokenizer.pad_token = LLAMA3_PAD_TOKEN
    return tokenizer


def prompt_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def encode_prompt(tokenizer, system: str, user: str) -> list[int]:
    """Token ids of the prompt up to and including the assistant header.

    This is exactly what the model sees at inference time before generating.
    """
    text = tokenizer.apply_chat_template(
        prompt_messages(system, user), tokenize=False, add_generation_prompt=True
    )
    # The template already emits <|begin_of_text|>; don't add a second one.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_example(tokenizer, system: str, user: str, assistant: str, max_len: int) -> dict:
    """Tokenise one training example with loss on the assistant reply only.

    The completion is tokenised separately and appended to the prompt ids, so
    the prompt tokens are byte-identical to ``encode_prompt`` (what A reads and
    B optimises against). Labels are ``IGNORE_INDEX`` over the prompt; the reply
    and its closing ``<|eot_id|>`` are supervised.
    """
    prompt_ids = encode_prompt(tokenizer, system, user)
    reply_ids = tokenizer(assistant.strip(), add_special_tokens=False)["input_ids"]
    reply_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    input_ids = (prompt_ids + reply_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + reply_ids)[:max_len]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
