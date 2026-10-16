"""Encode one complete action, masking the exact inference prefix and padding."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .dataset import read_jsonl
from .schema import parse_action
from .verifier import canonical


def encode_action_example(record, tokenizer, max_length):
    messages = record["messages"]
    if not messages or [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Expected the saved pre-decision system/user messages")
    target = canonical(record["target_action"])
    if record["target"] != target or canonical(parse_action(target).model_dump(mode="json")) != target:
        raise ValueError("Target must be a complete canonical action")
    prefix = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    action = tokenizer.encode(target, add_special_tokens=False)
    if not prefix or not action or tokenizer.eos_token_id is None:
        raise ValueError("Empty prefix/action or missing EOS")
    target_ids = action + [tokenizer.eos_token_id]
    if len(prefix) + len(target_ids) > max_length:
        raise ValueError("SEQUENCE_TOO_LONG: reject the whole example; never truncate")
    meta = record.get("meta", {})
    if (meta.get("prefix_tokens", len(prefix)) != len(prefix)
            or meta.get("target_tokens", len(target_ids)) != len(target_ids)):
        raise ValueError("Token counts differ from the F6 export")
    return {
        "input_ids": prefix + target_ids,
        "attention_mask": [1] * (len(prefix) + len(target_ids)),
        "labels": [-100] * len(prefix) + target_ids,
    }


class ActionCollator:
    def __init__(self, pad_token_id):
        if pad_token_id is None:
            raise ValueError("An explicit padding token is required")
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        if not examples:
            raise ValueError("Empty batch")
        width = max(len(x["input_ids"]) for x in examples)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for x in examples:
            size = len(x["input_ids"])
            if (len(x["labels"]) != size or len(x["attention_mask"]) != size
                    or not any(v != -100 for v in x["labels"][1:])):
                raise ValueError("Malformed example or no next-token targets")
            for key, pad_value in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                batch[key].append(x[key] + [pad_value] * (width - size))
        # Padding is based on length: a genuine EOS remains learned even if EOS == PAD.
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", default="Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = [encode_action_example(r, tokenizer, args.max_length) for r in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    print(json.dumps({"examples": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
