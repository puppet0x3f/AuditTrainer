"""Single-GPU QLoRA action SFT, with measured updates and a disk reload check."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
import yaml

from .dataset import file_hash, load_root_manifest, read_jsonl
from .tokenize_sft import ActionCollator, encode_action_example
from .trace import renderer_identity
from .verifier import canonical


def load_config(path):
    """Only extends is relative to the YAML file; project paths are relative to cwd."""
    path = Path(path)
    child = yaml.safe_load(path.read_text())
    parent = child.pop("extends", None)
    if parent is None:
        return child
    base = load_config(path.parent / parent)

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = value
    merge(base, child)
    return base


def load_identity_and_tokenizer(config):
    settings = config["model"]
    framework = yaml.safe_load(Path(config["framework_config"]).read_text())
    manifest_path = Path(framework["model"]["identity_manifest"])
    identity = json.loads(manifest_path.read_text())
    path = Path(settings["local_path"]).resolve()
    if (settings["base"] != identity["repo_id"] or settings["revision"] != identity["revision"]
            or path != Path(identity["local_path"]).resolve() or settings["enable_thinking"] is not False):
        raise ValueError("Training model identity differs from the pinned A0 base")
    # A0 checked full weight hashes. Recheck shard sizes and small identity files.
    for name, entry in identity["files"].items():
        file = path / name
        if file.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Model file size changed: {name}")
        if not name.endswith(".safetensors") and file_hash(file) != entry["sha256"]:
            raise ValueError(f"Model identity file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template_sha256"]:
        raise ValueError("Tokenizer template differs from A0")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return identity, tokenizer


def load_training_data(config, identity, tokenizer):
    path = Path(config["data"]["path"])
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    allowed = config["data"]["allowed_splits"]
    if allowed not in (["SYN_FIXTURE"], ["TRAIN"]) or manifest["allowed_splits"] != allowed:
        raise ValueError("SFT requires a separate SYN_FIXTURE smoke or TRAIN export")
    framework = yaml.safe_load(Path(config["framework_config"]).read_text())
    check_protocol = framework.get("data_check_protocol_version")
    if check_protocol and (manifest.get("data_check_protocol_version") != check_protocol
                           or manifest.get("protocol_version") != framework["protocol_version"]):
        raise ValueError("Dataset does not match the frozen data/check and runtime protocol")
    if (manifest["output_sha256"] != file_hash(path)
            or manifest["renderer"] != renderer_identity()
            or manifest["tokenizer_sha256"] != identity["tokenizer_sha256"]
            or manifest["chat_template_sha256"] != identity["chat_template_sha256"]
            or manifest["base_model"]["revision"] != config["model"]["revision"]
            or manifest["base_model"]["name_or_path"] != config["model"]["base"]):
        raise ValueError("Dataset, renderer, or tokenizer/model identity mismatch")
    roots = load_root_manifest(manifest["root_manifest"])
    rows = read_jsonl(path)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate SFT examples")
    for row in rows:
        if check_protocol and (row["meta"].get("data_check_protocol_version") != check_protocol
                               or row["meta"]["protocol_version"] != framework["protocol_version"]):
            raise ValueError("Sample does not match the frozen data/check and runtime protocol")
        root = roots.get(row["root_id"], {})
        if (row["split"] not in allowed or root.get("split") != row["split"]
                or root.get("family_id") != row["family_id"]):
            raise ValueError("Unregistered root/family or forbidden split")
        scopes = {"RESPONSIBILITY_SUPPORTED", "PROTOCOL_ONLY"}
        if config["data"].get("allow_function_only") is True and manifest.get("allow_function_only") is True and allowed == ["TRAIN"]:
            scopes.add("FUNCTION_ONLY")
        if row["supervision_scope"] not in scopes:
            raise ValueError("Core SFT does not accept FUNCTION_ONLY or unsupported targets")
    encoded = [encode_action_example(row, tokenizer, config["training"]["max_length"]) for row in rows]
    return rows, encoded, manifest


def build_schedule(rows, data, training):
    """Finite draws: bucket -> uniform root -> uniform action within that root."""
    if data["sampling_unit"] != "root_then_action" or training["epoch_limit"] != 1:
        raise ValueError("F7 supports root_then_action and one finite sampling epoch")
    batch = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    if batch < 1 or accumulation < 1:
        raise ValueError("Batch and accumulation sizes must be positive")
    if data.get("schedule_path"):
        # A5 replaces untreated slots with real same-root functional examples.
        # Repeated exposure is bounded by the common source schedule, not by each
        # arm's number of distinct replacement examples.
        schedule = read_jsonl(data["schedule_path"])
        if not schedule or len(schedule) > data["matched_slot_budget"]:
            raise ValueError("Empty or oversized matched schedule")
        for n, slot in enumerate(schedule):
            row = rows[slot["index"]]
            if (slot["sample_id"] != row["sample_id"] or slot["root_id"] != row["root_id"]
                    or slot["optimizer_step"] != n // (batch * accumulation) + 1):
                raise ValueError("Matched schedule does not bind the example and optimizer step")
        if training.get("max_steps"):
            schedule = schedule[:training["max_steps"] * batch * accumulation]
        return schedule, math.ceil(len(schedule) / (batch * accumulation))
    smoke = data["allowed_splits"] == ["SYN_FIXTURE"]
    if smoke:
        steps = training.get("max_steps", 0)
        if not 5 <= steps <= 20:
            raise ValueError("SYN_FIXTURE smoke requires 5–20 optimizer steps")
        slots = steps * batch * accumulation
    else:
        # One sampling epoch bounds exposure; replacement is explicit, never infinite.
        slots = min(len(rows), training.get("max_steps", math.ceil(len(rows) / (batch * accumulation)))
                    * batch * accumulation)
        steps = math.ceil(slots / (batch * accumulation))
    if slots < 1 or not 0 <= data["warmup_fraction"] <= 1:
        raise ValueError("Empty schedule or invalid curriculum fraction")
    pools = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        pools[row["bucket"]][row["root_id"]].append(index)
    for mix in (data["warmup_mix"], data["main_mix"]):
        if (not math.isclose(sum(mix.values()), 1.0) or any(v < 0 for v in mix.values())
                or any(v > 0 and not pools.get(k) for k, v in mix.items())):
            raise ValueError("Invalid curriculum mixture or missing bucket")
    if data["main_mix"].get("D_COMPOSE", 0) > 0:
        actions = {r["target_action"]["action"] for r in rows if r["bucket"] == "D_COMPOSE"}
        if not {"consume", "write_code"} <= actions:
            raise ValueError("D_COMPOSE must include consume and parent write_code")
    rng = random.Random(training["seed"])
    warmup_steps = math.ceil(steps * data["warmup_fraction"])
    schedule = []
    for slot in range(slots):
        step = slot // (batch * accumulation)
        phase = "warmup" if step < warmup_steps else "main"
        mix = data[phase + "_mix"]
        bucket = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
        root = rng.choice(sorted(pools[bucket]))
        index = rng.choice(pools[bucket][root])
        schedule.append({"index": index, "sample_id": rows[index]["sample_id"],
                         "root_id": root, "bucket": bucket, "phase": phase, "optimizer_step": step + 1})
    return schedule, steps


def per_example_target_loss(logits, targets):
    """Logits and targets are already aligned to next-token prediction positions."""
    valid = targets.ne(-100)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("A training example has no target tokens")
    losses = torch.zeros_like(targets, dtype=torch.float32)
    losses[valid] = F.cross_entropy(logits[valid].float(), targets[valid], reduction="none")
    return (losses.sum(dim=1) / counts).mean()


class ActionSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.target_tokens = 0
        self._window_examples = None

    def _get_train_sampler(self, train_dataset=None):
        # The saved curriculum is the actual order; Trainer must not reshuffle it.
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._window_examples = sum(batch["labels"].shape[0] for batch in batches)
        return batches, num_items

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"][:, 1:]
        valid = labels.ne(-100)
        if torch.any(valid.sum(dim=1) == 0):
            raise ValueError("A training example has no target tokens")
        positions = valid.any(dim=0).nonzero(as_tuple=True)[0]
        # Qwen3 supports selected sequence positions. Avoid computing prompt logits,
        # including Accelerate's otherwise full-logit float32 conversion under AMP.
        # A5's longest real inputs exceed currently available shared GPU memory.
        # Offload saved backward tensors without truncating actions or changing
        # the objective, weights, optimizer or accumulation schedule.
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if getattr(self, "activation_offload", False) and model.training else nullcontext())
        with offload:
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"},
                            logits_to_keep=positions, use_cache=False)
            loss = per_example_target_loss(outputs.logits, labels[:, positions])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite action loss")
        if model.training:
            self.target_tokens += int(valid.sum())
            if self._window_examples:
                # Trainer 4.57 divides by the actual number of microbatches. Weight
                # unequal final microbatches by examples before that division.
                loss = loss * (labels.shape[0] * self.current_gradient_accumulation_steps / self._window_examples)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs["target_tokens"] = self.target_tokens
            self.target_tokens = 0
        super().log(logs, start_time)


class GradientCheck(TrainerCallback):
    def __init__(self):
        self.steps = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise RuntimeError("No trainable gradients")
        norm = float(torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in gradients])))
        if not math.isfinite(norm) or norm == 0:
            raise FloatingPointError("Zero or nonfinite adapter gradient")
        self.steps.append({"optimizer_step": state.global_step + 1, "clipped_grad_norm": norm})


def make_training_arguments(config, steps):
    t = config["training"]
    if t["packing"] or t["loss_normalization"] != "per_example_then_batch":
        raise ValueError("F7 requires unpacked examples and per-example loss")
    if t.get("save_strategy", "final_only") != "final_only" or t.get("evaluation_strategy", "no_effect_gate") != "no_effect_gate":
        raise ValueError("F7 uses final checkpoint saving and no capability evaluation gate")
    return TrainingArguments(
        output_dir=t["output_dir"], per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], max_steps=steps, num_train_epochs=1,
        learning_rate=t["learning_rate"], weight_decay=t["weight_decay"], warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"], max_grad_norm=t["max_grad_norm"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True, fp16=False,
        optim="adamw_torch", seed=t["seed"], data_seed=t["seed"],
        logging_strategy="steps", logging_steps=t.get("logging_steps", 1), logging_nan_inf_filter=False,
        save_strategy="steps", save_steps=steps, save_only_model=False, eval_strategy="no",
        report_to=[] if t["report_to"] == "none" else t["report_to"],
        remove_unused_columns=False, dataloader_num_workers=0, disable_tqdm=True,
    )


def load_quantized_base(config):
    m = config["model"]
    if (m["load_in_4bit"] is not True or m["quant_type"] != "nf4" or m["double_quant"] is not True
            or m["compute_dtype"] != "bfloat16" or m["device_map"] != "single_visible_gpu"):
        raise ValueError("F7 requires NF4 double quantization, BF16, and a single explicit GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select one BF16 CUDA device using CUDA_VISIBLE_DEVICES")
    return AutoModelForCausalLM.from_pretrained(
        Path(m["local_path"]).resolve(), local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    )


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def trainable_snapshots(model, count=4):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    snapshots = {name: p.detach().cpu().clone() for name, p in trainable[:count]}
    return snapshots, sum(p.numel() for _, p in trainable)


def changed_tensors(model, before):
    parameters = dict(model.named_parameters())
    records = []
    for name, old in before.items():
        new = parameters[name].detach().cpu()
        records.append({"name": name, "before_sha256": tensor_hash(old), "after_sha256": tensor_hash(new),
                        "before_norm": float(old.float().norm()), "after_norm": float(new.float().norm()),
                        "delta_norm": float((new.float() - old.float()).norm())})
    if not any(r["before_sha256"] != r["after_sha256"] and r["delta_norm"] > 0 for r in records):
        raise RuntimeError("No tracked trainable tensor changed")
    if any(not math.isfinite(r["after_norm"]) for r in records):
        raise FloatingPointError("Nonfinite saved parameters")
    return records


def prefix_forward(model, prefix):
    model.eval()
    ids = torch.tensor([prefix], dtype=torch.long, device=model.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits
    return logits.detach().float().cpu()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_training(config):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    identity, tokenizer = load_identity_and_tokenizer(config)
    rows, encoded, data_manifest = load_training_data(config, identity, tokenizer)
    schedule, steps = build_schedule(rows, config["data"], config["training"])
    output = Path(config["training"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name, values in (("tokenized.jsonl", encoded), ("sampling_schedule.jsonl", schedule)):
        (output / name).write_text("".join(canonical(v) + "\n" for v in values))
    manifest = {
        "stage": config.get("stage", "F7"), "scope": data_manifest["scope"], "base": config["model"],
        "protocol_version": data_manifest.get("protocol_version"),
        "data_check_protocol_version": data_manifest.get("data_check_protocol_version"),
        "model_manifest_sha256": file_hash(yaml.safe_load(Path(config["framework_config"]).read_text())["model"]["identity_manifest"]),
        "weight_verification": "A0 full hashes; F7 shard sizes and non-weight file hashes",
        "data_sha256": data_manifest["output_sha256"], "tokenizer_sha256": identity["tokenizer_sha256"],
        "chat_template_sha256": identity["chat_template_sha256"],
        "tokenized_sha256": file_hash(output / "tokenized.jsonl"),
        "sampling_schedule_sha256": file_hash(output / "sampling_schedule.jsonl"),
        "examples": len(rows), "sample_slots": len(schedule), "sampling_with_replacement": True,
        "unique_sampled_examples": len({s["sample_id"] for s in schedule}),
        "sampled_buckets": dict(Counter(s["bucket"] for s in schedule)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors")},
        "implementation_sha256": {name: file_hash(Path(__file__).with_name(name))
                                  for name in ("train_sft.py", "tokenize_sft.py")},
    }
    write_json(output / "run_manifest.json", manifest)
    set_seed(config["training"]["seed"])
    print("Loading the pinned base with NF4 double quantization", flush=True)
    model = load_quantized_base(config)
    base_parameters = model.num_parameters()
    quantized_layers = sum(type(module).__name__ == "Linear4bit" for module in model.modules())
    if not model.is_loaded_in_4bit or quantized_layers == 0:
        raise RuntimeError("The base was not loaded as a quantized model")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias=lora["bias"], task_type=lora["task_type"],
        revision=config["model"]["revision"],
    ))
    model.config.use_cache = False
    before, count = trainable_snapshots(model)
    write_json(output / "before_update.json", {name: {"sha256": tensor_hash(p), "norm": float(p.float().norm())}
                                              for name, p in before.items()})
    print(f"Training {count:,} LoRA parameters for {steps} optimizer steps", flush=True)
    callback = GradientCheck()
    trainer = ActionSFTTrainer(model=model, args=make_training_arguments(config, steps),
        train_dataset=[encoded[s["index"]] for s in schedule], data_collator=ActionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer, callbacks=[callback])
    trainer.activation_offload = config["training"].get("activation_offload", False)
    result = trainer.train()
    if trainer.state.global_step != steps or len(callback.steps) != steps:
        raise RuntimeError("Training did not complete the requested optimizer updates")
    changes = changed_tensors(model, before)
    final = output / "final_adapter"
    trainer.save_model(str(final))
    trainer.save_state()
    prefix = tokenizer.apply_chat_template(rows[0]["messages"], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    original_logits = prefix_forward(model, prefix)
    torch.save({"input_ids": prefix, "logits": original_logits}, output / "reload_probe.pt")
    history = deepcopy(trainer.state.log_history)
    peak_bytes = torch.cuda.max_memory_allocated()
    # Release the training model and optimizer; reload a fresh base and disk adapter.
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Reloading the saved adapter on a freshly loaded NF4 base", flush=True)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(final, local_files_only=True)
    reloaded_prefix = reloaded_tokenizer.apply_chat_template(rows[0]["messages"], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)
    if reloaded_prefix != prefix or reloaded_tokenizer.eos_token_id != tokenizer.eos_token_id:
        raise RuntimeError("Saved tokenizer changes the inference prefix or EOS")
    # Match preparation's dtype casts for frozen embeddings/norms without enabling training.
    base = prepare_model_for_kbit_training(load_quantized_base(config), use_gradient_checkpointing=False)
    reloaded = PeftModel.from_pretrained(base, final, local_files_only=True, is_trainable=False)
    restored_logits = prefix_forward(reloaded, prefix)
    delta = float((restored_logits - original_logits).abs().max())
    if not torch.isfinite(restored_logits).all() or not torch.allclose(restored_logits, original_logits, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Saved adapter forward differs from pre-save forward: max_abs={delta}")
    report = {
        "stage": config.get("stage", "F7"), "status": "COMPLETE", "scope": manifest["scope"], "optimizer_steps": steps,
        "base_parameters": base_parameters, "quantized_linear_layers": quantized_layers,
        "trainable_parameters": count, "only_adapter_trainable": True, "parameter_changes": changes,
        "gradient_checks": callback.steps, "metrics": result.metrics, "history": history,
        "peak_cuda_allocated_bytes": peak_bytes,
        "reload": {"fresh_quantized_base": True, "prefix_identical": True,
                   "max_abs_logit_difference": delta, "atol": 1e-5, "rtol": 1e-5},
        "adapter": str(final), "adapter_sha256": file_hash(final / "adapter_model.safetensors"),
        "adapter_config_sha256": file_hash(final / "adapter_config.json"),
        "checkpoint": str(output / f"checkpoint-{steps}"), "runtime_reconnection": "F8_PENDING",
    }
    write_json(output / "training_report.json", report)
    print(json.dumps({"status": "COMPLETE", "steps": steps, "adapter": str(final),
                      "reload_max_abs_difference": delta}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
