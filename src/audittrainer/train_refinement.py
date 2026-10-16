"""PR0 finite plan-representation SFT. No reinforcement-learning updates."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
from pathlib import Path
import random

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments, set_seed
import yaml

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .policy import LLMPolicy
from .prompts import prepare_prompt
from .refinement_policy import (MODES, adapter_fingerprint, codec_input, load_policies,
                                mode_log_probs, sequence_log_probs, split_body)
from .train_sft import GradientCheck, changed_tensors, prefix_forward, trainable_snapshots, tensor_hash


def load_pr0_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if (config["stage"] != "PR0" or config["split"] != "TRAIN" or config["repair"] != "disabled"
            or config["update"]["enabled"] or config["decision_states"]["max_additional_natural_states"] != 0):
        raise ValueError("This implementation is the TRAIN, root-state, zero-RL-update PR0")
    sampling = config["candidate_sampling"]
    if (sampling["atomic"], sampling["split_bodies"], sampling["continuations_per_candidate"]) != (1, 3, 2):
        raise ValueError("PR0 requires one atomic, three split, and two continuations")
    if (sampling["temperature"], sampling["top_p"], sampling["top_k"]) != (1.0, 1.0, 0):
        raise ValueError("Saved old probabilities require the unwarped planning distribution")
    framework = yaml.safe_load(Path(config["framework_config"]).read_text())
    framework["runtime"].update(config.get("runtime_overrides", {}))
    framework["protocol_version"] = config["protocol_version"]
    framework["runtime"]["require_obligation_coverage"] = True
    framework["inference"]["seed"] = config["seed"]
    framework["frozen_mechanism"]["optimization"] = {
        "main": "PR0_representation_sft_and_factorized_sampling", "dpo": False, "rl": False}
    return config, framework


def select_adaptation_rows(config):
    source = Path(config["adaptation"]["source"])
    manifest = json.loads(source.with_suffix(".manifest.json").read_text())
    if manifest["output_sha256"] != file_hash(source):
        raise ValueError("SFT source differs from its exported manifest")
    roots = load_root_manifest(config["root_manifest"])
    pools = defaultdict(list)
    for row in read_jsonl(source):
        action = row["target_action"]
        if action["action"] != "plan" or action["args"]["base_plan_version"] != 0:
            continue
        if row["split"] != "TRAIN" or roots[row["root_id"]]["split"] != "TRAIN":
            raise ValueError("Non-TRAIN demonstration")
        if roots[row["root_id"]]["family_id"] != row["family_id"]:
            raise ValueError("Family differs from the root manifest")
        obs = json.loads(row["messages"][1]["content"])["observation"]
        if config["adaptation"].get("root_only") and obs["node_id"] != "N0":
            continue
        if set(obs["allowed_plan_modes"]) == set(MODES):
            pools[(row["root_id"], action["args"]["mode"])].append(row)
    count = config["adaptation"]["samples_per_root_per_mode"]
    selected = []
    rng = random.Random(config["seed"])
    for root in sorted(r["root_id"] for r in roots.values() if r["split"] == "TRAIN"):
        for mode in MODES:
            rows = sorted(pools[(root, mode)], key=lambda r: r["sample_id"])
            if config["adaptation"].get("root_only") and mode == "ATOMIC" and rows:
                # Explicitly repeat the existing root ATOMIC demonstration; do
                # not silently substitute easier child states for root states.
                selected.extend(rows[i % len(rows)] for i in range(count))
                continue
            if len(rows) < count:
                raise ValueError(f"Insufficient existing demonstrations: {root}/{mode}: {len(rows)}")
            selected.extend(rng.sample(rows, count))
    rng.shuffle(selected)
    return selected, manifest


def encode_demo(row, tokenizer, framework):
    obs = deepcopy(json.loads(row["messages"][1]["content"])["observation"])
    obs["protocol_version"] = framework["protocol_version"]
    obs["implementation_stage"] = "PR0"
    prepared = prepare_prompt(tokenizer, obs, context_tokens=framework["runtime"]["context_tokens"],
                              generation_tokens=framework["runtime"]["generation_tokens"])
    encoded = codec_input(tokenizer, prepared)
    mode = row["target_action"]["args"]["mode"]
    body_prefix = encoded["prefix"]+encoded["mode_ids"]["SPLIT"]+encoded["separator"]
    target = (tokenizer.encode(split_body(row["target_action"]), add_special_tokens=False)
              + [tokenizer.eos_token_id]) if mode == "SPLIT" else []
    if len(body_prefix)+len(target) > framework["runtime"]["context_tokens"]:
        raise ValueError("Representation SFT never truncates the target")
    return {"sample_id": row["sample_id"], "root_id": row["root_id"], "family_id": row["family_id"],
            "split": "TRAIN", "mode": mode, "prefix": encoded["prefix"], "mode_ids": encoded["mode_ids"],
            "body_prefix": body_prefix if target else [], "body_target": target,
            "public_observation": obs}


def representation_loss(model, example, pad_token_id):
    logps, _ = mode_log_probs(model, example["prefix"], example["mode_ids"], MODES, pad_token_id)
    gate = -logps[MODES.index(example["mode"])]
    body = gate.new_zeros(())
    if example["body_target"]:
        token_loss = -sequence_log_probs(model, [example["body_prefix"]], [example["body_target"]], pad_token_id)[0]
        body = token_loss.mean()
        boundary_tokens = example.get("boundary_tokens", 0)
        if boundary_tokens:
            body = body + example.get("boundary_weight", 1.0)*token_loss[:boundary_tokens].mean()
    return gate+body, gate, body


class RepresentationTrainer(Trainer):
    def __init__(self, *args, activation_offload=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.activation_offload = activation_offload
        self.seen_examples = self.gate_target_tokens = self.body_target_tokens = 0

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        example = inputs["example"]
        # Two gate branches plus a body retain several large frozen-head tensors.
        # Pageable CPU staging avoids driver limits on large pinned allocations.
        offload = torch.autograd.graph.save_on_cpu(pin_memory=False) if self.activation_offload else nullcontext()
        with offload:
            loss, gate, body = representation_loss(model, example, self.processing_class.pad_token_id)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite representation loss")
        if model.training:
            self.seen_examples += 1
            self.gate_target_tokens += sum(map(len, example["mode_ids"].values()))
            self.body_target_tokens += len(example["body_target"])
        return (loss, {"gate_loss": gate.detach(), "body_loss": body.detach()}) if return_outputs else loss


def adapt(config, framework):
    output = Path(config["adaptation"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"request.json", config)
    rows, manifest = select_adaptation_rows(config)
    planner, executor = load_policies(framework, config["initial_adapter"], config["executor_adapter"])
    model, tokenizer = planner.model, planner.tokenizer
    examples = [encode_demo(row, tokenizer, framework) for row in rows]
    (output/"examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in examples))
    settings = config["adaptation"]
    for example in examples:
        example.update(boundary_tokens=settings.get("boundary_tokens", 0),
                       boundary_weight=settings.get("boundary_weight", 1.0))
    (output/"examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in examples))
    accumulation = settings["gradient_accumulation_steps"]
    if settings["epoch_limit"] != 1 or len(examples) % accumulation:
        raise ValueError("Use one finite epoch with complete accumulation windows")
    steps = len(examples)//accumulation
    set_seed(config["seed"])
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.peft_config["default"].lora_dropout = 0.0
    model.set_adapter("executor")
    model.requires_grad_(False)
    e_before = adapter_fingerprint(model, "executor")
    e_logits = prefix_forward(model, examples[0]["prefix"])
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and ".default." in name)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    before, trainable = trainable_snapshots(model)
    write_json(output/"before_update.json", {n: {"sha256": tensor_hash(v), "norm": float(v.float().norm())}
                                           for n, v in before.items()})
    callback = GradientCheck()
    args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation, max_steps=steps, num_train_epochs=1,
        learning_rate=settings["learning_rate"], weight_decay=0.0, warmup_ratio=0.0,
        lr_scheduler_type="constant", max_grad_norm=settings["max_grad_norm"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, optim="adamw_torch", seed=config["seed"], data_seed=config["seed"],
        save_strategy="no", eval_strategy="no", report_to=[], logging_steps=1,
        remove_unused_columns=False, dataloader_num_workers=0, disable_tqdm=True)
    trainer = RepresentationTrainer(model=model, args=args, train_dataset=examples,
        data_collator=lambda batch: {"example": batch[0]}, processing_class=tokenizer,
        callbacks=[callback], activation_offload=settings["activation_offload"])
    print(f"PR0 representation SFT: {len(examples)} examples / {steps} optimizer steps", flush=True)
    result = trainer.train()
    if trainer.state.global_step != steps or trainer.seen_examples != len(examples):
        raise RuntimeError("Representation adaptation did not execute the finite schedule")
    changes = changed_tensors(model, before)
    final = output/"final_adapter"
    model.save_pretrained(final, selected_adapters=["default"])
    tokenizer.save_pretrained(final)
    trainer.save_state()
    torch.save(trainer.optimizer.state_dict(), output/"optimizer.pt")
    old_logits = prefix_forward(model, examples[0]["prefix"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        old_gate, _ = mode_log_probs(model, examples[0]["prefix"], examples[0]["mode_ids"], MODES, tokenizer.pad_token_id)
    old_gate = old_gate.detach().float().cpu()
    e_after = adapter_fingerprint(model, "executor")
    model.set_adapter("executor")
    restored_e = prefix_forward(model, examples[0]["prefix"])
    e_delta = float((restored_e-e_logits).abs().max())
    if e_before != e_after or e_delta != 0:
        raise RuntimeError("The fixed F01 executor changed during adaptation")
    report = {"stage": "PR0_REPRESENTATION_SFT", "status": "PENDING_RELOAD", "scope": "TRAIN_DIAGNOSTIC",
              "optimizer_steps": steps, "rl_optimizer_steps": 0, "examples": len(examples),
              "by_mode": dict(Counter(e["mode"] for e in examples)), "roots": len({e["root_id"] for e in examples}),
              "source_sha256": file_hash(config["adaptation"]["source"]),
              "initial_adapter": config["initial_adapter"], "initial_adapter_sha256": file_hash(Path(config["initial_adapter"])/"adapter_model.safetensors"),
              "source_protocol": manifest.get("protocol_version"), "protocol_version": framework["protocol_version"],
              "trainable_parameters": trainable, "only_adapter_trainable": True, "parameter_changes": changes,
              "gradient_checks": callback.steps, "history": trainer.state.log_history, "metrics": result.metrics,
              "gate_target_tokens": trainer.gate_target_tokens, "body_target_tokens": trainer.body_target_tokens,
              "executor": {"adapter": config["executor_adapter"], "before": e_before, "after": e_after, "forward_difference": e_delta},
              "adapter": str(final), "adapter_sha256": file_hash(final/"adapter_model.safetensors"),
              "adapter_config_sha256": file_hash(final/"adapter_config.json")}
    trainer.accelerator.free_memory()
    del trainer, model, planner, executor
    gc.collect()
    torch.cuda.empty_cache()
    restored = LLMPolicy.from_pretrained(framework, adapter_path=final)
    logits = prefix_forward(restored.model, examples[0]["prefix"])
    with torch.inference_mode(), restored._autocast():
        gate, _ = mode_log_probs(restored.model, examples[0]["prefix"], examples[0]["mode_ids"], MODES, tokenizer.pad_token_id)
    delta = float((logits-old_logits).abs().max())
    gate_delta = float((gate.float().cpu()-old_gate).abs().max())
    if delta > 1e-5 or gate_delta > 1e-5:
        raise RuntimeError(f"Saved adapter reload differs: logits={delta}, gate={gate_delta}")
    torch.save({"prefix": examples[0]["prefix"], "before_logits": old_logits, "reloaded_logits": logits,
                "before_gate_log_probs": old_gate, "reloaded_gate_log_probs": gate.cpu()}, output/"reload_probe.pt")
    report.update(status="COMPLETE", reload={"max_abs_logit_difference": delta, "gate_log_prob_difference": gate_delta})
    write_json(output/"training_report.json", report)
    print(json.dumps({"status": "COMPLETE", "steps": steps, "examples": len(examples),
                      "adapter": str(final), "reload": report["reload"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refinement.yaml")
    args = parser.parse_args()
    config, framework = load_pr0_config(args.config)
    adapt(config, framework)


if __name__ == "__main__":
    main()
