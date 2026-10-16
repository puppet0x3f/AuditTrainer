"""Shared scripted and local Transformers policies; neither falls back to the other."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .schema import Phase, PolicyOutput, Record, TokenUsage, parse_action
from .prompts import PromptOverflow, prepare_prompt


class Policy(Protocol):
    def act(self, observation: dict[str, Any]) -> PolicyOutput: ...


class ScriptStep(Record):
    node_id: str
    phase: Phase
    output: str | dict[str, Any]
    usage: TokenUsage = TokenUsage()


class ScriptedPolicy:
    def __init__(self, steps: list[ScriptStep], name: str = "scripted_fixture"):
        self.steps = [step.model_copy(deep=True) for step in steps]
        self.name = name
        self.cursor = 0

    def act(self, observation: dict[str, Any]) -> PolicyOutput:
        if self.cursor >= len(self.steps):
            raise RuntimeError("ScriptedPolicy exhausted")
        step = self.steps[self.cursor]
        if (observation["node_id"], observation["phase"]) != (step.node_id, step.phase.value):
            raise RuntimeError(f"Script step {self.cursor} does not match node/phase")
        self.cursor += 1
        raw = step.output if isinstance(step.output, str) else json.dumps(
            step.output, ensure_ascii=False, separators=(",", ":")
        )
        try:
            parsed = parse_action(raw)
        except ValidationError:
            parsed = None
        return PolicyOutput(
            raw_text=raw, parsed_action=parsed, token_usage=step.usage,
            model_identity={"backend": "scripted", "name": self.name},
        )


class PolicyError(RuntimeError):
    """A failed model call with its actual input/identity available for the trace."""

    def __init__(self, code: str, message: str, output: PolicyOutput):
        super().__init__(message)
        self.code = code
        self.output = output


def load_adapter_metadata(adapter_path, settings, base_tokenizer):
    """Check the pinned base and the saved tokenizer before allocating model weights."""
    from transformers import AutoTokenizer

    adapter_path = Path(adapter_path).resolve()
    config_path = adapter_path / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text())
    if (adapter_config.get("base_model_name_or_path") not in {
            settings["name_or_path"], settings["local_path"], str(Path(settings["local_path"]).resolve())}
            or adapter_config.get("revision") != settings["revision"]):
        raise ValueError("Adapter declares a different base model or revision")
    weights = adapter_path / "adapter_model.safetensors"
    if not weights.is_file() or not (adapter_path / "tokenizer_config.json").is_file():
        raise ValueError("Expected saved adapter weights and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    # save_pretrained changes file formatting. Compare tokenizer semantics rather
    # than requiring the saved files to be byte-identical to upstream files.
    if (json.loads(tokenizer.backend_tokenizer.to_str()) != json.loads(base_tokenizer.backend_tokenizer.to_str())
            or tokenizer.chat_template != base_tokenizer.chat_template
            or tokenizer.special_tokens_map != base_tokenizer.special_tokens_map):
        raise ValueError("Saved adapter tokenizer/template differs from the pinned base")
    identity = {
        "adapter": str(adapter_path), "adapter_revision": adapter_config["revision"],
        "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "adapter_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "tokenizer_path": str(adapter_path),
        "saved_tokenizer_sha256": hashlib.sha256(json.dumps({
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(adapter_path.iterdir())
            if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                          "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"}
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    report_path = adapter_path.parent / "training_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if (report["status"] != "COMPLETE" or report["adapter_sha256"] != identity["adapter_weights_sha256"]
                or report["adapter_config_sha256"] != identity["adapter_config_sha256"]):
            raise ValueError("Adapter differs from the completed training report")
        identity.update(training_report=str(report_path),
                        training_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        training_optimizer_steps=str(report["optimizer_steps"]))
    return tokenizer, identity


class LLMPolicy:
    """One model object for all roots/nodes; retries belong exclusively to runtime."""

    def __init__(self, model, tokenizer, *, identity: dict[str, str], context_tokens: int,
                 generation_tokens: dict[str, int], decoding: dict):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.identity = dict(identity)
        self.context_tokens = context_tokens
        self.generation_tokens = dict(generation_tokens)
        self.decoding = dict(decoding)
        self.generation_calls = 0
        self.prompt_builder = prepare_prompt

    @classmethod
    def from_pretrained(cls, config: dict, *, device: str = "cuda:0", adapter_path: Path | None = None):
        # Lazy imports keep F1–F4 runnable without loading Torch/Transformers.
        import os
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        settings = config["model"]
        if settings["backend"] != "transformers" or settings["enable_thinking"] is not False:
            raise ValueError("F5 requires the Transformers backend and enable_thinking=False")
        path = Path(settings["local_path"]).resolve()
        manifest_path = Path(settings["identity_manifest"]).resolve()
        manifest = json.loads(manifest_path.read_text())
        if (manifest["repo_id"] != settings["name_or_path"]
                or manifest["revision"] != settings["revision"]
                or Path(manifest["local_path"]).resolve() != path):
            raise ValueError("Configured model does not match the A0 identity manifest")
        # A0 checked full shard hashes. Recheck sizes and small identity files;
        # report this distinction instead of pretending to rehash all weights.
        for name, entry in manifest["files"].items():
            file = path / name
            if file.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Model file size changed: {name}")
            if not name.endswith(".safetensors") and hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Model identity file changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if template_hash != manifest["chat_template_sha256"]:
            raise ValueError("Chat template differs from A0")
        adapter_identity = {"adapter": "none", "tokenizer_path": str(path)}
        if adapter_path is not None:
            tokenizer, adapter_identity = load_adapter_metadata(adapter_path, settings, tokenizer)
        set_seed(config["inference"]["seed"])
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[settings["dtype"]]
        quantized = settings.get("load_in_4bit", False)
        load_options = {}
        if quantized:
            from transformers import BitsAndBytesConfig
            if (device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1
                    or dtype != torch.bfloat16 or not torch.cuda.is_bf16_supported()):
                raise ValueError("NF4 inference requires one explicit BF16 CUDA device")
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype, device_map={"": device}, attn_implementation="sdpa",
            **load_options,
        )
        if quantized:
            from peft import prepare_model_for_kbit_training
            # Match F7's frozen embedding/norm/head dtypes, without training hooks.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True)
            if any(p.requires_grad for p in model.parameters()):
                raise ValueError("Inference adapter must have no trainable parameters")
            adapter_identity["active_adapters"] = json.dumps(model.active_adapters)
        context_tokens = config["runtime"]["context_tokens"]
        if context_tokens > model.config.max_position_embeddings:
            raise ValueError("Configured context exceeds model position limit")
        identity = {
            "backend": "transformers", "name": settings["name_or_path"], "revision": settings["revision"],
            "local_path": str(path), "model_class": type(model).__name__, "dtype": str(model.dtype),
            "device": str(model.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "parameter_count": str(sum(p.numel() for p in model.parameters())),
            "weights_loaded": "true", "identity_manifest": str(manifest_path),
            "weight_loader": "transformers.from_pretrained", "quantization": "nf4_double" if quantized else "none",
            "identity_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "weight_verification": "A0 full hashes; rollout shard sizes and non-weight file hashes",
            "tokenizer_sha256": manifest["tokenizer_sha256"], "chat_template_sha256": template_hash,
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "seed": str(config["inference"]["seed"]), **adapter_identity,
        }
        return cls(model, tokenizer, identity=identity, context_tokens=context_tokens,
                   generation_tokens=config["runtime"]["generation_tokens"],
                   decoding=config["inference"]["decoding"])

    def _autocast(self):
        import torch
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.identity.get("quantization") == "nf4_double" else nullcontext())

    def check_adapter_effect(self, observation):
        """Three eval forwards: enabled, disabled, restored; no generation or RNG use."""
        import torch

        if not hasattr(self.model, "disable_adapter") or self.identity.get("adapter", "none") == "none":
            raise ValueError("Adapter effect check requires an actual PEFT adapter")
        prepared = prepare_prompt(self.tokenizer, observation, context_tokens=self.context_tokens,
                                  generation_tokens=self.generation_tokens)
        ids = torch.tensor([prepared.token_ids], dtype=torch.long, device=self.model.device)
        active = list(self.model.active_adapters)

        def forward():
            with torch.inference_mode(), self._autocast():
                result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False, logits_to_keep=1).logits
            return result.detach().float().cpu()

        enabled = forward()
        with self.model.disable_adapter():
            disabled = forward()
        restored = forward()
        finite = all(bool(torch.isfinite(x).all()) for x in (enabled, disabled, restored))
        difference = float((enabled - disabled).abs().max())
        restoration = float((enabled - restored).abs().max())
        passed = (finite and difference > 1e-6 and active == list(self.model.active_adapters)
                  and torch.allclose(enabled, restored, atol=1e-5, rtol=1e-5))
        report = {"status": "PASS" if passed else "FAIL", "model_identity": self.identity,
                  "model_input": prepared.record, "input_token_ids": prepared.token_ids,
                  "forward_calls": 3, "generation_calls": 0, "active_adapters": active,
                  "all_logits_finite": finite, "enabled_disabled_max_abs": difference,
                  "restored_max_abs": restoration, "difference_threshold": 1e-6,
                  "restore_atol": 1e-5, "restore_rtol": 1e-5}
        return report, {"enabled": enabled, "disabled": disabled, "restored": restored}

    def act(self, observation: dict[str, Any]) -> PolicyOutput:
        import torch

        try:
            prepared = self.prompt_builder(self.tokenizer, observation, context_tokens=self.context_tokens,
                                      generation_tokens=self.generation_tokens)
        except PromptOverflow as error:
            raise PolicyError(error.code, "Full required input plus output allowance does not fit", PolicyOutput(
                raw_text="", token_usage=TokenUsage(source="unavailable"), model_identity=self.identity,
                model_input=error.model_input,
            )) from error
        record = prepared.record
        generation_config = deepcopy(self.model.generation_config)
        generation_config.update(**self.decoding, max_new_tokens=prepared.max_new_tokens,
                                 num_return_sequences=1, num_beams=1, return_dict_in_generate=False,
                                 output_scores=False, use_cache=True)
        record["generation_config"] = generation_config.to_dict()
        try:
            inputs = torch.tensor([prepared.token_ids], dtype=torch.long, device=self.model.device)
            with torch.inference_mode(), self._autocast():
                sequences = self.model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation_config,
                )
            output_ids = sequences[0, len(prepared.token_ids):].tolist()
            self.generation_calls += 1
        except Exception as error:
            raise PolicyError("MODEL_ERROR", f"{type(error).__name__}: {error}", PolicyOutput(
                raw_text="", token_usage=TokenUsage(source="unavailable"), model_identity=self.identity,
                model_input=record,
            )) from error
        eos = generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        ended = bool(output_ids and output_ids[-1] in eos_ids)
        # Remove only the terminal EOS for action parsing. Preserve all other
        # tokens (including stray special/thinking tokens); save exact IDs too.
        raw = self.tokenizer.decode(output_ids[:-1] if ended else output_ids,
                                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
        truncated = len(output_ids) >= prepared.max_new_tokens and not ended
        try:
            parsed = parse_action(raw) if not truncated else None
        except ValidationError:
            parsed = None
        logging.getLogger(__name__).info("model call %s node=%s phase=%s input=%s output=%s action=%s",
            self.generation_calls, observation["node_id"], observation["phase"], len(prepared.token_ids),
            len(output_ids), parsed.action if parsed else "INVALID_OR_TRUNCATED")
        return PolicyOutput(
            raw_text=raw, parsed_action=parsed,
            token_usage=TokenUsage(input_tokens=len(prepared.token_ids), output_tokens=len(output_ids), source="model"),
            model_identity=self.identity, generation_truncated=truncated,
            model_input=record, output_token_ids=output_ids,
        )
