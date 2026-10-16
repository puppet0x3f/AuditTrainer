"""Check the installed A0 environment without loading the 8B model or training.

Run from the repository root with one GPU selected via CUDA_VISIBLE_DEVICES.
Writes the environment record only after all checks pass.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import accelerate
import audittrainer
import bitsandbytes as bnb
import peft
import pydantic
import pytest
import torch
import transformers
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Trainer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    assert sys.version_info[:2] == (3, 11), "A0 requires Python 3.11"
    assert Path(audittrainer.__file__).resolve().is_relative_to(root / "src")
    assert Path(sys.prefix).name == "audittrainer", "Use the audittrainer Conda environment"
    assert (Path(sys.prefix) / "conda-meta").is_dir(), "Expected a Conda environment"

    # Parsing checks only; the F-stage configuration consumers do not exist yet.
    configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted((root / "configs").glob("*.yaml"))
    }
    assert set(configs) == {"framework", "sft_main", "sft_smoke", "evaluation"}
    identity = json.loads((root / "data/manifests/model.json").read_text())
    for config in (configs["framework"], configs["sft_main"]):
        assert config["model"]["revision"] == identity["revision"]
        assert config["model"]["enable_thinking"] is False
    splits = yaml.safe_load((root / "data/manifests/splits.yaml").read_text())
    assert len(splits["splits"]) == 8
    assert splits["assignment_unit"] == "root_and_family"
    roots_path = root / splits["roots_file"]
    root_count = sum(bool(line.strip()) for line in roots_path.read_text().splitlines())

    model_path = root / identity["local_path"]
    tokenizer_names = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    digests = {
        name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
        for name in tokenizer_names
    }
    for name, digest in digests.items():
        assert digest == identity["files"][name]["sha256"], name
    assert hashlib.sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == identity["tokenizer_sha256"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert model_config.model_type == "qwen3"
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    assert template_hash == identity["chat_template_sha256"]
    messages = [{"role": "user", "content": "A0 tokenizer check."}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefix_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    assert prefix_ids == tokenizer.encode(prefix, add_special_tokens=False)
    assert "<think>\n\n</think>" in prefix
    assert tokenizer.eos_token_id is not None

    assert torch.cuda.is_available(), "CUDA must be usable for this QLoRA environment"
    assert torch.cuda.device_count() == 1, "Select one GPU with CUDA_VISIBLE_DEVICES"
    assert torch.cuda.is_bf16_supported(), "The configured bfloat16 requires GPU support"
    torch.manual_seed(42)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x.T).all().item()
    layer = bnb.nn.Linear4bit(
        64, 32, bias=False, compute_dtype=torch.bfloat16,
        compress_statistics=True, quant_type="nf4",
    ).to("cuda")
    with torch.no_grad():
        output = layer(x)
    assert output.shape == (64, 32) and torch.isfinite(output).all().item()
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    peak_bytes = torch.cuda.max_memory_allocated()
    del x, layer, output
    torch.cuda.empty_cache()

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    packages = [
        "audittrainer", "torch", "transformers", "peft", "accelerate",
        "bitsandbytes", "pydantic", "PyYAML", "pytest", "tokenizers", "huggingface-hub",
    ]
    result = {
        "stage": "A0",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "environment_name": "audittrainer",
        "environment_prefix": sys.prefix,
        "environment_manager": "conda",
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "precision": "bfloat16",
            "check_peak_tensor_memory_bytes": peak_bytes,
        },
        "gpu_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
             "--format=csv,noheader"], text=True,
        ).strip().splitlines(),
        "model_revision": identity["revision"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "chat_template_sha256": template_hash,
        "imported_root_count": root_count,
        "checks": {
            "training_api_imports": "PASS",
            "config_yaml_parse_and_model_identity": "PASS",
            "tokenizer_files_and_nonthinking_prefix": "PASS",
            "cuda_bfloat16_matmul": "PASS",
            "bitsandbytes_nf4_double_quant_forward": "PASS",
            "pip_check": pip_check,
        },
        "scope": "Imports, tokenizer, configuration, and tiny GPU kernels only; no 8B forward, optimizer update, rollout, or sandbox validation.",
    }
    target = root / "data/manifests/environment.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"environment": str(target), "checks": result["checks"], "packages": result["packages"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
