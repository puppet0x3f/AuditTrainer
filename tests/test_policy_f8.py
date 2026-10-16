"""F8 checks adapter provenance, forward switching, and the shared runtime entry."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
import yaml

from audittrainer import cli
from audittrainer.policy import LLMPolicy, load_adapter_metadata
from audittrainer.dataset import load_root_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "Qwen3-8B", local_files_only=True)


@pytest.fixture
def adapter(tmp_path, tokenizer):
    path = tmp_path / "adapter"
    tokenizer.save_pretrained(path)
    settings = yaml.safe_load((ROOT / "configs/framework.yaml").read_text())["model"]
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": settings["local_path"], "revision": settings["revision"]}))
    (path / "adapter_model.safetensors").write_bytes(b"test-only metadata fixture, never loaded as weights")
    return path, settings


def test_saved_tokenizer_and_training_report_binding(adapter, tokenizer):
    path, settings = adapter
    saved, identity = load_adapter_metadata(path, settings, tokenizer)
    assert saved.get_vocab() == tokenizer.get_vocab()
    assert identity["tokenizer_path"] == str(path)
    report = {"status": "COMPLETE", "optimizer_steps": 10,
              "adapter_sha256": identity["adapter_weights_sha256"],
              "adapter_config_sha256": identity["adapter_config_sha256"]}
    (path.parent / "training_report.json").write_text(json.dumps(report))
    _, bound = load_adapter_metadata(path, settings, tokenizer)
    assert bound["training_optimizer_steps"] == "10"
    assert bound["training_report_sha256"]
    (path / "adapter_model.safetensors").write_bytes(b"wrong adapter")
    with pytest.raises(ValueError, match="training report"):
        load_adapter_metadata(path, settings, tokenizer)


@pytest.mark.parametrize("field", ["base_model_name_or_path", "revision", "template", "tokenizer"])
def test_mismatched_adapter_identity_is_rejected(adapter, tokenizer, field):
    path, settings = adapter
    if field in {"base_model_name_or_path", "revision"}:
        p = path / "adapter_config.json"
        config = json.loads(p.read_text()); config[field] = "wrong"
        p.write_text(json.dumps(config))
    else:
        saved = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if field == "template":
            saved.chat_template += "wrong"
        else:
            saved.add_tokens(["test_extra_token_f8"])
        saved.save_pretrained(path)
    with pytest.raises(ValueError, match="different|differs"):
        load_adapter_metadata(path, settings, tokenizer)


def test_missing_adapter_files_are_rejected(adapter, tokenizer):
    path, settings = adapter
    (path / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="weights and tokenizer"):
        load_adapter_metadata(path, settings, tokenizer)


@pytest.mark.parametrize("changed", [False, True])
def test_real_peft_switch_detects_effect_and_restores_adapter(monkeypatch, changed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                           lora_dropout=0.1, task_type="CAUSAL_LM"))
    if changed:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.1)
    prepared = SimpleNamespace(token_ids=[1, 2, 3], record={"messages": []})
    monkeypatch.setattr("audittrainer.policy.prepare_prompt", lambda *a, **k: prepared)
    policy = LLMPolicy(model, None, identity={"adapter": "test-only-peft"}, context_tokens=64,
                       generation_tokens={}, decoding={})
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    report, logits = policy.check_adapter_effect({})
    assert report["status"] == ("PASS" if changed else "FAIL")
    assert report["restored_max_abs"] == 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert policy.generation_calls == 0 and model.active_adapters == ["default"]
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
    torch.testing.assert_close(logits["enabled"], logits["restored"], atol=0, rtol=0)


@pytest.mark.parametrize("args", [[], ["--policy", "sft"], ["--policy", "base", "--adapter", "missing"]])
def test_policy_and_adapter_arguments_are_explicit(monkeypatch, args):
    monkeypatch.setattr(cli.LLMPolicy, "from_pretrained", lambda *a, **k: pytest.fail("must not load"))
    with pytest.raises(SystemExit) as error:
        cli.main(["rollout", "--split", "DEV_ENGINEERING", *args])
    assert error.value.code == 2


@pytest.mark.parametrize("probe_passes", [True, False])
def test_sft_rollout_records_f8_and_probe_before_shared_runtime(tmp_path, monkeypatch, probe_passes):
    from audittrainer.schema import PolicyOutput, TokenUsage
    identity = {"backend": "test_fake", "adapter": "test-only-adapter", "adapter_weights_sha256": "test-hash"}

    class FakePolicy:
        generation_calls = 0

        def __init__(self):
            self.identity = identity
            self.probed = False

        def check_adapter_effect(self, observation):
            assert observation["root_id"] == "dev01_stable_unique"
            assert observation["phase"] == "NEW"
            self.probed = True
            return {"status": "PASS" if probe_passes else "FAIL", "forward_calls": 3,
                    "enabled_disabled_max_abs": 1.0 if probe_passes else 0.0, "restored_max_abs": 0}, {}

        def act(self, observation):
            assert self.probed
            self.generation_calls += 1
            return PolicyOutput(raw_text='{"action":"abstain","args":{"reason_code":"OTHER","evidence_ids":[]}}',
                                model_identity=identity, token_usage=TokenUsage(input_tokens=1, output_tokens=1, source="model"))

    policy = FakePolicy()
    def load(config, *, device, adapter_path):
        assert config["model"]["load_in_4bit"] and adapter_path == Path("test-adapter")
        return policy
    monkeypatch.setattr(cli.LLMPolicy, "from_pretrained", load)
    monkeypatch.setattr(cli, "ScriptedPolicy", lambda *a, **k: pytest.fail("must not use script"))
    output = tmp_path / "run"
    status = cli.main(["rollout", "--policy", "sft", "--adapter", "test-adapter", "--load-in-4bit",
                      "--split", "DEV_ENGINEERING", "--limit", "2", "--run-dir", str(output)])
    result = json.loads((output / "result.json").read_text())
    assert result["stage"] == "F8"
    if not probe_passes:
        assert status == 1 and result["status"] == "ADAPTER_CHECK_ERROR"
        assert policy.generation_calls == 0
        return
    assert status == 0 and result["generation_calls"] == 2
    assert len(result["results"]) == 2 and result["adapter_check"]["status"] == "PASS"
    for r in result["results"]:
        root_result = json.loads((Path(r["run_dir"]) / "result.json").read_text())
        assert root_result["stage"] == "F8"
        events = [json.loads(s) for s in (Path(r["run_dir"]) / "trace.jsonl").read_text().splitlines()]
        assert all(e["model_identity"] == identity for e in events if e["event_kind"] == "POLICY")


def test_development_roots_have_fixed_sources_and_no_training_family_overlap():
    roots = load_root_manifest(ROOT / "data/manifests/roots.jsonl")
    training_families = {r["family_id"] for r in roots.values() if r["split"] in {"TRAIN", "SYN_FIXTURE"}}
    development = [r for r in roots.values() if r["split"] == "DEV_ENGINEERING"]
    assert len(development) >= 2
    for row in development:
        path = ROOT / row["fixture_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["source_hash"]
        assert row["family_id"] not in training_families
        data = json.loads(path.read_text())
        assert "script" not in data and "completion_script" not in data
        assert all(s["visibility"] == "VISIBLE" for s in data["suites"])
