"""A3 checks executable mechanism choice and data protocol separation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer
import yaml

from audittrainer.cli import fixture_verifier
from audittrainer.dataset import collect_fixture_suite, export_sft, file_hash, read_jsonl
from audittrainer.policy import ScriptedPolicy, ScriptStep
from audittrainer.prompts import OBSERVATION_FIELDS
from audittrainer.runtime import Limits, RepairRuntime
from audittrainer.schema import ACTION_ADAPTER, ContractSpec, RootTask
from audittrainer.trace import renderer_identity
from audittrainer.train_sft import load_config, load_training_data

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/roots.jsonl"


@pytest.fixture(scope="module")
def frozen():
    return yaml.safe_load((ROOT / "configs/frozen_mechanism.yaml").read_text())


def test_frozen_entrypoints_preserve_renderer_and_use_one_main_protocol(frozen):
    record = frozen["frozen_mechanism"]
    assert record["renderer"] == renderer_identity()
    assert record["observation_fields"] == list(OBSERVATION_FIELDS)
    assert set(record["actions"]) == set(ACTION_ADAPTER.json_schema()["discriminator"]["mapping"])
    identity = json.loads((ROOT / "data/manifests/model.json").read_text())
    assert record["chat_template_sha256"] == identity["chat_template_sha256"]
    assert record["tokenizer_sha256"] == identity["tokenizer_sha256"]
    assert frozen["data_check_protocol_version"] in (ROOT / record["data_check_protocol"]).read_text()
    main = load_config(ROOT / record["training_config"])
    evaluation = yaml.safe_load((ROOT / record["evaluation_config"]).read_text())
    assert main["framework_config"] == evaluation["framework_config"] == "configs/frozen_mechanism.yaml"
    assert main["data"]["allowed_splits"] == ["TRAIN"]
    assert main["model"]["revision"] == frozen["model"]["revision"]
    assert evaluation["max_policy_calls"] == frozen["runtime"]["max_policy_calls"]
    assert load_config(ROOT / "configs/sft_smoke.yaml")["framework_config"] == "configs/framework.yaml"


def make_runtime(data, frozen, directory):
    return RepairRuntime(
        RootTask.model_validate(data["task"]), ContractSpec.model_validate(data["contract"]),
        ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(s)) for s in data["script"]]), directory,
        limits=Limits(**{k: frozen["runtime"][k] for k in Limits.model_fields}),
        protocol_version=frozen["protocol_version"], verifier=fixture_verifier(data, frozen),
        repair_config=frozen["repair"],
    )


def root_review(runtime):
    for _ in range(64):
        if runtime.active_node_id == "N0" and runtime.node.phase.value == "REVIEW":
            return
        assert runtime.stop_reason is None
        runtime.step()
    pytest.fail("Root review was not reached")


def test_r1_owner_screening_keeps_f4_freeze_and_visible_acceptance(frozen, repair_fixture, tmp_path):
    runtime = make_runtime(repair_fixture, frozen, tmp_path / "repair")
    root_review(runtime)
    assert runtime.repair_manager.candidate_strategy == "responsibility_filtered"
    assert [c["node_id"] for c in runtime.build_observation()["responsibility_candidates"]] == ["N0"]
    children = {n: c.artifact for n, c in runtime.contexts.items() if n != "N0"}
    assert runtime.step().host_result.code == "REPAIR_AUTHORIZED"
    assert set(runtime.repair_manager.ticket.frozen_hashes) == {f"nodes/{n}.py" for n in children}
    result = runtime.run()
    assert result["functional_status"] == "PASS"
    assert result["usage"]["patch_attempts"] == 1
    # Four original verifications + one anchor replay + one candidate check.
    assert result["usage"]["verify_calls"] == 6
    assert {n: runtime.contexts[n].artifact for n in children} == children
    manifest = json.loads((runtime.run_dir / "trace_manifest.json").read_text())
    assert manifest["protocol_version"] == frozen["protocol_version"]


def test_failed_child_precedes_consumer_and_stale_failure_is_excluded(frozen, repair_fixture, tmp_path):
    data = deepcopy(repair_fixture)
    old = data["script"][2]["output"]["args"]["source"]
    source = old.replace("+ x", "+ 0")
    data = json.loads(json.dumps(data).replace(hashlib.sha256(old.encode()).hexdigest(),
                                             hashlib.sha256(source.encode()).hexdigest()))
    data["script"][2]["output"]["args"]["source"] = source
    data["script"][5]["output"]["args"].update(
        artifact_hash=hashlib.sha256(source.encode()).hexdigest(), accept_unverified=True)
    runtime = make_runtime(data, frozen, tmp_path / "failed_child")
    root_review(runtime)
    calls = runtime.budget.verify_calls
    candidates = runtime.repair_manager.candidates()
    assert [c["node_id"] for c in candidates] == ["N0_p1_prefix", "N0"]
    child = runtime.contexts["N0_p1_prefix"]
    child.node = child.node.model_copy(update={
        "last_evidence": child.node.last_evidence.model_copy(update={"fingerprint": "0" * 64})})
    assert [c["node_id"] for c in runtime.repair_manager.candidates()] == ["N0"]
    assert runtime.budget.verify_calls == calls  # Screening itself does not run tests.


@pytest.fixture(scope="module")
def frozen_export(frozen, tmp_path_factory):
    directory = tmp_path_factory.mktemp("a3_data")
    collect_fixture_suite(directory / "traces", config=frozen, root_manifest=MANIFEST,
                          review_file=ROOT / "data/fixtures/f6_reviews.json")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "Qwen3-8B", local_files_only=True)
    out = directory / "smoke.jsonl"
    export_sft(directory / "traces", out, root_manifest=MANIFEST, config=frozen,
               max_length=8192, limit=32, tokenizer=tokenizer)
    return out, tokenizer


def test_frozen_export_loads_with_versions_but_remains_fixture_only(frozen, frozen_export):
    path, tokenizer = frozen_export
    config = load_config(ROOT / "configs/sft_main.yaml")
    config["data"].update(path=str(path), allowed_splits=["SYN_FIXTURE"])
    identity = json.loads((ROOT / "data/manifests/model.json").read_text())
    rows, encoded, manifest = load_training_data(config, identity, tokenizer)
    assert len(rows) == len(encoded) == 32
    assert manifest["scope"] == "SMOKE_ONLY"
    assert manifest["data_check_protocol_version"] == frozen["data_check_protocol_version"]
    assert {r["meta"]["protocol_version"] for r in rows} == {frozen["protocol_version"]}
    config["data"]["allowed_splits"] = ["TRAIN"]
    with pytest.raises(ValueError, match="separate"):
        load_training_data(config, identity, tokenizer)


@pytest.mark.parametrize("mismatch", ["manifest", "sample"])
def test_loader_rejects_incompatible_protocol(frozen_export, tmp_path, mismatch):
    original, tokenizer = frozen_export
    manifest = json.loads(original.with_suffix(".manifest.json").read_text())
    rows = read_jsonl(original)
    if mismatch == "manifest":
        manifest["data_check_protocol_version"] = "old"
    else:
        rows[0]["meta"]["data_check_protocol_version"] = "old"
    path = tmp_path / "incompatible.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest["output_sha256"] = file_hash(path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    config = load_config(ROOT / "configs/sft_main.yaml")
    config["data"].update(path=str(path), allowed_splits=["SYN_FIXTURE"])
    identity = json.loads((ROOT / "data/manifests/model.json").read_text())
    with pytest.raises(ValueError, match="frozen data/check"):
        load_training_data(config, identity, tokenizer)


def test_old_trace_cannot_be_relabelled_by_new_export_config(frozen, frozen_export, tmp_path):
    _, tokenizer = frozen_export
    result = export_sft(ROOT / "runs/f6_fixture_suite/fx01_integer_sum", tmp_path / "old.jsonl",
                        root_manifest=MANIFEST, config=frozen, tokenizer=tokenizer, max_length=8192)
    assert result["samples"] == 0
    assert result["rejection_reasons"]["RUNTIME_PROTOCOL_MISMATCH"] > 0
