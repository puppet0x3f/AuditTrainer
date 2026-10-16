import json
from pathlib import Path

import pytest
import yaml

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.mechanism import (GenerationPolicy, Session, cost, generation_arm,
    generation_summary, paired, source_links)
from audittrainer.mechanism_checks import audit_components
from audittrainer.mechanism_repair import (ReferencePolicy, failure_snapshot, make_candidate,
    repair_arm, regression_count, verify_all)
from audittrainer.mechanism_tasks import CATALOG, definitions
from audittrainer.schema import RepairTicket


@pytest.fixture
def inputs():
    config = yaml.safe_load(Path("configs/mechanism.yaml").read_text())
    framework = yaml.safe_load(Path(config["framework_config"]).read_text())
    roots = [r for r in load_root_manifest(config["root_manifest"]).values() if r["split"] == "DEV_MECHANISM"]
    tests = {r["root_id"]: r for r in read_jsonl(config["test_manifest"])}
    return config, framework, roots, tests


@pytest.fixture
def failure(inputs, tmp_path):
    config, framework, roots, tests = inputs
    return failure_snapshot(roots[0], tests, framework, config, tmp_path / "setup")


def test_fixed_pool_and_no_split_overlap(inputs):
    config, _, roots, tests = inputs
    assert len(roots) == config["limit"] == len(CATALOG) == 20
    assert len({r["family_id"] for r in roots}) == 1
    assert all(r["split"] == "DEV_MECHANISM" for r in roots)
    for definition, private, reference in definitions():
        assert len(private["suites"][0]["cases"]) == 5
        assert all(s["visibility"] == "VISIBLE" for s in definition["suites"])
        assert reference["fault_node"] in {"N0", "N0_p1_a", "N0_p1_b"}
        assert definition["task"]["root_id"] in tests


def test_known_oracle_examples_are_not_only_empty_cases():
    definitions_list = list(definitions())
    # Independently calculated from [3,-2,3,0,5,-2].
    expected = {0: "2", 1: "43", 2: "45", 3: "8", 4: "0", 5: "4", 5+1: "125"}
    for index, value in expected.items():
        assert definitions_list[index][0]["suites"][0]["cases"][0]["expected"] == value


def test_reference_pass_and_fixed_failure_reproducible(failure, inputs, tmp_path):
    config = inputs[0]
    assert failure["evidence"]["N0"].status == "FAIL"
    assert failure["evidence"]["N0_p1_a"].status == "FAIL"
    assert failure["evidence"]["N0_p1_b"].status == "PASS"
    assert failure["candidates"] == ["N0_p1_a", "N0"]
    snap, artifacts, _ = make_candidate(failure, failure["public"]["sources"], tmp_path / "replay")
    measured = cost()
    evidence = verify_all(failure, snap, artifacts, measured, 24)
    assert [e.status for e in evidence.values()] == ["FAIL", "PASS", "FAIL"]
    assert measured["verify_calls"] == 3 and measured["executed_cases"] == 4
    assert regression_count(failure["evidence"], evidence) == 0


def test_ticket_freezes_complement_and_all_arms_validate_interfaces(failure, tmp_path):
    ticket = RepairTicket(ticket_id="T1", snapshot_id="frozen", target={"node_id": "N0_p1_a"},
        boundary_kind="OWNER_MODULE", editable_files=["nodes/N0_p1_a.py"], editable_plan_nodes=[], rebind_edges=[],
        revalidate_nodes=["N0"], frozen_hashes={}, evidence_ids=["E1"], remaining_attempts=3)
    with pytest.raises(ValueError, match="FROZEN_COMPLEMENT"):
        make_candidate(failure, {"nodes/N0.py": failure["public"]["sources"]["nodes/N0.py"]}, tmp_path / "illegal", ticket=ticket)
    with pytest.raises(ValueError, match="INVALID_SOURCE_BOUNDARY"):
        make_candidate(failure, {"../oracle.json": "x"}, tmp_path / "outside")
    with pytest.raises(ValueError, match="parameters"):
        make_candidate(failure, {"nodes/N0_p1_a.py": "def transform(wrong):\n    return []\n"}, tmp_path / "bad_signature")


@pytest.mark.parametrize("arm", ["R0", "R1", "R2"])
def test_repair_same_snapshot_visible_selection_and_private_isolation(failure, inputs, tmp_path, monkeypatch, arm):
    captured = []
    reference = json.loads(Path(inputs[2][0]["reference_path"]).read_text())
    def ask(self, instruction, payload):
        captured.append(payload)
        self.cost["model_calls"] += 1
        if "candidates" in instruction:
            return {"candidates": ["N0_p1_a"], "abstain": False}, None
        return {"replacements": {"nodes/N0_p1_a.py": reference["a"]}, "abstain": False}, None
    monkeypatch.setattr(Session, "ask", ask)
    record = repair_arm(failure, inputs[0], object(), arm, tmp_path / arm)
    assert record["status"] == "COMPLETED", record["error"]
    assert record["located"] and record["private_pass"] and record["visible_pass"]
    assert record["changed_files"] == 1 and not record["regressions"]
    assert record["online_cost"]["model_calls"] == 2
    assert record["online_cost"]["verify_calls"] == (6 if arm == "R2" else 3)
    assert record["authorized_files"] == (1 if arm == "R2" else 3)
    assert all(p["visible_evidence"] == failure["public"]["visible_evidence"] for p in captured)
    assert not any("fault_node" in json.dumps(p) or '"PRIVATE"' in json.dumps(p) for p in captured)
    selection = json.loads((tmp_path / arm / "selection.json").read_text())
    assert selection["private_feedback_used"] is False


def test_repeated_abstention_scores_independent_copies(failure, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(Session, "ask", lambda *a: ({"candidates": [], "abstain": True}, None))
    records = [repair_arm(failure, inputs[0], object(), arm, tmp_path / arm) for arm in ("R0", "R1", "R2")]
    assert all(r["status"] == "COMPLETED" and r["abstained"] and not r["private_pass"] for r in records)


def test_generation_executes_model_actions_and_audits_functions(inputs, tmp_path):
    config, framework, roots, tests = inputs
    definition = json.loads(Path(roots[0]["fixture_path"]).read_text())
    reference = json.loads(Path(roots[0]["reference_path"]).read_text())
    class Model:
        def act(self, observation):
            return ReferencePolicy(definition, reference).act(observation)
    record = generation_arm(roots[0], tests, framework, config, Model(), "hierarchical", tmp_path / "hierarchical")
    assert record["status"] == "COMPLETED", record["error"]
    assert record["scores"]["private"]["status"] == "PASS"
    assert record["metrics"]["source_links"]["coverage"] == 1
    assert record["metrics"]["local_bound_checks"] == 2
    assert record["component_audit"]["passed"] == 2
    assert record["component_audit"]["output_insensitive"] == 0
    assert record["online_cost"]["model_calls"] == 6


def test_posthoc_does_not_generate_plan_before_source(inputs):
    config = inputs[0]
    definition = json.loads(Path(inputs[2][0]["fixture_path"]).read_text())
    o = {"phase": "NEW", "node_id": "N0", "incoming_contract": definition["contract"]}
    output = GenerationPolicy(None, "posthoc", definition).act(o)
    assert json.loads(output.raw_text)["args"]["obligation_map"] == {}
    assert output.model_identity["backend"] == "a2_fixed_protocol_control"


def test_failures_retained_in_summary_and_zero_success_cost():
    rows = [{"root_id": "r1", "arm": a, "status": "ERROR", "stop_reason": "INVALID", "metrics": None,
             "scores": None, "online_cost": {**cost(), "input_tokens": 10}, "scoring_cost": cost()} for a in ("a", "b")]
    summary = generation_summary(rows, ["a", "b"])
    assert summary["a"]["roots"] == summary["a"]["errors"] == 1
    assert summary["a"]["tokens_per_private_success"] is None
    assert paired(rows, ["a", "b"], lambda r: False) == {"a_vs_b": {"neither": 1}}


def test_auxiliary_private_and_source_labels_rejected(inputs, tmp_path):
    session = Session(object(), inputs[0], tmp_path)
    for payload in ({"visibility": "PRIVATE"}, {"nested": {"fault_node": "N0"}}):
        with pytest.raises(ValueError, match="Offline labels"):
            session.ask("test", payload)


def test_conditions_are_trusted_and_cannot_override_normal_prompts():
    from audittrainer.mechanism import prepare_mechanism_prompt
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
        def encode(self, text, **kwargs):
            return list(text.encode())
    base = {"allowed_actions": ["plan"], "root_spec": "untrusted task text", "implementation_stage": "A2",
            "mechanism_instruction": "Function-only: obligation_map={}.", "budget_remaining": {"tokens": 100000}}
    def render(o):
        return prepare_mechanism_prompt(Tokenizer(), o, context_tokens=100000, generation_tokens={"control":100}).record["messages"]
    rendered = render(base)
    assert "Function-only: obligation_map={}." in rendered[0]["content"]
    assert "untrusted task text" not in rendered[0]["content"]
    assert "mechanism_instruction" not in rendered[1]["content"]
    base["implementation_stage"] = "F9"
    assert "Function-only: obligation_map={}." not in render(base)[0]["content"]
