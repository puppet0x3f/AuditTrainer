from copy import deepcopy
import json

import pytest

from audittrainer.runtime import Limits
from audittrainer.sandbox import ExecutionLimits, Sandbox
from audittrainer.schema import Phase
from audittrainer.verifier import NO_APPLICABLE_SUITE, OracleCase


def action(phase, name, args):
    return {"node_id": "N0", "phase": phase, "output": {"action": name, "args": args}}


def abstain(phase):
    return action(phase, "abstain", {"reason_code": "INSUFFICIENT_EVIDENCE", "evidence_ids": [], "suggested_change": None})


def test_real_atomic_finish_has_version_bound_evidence_and_exact_cost(make_f2_runtime):
    runtime = make_f2_runtime()
    result = runtime.run()
    assert (result["phase"], result["stop_reason"], result["functional_status"]) == ("TERMINAL", "FINISHED", "PASS")
    assert result["verification_performed"] is True
    assert result["responsibility_status"] == "UNKNOWN"
    assert result["usage"]["policy_calls"] == 4 and result["usage"]["verify_calls"] == 1
    assert result["usage"]["executed_cases"] == 6 and result["usage"]["execution_seconds"] > 0
    assert result["node_result"]["artifact"] == result["artifact"]
    events = [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]
    assert [e["parsed_action"]["action"] for e in events] == ["plan", "write_code", "verify", "finish"]
    assert all(not e["observation_before"]["visible_evidence"] for e in events[:3])
    assert events[3]["observation_before"]["visible_evidence"][0]["status"] == "PASS"
    assert len(list((runtime.run_dir / "states").glob("*.json"))) == 5
    assert len(list((runtime.run_dir / "evidence").glob("*.json"))) == 1


@pytest.mark.parametrize("source,reason", [
    ("def solve(text):\n    a, b = map(int, text.split())\n    return str(a - b)", "WRONG_OUTPUT"),
    ("def solve(text):\n    while True: pass", "TIMEOUT"),
])
def test_finish_preserves_candidate_failures(make_f2_runtime, source, reason):
    runtime = make_f2_runtime(source=source, sandbox=Sandbox(ExecutionLimits(per_case_timeout_seconds=1.0)))
    result = runtime.run()
    assert (result["stop_reason"], result["functional_status"]) == ("FINISHED", "FAIL")
    assert runtime.node.last_evidence.details["reason"] == reason
    assert result["usage"]["verify_calls"] == result["usage"]["executed_cases"] == 1


@pytest.mark.parametrize("kind", ["empty", "missing"])
def test_finish_preserves_unknown(make_f2_runtime, fixture_data, reviewed_suite, kind):
    script = deepcopy(fixture_data["script"] + fixture_data["completion_script"])
    suites = [reviewed_suite.model_copy(update={"cases": []})] if kind == "empty" else []
    if kind == "missing":
        script[2]["output"]["args"]["suite_id"] = NO_APPLICABLE_SUITE
    runtime = make_f2_runtime(script=script, suites=suites)
    result = runtime.run()
    assert (result["stop_reason"], result["functional_status"]) == ("FINISHED", "UNKNOWN")
    assert result["usage"]["executed_cases"] == 0


def test_runner_error_terminates_as_error_and_is_not_candidate_fail(make_f2_runtime, tmp_path):
    runtime = make_f2_runtime(sandbox=Sandbox(bwrap=str(tmp_path / "absent")))
    result = runtime.run()
    assert (result["stop_reason"], result["functional_status"]) == ("VERIFIER_ERROR", "ERROR")
    assert runtime.budget.policy_calls == 3
    assert runtime.last_action_result.status == "ERROR"


def test_abstain_without_code_is_terminal_and_cannot_change_contract(make_f2_runtime):
    runtime = make_f2_runtime(script=[abstain("NEW")])
    original = runtime.node.incoming_contract.model_dump()
    result = runtime.run()
    assert result["stop_reason"] == "ABSTAINED" and result["artifact"] is None
    assert result["functional_status"] is None and result["verification_performed"] is False
    assert runtime.node.incoming_contract.model_dump() == original


def test_finish_before_verification_is_rejected(make_f2_runtime, fixture_data):
    premature = action("READY_VERIFY", "finish", {"artifact_version": 1})
    runtime = make_f2_runtime(script=[*fixture_data["script"], premature, abstain("READY_VERIFY")])
    runtime.step()
    runtime.step()
    assert runtime.step().host_result.code == "ACTION_NOT_ALLOWED"
    assert runtime.node.phase == Phase.READY_VERIFY and runtime.node.last_evidence is None
    assert runtime.run()["stop_reason"] == "ABSTAINED"


@pytest.mark.parametrize("args,code", [
    ({"artifact_version": 2}, "ARTIFACT_VERSION_MISMATCH"),
    ({"artifact_version": 1, "status": "PASS"}, "INVALID_ACTION_JSON"),
])
def test_finish_cannot_forge_version_or_verdict(make_f2_runtime, fixture_data, args, code):
    script = deepcopy(fixture_data["script"] + fixture_data["completion_script"])
    script[1]["output"]["args"]["source"] = "def solve(text): return 'wrong'"
    script[3]["output"]["args"] = args
    script.append(fixture_data["completion_script"][1])
    runtime = make_f2_runtime(script=script)
    for _ in range(3):
        runtime.step()
    evidence = runtime.node.last_evidence
    assert runtime.step().host_result.code == code
    assert runtime.node.last_evidence == evidence
    assert runtime.run()["functional_status"] == "FAIL"


def test_finish_rejects_changed_source_and_summary_does_not_reuse_pass(make_f2_runtime, fixture_data):
    runtime = make_f2_runtime(script=fixture_data["script"] + fixture_data["completion_script"] + [abstain("REVIEW")])
    for _ in range(3):
        runtime.step()
    (runtime.run_dir / runtime.artifact.module_path).write_text("def solve(text): return 'changed'")
    assert runtime.step().host_result.code == "STALE_OR_MISSING_EVIDENCE"
    assert runtime.node.phase == Phase.REVIEW
    result = runtime.run()
    assert result["functional_status"] == "UNKNOWN" and result["evidence_current"] is False
    assert result["node_result"]["evidence"]["status"] == "PASS" # Historical evidence is retained.


def test_source_changed_before_verify_is_an_infrastructure_binding_error(make_f2_runtime):
    runtime = make_f2_runtime()
    runtime.step()
    runtime.step()
    (runtime.run_dir / runtime.artifact.module_path).write_text("def solve(text): return 'changed'")
    result = runtime.run()
    assert (result["stop_reason"], result["functional_status"]) == ("VERIFIER_ERROR", "ERROR")
    assert result["usage"]["executed_cases"] == 0


def test_private_oracle_is_not_available_to_policy_or_candidate(make_f2_runtime, fixture_data, reviewed_suite):
    private = reviewed_suite.model_copy(update={
        "suite_id": "PRIVATE_ONLY", "visibility": "PRIVATE",
        "cases": [OracleCase(case_id="private_case", args=["PRIVATE_INPUT"], expected="PRIVATE_EXPECTED")],
    })
    script = deepcopy(fixture_data["script"] + fixture_data["completion_script"])
    script[2]["output"]["args"]["suite_id"] = private.suite_id
    script[3] = abstain("READY_VERIFY")
    runtime = make_f2_runtime(script=script, suites=[reviewed_suite, private])
    runtime.step()
    runtime.step()
    assert runtime.build_observation()["available_suites"] == [reviewed_suite.suite_id]
    assert runtime.step().host_result.code == "SUITE_UNAVAILABLE"
    assert runtime.budget.verify_calls == 0
    with pytest.raises(ValueError, match="Private suites"):
        runtime.verifier.verify(runtime.run_dir, runtime.task.root_id, runtime.node.incoming_contract,
                                runtime.artifact, private.suite_id, "E_PRIVATE")
    assert runtime.run()["stop_reason"] == "ABSTAINED"
    trace = runtime.trace.path.read_text()
    assert "PRIVATE_EXPECTED" not in trace and "PRIVATE_INPUT" not in trace
    # An explicit offline host check never changes the historical observation.
    runtime.verifier.verify(runtime.run_dir, runtime.task.root_id, runtime.node.incoming_contract,
                            runtime.artifact, private.suite_id, "E_PRIVATE", allow_private=True)
    assert runtime.node.last_evidence is None and runtime.trace.path.read_text() == trace


def test_verification_budget_counts_calls_and_still_allows_finish(make_f2_runtime, fixture_data):
    script = fixture_data["script"] + [fixture_data["completion_script"][0]]
    repeated = deepcopy(fixture_data["completion_script"][0])
    repeated["phase"] = "REVIEW"
    script += [repeated, fixture_data["completion_script"][1]]
    runtime = make_f2_runtime(script=script, limits=Limits(max_verify_calls=1))
    for _ in range(3):
        runtime.step()
    assert runtime.step().host_result.code == "VERIFY_BUDGET_EXHAUSTED"
    result = runtime.run()
    assert result["functional_status"] == "PASS" and result["usage"]["verify_calls"] == 1


def test_wrong_target_does_not_execute(make_f2_runtime, fixture_data):
    script = deepcopy(fixture_data["script"] + fixture_data["completion_script"])
    script[2]["output"]["args"]["target_ref"] = "../someone-else.py"
    runtime = make_f2_runtime(script=script)
    runtime.step()
    runtime.step()
    assert runtime.step().host_result.code == "TARGET_MISMATCH"
    assert runtime.budget.verify_calls == 0 and runtime.node.last_evidence is None
