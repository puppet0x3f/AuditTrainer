from copy import deepcopy
import hashlib
import json

import pytest

from audittrainer.runtime import Limits
from audittrainer.schema import Phase, PolicyOutput, TokenUsage


def events(runtime):
    return [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]


def test_atomic_smoke_preserves_state_history_and_stops_before_verification(make_runtime, fixture_data):
    runtime = make_runtime()
    result = runtime.run()
    assert result["phase"] == "READY_VERIFY"
    assert result["stop_reason"] == "F1_READY_VERIFY"
    assert result["verification_performed"] is False
    assert result["functional_status"] is None
    assert result["node_count"] == 1
    assert result["usage"] == {"policy_calls": 2, "input_tokens": 0, "output_tokens": 0,
                               "verify_calls": 0, "executed_cases": 0, "execution_seconds": 0.0,
                               "patch_attempts": 0, "plan_revisions": 0, "nodes_created": 1}
    assert runtime.plan.children == [] and runtime.node.pending_edges == []
    source = (runtime.run_dir / result["artifact"]["module_path"]).read_bytes()
    assert source.decode() == fixture_data["script"][1]["output"]["args"]["source"]
    assert hashlib.sha256(source).hexdigest() == result["artifact"]["source_hash"]
    assert list((runtime.run_dir / "artifacts").iterdir())[0].name == "N0"
    trace = events(runtime)
    assert [e["event_kind"] for e in trace] == ["POLICY", "POLICY", "HOST"]
    assert [e["observation_before"]["phase"] for e in trace] == ["NEW", "IMPLEMENT", "READY_VERIFY"]
    assert trace[0]["observation_before"]["current_plan"] is None
    assert trace[1]["observation_before"]["own_code"] is None
    for event in trace:
        assert event["root_id"] == "fx01_integer_sum" and event["node_id"] == "N0"
        assert event["observation_before"]["visible_evidence"] == []
        assert (runtime.run_dir / event["state_after_ref"]).exists()
    assert len(list((runtime.run_dir / "states").glob("*.json"))) == 4
    assert json.loads((runtime.run_dir / "states/S0000.json").read_text())["node"]["phase"] == "NEW"


@pytest.mark.parametrize("raw,code", [
    ("not JSON", "INVALID_ACTION_JSON"),
    ('{"action":"unknown","args":{}}', "INVALID_ACTION_JSON"),
    ('{"action":"finish","args":{"artifact_version":1}}', "ACTION_NOT_ALLOWED"),
])
def test_invalid_actions_leave_business_state_unchanged_and_charge_calls(make_runtime, raw, code):
    runtime = make_runtime(script=[{"node_id": "N0", "phase": "NEW", "output": raw}])
    before = runtime.node.model_dump(mode="json")
    event = runtime.step()
    assert event.host_result.code == code
    assert runtime.node.model_dump(mode="json") == before
    assert runtime.plan is None and runtime.artifact is None
    assert runtime.budget.policy_calls == 1
    assert event.raw_output == raw
    assert not (runtime.run_dir / "artifacts").exists()
    assert runtime.build_observation()["last_action_result"]["code"] == code


def test_one_feedback_retry_can_recover(make_runtime, fixture_data):
    bad = {"node_id": "N0", "phase": "NEW", "output": "{bad"}
    runtime = make_runtime(script=[bad, *fixture_data["script"]])
    result = runtime.run()
    assert result["phase"] == "READY_VERIFY" and result["usage"]["policy_calls"] == 3
    trace = events(runtime)
    assert trace[1]["observation_before"]["last_action_result"]["code"] == "INVALID_ACTION_JSON"
    assert trace[0]["parsed_action"] is None


def test_repeated_invalid_output_exhausts_retry_budget(make_runtime):
    bad = {"node_id": "N0", "phase": "NEW", "output": "{bad"}
    runtime = make_runtime(script=[bad, bad, bad])
    result = runtime.run()
    assert result["stop_reason"] == "PROTOCOL_RETRIES_EXHAUSTED"
    assert result["phase"] == "TERMINAL"
    assert runtime.policy.cursor == 2
    assert [e["raw_output"] for e in events(runtime)[:2]] == ["{bad", "{bad"]


@pytest.mark.parametrize("calls,expected_calls", [(0, 0), (1, 1)])
def test_root_call_budget_stops_without_more_decisions(make_runtime, calls, expected_calls):
    runtime = make_runtime(limits=Limits(max_policy_calls=calls))
    result = runtime.run()
    assert result["stop_reason"] == "BUDGET_EXHAUSTED"
    assert runtime.policy.cursor == expected_calls
    assert runtime.budget.policy_calls == expected_calls
    assert runtime.artifact is None


def test_exact_budget_can_save_source(make_runtime):
    runtime = make_runtime(limits=Limits(max_policy_calls=2))
    assert runtime.run()["phase"] == "READY_VERIFY"


def test_actual_tokens_count_even_when_over_budget(make_runtime, fixture_data):
    script = deepcopy(fixture_data["script"])
    script[0]["usage"] = {"input_tokens": 4, "output_tokens": 3, "source": "scripted"}
    runtime = make_runtime(script=script, limits=Limits(max_total_tokens=6))
    before = runtime.node.model_dump()
    assert runtime.step().host_result.code == "TOKEN_BUDGET_EXCEEDED"
    assert runtime.node.model_dump() == before and runtime.plan is None
    assert runtime.budget.total_tokens == 7
    assert runtime.run()["stop_reason"] == "BUDGET_EXHAUSTED"
    assert runtime.policy.cursor == 1


@pytest.mark.parametrize("change,expected", [
    ({"contract_version": 2}, "CONTRACT_VERSION_MISMATCH"),
    ({"source": "def solve(text)\n    return '0'"}, "INVALID_SOURCE"),
    ({"source": "def different(text): return '0'"}, "INTERFACE_MISMATCH"),
    ({"source": "def solve(other): return '0'"}, "INTERFACE_MISMATCH"),
    ({"declared_dependencies": [{"child_id": "N0_1", "artifact_version": 1, "artifact_hash": "a" * 64, "alias": "child"}]}, "UNAPPROVED_DEPENDENCY"),
    ({"file": "../N0_1.py"}, "INVALID_ACTION_JSON"),
])
def test_bad_write_preserves_node_and_creates_no_artifact(make_runtime, fixture_data, change, expected):
    script = deepcopy(fixture_data["script"])
    script[1]["output"]["args"].update(change)
    runtime = make_runtime(script=script)
    runtime.step()
    before = runtime.node.model_dump()
    assert runtime.step().host_result.code == expected
    assert runtime.node.model_dump() == before
    assert runtime.artifact is None and not (runtime.run_dir / "artifacts").exists()


def test_plan_version_and_unsupported_split_are_recorded(make_runtime, fixture_data):
    script = deepcopy(fixture_data["script"][:1])
    script[0]["output"]["args"].update(base_plan_version=2, ticket_id="T2")
    runtime = make_runtime(script=script)
    assert runtime.step().host_result.code == "PLAN_VERSION_MISMATCH"
    script[0]["output"]["args"].update(
        base_plan_version=0, ticket_id=None, mode="SPLIT",
        children=[{"local_key": "sum", "contract": fixture_data["contract"]}],
    )
    runtime = make_runtime(script=script)
    assert runtime.step().host_result.code == "F1_UNSUPPORTED_MODE"
    assert runtime.node.phase == Phase.NEW and runtime.plan is None


def test_obligation_omission_is_audited_without_rewriting_contract(make_runtime, fixture_data):
    script = deepcopy(fixture_data["script"])
    del script[0]["output"]["args"]["obligation_map"]["O_EMPTY"]
    runtime = make_runtime(script=script)
    result = runtime.step().host_result
    assert result.status == "APPLIED"
    assert result.details["missing_obligation_ids"] == ["O_EMPTY"]
    assert result.details["semantic_coverage"] == "UNKNOWN"
    assert len(runtime.node.incoming_contract.ensures) == 3


def test_saved_source_cannot_be_replaced_by_write_code(make_runtime, fixture_data):
    script = deepcopy(fixture_data["script"])
    second_write = deepcopy(script[1])
    second_write["phase"] = "READY_VERIFY"
    second_write["output"]["args"]["source"] = "def solve(text): return 'changed'"
    runtime = make_runtime(script=[*script, second_write])
    runtime.step()
    runtime.step()
    original = runtime.artifact
    assert runtime.step().host_result.code == "ACTION_NOT_ALLOWED"
    assert runtime.artifact == original
    assert (runtime.run_dir / original.module_path).read_text() == fixture_data["script"][1]["output"]["args"]["source"]


def test_policy_cannot_mutate_observation_or_supply_a_different_parsed_action(make_runtime, fixture_data):
    class MisbehavingPolicy:
        def act(self, observation):
            observation["incoming_contract"]["ensures"].clear()
            observation["root_spec"] = "INJECTED"
            from audittrainer.schema import parse_action
            return PolicyOutput(
                raw_text="bad raw JSON", token_usage=TokenUsage(), model_identity={"backend": "scripted"},
                parsed_action=parse_action(json.dumps(fixture_data["script"][0]["output"])),
            )
    runtime = make_runtime(policy=MisbehavingPolicy())
    event = runtime.step()
    assert event.host_result.code == "INVALID_ACTION_JSON"
    assert len(event.observation_before["incoming_contract"]["ensures"]) == 3
    assert event.observation_before["root_spec"] != "INJECTED"
    assert runtime.plan is None


def test_policy_failure_is_visible_and_charged(make_runtime):
    runtime = make_runtime(script=[])
    result = runtime.run()
    assert result["stop_reason"] == "POLICY_ERROR"
    assert runtime.budget.policy_calls == 1
    event = events(runtime)[0]
    assert event["host_result"]["status"] == "ERROR"
    assert event["usage"]["source"] == "unavailable"
    assert event["raw_output"] is None


def test_existing_run_directory_is_not_overwritten(make_runtime):
    runtime = make_runtime()
    from audittrainer.runtime import AtomicRuntime
    from audittrainer.schema import ContractSpec
    contract = ContractSpec.model_validate(runtime.node.incoming_contract.model_dump(exclude={"contract_id", "version"}))
    with pytest.raises(FileExistsError):
        AtomicRuntime(runtime.task, contract, runtime.policy, runtime.run_dir)
