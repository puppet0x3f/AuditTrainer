from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from audittrainer.schema import ACTION_ADAPTER, parse_action


def action_examples(fixture_data):
    return [
        fixture_data["script"][0]["output"],
        fixture_data["script"][1]["output"],
        {"action": "verify", "args": {"target_ref": "N0@v1", "suite_id": "visible_sum"}},
        {"action": "consume", "args": {
            "child_id": "N0_1", "artifact_version": 1, "artifact_hash": "a" * 64,
            "alias": "prefix_sums", "evidence_ids": ["E1"],
        }},
        {"action": "reject", "args": {
            "child_id": "N0_1", "artifact_version": 1, "reason_code": "LOCAL_CHECK_FAILED", "evidence_ids": ["E1"],
        }},
        {"action": "request_child_repair", "args": {
            "child_id": "N0_1", "artifact_version": 1, "evidence_ids": ["E1"], "instruction": "Correct the empty case",
        }},
        {"action": "localize", "args": {
            "target_type": "HANDOFF", "target_id": "N0<-N0_1", "evidence_ids": ["E1"],
            "boundary_kind": "CONSUMER_MODULE", "rationale": "The parent uses the wrong arguments",
        }},
        {"action": "repair", "args": {
            "ticket_id": "T1", "replacements": [{"file": "nodes/N0.py", "base_hash": "b" * 64, "source": "def solve(text): return text"}],
            "dependency_updates": [{"consumer_node": "N0", "child_id": "N0_1", "artifact_ref": {"kind": "CANDIDATE", "node_id": "N0_1"}, "alias": "prefix_sums"}],
        }},
        {"action": "finish", "args": {"artifact_version": 1}},
        {"action": "abstain", "args": {"reason_code": "INSUFFICIENT_EVIDENCE", "evidence_ids": []}},
    ]


@pytest.mark.parametrize("index", range(10))
def test_all_action_payloads_and_roundtrip(fixture_data, index):
    example = action_examples(fixture_data)[index]
    action = parse_action(json.dumps(example))
    assert action.action == example["action"]
    assert parse_action(action.model_dump_json()) == action
    assert len(ACTION_ADAPTER.json_schema()["discriminator"]["mapping"]) == 10


@pytest.mark.parametrize("raw", [
    "{broken", "{}", "[]", '{"action":"launch_subagent","args":{}}',
    '{"action":"finish","args":{"artifact_version":true}}',
    '{"action":"finish","args":{"artifact_version":"1"}}',
    '{"action":"finish","args":{"artifact_version":0}}',
    '{"action":"finish","args":{"artifact_version":1,"verdict":"PASS"}}',
    '{"action":"finish","args":{"artifact_version":1},"node_id":"N0_2"}',
    '```json\n{"action":"finish","args":{"artifact_version":1}}\n```',
    '{"action":"finish","args":{"artifact_version":1}} {}',
])
def test_rejects_invalid_or_privileged_payloads(raw):
    with pytest.raises(ValidationError):
        parse_action(raw)


def test_split_contracts_have_no_host_assigned_identity(fixture_data):
    action = deepcopy(fixture_data["script"][0]["output"])
    action["args"].update(mode="SPLIT", children=[{"local_key": "sum", "contract": fixture_data["contract"]}])
    assert parse_action(json.dumps(action)).args.children[0].local_key == "sum"
    action["args"]["children"][0]["contract"]["contract_id"] = "MODEL_ASSIGNED_ID"
    with pytest.raises(ValidationError):
        parse_action(json.dumps(action))


def test_initial_plan_and_revision_shapes(fixture_data):
    action = deepcopy(fixture_data["script"][0]["output"])
    action["args"]["base_plan_version"] = 1
    with pytest.raises(ValidationError):
        parse_action(json.dumps(action))
    action["args"]["ticket_id"] = "T1"
    assert parse_action(json.dumps(action)).args.base_plan_version == 1
    action["args"]["children"] = [{"local_key": "sum", "contract": fixture_data["contract"]}]
    with pytest.raises(ValidationError):
        parse_action(json.dumps(action))
