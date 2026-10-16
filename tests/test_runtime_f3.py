from copy import deepcopy
import hashlib
import json

import pytest

from audittrainer.runtime import Limits
from audittrainer.sandbox import Sandbox
from audittrainer.schema import Phase


PREFIX = "N0_p1_prefix"
QUERY = "N0_p1_query"
DIFFERENCE = "N0_p1_query_p1_difference"


def advance(runtime, node_id, phase):
    for _ in range(100):
        if runtime.active_node_id == node_id and runtime.node.phase.value == phase:
            return
        assert runtime.stop_reason is None, runtime.stop_reason
        runtime.step()
    pytest.fail("Script did not reach requested node/phase")


def step(node, phase, name, **args):
    return dict(node_id=node, phase=phase, output=dict(action=name, args=args))


def abandon(node, phase):
    return step(node, phase, "abstain", reason_code="INSUFFICIENT_EVIDENCE", evidence_ids=[], suggested_change=None)


def bad_prefix(fixture):
    data = deepcopy(fixture)
    source = "def prefix_sums(xs):\n    return [0]\n"
    data["script"][2]["output"]["args"]["source"] = source
    new_hash = hashlib.sha256(source.encode()).hexdigest()
    data["script"][5]["output"]["args"]["artifact_hash"] = new_hash
    data["script"][16]["output"]["args"]["declared_dependencies"][0]["artifact_hash"] = new_hash
    return data


def test_recursive_smoke_has_explicit_parent_decisions_and_real_grandchild_calls(make_f3_runtime, recursive_fixture):
    runtime = make_f3_runtime()
    result = runtime.run()
    assert (result["stop_reason"], result["functional_status"], result["node_count"]) == ("FINISHED", "PASS", 4)
    assert result["usage"]["policy_calls"] == 19
    assert result["usage"]["verify_calls"] == 4 and result["usage"]["executed_cases"] == 12
    assert result["trace_events"] == 25
    assert all(node["functional_status"] == "PASS" for node in result["node_results"].values())
    assert runtime.contexts[DIFFERENCE].node.parent_id == QUERY
    events = [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]
    decisions = [e for e in events if e["event_kind"] == "POLICY"]
    assert [(e["node_id"], e["parsed_action"]) for e in decisions] == [(s["node_id"], s["output"]) for s in recursive_fixture["script"]]
    assert all(e["observation_before"]["phase"] != "WAIT_CHILD" for e in decisions)
    returned = [e for e in events if e["host_result"]["code"] == "CHILD_RETURNED"]
    assert len(returned) == 3
    for event in returned:
        next_event = events[events.index(event) + 1]
        assert next_event["parsed_action"]["action"] == "consume"
        assert next_event["observation_before"]["returned_child"]["result"]["functional_status"] == "PASS"
        assert next_event["observation_before"]["mode"] == "COMPOSE"
    for item in runtime.node.last_evidence.details["dependency_usage"]:
        assert item["selected"] and item["linked"] and item["called"]
        assert item["call_count"] == 3 and item["dependency_supported"] == "UNKNOWN"
    query_usage = runtime.contexts[QUERY].node.last_evidence.details["dependency_usage"]
    assert query_usage[0]["child_id"] == DIFFERENCE and query_usage[0]["call_count"] == 4
    workspace = runtime.run_dir / "workspaces/N0/v1"
    manifest = json.loads((workspace / "manifest.json").read_text())
    assert set(manifest["modules"]) == {"N0", PREFIX, QUERY, DIFFERENCE}
    assert (workspace / "nodes/N0.py").read_text() == recursive_fixture["script"][16]["output"]["args"]["source"]
    assert "sys.stdout.write(solve(sys.stdin.read()))" in (workspace / "entry.py").read_text()
    assert set(json.loads((runtime.run_dir / f"workspaces/{QUERY}/v1/manifest.json").read_text())["modules"]) == {QUERY, DIFFERENCE}
    assert len((runtime.run_dir / "consumptions.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("accept", [False, True])
def test_failed_child_requires_explicit_unverified_consumption(make_f3_runtime, recursive_fixture, accept):
    data = bad_prefix(recursive_fixture)
    data["script"][5]["output"]["args"]["accept_unverified"] = accept
    runtime = make_f3_runtime(data)
    advance(runtime, "N0", "RESOLVE_CHILD")
    assert runtime.build_observation()["returned_child"]["result"]["functional_status"] == "FAIL"
    event = runtime.step()
    if not accept:
        assert event.host_result.code == "UNVERIFIED_CONSUMPTION"
        assert not runtime.context.consumptions and runtime.node.phase == Phase.RESOLVE_CHILD
    else:
        assert event.host_result.code == "CHILD_CONSUMED"
        assert runtime.context.consumptions[PREFIX].known_failed_consumed is True
        assert runtime.run()["functional_status"] == "FAIL"


def test_unknown_child_can_be_explicitly_consumed_without_inventing_local_pass(make_f3_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    data["suites"] = [s for s in data["suites"] if s.get("contract_key") != "prefix"]
    data["script"][3]["output"]["args"]["suite_id"] = "NO_APPLICABLE_SUITE"
    data["script"][5]["output"]["args"]["accept_unverified"] = True
    result = make_f3_runtime(data).run()
    assert result["functional_status"] == "PASS"
    assert result["node_results"][PREFIX]["functional_status"] == "UNKNOWN"


def test_infrastructure_error_cannot_be_consumed_and_unstarted_children_are_not_abstentions(make_f3_runtime, recursive_fixture, tmp_path):
    data = deepcopy(recursive_fixture)
    consume = deepcopy(data["script"][5])
    consume["output"]["args"]["accept_unverified"] = True
    data["script"] = data["script"][:4] + [consume, abandon("N0", "RESOLVE_CHILD")]
    runtime = make_f3_runtime(data)
    runtime.verifier.sandbox = Sandbox(bwrap=str(tmp_path / "missing-bwrap"))
    advance(runtime, "N0", "RESOLVE_CHILD")
    assert runtime.step().host_result.code == "CHILD_INFRASTRUCTURE_ERROR"
    result = runtime.run()
    assert result["node_results"][PREFIX]["functional_status"] == "ERROR"
    assert result["node_results"][QUERY]["termination_reason"] == "NOT_FINISHED"


def test_reject_preserves_delegation_even_if_parent_reimplements_correct_output(make_f3_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    data["script"][5] = step("N0", "RESOLVE_CHILD", "reject", child_id=PREFIX, artifact_version=1,
                              reason_code="OTHER", evidence_ids=["E0001"])
    write = data["script"][16]["output"]["args"]
    write["declared_dependencies"] = write["declared_dependencies"][1:]
    write["source"] = ("import json\nfrom nodes.N0_p1_query import range_sums as query\n"
                       "def solve(text):\n    request = json.loads(text)\n    sums = [0]\n"
                       "    for x in request['values']: sums.append(sums[-1] + x)\n"
                       "    return '\\n'.join(map(str, query(sums, request['ranges'])))\n")
    runtime = make_f3_runtime(data)
    advance(runtime, "N0", "RESOLVE_CHILD")
    original = runtime.node.incoming_contract.model_dump()
    assert runtime.step().host_result.code == "CHILD_REJECTED"
    assert runtime.node.incoming_contract.model_dump() == original
    assert runtime.build_observation()["unfulfilled_child_ids"] == [PREFIX]
    result = runtime.run()
    assert result["functional_status"] == "PASS" and result["responsibility_status"] == "VIOLATION"
    assert result["node_result"]["responsibility_details"]["unfulfilled_obligation_ids"] == ["O_RANGES"]
    assert runtime.build_observation()["pending_obligations"] == ["O_RANGES"]


def test_repair_request_reactivates_failed_child_without_granting_source_edit(make_f3_runtime, recursive_fixture):
    data = bad_prefix(recursive_fixture)
    request = step("N0", "RESOLVE_CHILD", "request_child_repair", child_id=PREFIX, artifact_version=1,
                   evidence_ids=["E0001"], instruction="保留原契约，修复前缀和错误")
    data["script"] = data["script"][:5] + [request, step(PREFIX,"REVIEW","finish",artifact_version=1),
        request, abandon("N0", "RESOLVE_CHILD")]
    runtime = make_f3_runtime(data)
    advance(runtime, "N0", "RESOLVE_CHILD")
    child = runtime.contexts[PREFIX]
    before = child.node.incoming_contract.model_dump(), child.artifact, child.node.last_evidence
    assert runtime.step().host_result.code == "CHILD_REPAIR_REQUESTED"
    assert runtime.budget.policy_calls == 6
    runtime.step() # Host re-enters the existing child; no extra policy call or node.
    observation = runtime.build_observation()
    assert runtime.active_node_id == PREFIX and observation["phase"] == "REVIEW"
    assert observation["child_repair_request"]["instruction"] == "保留原契约，修复前缀和错误"
    assert observation["visible_evidence"][0]["status"] == "FAIL"
    assert "write_code" not in observation["allowed_actions"] and "repair" not in observation["allowed_actions"]
    assert (child.node.incoming_contract.model_dump(), child.artifact, child.node.last_evidence) == before
    advance(runtime, "N0", "RESOLVE_CHILD")
    assert runtime.step().host_result.code == "CHILD_REPAIR_BUDGET_EXHAUSTED"
    assert runtime.run()["stop_reason"] == "ABSTAINED"
    assert runtime.contexts["N0"].repair_requests[PREFIX] == 1 and len(runtime.contexts) == 3


@pytest.mark.parametrize("limits", [Limits(max_depth=0), Limits(max_nodes=2), Limits(max_children=1)])
def test_invalid_split_cannot_partially_allocate_children(make_f3_runtime, recursive_fixture, limits):
    data = deepcopy(recursive_fixture)
    data["script"] = [data["script"][0], abandon("N0", "NEW")]
    runtime = make_f3_runtime(data, limits=limits)
    original = runtime.node.model_dump()
    event = runtime.step()
    assert event.host_result.code == "STRUCTURE_BUDGET_EXHAUSTED"
    assert runtime.node.model_dump() == original and len(runtime.contexts) == 1 and runtime.plan is None
    assert runtime.run()["stop_reason"] == "ABSTAINED"


def test_depth_limit_keeps_atomic_available_and_budget_is_shared(make_f3_runtime):
    runtime = make_f3_runtime(limits=Limits(max_depth=1))
    advance(runtime, QUERY, "NEW")
    observation = runtime.build_observation()
    assert observation["allowed_plan_modes"] == ["ATOMIC"] and observation["structure_budget_limited"]
    assert "abstain" in observation["allowed_actions"]
    assert runtime.step().host_result.code == "STRUCTURE_BUDGET_EXHAUSTED"
    assert DIFFERENCE not in runtime.contexts
    limited = make_f3_runtime(limits=Limits(max_policy_calls=3))
    result = limited.run()
    assert result["stop_reason"] == "BUDGET_EXHAUSTED" and result["usage"]["policy_calls"] == 3
    assert result["usage"]["verify_calls"] == 0
    assert all(c.node.phase == Phase.TERMINAL for c in limited.contexts.values())


def test_old_version_consumption_rejected_and_immutable_versions_retained(make_f3_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    replacement = deepcopy(data["script"][5])
    replacement["output"]["args"].update(artifact_version=2, accept_unverified=True)
    data["script"].insert(6, replacement)
    runtime = make_f3_runtime(data)
    advance(runtime, "N0", "RESOLVE_CHILD")
    child = runtime.contexts[PREFIX]
    old = child.artifact
    source = (runtime.run_dir / old.module_path).read_text()
    child.artifact = runtime.store.save_next(old, child.node.incoming_contract, source)
    child.node = child.node.model_copy(update={"artifact_ref": child.artifact.module_path})
    assert runtime.step().host_result.code == "ARTIFACT_VERSION_MISMATCH"
    assert runtime.step().host_result.code == "CHILD_CONSUMED"
    assert runtime.context.consumptions[PREFIX].artifact_version == 2
    assert (runtime.run_dir / old.module_path).read_text() == source
    assert runtime.store.load(PREFIX, 1) == old and runtime.store.load(PREFIX, 2) == child.artifact


def test_grandchild_version_change_invalidates_root_evidence(make_f3_runtime):
    runtime = make_f3_runtime()
    advance(runtime, "N0", "REVIEW")
    evidence = runtime.node.last_evidence
    grandchild = runtime.contexts[DIFFERENCE]
    grandchild.artifact = runtime.store.save_next(grandchild.artifact, grandchild.node.incoming_contract,
                                                "def range_sum(prefix, left, right):\n    return prefix[right] - prefix[left]\n")
    grandchild.node = grandchild.node.model_copy(update={"artifact_ref": grandchild.artifact.module_path})
    assert not runtime.verifier.is_current(evidence, runtime.run_dir, runtime.task.root_id, runtime.node.incoming_contract, runtime.artifact)
    assert runtime.step().host_result.code == "STALE_OR_MISSING_EVIDENCE"
    assert runtime.node.last_evidence == evidence
    assert (runtime.run_dir / f"evidence/{evidence.evidence_id}.json").exists()


def test_stale_transitive_dependency_rejects_write_without_saving_partial_artifact(make_f3_runtime):
    runtime = make_f3_runtime()
    advance(runtime, "N0", "IMPLEMENT")
    grandchild = runtime.contexts[DIFFERENCE]
    runtime.store.save_next(grandchild.artifact, grandchild.node.incoming_contract,
                            "def range_sum(prefix, left, right): return prefix[right] - prefix[left]")
    assert runtime.step().host_result.code == "STALE_DEPENDENCY"
    assert runtime.artifact is None and not (runtime.run_dir / "artifacts/N0/v1.py").exists()


@pytest.mark.parametrize("change,code", [
    ({"artifact_hash": "a" * 64}, "ARTIFACT_HASH_MISMATCH"),
    ({"child_id": QUERY}, "CHILD_NOT_PENDING"),
    ({"evidence_ids": ["invented"]}, "UNKNOWN_EVIDENCE"),
    ({"evidence_ids": []}, "MISSING_CHILD_EVIDENCE"),
])
def test_invalid_consumption_preserves_pending_child(make_f3_runtime, recursive_fixture, change, code):
    data = deepcopy(recursive_fixture)
    data["script"][5]["output"]["args"].update(change)
    runtime = make_f3_runtime(data)
    advance(runtime, "N0", "RESOLVE_CHILD")
    before = runtime.node.model_dump()
    assert runtime.step().host_result.code == code
    assert runtime.node.model_dump() == before and not runtime.context.consumptions


@pytest.mark.parametrize("source,code", [
    ("from nodes.N0_p1_query import range_sums as sibling\ndef prefix_sums(xs): return [0]", "UNAPPROVED_IMPORT"),
    ("import nodes.N0_p1_query as sibling\ndef prefix_sums(xs): return [0]", "DEPENDENCY_IMPORT_MISMATCH"),
])
def test_child_cannot_import_unconsumed_sibling(make_f3_runtime, recursive_fixture, source, code):
    data = deepcopy(recursive_fixture)
    data["script"][2]["output"]["args"]["source"] = source
    runtime = make_f3_runtime(data)
    advance(runtime, PREFIX, "IMPLEMENT")
    assert runtime.step().host_result.code == code and runtime.artifact is None


@pytest.mark.parametrize("linked,called", [(False, False), (True, False), (True, True)])
def test_selected_linked_called_are_distinct_and_calls_do_not_prove_consumption(make_f3_runtime, recursive_fixture, linked, called):
    data = deepcopy(recursive_fixture)
    imports = ("from nodes.N0_p1_prefix import prefix_sums as prefix\nfrom nodes.N0_p1_query import range_sums as query\n") if linked else ""
    discarded_calls = "    query(prefix(request['values']), request['ranges'])\n" if called else ""
    data["script"][16]["output"]["args"]["source"] = ("import json\n" + imports +
        "def solve(text):\n    request = json.loads(text)\n" + discarded_calls +
        "    return '\\n'.join(str(sum(request['values'][left:right])) for left, right in request['ranges'])\n")
    runtime = make_f3_runtime(data)
    assert runtime.run()["functional_status"] == "PASS"
    for item in runtime.node.last_evidence.details["dependency_usage"]:
        assert item["selected"] and item["linked"] is linked and item["called"] is called
        assert item["dependency_supported"] == "UNKNOWN"


def test_dynamic_import_cannot_bypass_direct_child_boundary(make_f3_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    data["script"][16]["output"]["args"]["source"] = (
        "import importlib\nfrom nodes.N0_p1_query import range_sums as query\n"
        "def solve(text):\n    importlib.import_module('nodes.N0_p1_query_p1_difference')\n    return ''\n")
    runtime = make_f3_runtime(data)
    assert runtime.run()["functional_status"] == "FAIL"
    case = runtime.node.last_evidence.details["cases"][0]
    assert case["reason"] == "CANDIDATE_EXCEPTION"
    assert "explicitly consumed direct children" in case["execution"]["stdout"]
