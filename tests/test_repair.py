from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from audittrainer.runtime import Limits
from audittrainer.schema import Phase
from audittrainer.verifier import OracleCase


PREFIX, QUERY, DIFFERENCE = "N0_p1_prefix", "N0_p1_query", "N0_p1_query_p1_difference"


def advance(runtime, node_id, phase):
    for _ in range(100):
        if runtime.active_node_id == node_id and runtime.node.phase.value == phase:
            return
        assert runtime.stop_reason is None, runtime.stop_reason
        runtime.step()
    pytest.fail("Requested state not reached")


def action(node, phase, name, **args):
    return dict(node_id=node, phase=phase, output=dict(action=name, args=args))


def abstain(node="N0", phase="REVIEW"):
    return action(node, phase, "abstain", reason_code="INSUFFICIENT_EVIDENCE", evidence_ids=[], suggested_change=None)


def test_parent_only_repair_replays_and_commits_without_changing_children(make_f4_runtime, repair_fixture):
    runtime = make_f4_runtime()
    advance(runtime, "N0", "REVIEW")
    original = runtime.artifact
    children = {n:c.artifact for n,c in runtime.contexts.items() if n != "N0"}
    assert runtime.node.last_evidence.status == "FAIL"
    assert runtime.build_observation()["mode"] == "LOCATE"
    assert runtime.step().host_result.code == "REPAIR_AUTHORIZED"
    assert runtime.build_observation()["mode"] == "REPAIR"
    ticket = runtime.repair_manager.ticket
    assert ticket.editable_files == ["nodes/N0.py"] and ticket.revalidate_nodes == ["N0"]
    assert set(ticket.frozen_hashes) == {f"nodes/{n}.py" for n in children}
    assert len(runtime.repair_manager.indexes()["module_owner"]) == 4
    assert runtime.step().host_result.code == "CODE_REPAIR_ACCEPTED"
    assert runtime.node.phase == Phase.REVIEW and runtime.artifact.version == 2
    assert runtime.artifact.source_hash != original.source_hash
    assert {n:runtime.contexts[n].artifact for n in children} == children
    assert runtime.store.load("N0", 1) == original
    calls = runtime.budget.verify_calls
    result = runtime.run()
    assert result["functional_status"] == "PASS" and runtime.budget.verify_calls == calls == 6
    assert result["usage"]["patch_attempts"] == 1 and result["usage"]["executed_cases"] == 14
    assert result["repair_history"][0]["root_checked"] is True
    assert (runtime.run_dir / "repairs/T1/snapshot/artifacts/N0/v1.py").read_text() == repair_fixture["script"][16]["output"]["args"]["source"]


def test_bad_patch_rolls_back_program_evidence_and_bindings_but_keeps_cost(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"][19]["output"]["args"]["replacements"][0]["source"] = data["script"][16]["output"]["args"]["source"]
    data["script"][20]["output"]["args"]["artifact_version"] = 1
    runtime = make_f4_runtime(data)
    advance(runtime, "N0", "REVIEW")
    artifact, evidence, bindings = runtime.artifact, runtime.node.last_evidence, deepcopy(runtime.context.consumptions)
    result = runtime.run()
    assert result["functional_status"] == "FAIL"
    assert runtime.artifact == artifact and runtime.node.last_evidence == evidence
    assert runtime.context.consumptions == bindings and runtime.node.phase == Phase.TERMINAL
    assert not (runtime.run_dir / "artifacts/N0/v2.py").exists()
    assert (runtime.run_dir / "repairs/T1/attempt_1/artifacts/N0/v2.py").exists()
    assert runtime.budget.patch_attempts == 1 and runtime.budget.verify_calls == 6
    assert result["repair_history"][0]["status"] == "ROLLED_BACK"


def test_wrong_location_with_explicit_candidate_rebind_still_fails_root(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"][18]["output"]["args"].update(target_type="NODE", target_id=PREFIX, boundary_kind="OWNER_MODULE")
    source = data["script"][2]["output"]["args"]["source"]
    data["script"][19]["output"]["args"].update(
        replacements=[dict(file=f"nodes/{PREFIX}.py",base_hash=hashlib.sha256(source.encode()).hexdigest(),source=source)],
        dependency_updates=[dict(consumer_node="N0",child_id=PREFIX,artifact_ref=dict(kind="CANDIDATE",node_id=PREFIX),alias="prefix")])
    data["script"][20]["output"]["args"]["artifact_version"] = 1
    runtime = make_f4_runtime(data)
    result = runtime.run()
    assert result["functional_status"] == "FAIL"
    attempt = runtime.run_dir / "repairs/T1/attempt_1"
    assert (attempt / f"artifacts/{PREFIX}/v2.py").exists()
    candidate = json.loads((attempt / "artifacts/N0/v2.json").read_text())
    assert candidate["dependency_bindings"][0]["artifact_version"] == 2
    assert runtime.contexts[PREFIX].artifact.version == runtime.artifact.version == 1
    assert runtime.context.consumptions[PREFIX].artifact_version == 1
    assert result["repair_history"][0]["reason"] == "VISIBLE_CHECK_NOT_PASS"


@pytest.mark.parametrize("change,code", [
    ({"file":"nodes/N0_p1_prefix.py"}, "OUTSIDE_REPAIR_BOUNDARY"),
    ({"file":"../outside.py"}, "OUTSIDE_REPAIR_BOUNDARY"),
    ({"base_hash":"a"*64}, "BASE_HASH_MISMATCH"),
])
def test_unauthorized_patch_is_rejected_before_candidate_write(make_f4_runtime, repair_fixture, change, code):
    data = deepcopy(repair_fixture)
    data["script"][19]["output"]["args"]["replacements"][0].update(change)
    data["script"][20] = abstain(phase="REPAIR_AUTHORIZED")
    runtime = make_f4_runtime(data)
    advance(runtime, "N0", "REPAIR_AUTHORIZED")
    before = runtime.node.model_dump()
    assert runtime.step().host_result.code == code
    assert runtime.node.model_dump() == before and runtime.budget.patch_attempts == 1
    assert not (runtime.run_dir / "repairs/T1/attempt_1").exists()
    assert runtime.run()["stop_reason"] == "ABSTAINED"


def test_rebind_does_not_grant_unrelated_dependency_access(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"][19]["output"]["args"]["dependency_updates"] = [
        dict(consumer_node="N0",child_id=DIFFERENCE,artifact_ref=dict(kind="EXISTING",version=1,hash="a"*64),alias="diff")]
    runtime = make_f4_runtime(data)
    advance(runtime, "N0", "REPAIR_AUTHORIZED")
    assert runtime.step().host_result.code == "UNAUTHORIZED_REBIND"
    assert runtime.artifact.version == 1


def test_all_fixed_visible_suites_are_checked_not_just_the_previously_selected_suite(make_f4_runtime):
    runtime = make_f4_runtime()
    suite = runtime.verifier.suites["FX02_ROOT_VISIBLE"]
    runtime.verifier.suites["ROOT_EXTRA"] = suite.model_copy(update={
        "suite_id":"ROOT_EXTRA", "cases":[OracleCase(case_id="extra",args=['{"values": [9, -4], "ranges": [[0, 1], [0, 2]]}'],expected="9\n5")]})
    result = runtime.run()
    assert result["functional_status"] == "PASS"
    assert {e["suite_id"] for e in result["repair_history"][0]["evidence"]} == {"FX02_ROOT_VISIBLE","ROOT_EXTRA"}
    assert runtime.budget.verify_calls == 7


def test_previously_passing_suite_cannot_regress(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    # Pass every primary-suite input, but regress a distinct previously passing suite.
    patch = data["script"][19]["output"]["args"]["replacements"][0]
    patch["source"] = patch["source"].replace('    sums = prefix', '    if request["values"] == [9, -4]: return "wrong"\n    sums = prefix')
    data["script"][20]["output"]["args"]["artifact_version"] = 1
    runtime = make_f4_runtime(data)
    suite = runtime.verifier.suites["FX02_ROOT_VISIBLE"]
    runtime.verifier.suites["ROOT_REGRESSION"] = suite.model_copy(update={
        "suite_id":"ROOT_REGRESSION", "cases":[OracleCase(case_id="single",args=['{"values":[9,-4],"ranges":[[0,2]]}'],expected="5")]})
    advance(runtime,"N0","REVIEW")
    manager = runtime.repair_manager
    baseline = manager._check("N0","ROOT_REGRESSION",update=False)
    assert baseline.status == "PASS"
    result = runtime.run()
    assert result["functional_status"] == "FAIL" and result["repair_history"][0]["status"] == "ROLLED_BACK"
    assert {e["suite_id"]:e["status"] for e in result["repair_history"][0]["evidence"]} == {"FX02_ROOT_VISIBLE":"PASS", "ROOT_REGRESSION":"FAIL"}


@pytest.mark.parametrize("change", ["config", "frozen_source"])
def test_snapshot_changes_invalidate_ticket(make_f4_runtime, change):
    runtime = make_f4_runtime()
    advance(runtime,"N0","REPAIR_AUTHORIZED")
    if change == "config":
        runtime.verifier.sandbox.limits = runtime.verifier.sandbox.limits.model_copy(update={"memory_mb":512})
    else:
        (runtime.run_dir / runtime.contexts[PREFIX].artifact.module_path).write_text("def prefix_sums(xs): return []")
    assert runtime.step().host_result.code == "STALE_TICKET"
    assert not (runtime.run_dir / "artifacts/N0/v2.py").exists()


def test_unstable_no_change_replay_does_not_issue_ticket(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"][16]["output"]["args"]["source"] = "import time\ndef solve(text): return str(time.time_ns())"
    runtime = make_f4_runtime(data)
    advance(runtime,"N0","REVIEW")
    assert runtime.step().host_result.code == "UNSTABLE_REPLAY"
    assert runtime.repair_manager.ticket is None and runtime.budget.patch_attempts == 0
    assert runtime.budget.verify_calls == 5


def test_patch_attempts_do_not_reset_when_a_new_ticket_is_issued(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    bad = deepcopy(data["script"][19])
    bad["output"]["args"]["replacements"][0]["source"] = data["script"][16]["output"]["args"]["source"]
    data["script"] = data["script"][:19] + [bad, deepcopy(data["script"][18]), abstain()]
    runtime = make_f4_runtime(data,limits=Limits(max_patch_attempts=1))
    advance(runtime,"N0","REPAIR_AUTHORIZED")
    assert runtime.step().host_result.code == "REPAIR_ROLLED_BACK"
    assert runtime.step().host_result.code == "PATCH_BUDGET_EXHAUSTED"
    assert runtime.run()["stop_reason"] == "ABSTAINED" and runtime.budget.patch_attempts == 1


def test_parent_expansion_needs_failed_local_attempt(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"] = data["script"][:4] + [action(PREFIX,"REVIEW","localize",
        target_type="CONTRACT",target_id="N0",evidence_ids=["E0001"],boundary_kind="PLAN_SUBTREE",rationale="尝试扩大边界")]
    runtime = make_f4_runtime(data)
    advance(runtime,PREFIX,"REVIEW")
    assert runtime.step().host_result.code == "UNSUPPORTED_EXPANSION"
    assert runtime.repair_manager.ticket is None


@pytest.mark.parametrize("succeeds", [True, False])
def test_plan_revision_rebuilds_subtree_and_keeps_incoming_contract(make_f4_runtime, succeeds):
    path = Path(__file__).resolve().parents[1] / "data/fixtures/fx04_plan_revision.json"
    data = json.loads(path.read_text())
    if not succeeds:
        root_write = next(s for s in data["script"][20:] if s["node_id"]=="N0" and s["output"]["action"]=="write_code")
        root_write["output"]["args"]["source"] = root_write["output"]["args"]["source"].replace('query(sums, request["ranges"])','query(sums, request["ranges"])[::-1]')
        data["script"].append(action("N0","REVIEW","finish",artifact_version=1))
    runtime = make_f4_runtime(data)
    incoming = runtime.node.incoming_contract
    result = runtime.run()
    assert runtime.node.incoming_contract == incoming
    assert runtime.budget.patch_attempts == runtime.budget.plan_revisions == 1
    assert runtime.budget.nodes_created == 7
    assert (runtime.run_dir / "plans/N0.v1.json").exists()
    assert (runtime.run_dir / f"artifacts/{PREFIX}/v1.py").exists()
    if succeeds:
        assert result["functional_status"] == "PASS" and runtime.artifact.version == 2
        assert set(runtime.context.children) == {"N0_p2_prefix","N0_p2_query"}
        assert runtime.contexts["N0_p2_prefix"].node.last_evidence.evidence_id == "E0006"
        assert runtime.plan.version == 2 and runtime.budget.verify_calls == 9
        assert result["repair_history"][0]["status"] == "PLAN_REVISION_ACCEPTED"
        assert (runtime.run_dir / "plans/N0.v2.json").exists()
    else:
        assert result["functional_status"] == "FAIL" and runtime.artifact.version == 1
        assert set(runtime.context.children) == {PREFIX,QUERY}
        assert runtime.node.last_evidence.evidence_id == "E0004"
        assert not (runtime.run_dir / "plans/N0.v2.json").exists()
        assert (runtime.run_dir / "repairs/T1/attempt_1/plans/N0.v2.json").exists()


def test_plan_cannot_supply_a_weaker_incoming_contract(make_f4_runtime):
    data = json.loads((Path(__file__).resolve().parents[1]/"data/fixtures/fx04_plan_revision.json").read_text())
    data["script"][19]["output"]["args"]["incoming_contract"] = {"ensures":[]}
    runtime = make_f4_runtime(data)
    advance(runtime,"N0","REPAIR_AUTHORIZED")
    original = runtime.node.incoming_contract
    assert runtime.step().host_result.code == "INVALID_ACTION_JSON"
    assert runtime.node.incoming_contract == original and runtime.plan.version == 1


def test_budget_exhaustion_during_reconstruction_rolls_back_and_keeps_global_usage(make_f4_runtime):
    data = json.loads((Path(__file__).resolve().parents[1]/"data/fixtures/fx04_plan_revision.json").read_text())
    runtime = make_f4_runtime(data,limits=Limits(max_policy_calls=22))
    result = runtime.run()
    assert result["stop_reason"] == "BUDGET_EXHAUSTED"
    assert result["run_dir"] == str(runtime.repair_manager.base_dir)
    assert runtime.plan.version == 1 and runtime.artifact.version == 1
    assert runtime.budget.policy_calls == 22 and runtime.budget.plan_revisions == 1
    assert result["repair_history"][0]["status"] == "ROLLED_BACK"


def test_requested_child_repair_returns_new_version_before_parent_consumes(make_f4_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    good = data["script"][2]["output"]["args"]["source"]
    bad = "def prefix_sums(xs): return [0]"
    data["script"][2]["output"]["args"]["source"] = bad
    request = action("N0","RESOLVE_CHILD","request_child_repair",child_id=PREFIX,artifact_version=1,
                     evidence_ids=["E0001"],instruction="修复前缀和实现，保留契约")
    localize = action(PREFIX,"REVIEW","localize",target_type="NODE",target_id=PREFIX,evidence_ids=["E0001"],
                      boundary_kind="OWNER_MODULE",rationale="前缀和局部验证失败")
    patch = action(PREFIX,"REPAIR_AUTHORIZED","repair",ticket_id="T1",
                   replacements=[dict(file=f"nodes/{PREFIX}.py",base_hash=hashlib.sha256(bad.encode()).hexdigest(),source=good)],dependency_updates=[])
    consume = deepcopy(data["script"][5])
    consume["output"]["args"].update(artifact_version=2,evidence_ids=["E0003"])
    tail = deepcopy(data["script"][6:])
    for s in tail:
        args = s["output"]["args"]
        if s["output"]["action"] == "consume":
            args["evidence_ids"] = [f"E{int(e[1:])+2:04d}" for e in args["evidence_ids"]]
        if s["node_id"] == "N0" and s["output"]["action"] == "write_code":
            args["declared_dependencies"][0]["artifact_version"] = 2
    data["script"] = data["script"][:5] + [request,localize,patch,action(PREFIX,"REVIEW","finish",artifact_version=2),consume] + tail
    runtime = make_f4_runtime(data)
    result = runtime.run()
    assert result["functional_status"] == "PASS"
    assert result["repair_history"][0]["root_checked"] is False
    assert runtime.contexts[PREFIX].artifact.version == runtime.context.consumptions[PREFIX].artifact_version == 2
    assert runtime.contexts[PREFIX].node.incoming_contract.version == 1
    assert runtime.budget.patch_attempts == 1 and runtime.budget.verify_calls == 6


def test_unknown_oracle_cannot_commit_a_patch(make_f4_runtime, fixture_data):
    data = deepcopy(fixture_data)
    data["suites"][0]["cases"] = []
    source = data["script"][1]["output"]["args"]["source"]
    data["script"] += [data["completion_script"][0],
        action("N0","REVIEW","localize",target_type="NODE",target_id="N0",evidence_ids=["E0001"],boundary_kind="OWNER_MODULE",rationale="证据不足的候选"),
        action("N0","REPAIR_AUTHORIZED","repair",ticket_id="T1",replacements=[dict(file="nodes/N0.py",base_hash=hashlib.sha256(source.encode()).hexdigest(),source=source)],dependency_updates=[]),
        data["completion_script"][1]]
    runtime = make_f4_runtime(data)
    result = runtime.run()
    assert result["functional_status"] == "UNKNOWN" and runtime.artifact.version == 1
    assert result["repair_history"][0]["status"] == "ROLLED_BACK"
    assert runtime.budget.executed_cases == 0 and runtime.budget.verify_calls == 3


def test_automatic_replay_cannot_exceed_root_verify_budget(make_f4_runtime, repair_fixture):
    data = deepcopy(repair_fixture)
    data["script"][20] = abstain()
    runtime = make_f4_runtime(data,limits=Limits(max_verify_calls=5))
    result = runtime.run()
    assert result["stop_reason"] == "ABSTAINED" and runtime.artifact.version == 1
    assert runtime.budget.verify_calls == 5 and runtime.budget.patch_attempts == 1
    assert "VERIFY_BUDGET_EXHAUSTED" in result["repair_history"][0]["reason"]


def test_one_evidenced_parent_expansion_rebuilds_parent_plan(make_f4_runtime, recursive_fixture):
    data = deepcopy(recursive_fixture)
    bad = "def prefix_sums(xs): return [0]"
    data["script"][2]["output"]["args"]["source"] = bad
    source = "import json\ndef solve(text):\n    request = json.loads(text)\n    return '\\n'.join(str(sum(request['values'][l:r])) for l,r in request['ranges'])"
    data["script"] = data["script"][:4] + [
        action(PREFIX,"REVIEW","localize",target_type="NODE",target_id=PREFIX,evidence_ids=["E0001"],boundary_kind="OWNER_MODULE",rationale="先修当前局部失败"),
        action(PREFIX,"REPAIR_AUTHORIZED","repair",ticket_id="T1",replacements=[dict(file=f"nodes/{PREFIX}.py",base_hash=hashlib.sha256(bad.encode()).hexdigest(),source=bad)],dependency_updates=[]),
        action(PREFIX,"REVIEW","localize",target_type="CONTRACT",target_id="N0",evidence_ids=["E0001"],boundary_kind="PLAN_SUBTREE",rationale="局部尝试失败后扩大至父计划"),
        action("N0","REPAIR_AUTHORIZED","plan",base_plan_version=1,ticket_id="T2",mode="ATOMIC",children=[],obligation_map={"O_RANGES":["SELF"]},retained_responsibilities=["complete_range_sums"],dataflow=[]),
        action("N0","IMPLEMENT","write_code",contract_version=1,source=source,declared_dependencies=[]),
        action("N0","READY_VERIFY","verify",target_ref="artifacts/N0/v1.py",suite_id="FX02_ROOT_VISIBLE"),
        action("N0","REVIEW","finish",artifact_version=1)]
    runtime = make_f4_runtime(data)
    result = runtime.run()
    assert result["functional_status"] == "PASS"
    assert runtime.repair_manager.expansions == 1 and runtime.budget.patch_attempts == 2
    assert runtime.plan.version == 2 and runtime.node.incoming_contract.version == 1
