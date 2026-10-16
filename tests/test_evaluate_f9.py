"""F9 checks frozen programs, private isolation, root denominators and APPS scope."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from audittrainer.apps import compare_stdio, convert_record, import_apps
from audittrainer.artifacts import ArtifactStore
from audittrainer.dataset import file_hash, load_root_manifest, read_jsonl
from audittrainer.evaluate import aggregate, capture_program, evaluate, load_task, score_program
from audittrainer.policy import ScriptedPolicy, ScriptStep
from audittrainer.schema import Artifact, Contract, ContractSpec, RootTask
from audittrainer.runtime import Limits, RepairRuntime
from audittrainer.verifier import canonical

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, **overrides):
    c = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    c.update(output_dir=str(tmp_path / "evaluation"), split="SYN_FIXTURE", limit=1, **overrides)
    return c


def test_first_snapshot_survives_real_bounded_repair(tmp_path):
    c = config(tmp_path)
    c.update(max_policy_calls=64, limit=6)
    result = evaluate(c, policy_name="script")
    rows = read_jsonl(result["root_results"])
    assert len(rows) == 6
    repaired = next(r for r in rows if r["root_id"] == "fx03_bounded_repair")
    assert repaired["first_scores"]["private"]["status"] == "FAIL"
    assert repaired["final_scores"]["private"]["status"] == "PASS"
    first, final = repaired["first_program"], repaired["final_program"]
    assert first["artifact"]["version"] == 1 and final["artifact"]["version"] == 2
    assert first["artifact"]["source_hash"] != final["artifact"]["source_hash"]
    assert first["captured_usage"]["patch_attempts"] == 0
    assert repaired["online_cost"]["patch_attempts"] == 1
    # Every first transitive bundle remains readable after later repairs/replanning.
    for row in rows:
        snap = row["first_program"]
        if snap:
            bundle = ArtifactStore(Path(snap["path"])).bundle(Artifact.model_validate(snap["artifact"]))
            assert bundle.manifest() == snap["bundle"]
    assert all(set(row) == set(rows[0]) for row in rows)


def test_private_failure_cannot_trigger_online_repair_or_leak(tmp_path, fixture_data):
    data = deepcopy(fixture_data)
    data["script"][1]["output"]["args"]["source"] = "def solve(text): return '10'\n"
    data["suites"][0]["cases"] = [{"case_id": "visible", "args": ["7 3"], "expected": "10"}]
    private = deepcopy(data["suites"][0])
    private.update(suite_id="PRIVATE_FIXED", visibility="PRIVATE",
                   cases=[{"case_id": "private", "args": ["98123765 0"], "expected": "98123765"}])
    data["suites"].append(private)
    path = tmp_path / "task.json"; path.write_text(json.dumps(data))
    row = load_root_manifest(ROOT / "data/manifests/roots.jsonl")["fx01_integer_sum"]
    row.update(fixture_path=str(path), source_hash=file_hash(path))
    manifest = tmp_path / "roots.jsonl"; manifest.write_text(canonical(row) + "\n")
    c = config(tmp_path); c.update(root_manifest=str(manifest), test_manifest=None)
    report = evaluate(c, policy_name="script")
    result = read_jsonl(report["root_results"])[0]
    assert result["online_functional_status"] == "PASS"
    assert result["final_scores"]["private"]["status"] == "FAIL"
    assert result["stop_reason"] == "FINISHED" and result["online_cost"]["policy_calls"] == 4
    assert result["online_cost"]["patch_attempts"] == 0
    trace = (Path(result["run_dir"]) / "trace.jsonl").read_text()
    assert "98123765" not in trace and "PRIVATE_FIXED" not in trace
    assert report["aggregate"]["roots"] == 1 and report["aggregate"]["BudgetedSuccess"] == 0
    assert report["aggregate"]["cost_per_budgeted_success"] is None
    assert result["scoring_cost"]["executed_cases"] == 4


def test_model_load_errors_retain_all_roots_and_uniform_fields(tmp_path):
    c = config(tmp_path); c["limit"] = 3
    def fail(*args, **kwargs):
        raise RuntimeError("test unavailable model")
    report = evaluate(c, policy_name="base", policy_factory=fail)
    rows = read_jsonl(report["root_results"])
    assert len(rows) == report["aggregate"]["roots"] == 3
    assert {r["stop_reason"] for r in rows} == {"MODEL_LOAD_ERROR"}
    assert all(r["error"] and r["first_program"] is None for r in rows)
    assert report["aggregate"]["online_cost"]["nodes_created"] == 0


@pytest.mark.parametrize("policy_name", ["base", "sft"])
def test_policy_reports_share_fields_and_budget_failures_stay_in_denominator(tmp_path, fixture_data, policy_name):
    steps = fixture_data["script"]
    class Policy(ScriptedPolicy):
        identity = {"backend": "test_scripted_interface"}
        def check_adapter_effect(self, observation):
            return {"status": "PASS", "forward_calls": 3}, {}
    p = Policy([ScriptStep.model_validate_json(json.dumps(s)) for s in steps])
    c = config(tmp_path); c["max_policy_calls"] = 2
    result = evaluate(c, policy_name=policy_name, adapter=Path("test_adapter") if policy_name == "sft" else None,
                      policy_factory=lambda *a, **k: p)
    row = read_jsonl(result["root_results"])[0]
    assert row["stop_reason"] == "BUDGET_EXHAUSTED"
    assert row["first_program"] and row["final_program"]
    assert row["online_cost"]["verify_calls"] == 0 and row["scoring_cost"]["verify_calls"] == 4
    assert result["aggregate"]["roots"] == 1
    assert set(row) == set(__import__('audittrainer.evaluate', fromlist=['empty_record']).empty_record(row, 'script'))


def test_missing_task_root_is_reported_then_later_root_still_runs(tmp_path):
    roots = list(load_root_manifest(ROOT / "data/manifests/roots.jsonl").values())[:2]
    roots[0]["fixture_path"] = str(tmp_path / "missing")
    path = tmp_path / "roots.jsonl"; path.write_text(''.join(canonical(r)+'\n' for r in roots))
    c = config(tmp_path); c.update(root_manifest=str(path), limit=2)
    result = evaluate(c, policy_name="script")
    rows = read_jsonl(result["root_results"])
    assert len(rows) == 2 and rows[0]["status"] == "ERROR"
    assert rows[1]["stop_reason"] == "FINISHED"


def apps_record(**changes):
    row = {"problem_id": 12, "question": "Return the input integer plus one.", "url": "https://example.test/problem/12",
           "input_output": json.dumps({"inputs": ["1\n", ["5"]], "outputs": ["2\n", ["6"]]}), "solutions": "must never enter task"}
    row.update(changes)
    return row


def test_apps_import_preserves_sources_and_disjoint_original_cases(tmp_path):
    record = apps_record()
    public, private = convert_record(record, original_split="train", split="DEV_ENGINEERING", visible_cases=1)
    assert public["suites"][0]["cases"][0]["args"] == ["1\n"]
    assert private["suites"][0]["cases"][0]["args"] == ["5"]
    assert 'solutions' not in canonical(public) and 'must never enter task' not in canonical(public)
    inputs = tmp_path / "input.jsonl"
    unsupported = apps_record(problem_id=13, input_output={"fn_name": "add", "inputs": [[1]], "outputs": [2]})
    inputs.write_text(canonical(record)+'\n'+canonical(unsupported)+'\n')
    roots = tmp_path / "roots.jsonl"; roots.write_text('')
    tests = tmp_path / "tests.jsonl"
    report = import_apps(inputs, tmp_path/'import', roots, tests, original_split='train', split='DEV_ENGINEERING', source='test-source')
    assert report['counts'] == {'imported':1,'rejected':1}
    row=read_jsonl(roots)[0]
    assert json.loads(Path(row['raw_path']).read_text()) == record
    assert row['source_task_id']=='12' and row['original_split']=='train' and row['io_type']=='stdin'
    assert report['rejected'][0]['reason']=='UNSUPPORTED_CALL_BASED'


def test_apps_single_case_can_remain_offline_without_fabricated_visible_test():
    record = apps_record(input_output={'inputs':['1'],'outputs':['2']})
    with pytest.raises(ValueError, match='DISJOINT'):
        convert_record(record, original_split='train', split='DEV_ENGINEERING')
    public, private = convert_record(record, original_split='train', split='DEV_ENGINEERING', visible_cases=0)
    assert public['suites']==[] and public['task']['visible_suite_ref'] is None
    assert len(private['suites'][0]['cases'])==1


@pytest.mark.parametrize('original,split', [('test','TRAIN'),('test','DEV_ENGINEERING'),('train','APPS_SEALED')])
def test_apps_original_test_split_cannot_be_consumed_as_training(original, split):
    with pytest.raises(ValueError, match='FORBIDDEN_SPLIT'):
        convert_record(apps_record(), original_split=original, split=split)


@pytest.mark.parametrize('actual,expected,match', [
    (' 2\n','2',True),('0.333333333\n','0.333333334',True),('NO','YES',False),
    ('2 1','1 2',True),('1 1 2','1 2',True),('1.0004 2','1 2',True),('2','3',False),
    (2,'2',False)])
def test_apps_legacy_comparison_is_explicit_including_permissive_fallbacks(actual,expected,match):
    assert compare_stdio(actual,expected) is match


def test_five_registered_development_tasks_and_all_fixture_sources():
    roots=load_root_manifest(ROOT/'data/manifests/roots.jsonl')
    assert len([r for r in roots.values() if r['split']=='DEV_ENGINEERING'])==5
    assert len([r for r in roots.values() if r['split']=='SYN_FIXTURE'])==6
    tests=read_jsonl(ROOT/'data/manifests/tests.jsonl')
    assert set(r['root_id'] for r in tests)==set(roots)


def test_actual_apps_record_executes_through_isolated_stdin_adapter(tmp_path):
    # Independent parity solution for the downloaded Cartesian-walk task. This
    # is a reference interface check, never counted as a model-generated result.
    row=load_root_manifest(ROOT/'data/manifests/roots.jsonl')['apps_train_0001']
    tests={r['root_id']:r for r in read_jsonl(ROOT/'data/manifests/tests.jsonl')}
    framework=yaml.safe_load((ROOT/'configs/framework.yaml').read_text())
    _,task,spec,online,full=load_task(row,tests,framework)
    assert not online.suites
    contract=Contract(**spec.model_dump(),contract_id='C_N0_1',version=1)
    source="""def solve(text):
    values = list(map(int, text.split()))
    answers = []
    for i in range(values[0]):
        n, m, k = values[1 + 3 * i:4 + 3 * i]
        if max(n, m) > k:
            answer = -1
        elif (n - m) % 2:
            answer = k - 1
        elif (k - n) % 2:
            answer = k - 2
        else:
            answer = k
        answers.append(str(answer))
    return '\\n'.join(answers)
"""
    artifact=ArtifactStore(tmp_path).save_initial('N0',contract,source)
    snapshot={'path':str(tmp_path),'artifact':artifact.model_dump(),'contract':contract.model_dump()}
    scores,cost=score_program(snapshot,full,task.root_id)
    assert scores['visible']['status']=='UNKNOWN'
    assert scores['private']['status']=='PASS'
    assert cost['executed_cases']==1
