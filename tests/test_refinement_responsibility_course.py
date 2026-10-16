from pathlib import Path
import json

import yaml

from audittrainer.refinement_responsibility_course import allocation, prepare, task_pack, CourseReference
from audittrainer.refinement_rollout import runtime_for


def test_calendar_summary_keeps_negative_maximum_for_composed_inputs():
    from audittrainer.training_tasks import oracle_b
    assert oracle_b('calendar', [-6, -9], 3) == {'breaks': [], 'max_gap': -3}
    assert oracle_b('calendar', [], 3) == {'breaks': [], 'max_gap': 0}
    assert oracle_b('calendar', [1], 3) == {'breaks': [], 'max_gap': 0}


def test_course_keeps_compositions_disjoint_and_corrects_calendar_tests(tmp_path):
    study = yaml.safe_load(Path('configs/refinement_responsibility_course.yaml').read_text())
    study.update(output_dir=str(tmp_path/'run'), data_dir=str(tmp_path/'data'))
    config, framework, selections = prepare(study)
    assert len(selections) == 24 and framework['runtime']['max_depth'] == 2
    request = json.loads((tmp_path/'run/request.json').read_text())
    assert request['train_composition_families'] == 20
    for parameter in (3, 4):
        path = tmp_path/'data'/f'train_course_calendar_matrix_{parameter:02d}'/'private_tests.json'
        suites = json.loads(path.read_text())['suites']
        assert all(len(s['cases']) == 8 for s in suites)


def test_joint_obligation_has_two_child_owners_and_explicit_dataflow():
    definition, _, _ = task_pack('paths', 'paths', 3, 'joint_obligation')
    plan = allocation(definition['contract'], definition, 'joint_obligation')['args']
    assert set(plan['obligation_map']) == {'PARSE', 'RESULT', 'FORMAT'}
    assert plan['obligation_map']['RESULT'] == ['child:prepare', 'child:aggregate']
    assert plan['retained_responsibilities'] == ['parse', 'format']
    assert plan['dataflow'][0]['to_parameter'] == 'prepared'


def test_recursive_reference_really_runs_grandchild_and_consumes_result(tmp_path):
    study = yaml.safe_load(Path('configs/refinement_responsibility_course.yaml').read_text())
    study.update(output_dir=str(tmp_path/'run'), data_dir=str(tmp_path/'data'))
    config, framework, _ = prepare(study)
    from audittrainer.dataset import load_root_manifest, read_jsonl
    rows = load_root_manifest(config['root_manifest'])
    row = rows['train_course_calendar_matrix_04']
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    definition = json.loads(Path(row['fixture_path']).read_text())
    reference = json.loads(Path(row['reference_path']).read_text())
    runtime, _ = runtime_for(row, tests, framework, CourseReference(definition, reference, 'recursive'), tmp_path/'runtime')
    result = runtime.run()
    assert result['functional_status'] == 'PASS' and result['node_count'] == 3
    child = runtime.contexts['N0_p1_compute']
    assert child.consumptions['N0_p1_compute_p1_prepare'].decision == 'CONSUME'
    assert runtime.contexts['N0'].consumptions['N0_p1_compute'].decision == 'CONSUME'
    assert '_prepare_child(items)' in (runtime.run_dir/child.artifact.module_path).read_text()
