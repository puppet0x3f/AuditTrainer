"""CPU checks for APPS extraction, executable teachers, and paired result accounting."""

import json

import pytest

from audittrainer.apps import convert_record
from audittrainer import apps_multitask as study
from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.verifier import canonical


def _record(id):
    return {'id': id, 'difficulty': 'interview', 'url': f'https://example.test/apps/{id}',
            'question': 'Read one integer and output its successor.',
            'input_output': json.dumps({'inputs': ['1\n', '9854321\n'],
                                        'outputs': ['2\n', '9854322\n']}),
            'solutions': json.dumps(['value = int(input())\nprint(value + 1)\n'])}


def test_local_stdin_adaptation_and_auto_visible_partition():
    source = study.source_to_solve('value = int(input())\nprint(value + 1)\n')
    namespace = {}
    exec(source, namespace)
    assert namespace['solve']('41\n') == '42\n'
    with pytest.raises(ValueError, match='DIRECT_STANDARD_STREAM'):
        study.source_to_solve('import sys\nprint(sys.stdin.read())\n')
    one = {**_record(7000), 'input_output': {'inputs': ['1\n'], 'outputs': ['2\n']}}
    public, private = convert_record(one, original_split='train', split='TRAIN', visible_cases=None)
    assert public['suites'] == [] and len(private['suites'][0]['cases']) == 1
    public, private = convert_record(_record(7000), original_split='train', split='TRAIN', visible_cases=None)
    assert len(public['suites'][0]['cases']) == len(private['suites'][0]['cases']) == 1


def test_extract_verified_train_and_heldout_then_report(tmp_path, monkeypatch):
    source = tmp_path/'train.jsonl'
    source.write_text(''.join(canonical(_record(i))+'\n' for i in (7000, 7001)))
    data, root = tmp_path/'data', tmp_path/'runs'
    ready = study.prepare(source, data, root,
                          quotas={'TRAIN': {'interview': 1}, 'VALIDATION': {'interview': 1}},
                          expected_sha=None)
    assert ready['counts'] == {'train': 1, 'validation': 1, 'retention': 177, 'steps': 1}
    roots = load_root_manifest(data/'roots.jsonl')
    assert {r['split'] for r in roots.values()} == {'TRAIN', 'VALIDATION'}
    assert len({r['family_id'] for r in roots.values()}) == 2
    target = next(r for r in read_jsonl(data/'training.jsonl') if r['kind'] == 'target')
    assert roots[target['root_id']]['split'] == 'TRAIN'
    assert '9854321' not in canonical(target['observation'])
    assert target['target_action']['action'] == 'write_code'
    assert target['target_tokens'] < 3072

    monkeypatch.setattr(study, 'DATA', data)
    monkeypatch.setattr(study, 'ROOT', root)
    heldout = next(r for r in roots.values() if r['split'] == 'VALIDATION')
    for condition in ('BEFORE', 'AFTER'):
        directory = root/'evaluation'/condition
        directory.mkdir(parents=True)
        for shard in range(1 if condition == 'BEFORE' else 2):
            (directory/f'shard_{shard}.json').write_text(json.dumps({'status': 'COMPLETE',
                'shard': shard, 'shards': 1 if condition == 'BEFORE' else 2}))
        for arm in ('NATURAL', 'ATOMIC', 'SPLIT'):
            path = directory/heldout['root_id']/arm/'case_result.json'
            path.parent.mkdir(parents=True)
            passed = condition == 'AFTER' and arm in {'NATURAL', 'SPLIT'}
            path.write_text(json.dumps({'root_id': heldout['root_id'],
                'run_dir': str(path.parent),
                'final_scores': {'private': {'status': 'PASS' if passed else 'FAIL',
                    'suites': [{'evidence_path': str(path.parent/'strict_evidence.json')}] if passed else []}},
                'first_scores': {'private': {'status': 'FAIL'}},
                'root_first_plan_registered': True,
                'registered_plan_modes': {'SPLIT': 1} if arm == 'SPLIT' else {'ATOMIC': 1},
                'usage': {'policy_calls': 4}}))
            if passed:
                (path.parent/'strict_evidence.json').write_text(json.dumps({'details': {'cases': [
                    {'expected': 'A\nB\n', 'execution': {
                        'value': 'A\nB\n' if arm == 'NATURAL' else 'B\nA\n'}}]}}))
            trace = path.parent/'runtime/trace.jsonl'
            trace.parent.mkdir()
            trace.write_text(json.dumps({'node_id': 'N0', 'observation_before': {'phase': 'NEW'},
                'parsed_action': {'action': 'plan', 'args': {'mode': 'SPLIT' if arm == 'SPLIT' else 'ATOMIC'}},
                'host_result': {'code': 'PLAN_REGISTERED'},
                'model_input': {'refinement': {'gate': {'mode_probabilities': {'SPLIT': .5}}}}})+'\n')
    result = study.report(shards=2)
    assert result['arms']['BEFORE_NATURAL']['private_pass'] == 0
    assert result['arms']['AFTER_NATURAL']['private_pass'] == 1
    assert result['arms']['AFTER_SPLIT']['root_split'] == 1
    assert result['arms']['AFTER_NATURAL']['private_strict_pass'] == 1
    assert result['arms']['AFTER_SPLIT']['private_pass'] == 1
    assert result['arms']['AFTER_SPLIT']['private_strict_pass'] == 0
    assert result['paired']['NATURAL']['wins'] == 1
    assert result['paired']['NATURAL']['strict_wins'] == 1


def test_planner_adapter_routing_and_forced_root_only():
    class Gate:
        def __init__(self, name):
            self.name = name
        def score_modes(self, observation):
            return self.name
    class Policy:
        root_gate = Gate('root')
        body = Gate('body')
        score_modes = Gate('first_compute').score_modes
        def atomic(self, gate):
            return 'atomic'
        def sample_split(self, gate):
            return 'split'
        def as_output(self, gate, candidate):
            return candidate
    planner = study.CurrentPlanner.__new__(study.CurrentPlanner)
    planner.policy = Policy()
    planner.cost = {}
    planner.identity = {}
    assert planner.score_modes({'node_id': 'N0'}) == 'root'
    assert planner.score_modes({'node_id': 'N0_p1_compute'}) == 'first_compute'
    assert planner.score_modes({'node_id': 'N0_p2_other'}) == 'body'
    planner.score_modes = lambda observation: {'mode_probabilities': {'ATOMIC': .5, 'SPLIT': .5}}
    planner.act = lambda observation: 'natural'
    forced = study._ForcedRoot(planner, 'SPLIT')
    assert forced.act({'node_id': 'N0', 'phase': 'NEW'}) == 'split'
    assert forced.act({'node_id': 'N0', 'phase': 'NEW'}) == 'natural'
    assert forced.act({'node_id': 'N0_p1_compute', 'phase': 'NEW'}) == 'natural'
