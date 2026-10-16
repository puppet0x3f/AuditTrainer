import json
from pathlib import Path
from audittrainer.dataset import read_jsonl
from audittrainer.refinement_handoff import direct_binding


def test_targets_are_new_train_and_retention_preserves_previous_supervision():
    r=json.loads(Path('runs/refinement/compute_sft/request.json').read_text())
    samples=read_jsonl('data/refinement/compute_sft/training.jsonl')
    targets=[x for x in samples if x['kind']=='target']
    assert {x['root_id'] for x in targets}=={x['root_id'] for x in r['fit']}
    assert not {x['root_id'] for x in targets}&{x['root_id'] for x in r['development']}
    assert all(x['split']=='TRAIN' for x in samples)
    retained=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/handoff_sft/training.jsonl')]
    assert [x for x in samples if x['kind']=='retention']==retained
    for x in retained:
        if x['target_action']['action']=='consume':
            expected=direct_binding(x['observation'])
            assert all(x['target_action']['args'][k]==v for k,v in expected.items())


def test_teacher_completes_root_and_uses_only_approved_dependencies():
    r=json.loads(Path('runs/refinement/compute_sft/request.json').read_text())
    for s in r['fit']:
        base=Path('runs/refinement/compute_sft/teacher')/s['root_id']
        result=json.loads((base/'result.json').read_text())
        assert all(v['status']=='PASS' for v in result['scores'].values())
    for x in read_jsonl('data/refinement/compute_sft/training.jsonl'):
        if x['kind']!='target':continue
        a=x['target_action']['args'];obs=x['observation']
        assert a['declared_dependencies']==obs['approved_dependencies']
        assert a['contract_version']==obs['incoming_contract']['version']
        symbol=obs['incoming_contract']['signature']['name']
        assert symbol in {'aggregate','compute','solve'}
        if symbol=='compute':assert 'json.dumps' not in a['source']
