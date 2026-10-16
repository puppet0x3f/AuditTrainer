import json
from pathlib import Path
from audittrainer.refinement_handoff import direct_binding


def data():
    return [json.loads(x) for x in Path('data/refinement/handoff_sft/training.jsonl').read_text().splitlines()]


def test_supervised_consumption_binds_current_artifact():
    targets=[x for x in data() if x['kind']=='target' and x['target_action']['action']=='consume']
    assert len(targets)==8
    for x in targets:
        b=direct_binding(x['observation']);a=x['target_action']['args']
        assert all(a[k]==v for k,v in b.items())
        assert not a['accept_unverified']


def test_target_roots_exclude_failure_evaluation_roots():
    r=json.loads(Path('runs/refinement/handoff_sft/request.json').read_text())
    targets=[x for x in data() if x['kind']=='target']
    assert {x['root_id'] for x in targets}==set(r['fit_roots'])
    assert not {x['root_id'] for x in targets}&{x['root_id'] for x in r['replay']}
    assert all(x['split']=='TRAIN' for x in data())
