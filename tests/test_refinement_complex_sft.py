import json
from pathlib import Path
from audittrainer.dataset import read_jsonl
from audittrainer.refinement_complex_tasks import TASKS, pack
from audittrainer.refinement_complex_sft import INPUTS, ROOT


def test_development_unchanged_and_training_disjoint():
    r=json.loads((ROOT/'request.json').read_text())
    samples=read_jsonl('data/refinement/complex_sft/training.jsonl')
    dev_ids={x['root_id'] for x in r['development']}
    assert all(x['split']=='TRAIN' and x['root_id'] not in dev_ids for x in samples)
    for name in TASKS:
        assert all(x not in TASKS[name]['cases'] for x in INPUTS[name])
        d,private,_=pack(name)
        original=json.loads(Path('data/refinement/complex_values',d['task']['root_id'],'task.json').read_text())
        assert d==original
        copied=json.loads(Path('data/refinement/complex_sft/development',d['task']['root_id'],'task.json').read_text())
        copied['task']['split']='VALIDATION'
        assert copied==original
    tests=read_jsonl(r['config']['test_manifest'])
    graph=next(x for x in tests if x['root_id']=='validation_internal_complex_weighted_routes')
    assert any(s['suite_id'].endswith('edge_order_private') for s in json.loads(Path(graph['path']).read_text())['suites'])


def test_complete_teacher_algorithms_and_retention():
    samples=read_jsonl('data/refinement/complex_sft/training.jsonl')
    targets=[x for x in samples if x['kind']=='target']
    assert len(targets)==28
    for name in TASKS:
        for variant in ['parent_aggregate','pipeline']:
            scores=json.loads((ROOT/'teacher'/('train_complex_sft_'+name)/variant/'result.json').read_text())
            assert all(v['status']=='PASS' for v in scores.values())
            group=[x for x in targets if x['sample_id'].startswith('train_complex_sft_'+name+'_'+variant+'_')]
            functions={x['observation']['incoming_contract']['signature']['name'] for x in group}
            assert functions==({'prepare','compute','solve','aggregate'} if variant=='pipeline' else {'prepare','compute','solve'})
    old=read_jsonl('data/refinement/compute_sft/training.jsonl')
    assert [x for x in samples if x['kind']=='retention']==[{**x,'kind':'retention'} for x in old]
