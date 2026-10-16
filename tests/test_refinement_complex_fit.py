import json
from pathlib import Path
from audittrainer.dataset import read_jsonl, file_hash


def test_budget_control_and_cached_baseline_are_matched():
    root=Path('runs/refinement/complex_fit');r=json.loads((root/'request.json').read_text())
    previous=Path('runs/refinement/complex_sft');old=json.loads((previous/'request.json').read_text())
    for key in ['framework','initial_executor','gate','body','seed','eval_seed','learning_rate','training_sha256','fit','development','retention','source_hashes']:
        assert r[key]==old[key]
    assert {k:v for k,v in r['config'].items() if k!='output_dir'}=={k:v for k,v in old['config'].items() if k!='output_dir'}
    for i in range(3):
        source=previous/'evaluation/BEFORE'/f'shard_{i}.json';cached=root/'evaluation/BEFORE'/f'shard_{i}.json'
        assert file_hash(source)==file_hash(cached)==r['cache_hashes'][str(source)]
        arm=json.loads(cached.read_text())
        assert arm['status']=='COMPLETE' and len(arm['results'])==8
        assert all(x['case']['status']=='COMPLETE' and not x['case']['errors'] for x in arm['results'])


def test_all_targets_retained_with_eight_equal_exposures():
    r=json.loads(Path('runs/refinement/complex_fit/request.json').read_text())
    p=Path('data/refinement/complex_sft/training.jsonl');data=read_jsonl(p)
    assert file_hash(p)==r['training_sha256']
    targets=[x for x in data if x['kind']=='target'];keep=[x for x in data if x['kind']=='retention']
    assert len(targets)==28 and len(keep)==46
    assert 2*r['steps']==8*len(targets)
    assert all(x['split']=='TRAIN' for x in data)
    assert not {x['root_id'] for x in data}&{x['root_id'] for x in r['development']}
