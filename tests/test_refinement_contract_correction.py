import json
from collections import Counter
from pathlib import Path

from audittrainer import refinement_contract_correction as study
from audittrainer.dataset import read_jsonl


def test_frozen_training_design_and_no_development_supervision():
    r=study.check()
    rows=read_jsonl(study.DATA/'training.jsonl')
    targets,retention=rows[:27],rows[27:]
    assert len(targets)==27 and len(retention)==150
    assert all(x['kind']=='target' and x['split']=='TRAIN' for x in targets)
    assert retention==[{**x,'kind':'retention'} for x in
                       read_jsonl('data/refinement/contract_course/training.jsonl')]
    assert all(x['root_id'].startswith('train_contract_correction_') for x in targets)
    assert not any('validation_cross_state_' in x['root_id'] or
                   '_fresh_' in x['root_id'] for x in rows)
    assert Counter((2*i+j)%27 for i in range(r['steps']) for j in (0,1))==Counter({i:8 for i in range(27)})
    assert r['evaluation_cases']==136
    assert json.loads((study.ROOT/'gpu_resources.json').read_text())['devices']==['0','1','2']


def test_reference_programs_and_new_contract_oracles():
    for name in study.CORRECTIONS:
        for variant in ['atomic','parent_aggregate','pipeline']:
            path=study.ROOT/'teacher'/('train_contract_correction_'+name)/variant/'result.json'
            assert all(s['status']=='PASS' for s in json.loads(path.read_text()).values())
    for name,(family,_,code) in study.FRESH.items():
        namespace={}
        exec(code,namespace)
        for row in study.INPUTS[family]:
            prepared,expected=study.fresh_oracle(name,row)
            assert namespace['aggregate'](prepared)==expected
        assert (study.DATA/('train_contract_correction_fresh_'+name)/'task.json').exists()
