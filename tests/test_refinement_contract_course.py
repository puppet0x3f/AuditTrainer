import inspect
import json
from collections import Counter
from pathlib import Path
from audittrainer import refinement_contract_course as study
from audittrainer.dataset import read_jsonl
from audittrainer.refinement_cross_state_tasks import VARIANTS,inputs as cross_inputs
from audittrainer.refinement_internal_transfer_tasks import CASES


def test_training_data_pairs_exposure_and_retention():
    r=study.check();xs=read_jsonl(study.DATA/'training.jsonl');targets=xs[:54]
    assert len(targets)==54 and all(x['kind']=='target' and x['split']=='TRAIN' for x in targets)
    assert xs[54:]==[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/root_retention/training.jsonl')]
    assert len({x['sample_id'] for x in xs})==150
    counts=Counter(j%54 for j in range(2*r['steps']));assert set(counts.values())=={8}
    for a,b in zip(study.NAMES[::2],study.NAMES[1::2]):assert study.specification(a)[0]==study.specification(b)[0]
    for name in study.NAMES:
        family,_,_=study.specification(name)
        if name in VARIANTS:assert not any(x in cross_inputs(name) for x in study.INPUTS[family])
        if name in CASES:assert not any(x in CASES[name] for x in study.INPUTS[family])
        for variant in ['atomic','parent_aggregate','pipeline']:
            scores=json.loads((study.ROOT/'teacher'/('train_contract_course_'+name)/variant/'result.json').read_text())
            assert all(x['status']=='PASS' for x in scores.values())


def test_unsupervised_contracts_absent_and_no_failed_trace_supervision():
    text=(study.DATA/'training.jsonl').read_text();xs=read_jsonl(study.DATA/'training.jsonl')
    for name in study.UNSUPERVISED:assert VARIANTS[name][1] not in text
    assert all(not x['root_id'].startswith('validation_cross_state_') for x in xs)
    assert all(not x['root_id'].startswith('validation_internal_') for x in xs)


def test_fixed_design_two_gpu_and_progress_interface():
    r=study.check()
    assert r['steps']==216 and r['learning_rate']==1e-5 and r['evaluation_cases']==112
    assert Counter(s['group'] for s in r['selection'])=={'fit':6,'development':6,'unsupervised_contract':6,'retention':10}
    assert [sum(i%2==shard for i in range(len(r['selection'])))*2 for shard in [0,1]]==[28,28]
    assert json.loads((study.ROOT/'gpu_resources.json').read_text())['devices']==['0','1']
    assert inspect.signature(study.study.train).parameters['progress'].default is False
