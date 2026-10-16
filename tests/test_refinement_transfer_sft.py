import json
from collections import Counter
from pathlib import Path
from audittrainer.dataset import read_jsonl,file_hash
from audittrainer.refinement_transfer_sft import INPUTS
from audittrainer.refinement_internal_transfer_tasks import CASES

ROOT=Path('runs/refinement/transfer_sft');DATA=Path('data/refinement/transfer_sft')


def test_independent_training_inputs_and_no_development_supervision():
    r=json.loads((ROOT/'request.json').read_text());data=read_jsonl(DATA/'training.jsonl')
    assert all(not any(x in CASES[name] for x in cases) for name,cases in INPUTS.items())
    assert all(x['split']=='TRAIN' for x in data)
    assert not {x['root_id'] for x in data}&{s['root_id'] for s in r['development']}
    targets=[x for x in data if x['kind']=='target']
    assert len(targets)==14 and len({x['sample_id'] for x in data})==len(data)
    for s in r['fit']:
        for variant in ['parent_aggregate','pipeline']:
            scores=json.loads((ROOT/'teacher'/s['root_id']/variant/'result.json').read_text())
            assert all(v['status']=='PASS' for v in scores.values())


def test_old_curriculum_retained_and_new_targets_equally_exposed():
    data=read_jsonl(DATA/'training.jsonl');old=read_jsonl('data/refinement/complex_sft/training.jsonl')
    assert [x for x in data if x['kind']=='retention']==[{**x,'kind':'retention'} for x in old]
    r=json.loads((ROOT/'request.json').read_text());assert r['steps']==56
    counts=Counter(j%14 for j in range(2*r['steps']))
    assert len(counts)==14 and set(counts.values())=={8}
    assert file_hash(DATA/'training.jsonl')==r['training_sha256']


def test_fixed_budget_baseline_and_frozen_routing_models():
    r=json.loads((ROOT/'request.json').read_text())
    assert r['initial_executor']=='runs/refinement/complex_fit/training/final_adapter'
    assert r['internal_gate']=='runs/refinement/internal_gate/training/final_adapter'
    assert r['evaluation_cases']==48 and len(r['selection'])==12
    assert Counter(s['group'] for s in r['selection'])=={'fit':2,'development':2,'retention':8}
    for k,p in r['adapters'].items():assert file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k]
    assert r['gate']=='runs/refinement/gate_refresh/training/final_adapter'
