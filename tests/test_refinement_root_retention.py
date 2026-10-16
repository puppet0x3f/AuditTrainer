import json
from pathlib import Path
from audittrainer.dataset import read_jsonl,file_hash
from audittrainer.refinement_root_retention import ROOT,DATA,PREVIOUS,INITIAL


def test_alias_change_preserves_contracts_and_correct_composition():
    original={x['sample_id']:x for x in read_jsonl('data/refinement/transfer_sft/training.jsonl')}
    targets=[x for x in read_jsonl(DATA/'training.jsonl') if x['kind']=='target']
    assert len(targets)==8
    for x in targets:
        old=original[x['sample_id'].removesuffix('_return_contract_alias')]
        expected=json.loads(json.dumps(old).replace('_compute_child','_prepare_child'))
        assert x['observation']==expected['observation']
        assert x['target_action']==expected['target_action']
        assert x['observation']['incoming_contract']==old['observation']['incoming_contract']
        assert x['split']=='TRAIN' and x['root_id'].startswith('train_complex_sft_')
        # Execute the actual root body against already-aggregated data: no distances available.
        source=x['target_action']['args']['source']
        ns={'_prepare_child':lambda items:{'queries':[{'count':3,'total':7,'farthest':'a'}]}}
        exec('\n'.join(l for l in source.splitlines() if not l.startswith('from nodes.')),ns)
        assert ns['solve']('{}')=='{"queries":[{"count":3,"total":7,"farthest":"a"}]}'


def test_full_retention_and_cached_pair_design():
    r=json.loads((ROOT/'request.json').read_text())
    original=read_jsonl('data/refinement/transfer_sft/training.jsonl')
    rows=read_jsonl(DATA/'training.jsonl')
    assert rows[8:]==[{**x,'kind':'retention'} for x in original]
    assert len({x['sample_id'] for x in rows})==96
    assert r['steps']==44 and r['learning_rate']==1e-5 and r['initial_executor']==INITIAL
    assert r['new_evaluation_cases']==24
    old=json.loads((PREVIOUS/'request.json').read_text())
    assert r['selection']==old['selection'] and r['eval_seed']==old['eval_seed']
    for i in range(3):
        assert (ROOT/'evaluation/BEFORE'/f'shard_{i}.json').read_bytes()==(PREVIOUS/'evaluation/AFTER'/f'shard_{i}.json').read_bytes()
    assert file_hash(DATA/'training.jsonl')==r['training_sha256']
    for k,p in r['adapters'].items():assert file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k]
