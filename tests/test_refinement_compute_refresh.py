import json
from pathlib import Path
from audittrainer.dataset import file_hash


def test_refresh_uses_same_states_new_executor_and_fresh_seeds():
    old=json.loads(Path('runs/refinement/compute_values/request.json').read_text())
    new=json.loads(Path('runs/refinement/compute_refresh/request.json').read_text())
    assert new['cases']==32 and new['repetitions']==4
    assert new['framework']==old['framework']
    assert new['plan_seed']!=old['plan_seed'] and new['execution_seed']!=old['execution_seed']
    assert new['adapters']['executor']=='runs/refinement/compute_sft/training/final_adapter'
    assert new['hashes']['executor']!=old['hashes']['executor']
    for k in ['gate','body']:assert new['hashes'][k]==old['hashes'][k]
    for a,b in zip(old['selection'],new['selection'],strict=True):
        assert a['root_id']==b['root_id']
        assert json.loads(Path(a['state_path']).read_text())['observation']==json.loads(Path(b['state_path']).read_text())['observation']
    for k,p in new['adapters'].items():assert file_hash(Path(p)/'adapter_model.safetensors')==new['hashes'][k]
