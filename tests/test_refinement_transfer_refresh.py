import json
from pathlib import Path
from audittrainer.dataset import file_hash
from audittrainer.refinement_transfer_refresh import ROOT,SOURCE,FIT


def test_current_executor_fresh_seeds_and_exact_states():
    r=json.loads((ROOT/'request.json').read_text());old=json.loads((SOURCE/'request.json').read_text())
    assert r['cases']==32 and r['repetitions']==4 and r['updates']==0 and r['arms']==['ATOMIC','SPLIT']
    assert r['execution_seed']!=old['execution_seed']
    assert r['adapters']['executor']==str(FIT/'training/final_adapter')
    for k in ['gate','body','internal_gate']:assert r['adapters'][k]==old['adapters'][k]
    assert len(r['selection'])==4
    for s,t in zip(r['selection'],old['selection']):
        assert s['root_id']==t['root_id']
        assert json.loads(Path(s['state_path']).read_text())==json.loads(Path(t['state_path']).read_text())
    for k,p in r['adapters'].items():assert file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k]
