import json
from pathlib import Path
from audittrainer.dataset import file_hash,read_jsonl


def test_frozen_executor_and_fresh_seed_design():
    r=json.loads(Path('runs/refinement/complex_refresh/request.json').read_text())
    fit=json.loads(Path('runs/refinement/complex_fit/request.json').read_text())
    old=json.loads(Path('runs/refinement/complex_values/request.json').read_text())
    assert r['framework']==fit['framework'] and r['cases']==32 and r['repetitions']==4 and r['updates']==0
    assert [s['root_id'] for s in r['selection']]==[s['root_id'] for s in fit['development']]
    assert r['adapters']['executor']=='runs/refinement/complex_fit/training/final_adapter'
    assert r['hashes']['executor']!=old['hashes']['executor']
    for k in ['gate','body']:assert r['hashes'][k]==old['hashes'][k]
    for k,p in r['adapters'].items():assert file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k]
    assert r['plan_seed']!=old['plan_seed'] and r['execution_seed']!=old['execution_seed']


def test_internal_states_and_edge_order_scoring_preserved():
    r=json.loads(Path('runs/refinement/complex_refresh/request.json').read_text())
    for s in r['selection']:
        state=json.loads(Path(s['state_path']).read_text());obs=state['observation']
        assert s['split']=='TRAIN' and obs['node_id']=='N0_p1_compute' and obs['phase']=='NEW'
        assert {o['id'] for o in obs['incoming_contract']['ensures']}=={'PREPARE','AGGREGATE'}
        assert state['root_event']['parsed_action']['args']['children'][0]['local_key']=='compute'
        assert obs['allowed_plan_modes']==['ATOMIC','SPLIT']
    graph=next(t for t in read_jsonl(r['config']['test_manifest']) if t['root_id']=='validation_internal_complex_weighted_routes')
    suites=json.loads(Path(graph['path']).read_text())['suites']
    assert any(s['suite_id'].endswith('edge_order_private') and s['visibility']=='PRIVATE' and len(s['cases'])==2 for s in suites)


def test_resume_keeps_scoring_error_without_regenerating(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import shutil
    from audittrainer import refinement_compute_values as probe
    source=Path('runs/refinement/complex_values');r=json.loads((source/'request.json').read_text())
    selected=r['selection'][0];rid=selected['root_id'];base=tmp_path/rid;base.mkdir()
    (tmp_path/'request.json').write_text(json.dumps(r))
    for name in ['gate.json',*[f'candidate_{i:02d}.json' for i in range(4)]]:shutil.copyfile(source/rid/name,base/name)
    for arm in ['ATOMIC','SPLIT']:
        for rep in range(4):
            c=json.loads((source/rid/arm/f'repeat_{rep:02d}'/'case_result.json').read_text())
            if arm=='SPLIT' and rep==0:c.update(status='ERROR',errors=['SCORING_ERROR'])
            folder=base/arm/f'repeat_{rep:02d}';folder.mkdir(parents=True)
            (folder/'case_result.json').write_text(json.dumps(c))
    def forbidden(*args,**kwargs):raise AssertionError('Cached continuation must not regenerate model output')
    policy=SimpleNamespace(cost={},identity={},atomic=lambda gate:None,score_modes=forbidden,sample_split=forbidden)
    monkeypatch.setattr(probe,'fingerprints',lambda policy:{'frozen':'unchanged'})
    probe.run(0,root=tmp_path,policy_context=lambda method:(r,policy,None,None,None),allow_scoring_errors=True)
    result=json.loads((base/'result.json').read_text());assert len(result['cases'])==8
    error=next(x for x in result['cases'] if x['arm']=='SPLIT' and x['rep']==0)
    assert error['case']['status']=='ERROR' and error['case']['errors']==['SCORING_ERROR']
