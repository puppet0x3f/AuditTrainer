import json

from audittrainer import refinement_contract_value_refresh as study


def test_frozen_new_contract_value_design():
    r=study.check()
    assert r['cases']==24 and r['updates']==0 and r['repetitions']==4
    assert r['arms']==['ATOMIC','SPLIT']
    assert [s['contract'] for s in r['selection']]==study.NAMES
    assert r['adapters']['executor']==study.EXECUTOR
    assert json.loads((study.ROOT/'gpu_resources.json').read_text())['devices']==['0','1']
    assert r['execution_seed']!=json.loads((study.SOURCE/'request.json').read_text())['eval_seed']


def test_fixed_root_states_and_reference_scores():
    r=study.check()
    for selected in r['selection']:
        state=json.loads((study.ROOT/selected['root_id']/'state.json').read_text())
        assert state['root_event']['host_result']['code']=='PLAN_REGISTERED'
        assert state['observation']['node_id']=='N0_p1_compute'
        scores=json.loads((study.ROOT/selected['root_id']/'teacher/result.json').read_text())
        assert scores['visible']['status']==scores['private']['status']=='PASS'
