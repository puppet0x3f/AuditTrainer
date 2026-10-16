import json

from audittrainer import refinement_anchor_value_refresh as study


def test_two_current_executor_anchor_states_and_gpu_budget():
    r=study.check()
    assert r['updates']==0 and r['cases']==16 and r['repetitions']==4
    assert r['arms']==['ATOMIC','SPLIT']
    assert [x['contract'] for x in r['selection']]==['routes_outside','polynomial_joint']
    assert r['adapters']['executor']==study.EXECUTOR
    assert r['adapters']['internal_gate']=='runs/refinement/transfer_gate/training/final_adapter'
    assert json.loads((study.ROOT/'gpu_resources.json').read_text())['devices']==['0','1']
    old=json.loads((study.SOURCE/'request.json').read_text())
    assert r['execution_seed']!=old['execution_seed']


def test_fixed_state_and_reference_root_scores():
    r=study.check()
    for selected in r['selection']:
        folder=study.ROOT/selected['root_id']
        state=json.loads((folder/'state.json').read_text())
        assert state['root_event']['host_result']['code']=='PLAN_REGISTERED'
        assert state['observation']['node_id']=='N0_p1_compute'
        scores=json.loads((folder/'teacher/result.json').read_text())
        assert scores['visible']['status']==scores['private']['status']=='PASS'
