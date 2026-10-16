import json

from audittrainer import refinement_gate_contrast as study
from audittrainer.refinement_internal_gate import target_for


def test_fixed_current_value_targets_and_three_gpu_schedule():
    r=study.check()
    assert r['steps']==r['updates']==32 and r['natural_cases']==40
    assert r['repetitions']==4 and r['shards']==3
    assert [x['root_id'] for x in r['selection']]==study.ROOTS
    assert [x['direction'] for x in r['targets']]==[
        'SPLIT','ATOMIC','ANCHOR','ANCHOR','ANCHOR']
    for target in r['targets']:
        assert target['target_p_split']==target_for(
            target['delta_raw'],target['initial_p_split'])
    assert [sum(i%3==shard for i in range(5))*4 for shard in range(3)]==[8,8,4]
    assert json.loads((study.ROOT/'gpu_resources.json').read_text())['devices']==['0','1','2']


def test_source_results_states_and_frozen_adapters():
    r=study.check()
    assert r['adapters']['executor']==study.EXECUTOR
    assert r['adapters']['internal_gate']==study.GATE
    for selected in r['selection']:
        state=json.loads((study.ROOT/selected['root_id']/'state.json').read_text())
        assert state['root_event']['host_result']['code']=='PLAN_REGISTERED'
        assert state['observation']['node_id']=='N0_p1_compute'
        assert selected['source_state_path']!=selected['state_path']
