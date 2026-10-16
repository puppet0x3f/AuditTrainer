from copy import deepcopy
import json
from pathlib import Path
import pytest
from audittrainer.refinement_complex_tasks import TASKS,oracle


@pytest.mark.parametrize('name',list(TASKS))
def test_reference_matches_independent_oracle_without_mutating_input(name):
    task=TASKS[name];ns={};exec(task['prepare_code']+'\n'+task['aggregate_code'],ns)
    for original in task['cases']:
        items=deepcopy(original);prepared,answer=oracle(name,deepcopy(items))
        actual=ns['prepare'](items);assert items==original and actual==prepared
        assert ns['aggregate'](actual)==answer and actual==prepared


def test_hand_calculated_nontrivial_answers():
    assert oracle('weighted_routes',TASKS['weighted_routes']['cases'][1])[1]=={'queries':[{'count':1,'total':0,'farthest':'s'},{'count':2,'total':2,'farthest':'b'},{'count':4,'total':9,'farthest':'c'}]}
    assert oracle('stock_events',TASKS['stock_events']['cases'][1])[1]=={'orders':[['a',4],['b',2]],'total':6,'rejected':[2]}
    assert oracle('coverage_queries',TASKS['coverage_queries']['cases'][1])[1]=={'queries':[{'covered':4,'exposure':6,'peak':2},{'covered':2,'exposure':2,'peak':1},{'covered':2,'exposure':2,'peak':1}]}
    assert oracle('polynomial_chain',TASKS['polynomial_chain']['cases'][1])[1]=={'values':[-4,0,6],'degree':1}


def test_fixed_selection_teacher_passes_and_budget_unchanged():
    base=Path('runs/refinement/complex_values');r=json.loads((base/'request.json').read_text());old=json.loads(Path('runs/refinement/compute_refresh/request.json').read_text())
    assert [s['structure'] for s in r['selection']]==list(TASKS)
    assert r['framework']==old['framework'] and r['hashes']==old['hashes']
    assert r['cases']==32 and r['repetitions']==4
    assert r['plan_seed']!=old['plan_seed'] and r['execution_seed']!=old['execution_seed']
    for s in r['selection']:
        assert s['split']=='VALIDATION'
        result=json.loads((base/'teacher'/s['root_id']/'result.json').read_text())
        assert all(x['status']=='PASS' for x in result['scores'].values())
        obs=json.loads(Path(s['state_path']).read_text())['observation']
        assert obs['node_id']=='N0_p1_compute' and obs['phase']=='NEW'


def test_value_report_accepts_development_cases_without_training_rewards(tmp_path):
    from audittrainer.refinement_compute_values import report
    source=Path('runs/refinement/compute_refresh')
    request=json.loads((source/'request.json').read_text())
    (tmp_path/'request.json').write_text(json.dumps(request))
    for s in request['selection']:
        group=json.loads((source/s['root_id']/'result.json').read_text())
        for x in group['cases']:x['case']['reward']=None
        out=tmp_path/s['root_id'];out.mkdir();(out/'result.json').write_text(json.dumps(group))
    report(tmp_path)
    result=json.loads((tmp_path/'result.json').read_text())
    assert all(g[arm]['success']==4 for g in result['summary'] for arm in ['ATOMIC','SPLIT'])


def test_edge_order_supplement_preserves_original_and_exposes_counterexample():
    base=Path('runs/refinement/complex_values');rid='validation_internal_complex_weighted_routes'
    raw=json.loads((base/rid/'result.json').read_text())
    original=[x for x in raw['cases'] if x['arm']=='SPLIT' and x['case']['final_scores']['private']['status']=='PASS']
    assert len(original)==1
    check=json.loads((base/'edge_order_check/result.json').read_text())
    assert len(check['results'])==8 and check['model_calls']==0
    assert all(x['checked_scores']['private']['status']!='PASS' for x in check['results'])
    target=next(x for x in check['results'] if x['arm']=='SPLIT' and x['rep']==original[0]['rep'])
    assert target['original_private']=='PASS'
    ns={};task=TASKS['weighted_routes'];exec(task['prepare_code']+'\n'+task['aggregate_code'],ns)
    for edges in [[['s','a',1],['a','b',1]],[['a','b',1],['s','a',1]]]:
        items={'edges':edges,'source':'s','limits':[2]};prep,answer=oracle('weighted_routes',items)
        assert ns['prepare'](items)==prep
        assert ns['aggregate'](prep)==answer=={'queries':[{'count':3,'total':3,'farthest':'b'}]}
