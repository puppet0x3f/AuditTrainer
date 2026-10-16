from copy import deepcopy
import json
from pathlib import Path
import pytest
from audittrainer.refinement_cross_state_tasks import VARIANTS,inputs,oracle,task_pack
from audittrainer.refinement_cross_state import ROOT,check
from audittrainer.dataset import read_jsonl,load_root_manifest


@pytest.mark.parametrize('name',list(VARIANTS))
def test_reference_oracle_and_fixture_match(name):
    d,private,ref=task_pack(name);ns={};exec(ref['prepare']+'\n'+ref['aggregate'],ns)
    r=check();rows=load_root_manifest(r['config']['root_manifest']);row=rows[d['task']['root_id']]
    assert json.loads(Path(row['fixture_path']).read_text())==d
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    assert json.loads(Path(tests[row['root_id']]['path']).read_text())==private
    for value in inputs(name):
        before=deepcopy(value);prepared,answer=oracle(name,value)
        assert ns['prepare'](value)==prepared and value==before
        original=deepcopy(prepared);assert ns['aggregate'](prepared)==answer and prepared==original
        if 'polynomials' in value:
            assert 0<=value['derivative']<=4
            assert all(0<=d<=4 and -3<=c<=3 for factor in value['polynomials'] for d,c in factor)


def test_hand_calculated_new_semantics():
    x={'edges':[['s','b',1],['s','a',1]],'source':'s','limits':[1,0]}
    assert oracle('routes_below',x)[1]['queries']==[{'count':1,'total':0,'nodes':['s']},{'count':0,'total':0,'nodes':[]}]
    assert oracle('routes_exact',x)[1]['queries'][0]=={'count':2,'total':2,'nodes':['a','b']}
    x={'initial':{'a':3,'b':1},'events':[],'targets':{'a':1,'b':2,'c':0}}
    assert oracle('stock_balance',x)[1]=={'balances':[['a',2],['b',-1],['c',0]],'total':1,'rejected':[]}
    assert oracle('stock_status',x)[1]=={'below':['b'],'equal':['c'],'above':['a'],'rejected':[]}
    x={'intervals':[[0,2],[1,3]],'queries':[[-1,4],[1,1]]}
    assert oracle('coverage_single',x)[1]['queries']==[{'single':2,'multiple':1},{'single':0,'multiple':0}]
    assert oracle('coverage_gaps',x)[1]['queries']==[{'uncovered':2,'covered':3},{'uncovered':0,'covered':0}]
    x={'polynomials':[[[2,1],[0,-1]]],'derivative':1,'points':[-2,0,1]}
    assert oracle('polynomial_signs',x)[1]=={'signs':[1,-1,0],'degree':2}
    assert oracle('polynomial_sum',x)[1]=={'sums':[-1,-1,2],'degree':2}


def test_no_supervised_contract_overlap_and_budget():
    r=check();assert len(r['selection'])==8 and r['cases']==96
    assert {s['known_algorithm_family'] for s in r['selection']}=={'weighted_routes','stock_events','coverage_queries','polynomial_chain'}
    old=json.loads(Path('runs/refinement/transfer_gate/request.json').read_text())
    assert r['execution_seed'] not in [old['natural_seed'],old['execution_seed']]
    corpus=Path('data/refinement/root_retention/training.jsonl').read_text()
    old_contracts=[json.loads(Path(s['state_path']).read_text())['observation']['incoming_contract'] for s in old['selection']]
    for s in r['selection']:
        assert VARIANTS[s['variant']][1] not in corpus
        state=json.loads(Path(s['state_path']).read_text())
        assert state['observation']['incoming_contract'] not in old_contracts
