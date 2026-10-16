import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from audittrainer.refinement_internal_transfer_tasks import VARIANTS,CASES,oracle,task_pack
from audittrainer.refinement_complex_tasks import TASKS


@pytest.mark.parametrize('name',list(VARIANTS))
def test_new_contract_reference_matches_oracle_and_preserves_inputs(name):
    d,private,ref=task_pack(name);ns={};exec(ref['prepare']+'\n'+ref['aggregate'],ns)
    assert d['task']['split']=='VALIDATION' and d['local_contracts']['compute']['ensures'][-1]['text']==VARIANTS[name][1]
    assert VARIANTS[name][1]!=TASKS[VARIANTS[name][0]]['aggregate']
    for original in CASES[name]:
        items=deepcopy(original);prepared,expected=oracle(name,deepcopy(items));actual=ns['prepare'](items)
        assert items==original and actual==prepared
        assert ns['aggregate'](actual)==expected and actual==prepared


def test_changed_semantics_hand_calculated():
    assert oracle('routes_outside',CASES['routes_outside'][1])[1]['queries']==[
        {'count':3,'total':7,'nearest':'b'},{'count':2,'total':6,'nearest':'a'},
        {'count':0,'total':0,'nearest':None},{'count':0,'total':0,'nearest':None}]
    assert oracle('stock_surplus',CASES['stock_surplus'][1])[1]=={'surplus':[['pear',3],['plum',3]],'total':6,'rejected':[0]}
    assert oracle('coverage_redundancy',CASES['coverage_redundancy'][2])[1]['queries']==[
        {'overlap':4,'excess':4,'peak':2},{'overlap':0,'excess':0,'peak':0},{'overlap':0,'excess':0,'peak':0}]
    assert oracle('polynomial_joint',CASES['polynomial_joint'][1])[1]=={'values':[-30,-2,55],'derivatives':[32,4,52],'degree':3}


@pytest.mark.parametrize('arm,expected',[('ATOMIC','ATOMIC'),('SPLIT','SPLIT'),('NATURAL','SPLIT')])
def test_only_first_choice_is_intervened_and_natural_is_sampled(monkeypatch,arm,expected):
    import torch
    from audittrainer.refinement_internal_transfer import BranchInternal
    from audittrainer.refinement_internal_gate import NODE
    monkeypatch.setattr(torch,'multinomial',lambda *a:torch.tensor([1]))
    obs={'node_id':NODE,'phase':'NEW'}
    policy=SimpleNamespace(cost={},identity={},score_modes=lambda o:{'mode_probabilities':{'ATOMIC':.4,'SPLIT':.6}},
        atomic=lambda g:'ATOMIC',sample_split=lambda g:'SPLIT',as_output=lambda g,c:c,continuation=lambda o:'FROZEN')
    wrapper=BranchInternal(policy,{'observation':obs},arm)
    assert wrapper.act(obs)==expected
    assert wrapper.act(obs)=='FROZEN'


def test_prespecified_selection_is_disjoint_and_teacher_passes():
    from audittrainer.dataset import read_jsonl
    base=Path('runs/refinement/internal_transfer');r=json.loads((base/'request.json').read_text())
    old=json.loads(Path('runs/refinement/internal_gate/request.json').read_text())
    assert r['cases']==48 and r['updates']==0 and len(r['selection'])==4
    assert not ({s['root_id'] for s in r['selection']}&{s['root_id'] for s in old['selection']})
    supervised=Path('data/refinement/complex_sft/training.jsonl').read_text()
    for s in r['selection']:
        state=json.loads(Path(s['state_path']).read_text())
        assert state['observation']['incoming_contract'] not in [json.loads(Path(t['state_path']).read_text())['observation']['incoming_contract'] for t in old['selection']]
        assert VARIANTS[s['variant']][1] not in supervised
        teacher=json.loads((base/'teacher'/s['root_id']/'result.json').read_text())
        assert all(x['status']=='PASS' for x in teacher['scores'].values())
    assert all(row['split']=='VALIDATION' for row in read_jsonl(r['config']['root_manifest']))
