"""Decisive checks for PR1 credit, data separation and authored edge semantics."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from audittrainer.refinement_pr1 import credit_terms,mode_surrogate,conditional_token_terms
from audittrainer.refinement_policy import two_level_advantages
from audittrainer.refinement_tasks import prepare,aggregate,definition
from audittrainer.train_refinement import select_adaptation_rows,load_pr0_config,representation_loss
from test_refinement_pr0 import tiny


def test_root_mode_balance_and_boundary_supervision():
    c,_=load_pr0_config('configs/refinement_support.yaml'); rows,_=select_adaptation_rows(c)
    assert Counter((json.loads(r['messages'][1]['content'])['observation']['node_id'],r['target_action']['args']['mode']) for r in rows)=={('N0','ATOMIC'):100,('N0','SPLIT'):100}
    assert len({r['sample_id'] for r in rows})==150
    model=tiny(); e={'prefix':[1,2,3],'mode_ids':{'ATOMIC':[4,5],'SPLIT':[6,7]},'mode':'SPLIT',
                    'body_prefix':[1,2,3,6,7,8],'body_target':[9,10,11],'boundary_tokens':2,'boundary_weight':1.}
    loss,gate,body=representation_loss(model,e,0)
    logits=model(input_ids=torch.tensor([e['body_prefix']+e['body_target']])).logits[0,5:8].float()
    ce=F.cross_entropy(logits,torch.tensor(e['body_target']),reduction='none')
    torch.testing.assert_close(body,ce.mean()+ce[:2].mean()); loss.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_b3_clips_each_plan_before_averaging_gate_contributions():
    old=torch.tensor([.5,.5]).log(); current=torch.tensor([.2,.8]).log().requires_grad_()
    value=mode_surrogate(current,old,[0.,0.],[-1.,0.,1.],'B3',.2)
    # Positive plan clips at 1.2, negative plan keeps ratio 1.6.
    assert value.item()==pytest.approx(.5*(-1.6+1.2)/3)
    value.backward(); assert current.grad[1]<0


def test_two_level_credit_can_reduce_split_gate_while_improving_best_body():
    group={'mode_probabilities':{'SPLIT':.4},'q_values':[.7,.85,.5,.45],
           'advantages':two_level_advantages(.4,.7,[.85,.5,.45])}
    mode,body=credit_terms(group,'B4',0.)
    assert mode==pytest.approx([.04,-.06]); assert body==pytest.approx([.375,-.15,-.225])
    mode,body=credit_terms(group,'B3',1.)
    assert mode==pytest.approx([-.3,-.4]); assert body==pytest.approx([-.15,-.5,-.55])


def test_token_sum_full_distribution_kl_and_gradient():
    torch.manual_seed(23); ref=torch.randn(4,7); x=(ref+.1*torch.randn(4,7)).requires_grad_()
    target=[1,3,2,0]; old=F.log_softmax(ref,dim=-1)[torch.arange(4),target].tolist()
    objective,kl,chosen=conditional_token_terms(x,ref,target,old,.3,.2)
    expected=(F.softmax(x,dim=-1)*(F.log_softmax(x,dim=-1)-F.log_softmax(ref,dim=-1))).sum()
    torch.testing.assert_close(kl,expected)
    assert kl>0; (-objective+.02*kl).backward(); assert x.grad.norm()>0
    initial,zero,_=conditional_token_terms(ref,ref,target,old,.3,.2)
    assert initial.item()==pytest.approx(4*.3); assert zero.item()==0
    rescored,zero,_=conditional_token_terms(ref,ref,target,None,.3,.2)
    torch.testing.assert_close(rescored,initial); assert zero.item()==0


@pytest.mark.parametrize('family,p,items,expected',[
 ('fifo',5,{'invoices':[['a','i1',0,10],['a','i1',0,8],['a','i2',5,7],['a','i3',1,50]],'payments':[['a',10]],'void':['i3'],'now':10},{'rows':[['a',5,0,1]],'total':5}),
 ('timesheets',5,[['a',0,12,2],['a',3,8,5],['b',9,22,1]],{'rows':[['a',0,10,16],['a',1,2,4],['b',0,1,1],['b',1,10,5],['b',2,2,2]],'total':28}),
 ('reconcile',10,{'prices':[['x',0,3,'A'],['x',0,5,'B']],'orders':[['o1','x',1,4],['o2','x',2,1]],'returns':[['o1',2],['o2',3]]},{'rows':[['B',2,10,1,9]],'total':9}),
 ('schedule',2,{'jobs':[['a',3,[]],['b',2,[]],['c',1,['a','b']]],'blackout':[[1,4],[3,5]]},{'rows':[['a',5,8,0],['b',5,7,1],['c',8,9,0]],'makespan':9}),
 ('reservations',2,{'stock':{'x':5},'events':[[0,'a','reserve','x',3],[1,'a','reserve','x',1],[2,'a','cancel','x',0],[3,'a','ship','x',0],[4,'b','reserve','x',4],[5,'b','ship','x',0]]},{'rows':[['x',1,0,4]],'rejected':2}),
 ('revisions',2,[['a',None,0,0,'one two'],['b','a',3,4,' THREE '],['c','a',99,0,' x']],{'rows':[['b','one THREE',2],['c','one two x',3]],'total':5})])
def test_authored_oracles_against_hand_calculated_edges(family,p,items,expected):
    before=deepcopy(items); value=prepare(family,items); saved=deepcopy(value)
    assert aggregate(family,value,p)==expected; assert items==before and value==saved


def test_task_families_and_private_cases_are_separate():
    rows=[json.loads(x) for x in Path('data/refinement/pr1/roots.jsonl').read_text().splitlines()]
    train={r['family_id'] for r in rows if r['split']=='TRAIN'}
    validation={r['family_id'] for r in rows if r['split']=='VALIDATION'}
    assert train.isdisjoint(validation)
    assert Counter(r['split'] for r in rows)=={'TRAIN':10,'VALIDATION':4}
    d,private=definition('schedule',2,'TRAIN')
    assert all(s['visibility']=='VISIBLE' for s in d['suites'])
    assert all(s['visibility']=='PRIVATE' for s in private['suites'])
    assert 'Optional reviewed subcontracts' in d['task']['spec']
