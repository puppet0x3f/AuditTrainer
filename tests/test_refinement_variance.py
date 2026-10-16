from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from audittrainer.evaluate import write_json
from audittrainer.refinement_pr1 import baseline_paths, load_baselines, load_config, load_training_groups
from audittrainer.refinement_variance import arm_config, recovery_counts
from audittrainer import refinement_rollout as rollout


def test_matched_rollout_budget_shared_optimizer_and_disjoint_audit_seeds():
    c,_=load_config('configs/refinement_variance.yaml')
    arms={a:arm_config(c,a) for a in ('wide','repeat','audit')}
    for a,s in arms.items():
        k=[rollout.split_candidate_count(s,r) for r in c['training_roots']]
        assert sum((v+1)*s['candidate_sampling']['continuations_per_candidate']+s['independent_baseline_repetitions'] for v in k)==120
        assert s['initial_adapter']==c['initial_adapter'] and s['optimizer_state']==c['optimizer_state']
        assert s['reference_adapter']!=s['initial_adapter']
        assert s['update']['epochs']*s['root_limit']//s['update']['gradient_accumulation_groups']==10
    assert arms['wide']['update']==arms['repeat']['update']
    assert arms['wide']['retention_offset']==arms['repeat']['retention_offset']==150
    assert not arms['audit']['update']['enabled']
    seeds={a:{s['seed']+s['candidate_sampling'].get('continuation_seed_offset',300000)+i*8+j for i in range(6) for j in range(s['candidate_sampling']['continuations_per_candidate'])}
           |{s['seed']+s.get('baseline_seed_offset',900000)+i*8+j for i in range(6) for j in range(s['independent_baseline_repetitions'])} for a,s in arms.items()}
    assert seeds['wide']<=seeds['repeat'] and seeds['audit'].isdisjoint(seeds['repeat'])


def test_collector_preserves_common_plans_and_pairs_suffix_seeds_when_repeats_change(tmp_path,monkeypatch):
    calls=[]
    def run(row,tests,fw,config,planner,executor,directory,seed,initial):
        calls.append((row['root_id'],seed))
        return {'run_dir':str(directory),'reward':{'reward':float(seed%2)}}
    monkeypatch.setattr(rollout,'run_case',run)
    monkeypatch.setattr(rollout,'runtime_for',lambda row,*args:(SimpleNamespace(build_observation=lambda:{'root_id':row['root_id']}),None))
    planner=SimpleNamespace(cost=Counter(),
        score_modes=lambda obs:{'mode_probabilities':{'ATOMIC':.75,'SPLIT':.25},'observation_sha256':obs['root_id']},
        atomic=lambda gate:{'parse_error':None,'body_output_ids':[]},
        sample_split=lambda gate:{'parse_error':None,'body_output_ids':[torch.initial_seed()]})
    rows=[{'root_id':r,'family_id':'f','split':'TRAIN'} for r in ('a','b')]
    configs=[]; captured=[]
    for name,m,counts in [('wide',2,{'a':8,'b':9}),('repeat',4,{})]:
        s={'seed':123,'output_dir':str(tmp_path/name),'candidate_sampling':{
            'split_bodies':3,'split_bodies_by_root':counts,'continuations_per_candidate':m,
            'candidate_seed_stride':32,'continuation_seed_stride':8}}
        rollout.collect(rows,{},None,s,planner,SimpleNamespace(cost=Counter()),0,1)
        configs.append(s);captured.append(calls[:]);calls.clear()
    for rid in ('a','b'):
        folders=[Path(c['output_dir'])/'groups'/rid for c in configs]
        for i in range(4):
            assert json.loads((folders[0]/f'candidate_{i}.json').read_text())==json.loads((folders[1]/f'candidate_{i}.json').read_text())
        for f in folders:
            g=json.loads((f/'group_result.json').read_text());assert all(q==.5 for q in g['q_values'])
    assert len(captured[0])==38 and len(captured[1])==32
    for captured_ in captured:
        assert {s for r,s in captured_ if r=='a'}.isdisjoint({s for r,s in captured_ if r=='b'})


def test_training_loader_averages_independent_baselines_and_keeps_failed_candidates(tmp_path):
    config={'root_manifest':str(tmp_path/'roots.jsonl'),'output_dir':str(tmp_path),'root_limit':1,
            'candidate_sampling':{'split_bodies':3,'continuations_per_candidate':4},'independent_baseline_repetitions':4}
    (tmp_path/'roots.jsonl').write_text(json.dumps({'root_id':'r','family_id':'f','split':'TRAIN'})+'\n')
    folder=tmp_path/'groups/r';folder.mkdir(parents=True);write_json(folder/'gate.json',{})
    for i in range(4):write_json(folder/f'candidate_{i}.json',{'body_output_ids':[1],'body_old_log_probs':[-.1],'parse_error':'invalid'})
    write_json(folder/'group_result.json',{'status':'COMPLETE','split':'TRAIN','root_id':'r','candidate_count':4,
        'continuations':16,'gate_path':str(folder/'gate.json'),'case_paths':[[str(i)]*4 for i in range(4)],'q_values':[0.,0.,.25,1.]})
    for p,reward in zip(baseline_paths(config,'r'),[1.,0.,0.,1.]):
        p.mkdir(parents=True)
        write_json(p/'case_result.json',{'root_id':'r','status':'COMPLETE','split':'TRAIN','reward':{'reward':reward}})
    group,gate,candidates,baseline=load_training_groups(config)[0]
    assert baseline==.5 and len(candidates)==4 and group['q_values'][0]==0
    path=baseline_paths(config,'r')[-1]/'case_result.json';d=json.loads(path.read_text());d['split']='VALIDATION';write_json(path,d)
    with pytest.raises(ValueError,match='Invalid independent baseline'):load_baselines(config,'r')


def test_recovery_is_separate_from_initial_plan_success():
    cases=[{'reward':{'y':y},'root_first_plan_registered':registered} for y,registered in [(1,True),(1,False),(0,False),(0,True)]]
    assert recovery_counts(cases)=={'success_with_first_plan_registered':1,'success_after_first_plan_rejection':1}
