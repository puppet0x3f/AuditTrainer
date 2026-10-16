from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from audittrainer.refinement_first_credit import adoption, evaluation_seed, metrics, outcome_records, training_config
from audittrainer.refinement_pr1 import credited_return, credit_terms, load_config, load_training_groups


def test_credit_keeps_full_accepted_returns_and_zeros_rejected_recovery():
    c={'split':'TRAIN','status':'COMPLETE','root_first_plan_registered':True,'reward':{'reward':-.2}}
    assert credited_return(c,'first_registered')==-.2
    c['root_first_plan_registered']=False; c['reward']['reward']=1.
    assert credited_return(c,'first_registered')==0.
    assert credited_return(c,'full_return')==1.
    c['split']='VALIDATION'
    with pytest.raises(ValueError,match='TRAIN'): credited_return(c,'first_registered')
    c['split']='TRAIN'; c['reward']=None
    with pytest.raises(ValueError,match='TRAIN'): credited_return(c,'first_registered')


def test_candidate_and_baseline_relabeling_use_the_same_rule_and_keep_every_draw(tmp_path):
    rid='train_transfer_calendar_matrix_03'
    source,_=load_config('runs/refinement/B3_success_sft/batch_91000713/config.yaml')
    c=deepcopy(source);c.update(output_dir=str(tmp_path),root_limit=1,reward_credit='first_registered')
    rows=[json.loads(l) for l in Path(c['root_manifest']).read_text().splitlines()]
    manifest=tmp_path/'roots.jsonl';manifest.write_text(json.dumps(next(r for r in rows if r['root_id']==rid))+'\n')
    c['root_manifest']=str(manifest)
    folder=tmp_path/'groups'/rid;folder.mkdir(parents=True)
    paths=[]
    for i in range(9):
        repeats=[]
        for j in range(4):
            d=folder/f'candidate_{i}_repeat_{j}';d.mkdir()
            # Candidate 1 succeeds only after initial rejection; all draws stay.
            row={'root_id':rid,'split':'TRAIN','status':'COMPLETE','root_first_plan_registered':i!=1,'reward':{'reward':1.}}
            (d/'case_result.json').write_text(json.dumps(row));repeats.append(str(d))
        paths.append(repeats)
        (folder/f'candidate_{i}.json').write_text(json.dumps({'body_output_ids':[1],'body_old_log_probs':[-.1]}))
    (folder/'gate.json').write_text('{}')
    g={'status':'COMPLETE','split':'TRAIN','root_id':rid,'candidate_count':9,'continuations':36,
       'case_paths':paths,'gate_path':str(folder/'gate.json'),'q_values':[1.]*9,'mode_probabilities':{'SPLIT':.2}}
    saved=json.dumps(g);(folder/'group_result.json').write_text(saved)
    for j in range(4):
        d=tmp_path/'independent_baseline'/rid/f'repeat_{j:02d}';d.mkdir(parents=True)
        (d/'case_result.json').write_text(json.dumps({'root_id':rid,'split':'TRAIN','status':'COMPLETE',
            'root_first_plan_registered':j==0,'reward':{'reward':1.}}))
    assigned,_,candidates,baseline=load_training_groups(c)[0]
    assert len(candidates)==9 and assigned['q_values']==[1.,0.,1.,1.,1.,1.,1.,1.,1.]
    assert baseline==.25
    assert credit_terms(assigned,'B3',baseline)[1][0]==-.25
    assert assigned['credit_assignment']['original_baseline']==1.
    assert (folder/'group_result.json').read_text()==saved
    c.pop('reward_credit')
    original,_,_,baseline=load_training_groups(c)[0]
    assert original['q_values']==[1.]*9 and baseline==1.


def test_only_credit_assignment_changes_and_evaluation_seeds_are_fresh():
    s=yaml.safe_load(Path('configs/refinement_first_credit.yaml').read_text())
    previous=json.loads(Path('runs/refinement/B3_success_sft/result.json').read_text())
    used={r['seed'] for rows in previous['outcomes'].values() for r in rows}
    for seed in s['batch_seeds']:
        old,_=load_config(f'runs/refinement/B3_success_sft/batch_{seed}/config.yaml')
        new=training_config(s,seed)
        assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'output_dir','reward_credit'}
        assert new['update']['epochs']==20 and 'optimizer_state' not in new
        for f in Path(old['output_dir']).glob('**/case_result.json'):
            used.add(json.loads(f.read_text())['seed'])
    fresh={evaluation_seed(s,i,j) for i in range(3) for j in range(16)}
    assert len(fresh)==48 and not fresh&used


def test_retry_metric_counts_invalid_root_attempts_without_child_planning(tmp_path):
    (tmp_path/'runtime').mkdir()
    events=[{'event_kind':'POLICY','node_id':node,'observation_before':{'phase':phase}}
            for node,phase in [('N0','NEW'),('N0','NEW'),('N0_child','NEW'),('N0','IMPLEMENT')]]
    (tmp_path/'runtime/trace.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    case={'run_dir':str(tmp_path),'root_id':'root','seed':1,'root_first_plan_registered':False,
          'final_scores':{'private':{'status':'PASS'}}}
    rows=outcome_records([case]);m=metrics(rows)
    assert m['final_successes']==1 and m['first_plan_successes']==0
    assert m['root_planning_retries']==1 and m['successes_after_initial_rejection']==1


def test_adoption_requires_repeated_first_plan_gain_without_final_regression():
    good={'first_plan_success':{'left_passes':35,'right_passes':36},'final_success':{'left_passes':37,'right_passes':37}}
    paired={'a':deepcopy(good),'b':deepcopy(good)}
    assert adoption(paired)
    paired['b']['first_plan_success']['right_passes']=35
    assert not adoption(paired)
    paired['b']=deepcopy(good);paired['b']['final_success']['right_passes']=36
    assert not adoption(paired)
