import json
from pathlib import Path

import pytest
import yaml

from audittrainer.refinement_curriculum import repeated_contrasts
from audittrainer.refinement_pr1 import success_imitation_targets
from audittrainer.refinement_success_sft import batch_config, natural_seed


def feedback(tmp_path, outcomes, p=.2):
    paths=[]
    for i, repeats in enumerate(outcomes):
        row=[]
        for j, (success, registered, violation) in enumerate(repeats):
            d=tmp_path/f'{i}_{j}'; d.mkdir(parents=True, exist_ok=True)
            (d/'case_result.json').write_text(json.dumps({'status':'COMPLETE',
                'root_first_plan_registered':registered, 'reward':{'y':success,'v':violation}}))
            row.append(str(d))
        paths.append(row)
    return {'split':'TRAIN','status':'COMPLETE','case_paths':paths}, {'mode_probabilities':{'SPLIT':p}}


def test_success_imitation_corrects_stratification_instead_of_rewarding_more_split_draws(tmp_path):
    win=(1,True,0); lose=(0,True,0)
    group,gate=feedback(tmp_path, [[win,lose],[win,win],[lose,lose]])
    target=success_imitation_targets(group,gate)
    assert target['weights']==pytest.approx([.8,.2,0])
    assert target['success_counts']==[1,2,0]
    # Doubling the conditional draw budget must not manufacture SPLIT mass.
    group,gate=feedback(tmp_path/'double', [[win,lose],[win,win],[lose,lose],[win,win],[lose,lose]])
    target=success_imitation_targets(group,gate)
    assert target['weights'][0]==pytest.approx(.8)
    assert sum(target['weights'][1:])==pytest.approx(.2)


def test_retry_successes_and_confirmed_violations_do_not_become_imitation_targets(tmp_path):
    group,gate=feedback(tmp_path, [[(1,True,0)],[(1,False,0)],[(1,True,1)]])
    assert success_imitation_targets(group,gate)['weights']==[1.,0.,0.]
    group['split']='VALIDATION'
    with pytest.raises(ValueError,match='TRAIN'):
        success_imitation_targets(group,gate)


def test_no_success_is_retained_without_fabricating_a_positive_target(tmp_path):
    group,gate=feedback(tmp_path, [[(0,True,0)],[(1,False,0)]])
    targets=success_imitation_targets(group,gate)
    assert targets['success_mass']==0 and targets['weights']==[0.,0.]


def test_two_independent_shared_batches_and_fresh_evaluation_seeds():
    study=yaml.safe_load(Path('configs/refinement_success_sft.yaml').read_text())
    train=set(); candidates=set()
    for seed in study['batch_seeds']:
        c=batch_config(study,seed)
        assert c['initial_adapter']=='runs/refinement/PR1C_support/adaptation/final_adapter'
        assert 'optimizer_state' not in c and 'reference_adapter' not in c
        assert c['update']['epochs']==20 and c['update']['gradient_accumulation_groups']==3
        assert 3*((c['candidate_sampling']['split_bodies']+1)*4+c['independent_baseline_repetitions'])==120
        for i in range(3):
            train.update(seed+300000+i*8+j for j in range(4))
            train.update(seed+900000+i*8+j for j in range(4))
            candidates.update(seed+200000+i*32+j for j in range(8))
    assert len(train)==48 and len(candidates)==48
    natural={natural_seed(study,i,j) for i in range(3) for j in range(16)}
    probes={study['evaluation_seed']+2000000+i*1000+j for i in range(3) for j in range(24)}
    calibration={study['calibration_seed']+500000+i*4+j for i in range(3) for j in range(4)}
    domains=[train,candidates,natural,probes,calibration]
    assert all(not a&b for i,a in enumerate(domains) for b in domains[i+1:])


def test_predeclared_good_allocations_come_from_prior_train_calibration():
    study=yaml.safe_load(Path('configs/refinement_success_sft.yaml').read_text())
    rows={r['root_id']:r for r in json.loads(Path(study['calibration_source']).read_text())['reference']}
    for rid in study['root_ids']:
        row=rows[rid]
        values={name:sum(v['returns'])/4 for name,v in row['variants'].items()}
        assert repeated_contrasts(row['variants'])
        assert set(study['preferred_variants'][rid])=={k for k,v in values.items() if v==max(values.values())}
