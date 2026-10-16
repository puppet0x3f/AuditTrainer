from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from audittrainer.refinement_first_credit import evaluation_seed
from audittrainer.refinement_sft_replication import compare, continued_support, evaluation_setup, models


def test_same_execution_settings_slots_and_existing_checkpoints():
    settings=yaml.safe_load(Path('configs/refinement_sft_replication.yaml').read_text())
    original,evaluation,config,framework=evaluation_setup(settings)
    assert {k for k in original.keys()|evaluation.keys() if original.get(k)!=evaluation.get(k)}=={'output_dir'}
    saved=json.loads(Path('runs/refinement/B3_first_credit/batch_91000713/B3/request.json').read_text())
    assert config==saved['config']
    assert config['reward']=={'gamma':1.,'cost_weight':0.,'confirmed_violation_weight':.2}
    assert framework['runtime']['max_depth']==1
    selected=models(settings)
    assert set(selected)=={'B0','SUCCESS_SFT_91000713','SUCCESS_SFT_92000713','B3_91000713','B3_92000713'}
    roots=sorted(json.loads(Path('runs/refinement/B3_first_credit/evaluation/B3_91000713/result.json').read_text())['by_root'])
    expected={(rid,evaluation_seed(evaluation,i,j)) for i,rid in enumerate(roots) for j in range(16)}
    for seed in original['batch_seeds']:
        old=json.loads(Path(f'runs/refinement/B3_first_credit/evaluation/B3_{seed}/result.json').read_text())
        assert expected=={(r['root_id'],r['seed']) for r in old['records']}
        assert selected[f'B3_{seed}']['adapter_sha256']==old['adapter_sha256']


def test_paired_success_and_first_acceptance_are_distinct_and_right_means_b3():
    sft=[{'root_id':'r','seed':i,'success':s,'first_plan_success':f} for i,(s,f) in enumerate([(0,0),(1,1),(0,0)])]
    b3=[{'root_id':'r','seed':i,'success':s,'first_plan_success':f} for i,(s,f) in enumerate([(1,0),(0,0),(1,1)])]
    p=compare(sft,b3,{'r'})
    assert p['final_success']=={'runs':3,'left_passes':1,'right_passes':2,'wins':2,'losses':1,'ties':0}
    assert p['first_plan_success']['left_passes']==p['first_plan_success']['right_passes']==1
    assert p['first_plan_success']['ties']==1
    b3[0]['seed']=99
    with pytest.raises(ValueError,match='Unpaired'):compare(sft,b3,{'r'})


def test_replication_requires_final_success_advantage_in_each_batch():
    good={'B3_vs_SUCCESS_SFT':{'final_success':{'left_passes':30,'right_passes':31},
                            'first_plan_success':{'left_passes':30,'right_passes':29}}}
    paired={'a':deepcopy(good),'b':deepcopy(good)}
    assert continued_support(paired)  # Diagnostics cannot override the primary criterion.
    paired['b']['B3_vs_SUCCESS_SFT']['final_success']['right_passes']=30
    assert not continued_support(paired)
    paired['b']['B3_vs_SUCCESS_SFT']['final_success']['right_passes']=29
    assert not continued_support(paired)
