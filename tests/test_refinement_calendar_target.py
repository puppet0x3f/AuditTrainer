import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from audittrainer.refinement_calendar_target import prepare, target_weights
from audittrainer.refinement_curriculum_rl import plan_key
from audittrainer.refinement_rollout import runtime_for
from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.schema import PolicyOutput, TokenUsage
from audittrainer.verifier import canonical


def test_target_is_prior_winner_and_registers_on_same_public_state(tmp_path):
    study=yaml.safe_load(Path('configs/refinement_calendar_target.yaml').read_text())
    study['output_dir']=str(tmp_path/'study')
    config,framework,demo=prepare(study)
    rows=load_root_manifest(config['root_manifest']);row=rows[study['root_id']]
    tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    policy=SimpleNamespace(act=lambda o:PolicyOutput(raw_text=canonical(demo['target_action']),
        token_usage=TokenUsage(),model_identity={'backend':'test'}))
    runtime,_=runtime_for(row,tests,framework,policy,tmp_path/'registration')
    event=runtime.step()
    assert event.host_result.code=='PLAN_REGISTERED'
    assert event.observation_before==json.loads(demo['messages'][1]['content'])['observation']
    definition=json.loads(Path(row['fixture_path']).read_text())
    prior=json.loads(Path(study['source_fixed_plan']).read_text())
    assert plan_key(demo['target_action'],definition)==plan_key(prior['fixed_plans'][row['root_id']]['parent_aggregate'],definition)
    request=json.loads((tmp_path/'study/request.json').read_text())
    assert request['prior_plan_returns']=={'atomic':4/12,'parent_aggregate':1.,'pipeline':5/12}
    assert request['planned']['optimizer_steps']==60
    assert request['planned']['natural_evaluations']==192


def test_target_ablations_do_not_change_common_constraints():
    assert target_weights('GATE_SFT')==(1.,0.)
    assert target_weights('BODY_SFT')==(0.,1.)
    assert target_weights('JOINT_SFT')==(1.,1.)
    study=yaml.safe_load(Path('configs/refinement_calendar_target.yaml').read_text())
    seeds={s+i for s in study['evaluation_seeds'] for i in range(study['natural_repetitions'])}
    probe={study['probe_seed']+2000000+i for i in range(study['conditional_split_samples'])}
    assert len(seeds)==48 and not seeds.intersection(probe)
    assert not seeds.intersection(range(170600713,170600725))
