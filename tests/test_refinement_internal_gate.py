import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from audittrainer.refinement_internal_gate import NaturalInternal, target_for, NODE


def test_value_targets_do_not_turn_scoring_errors_or_ties_into_labels():
    assert target_for(0, .18) == .18
    assert target_for(-.75, .18, scoring_error=True) == .18
    assert target_for(.25, .18) == .18
    assert target_for(.75, .18) > .99
    assert target_for(-.5, .18) < .01


def test_only_first_compute_uses_adapted_gate():
    calls = []
    obs = {'node_id': NODE, 'phase': 'NEW'}
    policy = SimpleNamespace(cost={}, identity={}, act=lambda o:calls.append('adapted'),
                             continuation=lambda o:calls.append('frozen'))
    wrapper = NaturalInternal(policy, {'observation':obs})
    wrapper.act(obs)
    wrapper.act(obs)  # A protocol retry must retain the continuation used in values.
    wrapper.act({'node_id': NODE+'_p1_prepare', 'phase':'NEW'})
    assert calls == ['adapted','frozen','frozen']
    independent_run = NaturalInternal(policy, {'observation':obs})
    independent_run.act(obs)
    assert calls[-1] == 'adapted'


def test_fixed_state_mismatch_cannot_silently_receive_old_value():
    policy = SimpleNamespace(cost={},identity={})
    wrapper = NaturalInternal(policy, {'observation':{'node_id':NODE,'phase':'NEW','budget':1}})
    with pytest.raises(AssertionError):
        wrapper.act({'node_id':NODE,'phase':'NEW','budget':2})


def test_experiment_separates_new_seed_evaluation_and_anchors():
    r = json.loads(Path('runs/refinement/internal_gate/request.json').read_text())
    old = json.loads(Path('runs/refinement/complex_refresh/request.json').read_text())
    assert r['adapters']==old['adapters'] and r['hashes']==old['hashes']
    assert r['natural_seed'] not in [old['plan_seed'],old['execution_seed']]
    assert [t['direction'] for t in r['targets']]==['ANCHOR','ANCHOR','SPLIT','SPLIT']
    assert r['config']['reward']['cost_weight']==0


def test_loaded_policy_identity_obeys_output_protocol(monkeypatch):
    from audittrainer import refinement_internal_gate as study
    from audittrainer.schema import PolicyOutput
    model = SimpleNamespace(load_adapter=lambda *a,**k:None,set_adapter=lambda *a:None,requires_grad_=lambda *a:None)
    llm = SimpleNamespace(model=model,tokenizer=None,identity={'adapter':'test'})
    monkeypatch.setattr(study,'load_policies',lambda *a:(SimpleNamespace(model=model,llm=llm),None))
    _,policy,_,_,_ = study.context()
    PolicyOutput(raw_text='{}',model_identity=policy.identity,token_usage={'source':'model'})
    assert json.loads(policy.identity['frozen_adapter_paths'])['body'].endswith('MULTI_BODY/final_adapter')
