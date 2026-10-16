from types import SimpleNamespace
from audittrainer.refinement_separate_gate import RootGatePolicy
from audittrainer.refinement_policy import RefinementPolicy


def test_only_root_mode_uses_gate_copy(monkeypatch):
    calls=[]
    policy=object.__new__(RootGatePolicy)
    def gate(obs):calls.append(('gate',obs['node_id']));return 'gate_result'
    def body(obs):calls.append(('body',obs['node_id']));return 'body_result'
    monkeypatch.setattr(RefinementPolicy,'score_modes',lambda self,obs:gate(obs))
    policy.body=SimpleNamespace(score_modes=body)
    assert policy.score_modes({'node_id':'N0'})=='gate_result'
    assert policy.score_modes({'node_id':'N0_p1_compute'})=='body_result'
    assert calls==[('gate','N0'),('body','N0_p1_compute')]


def test_every_split_body_uses_frozen_adapter(monkeypatch):
    policy=object.__new__(RootGatePolicy);seen=[]
    monkeypatch.setattr(RefinementPolicy,'sample_split',lambda self,gate:(_ for _ in ()).throw(AssertionError('Trainable gate must not generate a body')))
    policy.body=SimpleNamespace(sample_split=lambda gate:seen.append(gate) or 'frozen_candidate')
    for node in ['N0','N0_p1_compute']:
        gate={'observation':{'node_id':node}}
        assert policy.sample_split(gate)=='frozen_candidate'
        assert seen[-1] is gate
