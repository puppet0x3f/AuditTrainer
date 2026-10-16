from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from audittrainer import refinement_policy as policy
from audittrainer.refinement_parent_composition import selected_state, summary_prompt
from audittrainer.refinement_pr1 import load_config
from audittrainer.refinement_summary_training import batch_path, repeated_benefit


def test_summary_applied_to_executor_only(monkeypatch):
    class Model:
        def load_adapter(self,*a,**kw): pass
        def requires_grad_(self,*a): pass
        def set_adapter(self,*a): pass
    class LLM:
        def __init__(self,model,tokenizer,**kwargs):
            self.model=model; self.tokenizer=tokenizer; self.prompt_builder='original'
        @classmethod
        def from_pretrained(cls,*a,**kw):
            return SimpleNamespace(model=Model(),tokenizer=None,identity={},context_tokens=8192,
                                   generation_tokens=2048,decoding={},prompt_builder='original')
    monkeypatch.setattr(policy,'LLMPolicy',LLM)
    monkeypatch.setattr(policy,'load_adapter_metadata',lambda *a:(None,{}))
    monkeypatch.setattr(policy,'RefinementPolicy',lambda x:x)
    monkeypatch.setattr(policy,'FrozenExecutor',lambda x:x)
    for enabled in (False,True):
        planner,executor=policy.load_policies({'model':{},'parent_composition_summary':enabled},'planner','executor')
        assert planner.prompt_builder=='original'
        assert executor.prompt_builder==(summary_prompt if enabled else 'original')


def test_frozen_summary_phase_scope(monkeypatch):
    settings=yaml.safe_load(Path('configs/refinement_parent_composition.yaml').read_text())
    observation=selected_state(settings['source_cases'][0],0)['observation']
    monkeypatch.setattr('audittrainer.refinement_parent_composition.prepare_prompt',lambda t,o,**kw:o)
    for phase,mode in [('NEW',None),('IMPLEMENT','ATOMIC'),('RESOLVE_CHILD','COMPOSE'),('IMPLEMENT','COMPOSE')]:
        o=deepcopy(observation); o.update(phase=phase,mode=mode)
        result=summary_prompt(None,o)
        assert ('Parent composition summary' in result['root_spec'])==(phase=='IMPLEMENT' and mode=='COMPOSE')
        assert o['root_spec']==observation['root_spec']


def test_new_configs_preserve_training_and_budget():
    study=yaml.safe_load(Path('configs/refinement_summary_training.yaml').read_text())
    old,_=load_config('runs/refinement/B3_success_sft/batch_91000713/config.yaml')
    for seed in study['batch_seeds']:
        new,framework=load_config(batch_path(study,seed))
        for key in ('update','reward','runtime_overrides','candidate_sampling','initial_adapter','executor_adapter'):
            assert new[key]==old[key]
        assert not new.get('optimizer_state') and new['reward_credit']=='full_return'
        assert framework['parent_composition_summary'] is True
        assert 3*((1+new['candidate_sampling']['split_bodies'])*4+4)==120
    assert set(study['batch_seeds']).isdisjoint(study['evaluation_seeds'])
    good={str(i):{'final_success':{'wins':2,'losses':1}} for i in range(4)}
    assert repeated_benefit(good)
    good['3']['final_success']['losses']=2
    assert not repeated_benefit(good)
