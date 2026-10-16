import json
from types import SimpleNamespace
from audittrainer import refinement_transfer_gate as study


def test_current_values_and_anchor_targets():
    r=study.preflight()
    assert r['steps']==32 and r['learning_rate']==1e-5 and r['kl_beta']==.1
    assert [x['direction'] for x in r['targets']]==['SPLIT','ANCHOR','ANCHOR','SPLIT']
    for t in r['targets']:
        if t['direction']=='ANCHOR':assert t['target_p_split']==t['initial_p_split']
        else:assert t['delta_raw']==1 and .99<t['target_p_split']<1
    source=json.loads((study.SOURCE/'request.json').read_text())
    assert r['selection']==source['selection'] and r['natural_seed']!=source['execution_seed']
    assert r['natural_cases']==32 and r['config']['reward']['cost_weight']==0


def test_loads_existing_internal_gate_and_freezes_continuations(monkeypatch):
    calls=[];loaded=[]
    model=SimpleNamespace(load_adapter=lambda p,**kw:loaded.append((kw['adapter_name'],p)),set_adapter=lambda a:None,requires_grad_=lambda v:calls.append(('grad',v)))
    llm=SimpleNamespace(model=model,tokenizer=None,identity={'adapter':'test'})
    def load(framework,gate,executor):
        calls.append(('load',gate,executor));return SimpleNamespace(model=model,llm=llm),None
    monkeypatch.setattr(study,'load_policies',load)
    r,policy,_,_,_=study.context('BEFORE')
    assert calls[0]==('load',r['adapters']['internal_gate'],r['adapters']['executor'])
    assert ('grad',False) in calls
    assert loaded==[('body',r['adapters']['body']),('root_gate',r['adapters']['gate'])]
    study.context('AFTER')
    assert ('load',str(study.ROOT/'training/final_adapter'),r['adapters']['executor']) in calls
    assert isinstance(policy.identity['frozen_adapter_paths'],str)
