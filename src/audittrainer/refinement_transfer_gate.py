"""Current-value calibration of the existing independent internal gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/transfer_gate')
SOURCE=Path('runs/refinement/transfer_refresh')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','natural'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from . import refinement_internal_gate as base
from .refinement_internal_gate import target_for,InternalGatePolicy,fingerprints
from .refinement_policy import load_policies,active_adapter,mode_log_probs
from .dataset import file_hash,read_jsonl,load_root_manifest
from .evaluate import write_json


def prepare():
    if (ROOT/'request.json').exists():return
    source=json.loads((SOURCE/'request.json').read_text());result=json.loads((SOURCE/'result.json').read_text())
    assert result['status']=='COMPLETE' and not result['errors']
    ROOT.mkdir(parents=True,exist_ok=True);targets=[]
    by_id={g['root_id']:g for g in result['groups']}
    for s in source['selection']:
        summary=next(x for x in result['summary'] if x['root_id']==s['root_id']);g=by_id[s['root_id']]
        scoring_error=any(x['case']['errors'] for x in g['cases']);delta=summary['delta_success'];prior=summary['p_split']
        targets.append({'root_id':s['root_id'],'q_atomic':summary['ATOMIC']['success']/4,'q_split_raw':summary['SPLIT']['success']/4,
            'delta_raw':delta,'initial_p_split':prior,'scoring_error':scoring_error,
            'target_p_split':target_for(delta,prior,scoring_error),
            'direction':'ANCHOR' if scoring_error or abs(delta)<=.25 else 'SPLIT' if delta>0 else 'ATOMIC'})
    r={**source,'config':{**source['config'],'output_dir':str(ROOT)},'source':str(SOURCE),
        'source_result_sha256':file_hash(SOURCE/'result.json'),'source_request_sha256':file_hash(SOURCE/'request.json'),
        'state_hashes':{s['state_path']:file_hash(s['state_path']) for s in source['selection']},
        'targets':targets,'steps':32,'learning_rate':1e-5,'kl_beta':.1,
        'gate_seed':571000713,'plan_seed':572000713,'natural_seed':573000713,'natural_cases':32,'cases':32,'updates':32,
        'objective':'Equal per-state soft mode CE; target sigmoid(logit(current internal-gate prior)+delta/.1). Tied/uncertain states anchor current probabilities. Root-success reward, zero cost weight. Not success frequency as mode target.',
        'design':'Copy existing internal_gate for first compute only. Freeze root_retention executor, root gate and BODY; later decisions and retries unchanged. One fixed final checkpoint. Four before/after seeded BODY probes. BEFORE/AFTER natural each four states x four new paired seeds.',
        'scope':'Known development fit on current executor values, not unseen-contract generalization. Cached values contain all original retry outcomes, including one coverage SPLIT attempt falling back to ATOMIC.',
        'decision':'Assess opposite state roles: increase route/polynomial invocation, preserve tied anchors; inspect fresh paired natural root success, split positions, costs, and anchor drift. No adaptive checkpoint selection or enlarged RL budget.'}
    write_json(ROOT/'request.json',r)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})


def preflight():
    r=json.loads((ROOT/'request.json').read_text())
    assert file_hash(SOURCE/'result.json')==r['source_result_sha256']
    assert file_hash(SOURCE/'request.json')==r['source_request_sha256']
    assert all(file_hash(p)==h for p,h in r['state_hashes'].items())
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    assert r['config']['reward']['cost_weight']==0
    assert [t['direction'] for t in r['targets']]==['SPLIT','ANCHOR','ANCHOR','SPLIT']
    assert r['adapters']['executor']=='runs/refinement/root_retention/training/final_adapter'
    assert r['adapters']['internal_gate']=='runs/refinement/internal_gate/training/final_adapter'
    return r


def context(method='BEFORE'):
    r=json.loads((ROOT/'request.json').read_text());a=r['adapters']
    initial=a['internal_gate'] if method=='BEFORE' else str(ROOT/'training/final_adapter')
    model_policy,executor=load_policies(r['framework'],initial,a['executor'])
    for name,path in [('body',a['body']),('root_gate',a['gate'])]:
        model_policy.model.load_adapter(path,adapter_name=name,is_trainable=False,local_files_only=True)
    model_policy.model.set_adapter('default');model_policy.model.requires_grad_(False)
    policy=InternalGatePolicy(model_policy.llm);policy.identity['frozen_adapter_paths']=json.dumps(a,sort_keys=True)
    return r,policy,executor,load_root_manifest(r['config']['root_manifest']),{x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}


def train():
    import torch
    from transformers import set_seed
    from .refinement_pr1 import enable_planner
    from .train_sft import trainable_snapshots, changed_tensors, prefix_forward
    preflight(); r, policy, _, _, _ = context(); before = fingerprints(policy)
    set_seed(r['gate_seed']); model = policy.model; tok = policy.tokenizer
    states = [json.loads(Path(s['state_path']).read_text()) for s in r['selection']]
    gates = [policy.score_modes(s['observation']) for s in states]
    root_before = [policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities'] for s in states]
    for gate, target in zip(gates, r['targets']):
        assert abs(gate['mode_probabilities']['SPLIT']-target['initial_p_split']) < 1e-6
    body_baselines = []
    for i, gate in enumerate(gates):
        set_seed(r['plan_seed']+i*100)
        body_baselines.append(policy.sample_split(gate))
    set_seed(r['gate_seed'])
    out = ROOT/'training'; out.mkdir(exist_ok=False)
    write_json(out/'request.json', {'steps': r['steps'], 'lr': r['learning_rate'], 'targets': r['targets']})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    model.peft_config['default'].lora_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False; enable_planner(model)
    snapshots, trainable = trainable_snapshots(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=r['learning_rate'], weight_decay=0.)
    history = []
    for step in range(r['steps']):
        enable_planner(model); optimizer.zero_grad(set_to_none=True); losses = []
        for gate, record in zip(gates, r['targets']):
            target = record['target_p_split']; modes = list(gate['mode_log_probs'])
            with policy.llm._autocast():
                lp, _ = mode_log_probs(model, gate['prefix_ids'], gate['mode_token_ids'], modes, tok.pad_token_id)
                loss = -target*lp[modes.index('SPLIT')]-(1-target)*lp[modes.index('ATOMIC')]
            assert torch.isfinite(loss)
            (loss/len(gates)).backward(); losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.); assert torch.isfinite(norm) and norm > 0
        optimizer.step(); item = {'step': step+1, 'per_state_ce': losses, 'gradient_norm': float(norm)}
        history.append(item); print(json.dumps(item), flush=True)
    final = out/'final_adapter'; model.save_pretrained(final, selected_adapters=['default']); tok.save_pretrained(final)
    changes = changed_tensors(model, snapshots); original = prefix_forward(model, gates[0]['prefix_ids'])
    model.load_adapter(str(final), adapter_name='reload_check', is_trainable=False, local_files_only=True)
    with active_adapter(model, 'reload_check'): restored = prefix_forward(model, gates[0]['prefix_ids'])
    reload_delta = float((original-restored).abs().max()); assert reload_delta <= 1e-5
    final_probs = {s['root_id']: policy.score_modes(g['observation'])['mode_probabilities']['SPLIT'] for s,g in zip(r['selection'],gates)}
    root_after = [policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities'] for s in states]
    assert root_before == root_after
    after = fingerprints(policy)
    assert all(before[k] == after[k] for k in ['root_gate','body','executor']) and before['default'] != after['default']
    probes = []
    for i,(s,g) in enumerate(zip(r['selection'], gates)):
        set_seed(r['plan_seed']+i*100); candidate = policy.sample_split(g)
        prior = body_baselines[i]
        assert candidate['body_output_ids'] == prior['body_output_ids'] and candidate['raw_action'] == prior['raw_action']
        probes.append({'root_id': s['root_id'], 'tokens_identical': True})
    write_json(out/'result.json', {'status': 'COMPLETE', 'steps': r['steps'], 'exposures': r['steps']*4,
        'history': history, 'final_probs': final_probs, 'frozen_before': before, 'frozen_after': after,
        'root_mode_probabilities_unchanged': True, 'reload_max_abs_difference': reload_delta,
        'parameter_changes': changes, 'trainable_parameters': trainable, 'body_checks': probes,
        'adapter_sha256': file_hash(final/'adapter_model.safetensors')})


def configure():
    base.ROOT=ROOT;base.context=context


def report():
    preflight();configure();base.report()
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text());pairs=[];errors=[]
    keyed={m:{(x['case']['root_id'],x['case']['seed']):x for x in cs} for m,cs in result['choices'].items()}
    assert keyed['BEFORE'].keys()==keyed['AFTER'].keys()
    for key,x in keyed['BEFORE'].items():
        y=keyed['AFTER'][key]
        def observations(z):
            es=read_jsonl(Path(z['case']['run_dir'])/'runtime/trace.jsonl')
            first=next(e for e in es if e.get('event_kind')=='POLICY' and e['node_id']==base.NODE)
            root=next(e for e in es if e.get('event_kind')=='POLICY' and e['node_id']=='N0')
            return first['observation_before'],root['parsed_action']
        assert observations(x)==observations(y)
        pairs.append({'root_id':key[0],'seed':key[1],'same_initial_observation_and_root_plan':True})
        for m,z in [('BEFORE',x),('AFTER',y)]:
            if z['case']['errors']:errors.append({'method':m,'root_id':key[0],'seed':key[1],'errors':z['case']['errors']})
    result['paired']={'wins':sum(x['wins'] for x in result['by_root']),'losses':sum(x['losses'] for x in result['by_root'])}
    result['paired']['ties']=16-result['paired']['wins']-result['paired']['losses']
    result['errors']=errors
    write_json(ROOT/'result.json',result)
    write_json(ROOT/'validation.json',{'pairs':pairs,'errors':errors,'source_values_states_adapters_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','train','natural','report'])
    p.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE');p.add_argument('--shard',type=int,choices=range(3),default=0);a=p.parse_args()
    if a.command=='prepare':prepare();preflight()
    elif a.command=='check':print(json.dumps({'status':'READY','steps':preflight()['steps'],'model_loaded':False}))
    elif a.command=='train':train()
    elif a.command=='natural':preflight();configure();base.natural(a.method,a.shard)
    else:report()
