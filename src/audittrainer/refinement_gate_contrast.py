"""Current-executor positive/negative internal-gate calibration and natural runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path('runs/refinement/gate_contrast')
DATA=Path('data/refinement/gate_contrast')
SOURCES=[Path('runs/refinement/anchor_value_refresh'),
         Path('runs/refinement/contract_value_refresh')]
GATE='runs/refinement/transfer_gate/training/final_adapter'
EXECUTOR='runs/refinement/contract_correction/training/final_adapter'
ROOTS=[
    'validation_internal_transfer_routes_outside',
    'train_contract_correction_fresh_coverage_exact_two',
    'train_contract_correction_fresh_stock_absolute',
    'train_contract_correction_fresh_polynomial_difference',
    'validation_internal_transfer_polynomial_joint']
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','natural'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json',
                         'train' if sys.argv[1]=='train' else 'collect')

from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import write_json
from .refinement_internal_gate import InternalGatePolicy,NaturalInternal,NODE,fingerprints,target_for
from .refinement_policy import load_policies,active_adapter,mode_log_probs
from .refinement_rollout import run_case
from .verifier import canonical


def prepare():
    assert not (ROOT/'request.json').exists(), 'request.json already exists; frozen experiment will not be overwritten'
    packages=[]
    for source in SOURCES:
        request=json.loads((source/'request.json').read_text())
        result=json.loads((source/'result.json').read_text())
        assert result['status']=='COMPLETE' and not result['errors'] and result['updates']==0
        packages.append((source,request,result))
    assert packages[0][1]['framework']==packages[1][1]['framework']
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    rows=[];tests=[];selection=[];targets=[]
    for rid in ROOTS:
        source,request,result=next((s,q,o) for s,q,o in packages
                                   if any(x['root_id']==rid for x in q['selection']))
        selected=next(x for x in request['selection'] if x['root_id']==rid)
        summary=next(x for x in result['summary'] if x['root_id']==rid)
        group=next(x for x in result['groups'] if x['root_id']==rid)
        assert all(not x['case']['errors'] for x in group['cases'])
        manifest=load_root_manifest(request['config']['root_manifest'])
        test_by_id={x['root_id']:x for x in read_jsonl(request['config']['test_manifest'])}
        row=manifest[rid];test=test_by_id[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(test['path'])==test['sha256']
        state=json.loads(Path(selected['state_path']).read_text())
        assert state['observation']['node_id']==NODE
        assert state['root_event']['host_result']['code']=='PLAN_REGISTERED'
        folder=ROOT/rid;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'state.json',state)
        rows.append(row);tests.append(test)
        selection.append({'root_id':rid,'state_path':str(folder/'state.json'),
                          'source_state_path':selected['state_path'],
                          'source':str(source)})
        delta=summary['delta_success'];prior=summary['p_split']
        targets.append({'root_id':rid,'q_atomic':summary['ATOMIC']['success']/4,
                        'q_split_raw':summary['SPLIT']['success']/4,
                        'delta_raw':delta,'initial_p_split':prior,
                        'target_p_split':target_for(delta,prior),
                        'direction':'ANCHOR' if abs(delta)<=.25 else
                                    'SPLIT' if delta>0 else 'ATOMIC'})
    assert [x['direction'] for x in targets]==[
        'SPLIT','ATOMIC','ANCHOR','ANCHOR','ANCHOR']
    for label,values in [('roots',rows),('tests',tests)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    current=packages[1][1]
    adapters={**current['adapters'],'internal_gate':GATE,
              'executor':EXECUTOR,'initial':GATE}
    request={'framework':current['framework'],
             'config':{**current['config'],'root_manifest':str(DATA/'roots.jsonl'),
                       'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},
             'adapters':adapters,
             'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
             'selection':selection,'targets':targets,
             'source_hashes':{str(s):{'request':file_hash(s/'request.json'),
                                      'result':file_hash(s/'result.json')}
                              for s,_,_ in packages},
             'state_hashes':{x['source_state_path']:file_hash(x['source_state_path'])
                             for x in selection},
             'steps':32,'learning_rate':1e-5,'kl_beta':0.1,
             'gate_seed':631000713,'plan_seed':632000713,
             'natural_seed':633000713,'repetitions':4,'natural_cases':40,
             'shards':3,'updates':32,
             'objective':'Equal exposure soft gate-mode CE. Target sigmoid(logit(current p_split)+delta/0.1); root success only, cost weight zero. Three tied states anchor their current probability. No adaptive checkpoint.',
             'scope':'Known development states on fixed root plans. Frozen BODY, current contract_correction executor, root gate, and later decisions. Natural comparison is same-state paired development, not unseen-contract generalization.',
             'decision':'Require coverage P(SPLIT) to fall while route P(SPLIT) stays high; compare paired natural root success, split locations, and cost. Do not update gate from ties as if they were negative success labels.'}
    write_json(ROOT/'request.json',request)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],
                                          'min_free_mib':{'train':22000,'collect':16000}})


def check():
    r=json.loads((ROOT/'request.json').read_text())
    assert r['updates']==r['steps']==32 and r['natural_cases']==40
    assert [x['root_id'] for x in r['selection']]==ROOTS
    assert [x['direction'] for x in r['targets']]==[
        'SPLIT','ATOMIC','ANCHOR','ANCHOR','ANCHOR']
    assert r['config']['reward']['cost_weight']==0
    assert r['adapters']['internal_gate']==GATE and r['adapters']['executor']==EXECUTOR
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k]
               for k,p in r['adapters'].items())
    for source,hashes in r['source_hashes'].items():
        assert file_hash(Path(source)/'request.json')==hashes['request']
        assert file_hash(Path(source)/'result.json')==hashes['result']
    assert all(file_hash(p)==h for p,h in r['state_hashes'].items())
    rows=load_root_manifest(r['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    for selected in r['selection']:
        rid=selected['root_id'];row=rows[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(tests[rid]['path'])==tests[rid]['sha256']
        state=json.loads(Path(selected['state_path']).read_text())
        assert state['observation']['node_id']==NODE
    return r


def context(method='BEFORE'):
    r=check();a=r['adapters']
    initial=a['internal_gate'] if method=='BEFORE' else str(ROOT/'training/final_adapter')
    model_policy,executor=load_policies(r['framework'],initial,a['executor'])
    for name,path in [('body',a['body']),('root_gate',a['gate'])]:
        model_policy.model.load_adapter(path,adapter_name=name,
                                        is_trainable=False,local_files_only=True)
    model_policy.model.set_adapter('default');model_policy.model.requires_grad_(False)
    policy=InternalGatePolicy(model_policy.llm)
    policy.identity['frozen_adapter_paths']=json.dumps(a,sort_keys=True)
    rows=load_root_manifest(r['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    return r,policy,executor,rows,tests


def train():
    import torch
    from tqdm import tqdm
    from transformers import set_seed
    from .refinement_pr1 import enable_planner
    from .train_sft import trainable_snapshots,changed_tensors,prefix_forward
    r,policy,_,_,_=context();frozen_before=fingerprints(policy)
    set_seed(r['gate_seed']);model=policy.model;tok=policy.tokenizer
    states=[json.loads(Path(x['state_path']).read_text()) for x in r['selection']]
    gates=[policy.score_modes(s['observation']) for s in states]
    for gate,target in zip(gates,r['targets']):
        assert abs(gate['mode_probabilities']['SPLIT']-target['initial_p_split'])<1e-6
    root_before=[policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities']
                 for s in states]
    body_baselines=[]
    for i,gate in enumerate(gates):
        set_seed(r['plan_seed']+i*100)
        body_baselines.append(policy.sample_split(gate))
    set_seed(r['gate_seed'])
    out=ROOT/'training';out.mkdir(exist_ok=False)
    write_json(out/'request.json',{'steps':r['steps'],'lr':r['learning_rate'],
                                   'targets':r['targets']})
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.peft_config['default'].lora_dropout=0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    enable_planner(model)
    snapshots,trainable=trainable_snapshots(model)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=r['learning_rate'],weight_decay=0.)
    history=[]
    for step in tqdm(range(r['steps']),desc='内部门控SFT',unit='步',dynamic_ncols=True):
        enable_planner(model);optimizer.zero_grad(set_to_none=True);losses=[]
        for gate,record in zip(gates,r['targets']):
            target=record['target_p_split'];modes=list(gate['mode_log_probs'])
            with policy.llm._autocast():
                lp,_=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],
                                   modes,tok.pad_token_id)
                loss=-target*lp[modes.index('SPLIT')]-(1-target)*lp[modes.index('ATOMIC')]
            assert torch.isfinite(loss)
            (loss/len(gates)).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        assert torch.isfinite(norm) and norm>0
        optimizer.step()
        history.append({'step':step+1,'per_state_ce':losses,'gradient_norm':float(norm)})
    final=out/'final_adapter';model.save_pretrained(final,selected_adapters=['default'])
    tok.save_pretrained(final)
    changes=changed_tensors(model,snapshots)
    original=prefix_forward(model,gates[0]['prefix_ids'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,
                       local_files_only=True)
    with active_adapter(model,'reload_check'):
        restored=prefix_forward(model,gates[0]['prefix_ids'])
    delta=float((original-restored).abs().max());assert delta<=1e-5
    after=fingerprints(policy)
    assert all(frozen_before[k]==after[k] for k in ['root_gate','body','executor'])
    assert frozen_before['default']!=after['default']
    root_after=[policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities']
                for s in states]
    assert root_before==root_after
    probes=[]
    for i,(selected,gate) in enumerate(zip(r['selection'],gates)):
        set_seed(r['plan_seed']+i*100)
        candidate=policy.sample_split(gate)
        prior=body_baselines[i]
        assert candidate['body_output_ids']==prior['body_output_ids']
        assert candidate['raw_action']==prior['raw_action']
        probes.append({'root_id':selected['root_id'],'tokens_identical':True})
    final_probs={selected['root_id']:policy.score_modes(state['observation'])['mode_probabilities']['SPLIT']
                 for selected,state in zip(r['selection'],states)}
    write_json(out/'result.json',{'status':'COMPLETE','steps':r['steps'],
        'exposures':r['steps']*len(gates),'history':history,
        'final_probs':final_probs,'frozen_before':frozen_before,'frozen_after':after,
        'root_mode_probabilities_unchanged':True,'reload_max_abs_difference':delta,
        'parameter_changes':changes,'trainable_parameters':trainable,
        'body_checks':probes,'adapter_sha256':file_hash(final/'adapter_model.safetensors')})


def natural(method,shard):
    from tqdm import tqdm
    r,policy,executor,rows,tests=context(method)
    frozen=fingerprints(policy);cases=[];gates=[]
    jobs=[(i,rep) for i in range(len(r['selection'])) if i%r['shards']==shard
          for rep in range(r['repetitions'])]
    for i,rep in tqdm(jobs,desc=f'{method} 自然分片{shard}',unit='条',dynamic_ncols=True):
        selected=r['selection'][i];rid=selected['root_id']
        state=json.loads(Path(selected['state_path']).read_text())
        folder=ROOT/'natural'/method/rid;folder.mkdir(parents=True,exist_ok=True)
        if rep==0:
            gate=policy.score_modes(state['observation'])
            write_json(folder/'gate.json',gate)
            gates.append({'root_id':rid,'p_split':gate['mode_probabilities']['SPLIT']})
        branch=NaturalInternal(policy,state)
        directory=folder/f'repeat_{rep:02d}'
        cached=(directory/'case_result.json').exists()
        case=run_case(rows[rid],tests,r['framework'],r['config'],branch,executor,
                      directory,r['natural_seed']+i*100+rep)
        assert case['status']=='COMPLETE' or case['errors']==['SCORING_ERROR']
        if not cached:assert branch.root_sent and branch.internal_sent
        events=read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
        first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==NODE)
        assert first['observation_before']==state['observation']
        cases.append({'case':case,'rep':rep,
                      'mode':first['model_input']['refinement']['selected_mode'],
                      'first_registered':first['host_result']['code']=='PLAN_REGISTERED'})
    assert fingerprints(policy)==frozen
    write_json(ROOT/'natural'/method/f'shard_{shard}.json',
               {'status':'COMPLETE','cases':cases,'gates':gates,'frozen':frozen})


def report():
    r=check();training=json.loads((ROOT/'training/result.json').read_text())
    assert training['status']=='COMPLETE' and training['steps']==r['steps']
    arms={};frozen={};probs={}
    for method in ['BEFORE','AFTER']:
        pieces=[json.loads((ROOT/'natural'/method/f'shard_{i}.json').read_text())
                for i in range(r['shards'])]
        assert all(x['frozen']==pieces[0]['frozen'] for x in pieces)
        frozen[method]=pieces[0]['frozen']
        arms[method]=[x for piece in pieces for x in piece['cases']]
        probs[method]={x['root_id']:x['p_split'] for piece in pieces
                       for x in piece['gates']}
        assert len(arms[method])==r['natural_cases']//2
    assert all(frozen['BEFORE'][k]==frozen['AFTER'][k]
               for k in ['root_gate','body','executor'])
    keyed={method:{(x['case']['root_id'],x['case']['seed']):x for x in values}
           for method,values in arms.items()}
    assert keyed['BEFORE'].keys()==keyed['AFTER'].keys()
    pairs=[];errors=[];by_root=[]
    for selected in r['selection']:
        rid=selected['root_id'];item={'root_id':rid}
        for method in ['BEFORE','AFTER']:
            rows=[x for x in arms[method] if x['case']['root_id']==rid]
            assert len(rows)==r['repetitions']
            item[method]={'success':sum(x['case']['final_scores']['private']['status']=='PASS'
                                        for x in rows),
                'split':sum(x['mode']=='SPLIT' for x in rows),
                'p_split':probs[method][rid],
                'mean_tokens':sum(x['case']['usage']['input_tokens']+
                                  x['case']['usage']['output_tokens'] for x in rows)/len(rows),
                'first_registered':sum(x['first_registered'] for x in rows)}
        item['wins']=sum(keyed['AFTER'][(rid,x['case']['seed'])]['case']['final_scores']['private']['status']=='PASS'
                         and x['case']['final_scores']['private']['status']!='PASS'
                         for x in arms['BEFORE'] if x['case']['root_id']==rid)
        item['losses']=sum(keyed['AFTER'][(rid,x['case']['seed'])]['case']['final_scores']['private']['status']!='PASS'
                           and x['case']['final_scores']['private']['status']=='PASS'
                           for x in arms['BEFORE'] if x['case']['root_id']==rid)
        by_root.append(item)
    for key,before in keyed['BEFORE'].items():
        after=keyed['AFTER'][key]
        def first_and_root(x):
            events=read_jsonl(Path(x['case']['run_dir'])/'runtime/trace.jsonl')
            first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==NODE)
            root=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']=='N0')
            return first['observation_before'],root['parsed_action']
        assert first_and_root(before)==first_and_root(after)
        pairs.append({'root_id':key[0],'seed':key[1],
                      'same_initial_observation_and_root_plan':True})
        for method,value in [('BEFORE',before),('AFTER',after)]:
            if value['case']['errors']:
                errors.append({'method':method,'root_id':key[0],
                               'seed':key[1],'errors':value['case']['errors']})
    totals={method:{'runs':len(values),
        'success':sum(x[method]['success'] for x in by_root),
        'split':sum(x[method]['split'] for x in by_root),
        'mean_tokens':sum(x[method]['mean_tokens'] for x in by_root)/len(by_root)}
        for method,values in arms.items()}
    paired={'wins':sum(x['wins'] for x in by_root),
            'losses':sum(x['losses'] for x in by_root)}
    paired['ties']=len(pairs)-paired['wins']-paired['losses']
    write_json(ROOT/'result.json',{'status':'COMPLETE','natural_cases':r['natural_cases'],
        'updates':r['updates'],'by_root':by_root,'totals':totals,'paired':paired,
        'errors':errors,'training':training,'choices':arms,'frozen':frozen})
    write_json(ROOT/'validation.json',{'pairs':pairs,'all_controls_frozen':True,
        'errors':errors})
    print(json.dumps({'by_root':by_root,'totals':totals,'paired':paired,
                      'errors':errors},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','train','natural','report'])
    parser.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE')
    parser.add_argument('--shard',type=int,choices=range(3),default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare();check()
    elif args.command=='check':
        r=check();print(json.dumps({'status':'READY','steps':r['steps'],
                                    'natural_cases':r['natural_cases'],
                                    'model_loaded':False}))
    elif args.command=='train':train()
    elif args.command=='natural':natural(args.method,args.shard)
    else:report()
