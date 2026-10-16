"""Finite interface control, fresh current-policy values, then gate calibration."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml

DEFAULT='configs/refinement_interface_gate.yaml'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'interface','values','train','natural','probe'}:
    from .gpu_resource import acquire
    path=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else DEFAULT
    study=yaml.safe_load(Path(path).read_text())
    _reservation=acquire(Path(study['output_dir'])/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')

from .dataset import load_root_manifest, read_jsonl, file_hash
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_parent_composition import summary_prompt
from .refinement_policy import atomic_plan, load_policies, adapter_fingerprint
from .refinement_rollout import run_case, runtime_for
from .schema import PolicyOutput, TokenUsage
from .verifier import canonical

INTERNAL='N0_p1_compute'

def shape(values):
    types={type(v).__name__ for v in values}
    if len(types)!=1:return 'native value (types: '+', '.join(sorted(types))+')'
    kind=next(iter(types))
    if kind=='dict':
        keys=[set(v) for v in values]
        if keys and all(k==keys[0] for k in keys):
            return 'dict {'+', '.join(k+': '+shape([v[k] for v in values]) for k in sorted(keys[0]))+'}'
        return 'dict'
    return {'str':'str (native string data)', 'list':'list', 'NoneType':'None'}.get(kind,kind)

def bare(contract):
    return {k:v for k,v in contract.items() if k not in {'contract_id','version'}}

def interface_note(observation,definition):
    catalog={'root':definition['contract'],**definition['local_contracts']}
    entries=[]
    # Only public VISIBLE suites; never read private tests or emit example values.
    for key,contract in catalog.items():
        suites=[s for s in definition['suites'] if s['visibility']=='VISIBLE' and
                (s['check_kind']=='ROOT' if key=='root' else s.get('contract_key')==key)]
        cases=[c for s in suites for c in s['cases']]
        if not cases:continue
        params=contract['signature']['parameters']
        inputs={p['name']:shape([c['args'][i] for c in cases]) for i,p in enumerate(params)}
        result=shape([c['expected'] for c in cases])
        if key=='root':
            result='str containing exactly one JSON serialization of '+shape([json.loads(c['expected']) for c in cases])
        entries.append({'contract':bare(contract),'function':contract['signature']['name'],
                        'parameters':inputs,'returns':result})
    current=next((e for e in entries if e['contract']==bare(observation['incoming_contract'])),None)
    # Child interfaces are identified by exact contract, never by an inferred name.
    return {'source':'Structural types and fixed fields from public visible checks; no example values.',
            'current_function':{k:v for k,v in current.items() if k!='contract'} if current else {
                'signature':observation['incoming_contract']['signature'],
                'requires':observation['incoming_contract']['requires'],
                'returns':'Follow this exact contract; internal return is native, not JSON encoded.'},
            'public_interfaces':[{k:v for k,v in e.items() if k!='contract'} for e in entries],
            'serialization_owner':'Only root solve serializes the final result once. Internal functions return native values; a native string is allowed when it is the actual intermediate data. Do not json.dumps an internal object or convert a dictionary result into a tuple/list.'}

def prompt_for(definition,condition):
    def builder(tokenizer,observation,**kwargs):
        obs=deepcopy(observation)
        if condition=='CLARIFIED' and obs['phase']=='IMPLEMENT':
            obs['root_spec']+='\nExecution interface clarification:\n'+canonical(interface_note(obs,definition))
        return summary_prompt(tokenizer,obs,**kwargs)
    return builder

class FixedPlanner:
    def __init__(self,base,plans):
        self.cost=base.cost;self.plans=plans
        self.identity={'backend':'fixed_MULTIBODY_plans','learned_planner':canonical(base.identity)}
    def act(self,obs):
        action=self.plans.get(obs['node_id'],atomic_plan(obs['incoming_contract']))
        return PolicyOutput(raw_text=canonical(action),token_usage=TokenUsage(),model_identity=self.identity)

def prepare(study):
    source=Path(study['source_dir']);request=json.loads((source/'request.json').read_text())
    output=Path(study['output_dir']);output.mkdir(parents=True,exist_ok=True)
    request={'study':study,'config':request['config'],'framework':request['framework'],
        'planner_adapter':request['adapters']['MULTI_BODY'],
        'scope':'TRAIN interface selection and gate values; VALIDATION only final evaluation',
        'adoption_rule':'Clarified only if total wins increase and neither TRAIN state loses; otherwise original.',
        'value_sampling':'4 fresh SPLIT plans x 3 continuations and 12 paired ATOMIC continuations per TRAIN root; include rejection and recovery.',
        'gate_objective':'mean(-p_split * thresholded_delta + beta KL(mode || frozen MULTI_BODY)) + TRAIN body CE retention',
        'fixed_plan_sources':{}}
    for root in study['interface_roots']:
        request['fixed_plan_sources'][root]={node:str(source/'evaluation/MULTI_BODY/conditional'/f'{root}_{node}'/'sample_00/candidate.json') for node in ['N0',INTERNAL]}
    guarded_json(output/'request.json',request)
    guarded_json(output/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000,'train':22000}})
    return request

def context(study,method='BEFORE'):
    root=Path(study['output_dir']);r=json.loads((root/'request.json').read_text())
    adapter=r['planner_adapter'] if method=='BEFORE' else str(root/'training/final_adapter')
    planner,executor=load_policies(r['framework'],adapter,r['config']['executor_adapter'])
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    return root,r,planner,executor,rows,tests

def success(case):return case['final_scores']['private']['status']=='PASS'

def checked(case):
    if case['status']!='COMPLETE':raise ValueError('Infrastructure failure: '+case['run_dir'])
    return case

def frozen(planner):return {n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}

def interface(study,condition):
    root,r,planner,executor,rows,tests=context(study);before=frozen(planner);cases=[]
    for i,rid in enumerate(study['interface_roots']):
        definition=json.loads(Path(rows[rid]['fixture_path']).read_text())
        executor.llm.prompt_builder=prompt_for(definition,condition)
        plans={n:json.loads(Path(p).read_text())['parsed_action'] for n,p in r['fixed_plan_sources'][rid].items()}
        wrapped=FixedPlanner(planner,plans)
        for rep in range(4):
            c=checked(run_case(rows[rid],tests,r['framework'],r['config'],wrapped,executor,
                root/'interface'/condition/rid/f'repeat_{rep:02d}',study['interface_seed']+i*100+rep,
                reset_seed_after_root_plan=True,reset_seed_after_plan_node=INTERNAL))
            for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl'):
                if e['host_result']['code']=='PLAN_REGISTERED':
                    expected=plans.get(e['node_id'],atomic_plan(e['observation_before']['incoming_contract']))
                    assert e['parsed_action']==expected
            cases.append(c);print(json.dumps({'stage':'interface','condition':condition,'root':rid,'rep':rep,'pass':success(c)}),flush=True)
    after=frozen(planner);assert before==after
    write_json(root/'interface'/condition/'result.json',{'cases':cases,'frozen_before':before,'frozen_after':after})

def select_interface(study):
    root=Path(study['output_dir']);arms={a:json.loads((root/'interface'/a/'result.json').read_text()) for a in ['ORIGINAL','CLARIFIED']}
    counts={rid:{a:sum(success(c) for c in r['cases'] if c['root_id']==rid) for a,r in arms.items()} for rid in study['interface_roots']}
    adopt=all(v['CLARIFIED']>=v['ORIGINAL'] for v in counts.values()) and sum(v['CLARIFIED']-v['ORIGINAL'] for v in counts.values())>0
    result={'status':'COMPLETE','runs':16,'by_root':counts,'selected':'CLARIFIED' if adopt else 'ORIGINAL'}
    guarded_json(root/'interface/result.json',result);print(json.dumps(result),flush=True)

def values(study,shard):
    from transformers import set_seed
    root,r,planner,executor,rows,tests=context(study);before=frozen(planner)
    condition=json.loads((root/'interface/result.json').read_text())['selected'];records=[]
    for i,rid in enumerate(study['value_roots']):
        if i%2!=shard:continue
        assert rows[rid]['split']=='TRAIN'
        base=root/'values'/rid;base.mkdir(parents=True,exist_ok=True)
        definition=json.loads(Path(rows[rid]['fixture_path']).read_text());executor.llm.prompt_builder=prompt_for(definition,condition)
        state=next(s for s in read_jsonl('data/refinement/body_transfer/states.jsonl') if s['root_id']==rid and s['node_id']=='N0')
        obs=json.loads(state['messages'][1]['content'])['observation'];gate=planner.score_modes(obs);write_json(base/'gate.json',gate)
        candidates=[]
        for j in range(4):
            path=base/f'candidate_{j:02d}.json'
            if path.exists():candidate=json.loads(path.read_text())
            else:
                set_seed(study['value_seed']+i*10000+j);candidate=planner.sample_split(gate);write_json(path,candidate)
            candidates.append(candidate)
        cases={a:[] for a in ['ATOMIC','SPLIT']}
        for j in range(4):
            for rep in range(3):
                seed=study['value_seed']+i*10000+100+j*3+rep
                for arm in ['ATOMIC','SPLIT']:
                    candidate=planner.atomic(gate) if arm=='ATOMIC' else candidates[j]
                    c=checked(run_case(rows[rid],tests,r['framework'],r['config'],planner,executor,
                        base/arm/f'plan_{j:02d}_repeat_{rep:02d}',seed,initial=(gate,candidate)))
                    cases[arm].append(c)
                    print(json.dumps({'stage':'values','root':rid,'arm':arm,'plan':j,'rep':rep,'pass':success(c)}),flush=True)
        qa=sum(map(success,cases['ATOMIC']))/12;qs=sum(map(success,cases['SPLIT']))/12
        record={'root_id':rid,'q_atomic':qa,'q_split':qs,'delta':qs-qa,'p_split':gate['mode_probabilities']['SPLIT'],'cases':cases}
        write_json(base/'result.json',record);records.append(record)
    assert frozen(planner)==before
    write_json(root/'values'/f'shard_{shard}.json',{'status':'COMPLETE','records':records,'frozen':before,'condition':condition})

def value_records(study):
    root=Path(study['output_dir'])
    records=[json.loads((root/'values'/rid/'result.json').read_text()) for rid in study['value_roots']]
    assert all(len(r['cases'][a])==12 for r in records for a in ['ATOMIC','SPLIT'])
    return records

def train(study):
    import torch
    import torch.nn.functional as F
    from transformers import set_seed
    from .refinement_policy import active_adapter,mode_log_probs
    from .refinement_pr1 import causal_logits,enable_planner
    from .train_refinement import encode_demo
    from .train_sft import changed_tensors,prefix_forward,trainable_snapshots
    values_=value_records(study)
    root,r,planner,executor,rows,tests=context(study);output=root/'training';output.mkdir(exist_ok=False)
    set_seed(study['train_seed']);model=planner.model;tok=planner.tokenizer
    states=[s for s in read_jsonl('data/refinement/body_transfer/states.jsonl') if s['split']=='TRAIN']
    examples=[encode_demo(s,tok,r['framework']) for s in states]
    gates=[json.loads((root/'values'/v['root_id']/'gate.json').read_text()) for v in values_]
    deltas=[v['delta'] if abs(v['delta'])>study['close_margin'] else 0. for v in values_]
    if not any(deltas):raise ValueError('All TRAIN differences are within close margin; do not force a gate update')
    frozen_executor=adapter_fingerprint(model,'executor')
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.peft_config['default'].lora_dropout=0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    enable_planner(model);before,trainable=trainable_snapshots(model)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=study['learning_rate'],weight_decay=0.)
    history=[]
    guarded_json(output/'request.json',{'study':study,'values':[{k:v[k] for k in ['root_id','q_atomic','q_split','delta']} for v in values_],
        'effective_deltas':deltas,'initial_adapter':r['planner_adapter'],'body_states':[s['sample_id'] for s in states]})
    for step in range(study['optimizer_steps']):
        optimizer.zero_grad(set_to_none=True);enable_planner(model);total=0.
        for gate,delta in zip(gates,deltas):
            modes=list(gate['mode_log_probs'])
            with planner.llm._autocast():
                lp,_=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],modes,tok.pad_token_id)
                ref=torch.tensor([gate['mode_log_probs'][m] for m in modes],device=model.device)
                kl=(lp.exp()*(lp-ref)).sum()
                loss=(-lp[modes.index('SPLIT')].exp()*delta+study['kl_weight']*kl)/len(gates)
            loss.backward();total+=float(loss.detach());del loss,lp
        e=examples[step%len(examples)]
        with planner.llm._autocast():
            logits=causal_logits(model,e['body_prefix'],e['body_target'])
            body=F.cross_entropy(logits.float(),torch.tensor(e['body_target'],device=model.device))
        (study['body_weight']*body).backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Invalid calibration gradient')
        optimizer.step();record={'step':step+1,'gate_objective':total,'body_ce':float(body.detach()),'gradient_norm':float(norm)}
        history.append(record);print(json.dumps(record),flush=True);del logits,body
    final=output/'final_adapter';model.save_pretrained(final,selected_adapters=['default']);tok.save_pretrained(final)
    changes=changed_tensors(model,before);original=prefix_forward(model,examples[0]['prefix'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'):restored=prefix_forward(model,examples[0]['prefix'])
    reload_delta=float((original-restored).abs().max())
    assert frozen_executor==adapter_fingerprint(model,'executor') and reload_delta<=1e-5
    write_json(output/'result.json',{'status':'COMPLETE','steps':len(history),'history':history,'parameter_changes':changes,
        'trainable_parameters':trainable,'reload_max_abs_difference':reload_delta,'executor_frozen':frozen_executor,
        'effective_deltas':deltas,'adapter_sha256':file_hash(final/'adapter_model.safetensors')})

def natural(study,method):
    root,r,planner,executor,rows,tests=context(study,method);before=frozen(planner)
    condition=json.loads((root/'interface/result.json').read_text())['selected'];cases=[];gates=[]
    selected=[rows[rid] for rid in study['value_roots']]+[row for row in rows.values() if row['split']=='VALIDATION']
    states=read_jsonl('data/refinement/body_transfer/states.jsonl')
    for i,row in enumerate(selected):
        rid=row['root_id'];definition=json.loads(Path(row['fixture_path']).read_text())
        executor.llm.prompt_builder=prompt_for(definition,condition)
        state=next(s for s in states if s['root_id']==rid and s['node_id']=='N0')
        gate=planner.score_modes(json.loads(state['messages'][1]['content'])['observation'])
        gates.append({'root_id':rid,'split':row['split'],'p_split':gate['mode_probabilities']['SPLIT']})
        for rep in range(4 if row['split']=='TRAIN' else 2):
            c=checked(run_case(row,tests,r['framework'],r['config'],planner,executor,
                root/'natural'/method/rid/f'repeat_{rep:02d}',study['natural_seed']+i*100+rep))
            assert row['split']=='TRAIN' or c['reward'] is None
            cases.append(c);print(json.dumps({'stage':'natural','method':method,'root':rid,'rep':rep,'pass':success(c)}),flush=True)
    assert frozen(planner)==before
    write_json(root/'natural'/method/'result.json',{'status':'COMPLETE','cases':cases,'gates':gates,'frozen':before,'condition':condition})

def probe(study,method,shard=0,shards=1):
    import torch
    from transformers import set_seed
    from .refinement_body_transfer import at_state,reviewed_plans
    from .refinement_curriculum_rl import plan_key
    root,r,planner,executor,rows,tests=context(study,method);before=frozen(planner);records=[]
    source=json.loads((Path(study['source_dir'])/'request.json').read_text())
    states=read_jsonl('data/refinement/body_transfer/states.jsonl')
    for i,state in enumerate(states):
        if i%shards!=shard:continue
        base=root/'probe'/method/state['sample_id'];base.mkdir(parents=True,exist_ok=True)
        obs=json.loads(state['messages'][1]['content'])['observation'];definition=json.loads(Path(rows[state['root_id']]['fixture_path']).read_text())
        gate=planner.score_modes(obs);write_json(base/'gate.json',gate)
        variants=reviewed_plans(obs,definition);target=plan_key(state['target_action'],definition)
        for rep in range(4):
            folder=base/f'sample_{rep:02d}';path=folder/'result.json'
            if path.exists():records.append(json.loads(path.read_text()));continue
            folder.mkdir(parents=True,exist_ok=True)
            seed=source['study']['probe_seed']+100*i+rep;set_seed(seed)
            candidate=planner.sample_split(gate);write_json(folder/'candidate.json',candidate)
            policy=SimpleNamespace(act=lambda obs,c=candidate:planner.as_output(gate,c))
            runtime=at_state(state,rows[state['root_id']],tests,r['framework'],definition,folder/'registration',policy)
            event=runtime.step();registered=event.host_result.code=='PLAN_REGISTERED'
            key=plan_key(candidate['parsed_action'],definition) if registered else None
            record={**{k:state[k] for k in ['sample_id','root_id','split','node_id']},'seed':seed,'registered':registered,
                'reviewed':registered and key is not None and key in variants,'target_match':registered and key is not None and key==target}
            write_json(path,record);records.append(record)
        print(json.dumps({'stage':'probe','method':method,'state':i+1,'reviewed':sum(x['reviewed'] for x in records[-4:])}),flush=True)
    assert frozen(planner)==before
    write_json(root/'probe'/method/('result.json' if shards==1 else f'result_shard_{shard}.json'),{'status':'COMPLETE','records':records,'frozen':before})

def report(study):
    root=Path(study['output_dir']);interface_=json.loads((root/'interface/result.json').read_text());values_=value_records(study)
    natural_={m:json.loads((root/'natural'/m/'result.json').read_text()) for m in ['BEFORE','AFTER']}
    source_identity=json.loads((Path(study['source_dir'])/'evaluation/MULTI_BODY/result.json').read_text())['frozen_before']
    assert source_identity==natural_['BEFORE']['frozen']
    old=json.loads((Path(study['source_dir'])/'evaluation/MULTI_BODY/conditional_result.json').read_text())['records']
    probe_path=root/'probe/AFTER/result.json'
    if not probe_path.exists():
        pieces=[json.loads((root/'probe/AFTER'/f'result_shard_{i}.json').read_text()) for i in range(2)]
        assert pieces[0]['frozen']==pieces[1]['frozen']
        write_json(probe_path,{'status':'COMPLETE','records':[x for p in pieces for x in p['records']],'frozen':pieces[0]['frozen']})
    new=json.loads(probe_path.read_text())['records']
    assert len(old)==len(new)==192
    assert {(x['sample_id'],x['seed']) for x in old}=={(x['sample_id'],x['seed']) for x in new}
    body={m:{split:{key:sum(x[key] for x in records if x['split']==split) for key in ['reviewed','registered','target_match']} for split in ['TRAIN','VALIDATION']} for m,records in [('BEFORE',old),('AFTER',new)]}
    execution={};paired={}
    for split in ['TRAIN','VALIDATION']:
        execution[split]={m:{'runs':sum(c['split']==split for c in v['cases']),'successes':sum(success(c) for c in v['cases'] if c['split']==split)} for m,v in natural_.items()}
        a={(c['root_id'],c['seed']):success(c) for c in natural_['BEFORE']['cases'] if c['split']==split}
        b={(c['root_id'],c['seed']):success(c) for c in natural_['AFTER']['cases'] if c['split']==split}
        assert a.keys()==b.keys()
        paired[split]={'wins':sum(b[k]>a[k] for k in a),'losses':sum(b[k]<a[k] for k in a),'ties':sum(b[k]==a[k] for k in a)}
    gates={m:{g['root_id']:g['p_split'] for g in v['gates']} for m,v in natural_.items()}
    value_summary=[{**{k:v[k] for k in ['root_id','q_atomic','q_split','delta']},'p_before':gates['BEFORE'][v['root_id']],'p_after':gates['AFTER'][v['root_id']],
        'split_successes_by_plan':[sum(success(c) for c in v['cases']['SPLIT'][j*3:(j+1)*3]) for j in range(4)],
        'first_split_rejections':sum(not c['root_first_plan_registered'] for c in v['cases']['SPLIT']),
        'protocol_rejections':{a:sum(sum(c['protocol_rejections'].values()) for c in v['cases'][a]) for a in ['ATOMIC','SPLIT']}} for v in values_]
    from .refinement_body_transfer import reviewed_plans
    from .refinement_curriculum_rl import plan_key
    request=json.loads((root/'request.json').read_text());rows=load_root_manifest(request['config']['root_manifest'])
    for value in value_summary:
        rid=value['root_id'];definition=json.loads(Path(rows[rid]['fixture_path']).read_text())
        gate=json.loads((root/'values'/rid/'gate.json').read_text());variants=reviewed_plans(gate['observation'],definition)
        plans=[json.loads((root/'values'/rid/f'candidate_{j:02d}.json').read_text()) for j in range(4)]
        value['sampled_plan_variants']=[variants.get(plan_key(p['parsed_action'],definition)) if p['parsed_action'] else None for p in plans]
    root_choices={}
    for method,evaluation in natural_.items():
        choices=[]
        for c in evaluation['cases']:
            events=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
            first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']=='N0')
            action=first.get('parsed_action') or {}
            choices.append({'root_id':c['root_id'],'seed':c['seed'],'split':c['split'],
                'first_mode':action.get('args',{}).get('mode'),'first_registered':c['root_first_plan_registered'],'success':success(c)})
        root_choices[method]=choices
    result={'status':'COMPLETE','root_choices':root_choices,'interface':interface_,'values':value_summary,'natural':execution,'paired_natural':paired,'body':body,'gates':gates,
        'training':json.loads((root/'training/result.json').read_text()),'scope':'Finite TRAIN calibration with unchanged F01; existing validation development compositions, not an independent confirmation set.'}
    write_json(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='training'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','interface','select','values','train','natural','probe','report'])
    parser.add_argument('--config',default=DEFAULT);parser.add_argument('--condition',default='ORIGINAL')
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--shard',type=int,default=0);parser.add_argument('--method',default='BEFORE')
    args=parser.parse_args();study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':prepare(study)
    elif args.command=='interface':interface(study,args.condition)
    elif args.command=='select':select_interface(study)
    elif args.command=='values':values(study,args.shard)
    elif args.command=='probe':probe(study,args.method,args.shard,args.shards)
    else:globals()[args.command](study,args.method) if args.command=='natural' else globals()[args.command](study)
