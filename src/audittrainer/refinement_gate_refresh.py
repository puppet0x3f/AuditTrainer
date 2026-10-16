"""Fresh executor-conditioned values and stable independent root-gate calibration."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path('runs/refinement/gate_refresh')
DATA=Path('data/refinement/gate_refresh')
EXECUTOR='runs/refinement/executor_sft/training/final_adapter'
INITIAL='runs/refinement/separate_gate/CALIBRATED/final_adapter'
BODY='runs/refinement/body_transfer/MULTI_BODY/final_adapter'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'values','train','natural'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')

from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_policy import load_policies,load_adapter_metadata,active_adapter,mode_log_probs
from .refinement_separate_gate import RootGatePolicy,fingerprints
from .refinement_rollout import run_case,runtime_for
from .verifier import canonical


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/separate_gate/request.json').read_text())
    ex=json.loads(Path('runs/refinement/executor_sft/request.json').read_text())
    pools={};tests={}
    for config in [old['config'],ex['config']]:
        pools.update(load_root_manifest(config['root_manifest']));tests.update({t['root_id']:t for t in read_jsonl(config['test_manifest'])})
    train=old['value_roots']+[s['root_id'] for s in ex['selection'] if s['scope']=='FIT']
    transfer=[s['root_id'] for s in ex['selection'] if s['scope']=='TRANSFER'];selected=train+transfer
    assert len(train)==8 and len(transfer)==2 and all(pools[r]['split']=='TRAIN' for r in selected)
    for name,items in [('roots',[pools[r] for r in selected]),('tests',[tests[r] for r in selected])]:
        (DATA/f'{name}.jsonl').write_text(''.join(canonical(x)+'\n' for x in items))
    config=deepcopy(ex['config']);config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),output_dir=str(ROOT),executor_adapter=EXECUTOR)
    guarded_json(ROOT/'request.json',{'framework':ex['framework'],'config':config,'train_roots':train,'transfer_roots':transfer,
      'initial_gate':INITIAL,'body_adapter':BODY,'executor_adapter':EXECUTOR,'gate_seed':390000713,'plan_seed':391000713,'execution_seed':392000713,'natural_seed':393000713,
      'value_repetitions':8,'value_cases':128,'natural_repetitions':4,'natural_cases':80,'shards':3,'steps':32,'learning_rate':1e-5,'kl_beta':0.1,'uncertain_margin':0.125,
      'objective':'For |QS-QA|>1/8: exact maximizer of p*QS+(1-p)*QA-beta*KL(Bern(p)||original MULTI_BODY mode prior). Distill its two-mode distribution with CE. For close/tied states target initial separate-gate distribution, no directional label.',
      'scope':'Eight TRAIN roots only for calibration. Two new-parameter same-family variants excluded from values/optimization and used only for natural transfer evaluation. One final fixed-budget checkpoint.',
      'continuation':'Original MULTI_BODY for every SPLIT body and internal choice; new executor frozen. Root gate alone updated. All refusals/retries/failures retained.',
      'source_hashes':{n:file_hash(Path(p)/'adapter_model.safetensors') for n,p in [('initial_gate',INITIAL),('body',BODY),('executor',EXECUTOR)]}})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000,'train':22000}})
    print(json.dumps({'train_roots':train,'transfer_roots':transfer,'value_cases':128,'natural_cases':80}))


def context(method='BEFORE'):
    r=json.loads((ROOT/'request.json').read_text());adapter=INITIAL if method=='BEFORE' else str(ROOT/'training/final_adapter')
    base,executor=load_policies(r['framework'],adapter,EXECUTOR)
    base.model.load_adapter(BODY,adapter_name='body',is_trainable=False,local_files_only=True);base.model.set_adapter('default');base.model.requires_grad_(False)
    policy=RootGatePolicy(base.llm,BODY);_,metadata=load_adapter_metadata(Path(BODY),r['framework']['model'],base.tokenizer)
    policy.body.identity={**policy.body.identity,**metadata,'active_adapters':'["body"]'}
    executor.llm.identity['role']='frozen_executor_contract_sft'
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    return r,policy,executor,rows,tests


def values(shard):
    from transformers import set_seed
    r,policy,executor,rows,tests=context();before=fingerprints(policy);records=[]
    for i,rid in enumerate(r['train_roots']):
        if i%r['shards']!=shard:continue
        base=ROOT/'values'/rid;base.mkdir(parents=True,exist_ok=True)
        if (base/'gate.json').exists():gate=json.loads((base/'gate.json').read_text())
        else:
            runtime,_=runtime_for(rows[rid],tests,r['framework'],None,base/'decision');gate=policy.score_modes(runtime.build_observation());write_json(base/'gate.json',gate)
        prior=policy.body.score_modes(gate['observation'])['mode_probabilities']['SPLIT'];cases={a:[] for a in ['ATOMIC','SPLIT']}
        for rep in range(r['value_repetitions']):
            cp=base/f'candidate_{rep:02d}.json'
            if cp.exists():candidate=json.loads(cp.read_text())
            else:
                set_seed(r['plan_seed']+i*100+rep);candidate=policy.sample_split(gate);write_json(cp,candidate)
            for arm in cases:
                c=run_case(rows[rid],tests,r['framework'],r['config'],policy,executor,base/arm/f'repeat_{rep:02d}',r['execution_seed']+i*100+rep,
                    initial=(gate,policy.atomic(gate) if arm=='ATOMIC' else candidate))
                assert c['status']=='COMPLETE';cases[arm].append(c)
        qa=sum(c['final_scores']['private']['status']=='PASS' for c in cases['ATOMIC'])/8
        qs=sum(c['final_scores']['private']['status']=='PASS' for c in cases['SPLIT'])/8
        record={'root_id':rid,'q_atomic':qa,'q_split':qs,'delta':qs-qa,'initial_p_split':gate['mode_probabilities']['SPLIT'],'body_prior_p_split':prior,'cases':cases}
        write_json(base/'result.json',record);records.append(record)
    assert fingerprints(policy)==before
    write_json(ROOT/'values'/f'shard_{shard}.json',{'status':'COMPLETE','records':records,'frozen':before})


def target_for(record,beta,margin):
    if abs(record['delta'])<=margin:return record['initial_p_split']
    p=record['body_prior_p_split'];logit=math.log(p)-math.log1p(-p)+record['delta']/beta
    return 1/(1+math.exp(-logit))


def summarize_values():
    r=json.loads((ROOT/'request.json').read_text());pieces=[json.loads((ROOT/'values'/f'shard_{s}.json').read_text()) for s in range(r['shards'])]
    assert all(p['frozen']==pieces[0]['frozen'] for p in pieces)
    records=[json.loads((ROOT/'values'/rid/'result.json').read_text()) for rid in r['train_roots']];rows=[]
    for v in records:
        assert all(len(v['cases'][a])==8 for a in ['ATOMIC','SPLIT'])
        rows.append({k:v[k] for k in ['root_id','q_atomic','q_split','delta','initial_p_split','body_prior_p_split']})
        rows[-1]['target_p_split']=target_for(v,r['kl_beta'],r['uncertain_margin'])
        rows[-1]['direction']='UNCERTAIN' if abs(v['delta'])<=r['uncertain_margin'] else 'SPLIT' if v['delta']>0 else 'ATOMIC'
    result={'status':'COMPLETE','cases':128,'by_root':rows,'frozen':pieces[0]['frozen']}
    write_json(ROOT/'values/result.json',result);print(json.dumps(result))


def train():
    import torch
    from transformers import set_seed
    from .refinement_pr1 import enable_planner
    from .train_sft import trainable_snapshots,changed_tensors,prefix_forward
    r,policy,executor,_,_=context();values_=json.loads((ROOT/'values/result.json').read_text());before=fingerprints(policy)
    assert before==values_['frozen'];set_seed(r['gate_seed']);model=policy.model;tok=policy.tokenizer
    gates=[json.loads((ROOT/'values'/rid/'gate.json').read_text()) for rid in r['train_roots']];targets=[v['target_p_split'] for v in values_['by_root']]
    for gate in gates:
        assert policy.prepare(gate['observation'])[1]['prefix']==gate['prefix_ids']
    out=ROOT/'training';out.mkdir(exist_ok=False)
    guarded_json(out/'request.json',{'steps':r['steps'],'lr':r['learning_rate'],'targets':values_['by_root'],'initial_gate':INITIAL})
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.peft_config['default'].lora_dropout=0.;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False;enable_planner(model)
    snapshots,trainable=trainable_snapshots(model);params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=r['learning_rate'],weight_decay=0.);history=[]
    for step in range(r['steps']):
        enable_planner(model);optimizer.zero_grad(set_to_none=True);losses=[]
        for gate,target in zip(gates,targets):
            modes=list(gate['mode_log_probs'])
            with policy.llm._autocast():
                lp,_=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],modes,tok.pad_token_id)
                loss=-target*lp[modes.index('SPLIT')]-(1-target)*lp[modes.index('ATOMIC')]
            assert torch.isfinite(loss);(loss/len(gates)).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm) and norm>0
        optimizer.step();record={'step':step+1,'per_state_ce':losses,'gradient_norm':float(norm)};history.append(record);print(json.dumps(record),flush=True)
    final=out/'final_adapter';model.save_pretrained(final,selected_adapters=['default']);tok.save_pretrained(final)
    changes=changed_tensors(model,snapshots);original=prefix_forward(model,gates[0]['prefix_ids'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'):restored=prefix_forward(model,gates[0]['prefix_ids'])
    reload_delta=float((original-restored).abs().max());assert reload_delta<=1e-5
    final_probs={rid:policy.score_modes(g['observation'])['mode_probabilities']['SPLIT'] for rid,g in zip(r['train_roots'],gates)}
    after=fingerprints(policy);assert all(before[k]==after[k] for k in ['body','executor'])
    # Four small paired body checks against already sampled pre-calibration outputs.
    probes=[]
    for i in [0,1,6,7]:
        set_seed(r['plan_seed']+i*100);candidate=policy.sample_split(gates[i]);prior=json.loads((ROOT/'values'/r['train_roots'][i]/'candidate_00.json').read_text())
        assert candidate['body_output_ids']==prior['body_output_ids'] and candidate['raw_action']==prior['raw_action']
        probes.append({'root_id':r['train_roots'][i],'tokens_identical':True})
    write_json(out/'result.json',{'status':'COMPLETE','steps':r['steps'],'exposures':r['steps']*8,'history':history,'final_probs':final_probs,
      'frozen_before':before,'frozen_after':after,'reload_max_abs_difference':reload_delta,'parameter_changes':changes,'trainable_parameters':trainable,
      'body_checks':probes,'adapter_sha256':file_hash(final/'adapter_model.safetensors')})


def natural(method,shard):
    r,policy,executor,rows,tests=context(method);before=fingerprints(policy);cases=[];gates=[]
    for i,rid in enumerate(r['train_roots']+r['transfer_roots']):
        if i%r['shards']!=shard:continue
        base=ROOT/'natural'/method/rid;base.mkdir(parents=True,exist_ok=True)
        if (base/'gate.json').exists():gate=json.loads((base/'gate.json').read_text())
        else:
            runtime,_=runtime_for(rows[rid],tests,r['framework'],None,base/'decision');gate=policy.score_modes(runtime.build_observation());write_json(base/'gate.json',gate)
        gates.append({'root_id':rid,'p_split':gate['mode_probabilities']['SPLIT'],'scope':'TRAIN' if rid in r['train_roots'] else 'TRANSFER'})
        for rep in range(r['natural_repetitions']):
            c=run_case(rows[rid],tests,r['framework'],r['config'],policy,executor,base/f'repeat_{rep:02d}',r['natural_seed']+i*100+rep)
            assert c['status']=='COMPLETE';cases.append(c)
    assert fingerprints(policy)==before
    write_json(ROOT/'natural'/method/f'shard_{shard}.json',{'status':'COMPLETE','cases':cases,'gates':gates,'frozen':before})


def report():
    r=json.loads((ROOT/'request.json').read_text());arms={};probs={};frozen={}
    for method in ['BEFORE','AFTER']:
        pieces=[json.loads((ROOT/'natural'/method/f'shard_{s}.json').read_text()) for s in range(r['shards'])]
        assert all(p['frozen']==pieces[0]['frozen'] for p in pieces);frozen[method]=pieces[0]['frozen']
        cs=[c for p in pieces for c in p['cases']];assert len(cs)==40;choices=[]
        for c in cs:
            e=next(e for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl') if e['event_kind']=='POLICY' and e['node_id']=='N0')
            selected=e['model_input']['refinement']['selected_mode'];assert selected in {'ATOMIC','SPLIT'}
            choices.append({'root_id':c['root_id'],'seed':c['seed'],'mode':selected,
                            'success':c['final_scores']['private']['status']=='PASS','run_dir':c['run_dir']})
        arms[method]=choices;probs[method]={g['root_id']:g['p_split'] for p in pieces for g in p['gates']}
    assert all(frozen['BEFORE'][k]==frozen['AFTER'][k] for k in ['body','executor'])
    by_root=[]
    for rid in r['train_roots']+r['transfer_roots']:
        item={'root_id':rid,'scope':'TRAIN' if rid in r['train_roots'] else 'TRANSFER'}
        pair={}
        for method,cs in arms.items():
            subset=[c for c in cs if c['root_id']==rid];assert len(subset)==4;pair[method]={c['seed']:c for c in subset}
            item[method]={'success':sum(c['success'] for c in subset),'split':sum(c['mode']=='SPLIT' for c in subset),'p_split':probs[method][rid]}
        assert pair['BEFORE'].keys()==pair['AFTER'].keys()
        item['wins']=sum(pair['AFTER'][k]['success']>v['success'] for k,v in pair['BEFORE'].items())
        item['losses']=sum(pair['AFTER'][k]['success']<v['success'] for k,v in pair['BEFORE'].items());by_root.append(item)
    result={'status':'COMPLETE','values':json.loads((ROOT/'values/result.json').read_text()),'training':json.loads((ROOT/'training/result.json').read_text()),
            'natural_cases':80,'by_root':by_root,'choices':arms,'frozen':frozen}
    result['natural_summary']={scope:{method:{'success':sum(x[method]['success'] for x in by_root if x['scope']==scope),'split':sum(x[method]['split'] for x in by_root if x['scope']==scope),
       'runs':4*sum(x['scope']==scope for x in by_root)} for method in arms} for scope in ['TRAIN','TRANSFER']}
    write_json(ROOT/'result.json',result);print(json.dumps({'by_root':by_root,'summary':result['natural_summary']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','values','summarize_values','train','natural','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE');args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='values':values(args.shard)
    elif args.command=='summarize_values':summarize_values()
    elif args.command=='train':train()
    elif args.command=='natural':natural(args.method,args.shard)
    else:report()
