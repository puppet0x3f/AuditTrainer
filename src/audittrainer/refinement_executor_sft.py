"""Bounded executor-only SFT, fixed-plan controls, then fresh A/S values."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path('runs/refinement/executor_sft')
DATA=Path('data/refinement/executor_sft')
F01='runs/a5/train/F01_seed42/final_adapter'
GATE='runs/refinement/separate_gate/CALIBRATED/final_adapter'
BODY='runs/refinement/body_transfer/MULTI_BODY/final_adapter'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate','values'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')

from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import write_json,capture_program,score_program
from .refinement_generalization import guarded_json
from .refinement_execution_repair import local_checks
from .refinement_interface_gate import FixedPlanner
from .refinement_parent_composition import summary_prompt
from .refinement_policy import load_policies,adapter_fingerprint,active_adapter,atomic_plan,load_adapter_metadata
from .refinement_separate_gate import RootGatePolicy,fingerprints
from .refinement_responsibility_course import CourseReference,allocation
from .refinement_rollout import runtime_for,run_case
from .training_tasks import oracle_a,oracle_b
from .verifier import canonical

NEW_INPUTS={
 'polynomial': [[[4,7],[0,-2],[1,14],[4,-1]], [[2,3],[5,-7],[2,4],[0,0]],
                [[6,2],[6,-2]], [[3,-14],[0,7],[2,5]], [[5,21],[1,-7],[4,0]], [[0,1],[6,7],[2,14],[1,-1],[2,-7]]],
 'paths': [['/a/b/c/d/e/f/g','./oak/leaf','/'], ['x//y/./z','a/b/c/d/e/f/g/h'],
           ['///','././'], ['a/b/c/d/e/f/g/h/i/j/k/l/m/n','q'], ['a/./b//c','./'], ['a/b/c/d/e/f/g','q/r/s/t/u/v/w','x']]
}


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/execution_repair/request.json').read_text());framework=old['framework']
    train_rows=load_root_manifest(old['config']['root_manifest']);train_tests={r['root_id']:r for r in read_jsonl(old['config']['test_manifest'])}
    rows=list(train_rows.values());tests=list(train_tests.values());states=[];selections=[]
    for selection in old['selection']:
        row=train_rows[selection['root_id']];definition=json.loads(Path(row['fixture_path']).read_text());ref=json.loads(Path(row['reference_path']).read_text())
        policy=CourseReference(definition,ref,'parent_aggregate');out=ROOT/'teacher'/row['root_id']
        runtime,full=runtime_for(row,train_tests,framework,policy,out/'runtime');runtime.run()
        scores,_=score_program(capture_program(runtime,out/'program'),full,row['root_id']);assert scores['private']['status']=='PASS'
        fake={'run_dir':str(out),'final_scores':scores};checks=local_checks(fake,row,train_tests,framework)
        assert all(e['status']=='PASS' for e in checks['local']) and checks['parent_called']
        for event in read_jsonl(runtime.trace.path):
            action=event.get('parsed_action')
            if action and action['action']=='write_code':
                assert event['host_result']['code']=='SOURCE_SAVED'
                states.append({'sample_id':row['root_id']+'_'+event['node_id'],'root_id':row['root_id'],'split':'TRAIN','kind':'target',
                               'observation':event['observation_before'],'target_action':action})
        selections.append({'root_id':row['root_id'],'scope':'FIT','first':selection['first'],'repetitions':4})
        # Evaluation-only variants: no teacher samples, new modulus and inputs.
        first=selection['first'];rid=f'train_executor_check_{first}_matrix_07'
        d=json.loads(json.dumps(definition).replace(row['root_id'],rid).replace('divisible by 5','divisible by 7'))
        private=json.loads(Path(train_tests[row['root_id']]['path']).read_text());private=json.loads(json.dumps(private).replace(row['root_id'],rid))
        reference=deepcopy(ref);reference['aggregate']=reference['aggregate'].replace('% 5','% 7');reference['parameter']=7
        ns={};exec(reference['prepare']+'\n'+reference['aggregate'],ns)
        for suite in d['suites']+private['suites']:
            key=suite.get('contract_key','root');indices=[0,1] if suite['visibility']=='VISIBLE' else [2,3,4,5];cases=[]
            for i in indices:
                items=deepcopy(NEW_INPUTS[first][i]);raw=oracle_a(first,items)
                prepared=[p[1] for p in raw] if first=='polynomial' else [len(p) for p in raw];answer=oracle_b('matrix',prepared,7)
                assert ns['prepare'](deepcopy(items))==prepared and ns['aggregate'](deepcopy(prepared))==answer
                args=[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items]
                expected=json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer
                cases.append({'case_id':f'{key}_{i}','args':args,'expected':expected})
            suite['cases']=cases
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for name,value in [('task',d),('private_tests',private),('reference',reference)]:guarded_json(folder/f'{name}.json',value)
        rows.append({**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'executor_sft_untrained_parameter_check','original_split':'TRAIN','io_type':'solve_text',
                     'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')})
        tests.append({'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')})
        selections.append({'root_id':rid,'scope':'TRANSFER','first':first,'repetitions':4})
    # Two original task families are a focused forgetting check, not a broad benchmark.
    legacy=load_root_manifest('data/manifests/roots.jsonl');legacy_tests={r['root_id']:r for r in read_jsonl('data/manifests/tests.jsonl')}
    for rid in ['train_polynomial_03','train_paths_03']:
        rows.append(legacy[rid]);tests.append(legacy_tests[rid]);selections.append({'root_id':rid,'scope':'RETENTION','repetitions':2})
    # Twenty existing F01 write_code examples: root and prepare in each of ten families.
    old_data=read_jsonl('data/sft/a5/F01.jsonl');retention=[]
    for family in sorted({r['family_id'] for r in old_data}):
        for root_node in [False,True]:
            for r in old_data:
                if r['family_id']!=family or r['target_action']['action']!='write_code':continue
                obs=json.loads(r['messages'][1]['content'])['observation']
                if (obs['node_id']=='N0')!=root_node:continue
                if not root_node and obs['incoming_contract']['signature']['name']!='prepare':continue
                retention.append({'sample_id':r['sample_id'],'root_id':r['root_id'],'split':r['split'],'kind':'retention','observation':obs,'target_action':r['target_action']});break
    assert len(states)==4 and len(retention)==20 and all(r['split']=='TRAIN' for r in states+retention)
    for name,items in [('roots',rows),('tests',tests),('training',states+retention)]: (DATA/f'{name}.jsonl').write_text(''.join(canonical(x)+'\n' for x in items))
    config=deepcopy(old['config']);config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),output_dir=str(ROOT))
    guarded_json(ROOT/'request.json',{'framework':framework,'config':config,'selection':selections,'initial_executor':F01,'gate':GATE,'body':BODY,
        'steps':32,'learning_rate':2e-5,'seed':370000713,'eval_seed':371000713,'value_seed':372000713,'plan_seed':373000713,
        'training_examples':4,'retention_examples':20,'training_sha256':file_hash(DATA/'training.jsonl'),
        'objective':'Full write_code action CE, 2 target + 2 old-F01 examples per update, equal per-example token-mean loss. Original prompt, no clarification.',
        'selection_rule':'One final fixed-budget checkpoint. No evaluation checkpoint selection.',
        'eval_cases':40,'value_rule':'Only after both FIT and TRANSFER root success improve and retention does not regress: refresh 2 FIT states x A/S x 4 paired seeds x old/new executor =32 runs.',
        'scope':'Two TRAIN fit states, two untrained parameter/input variants of the same families, two original-task retention checks; not broad family generalization.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})
    print(json.dumps({'target_examples':4,'retention_examples':20,'updates':32,'fixed_plan_evaluations':40}))


def train(root=ROOT, data=DATA, initial_executor=F01, *, progress=False):
    import torch
    import torch.nn.functional as F
    from transformers import set_seed
    from .refinement_pr1 import enable_planner,causal_logits
    from .train_sft import trainable_snapshots,changed_tensors,prefix_forward
    r=json.loads((root/'request.json').read_text());set_seed(r['seed'])
    # The default slot holds an executor copy, not a planner adapter.
    policy,_=load_policies(r['framework'],initial_executor,initial_executor);model=policy.model;tok=policy.tokenizer
    rows=read_jsonl(data/'training.jsonl');examples=[]
    targets=[i for i,row in enumerate(rows) if row['kind']=='target']
    retention=[i for i,row in enumerate(rows) if row['kind']=='retention']
    assert targets and retention
    for row in rows:
        prompt=summary_prompt(tok,row['observation'],context_tokens=policy.llm.context_tokens,generation_tokens=policy.llm.generation_tokens)
        target=tok.encode(canonical(row['target_action']),add_special_tokens=False)+[tok.eos_token_id]
        assert len(prompt.token_ids)+len(target)<=policy.llm.context_tokens
        examples.append({'sample_id':row['sample_id'],'prefix':prompt.token_ids,'target':target})
    out=root/'training';out.mkdir(exist_ok=False);write_json(out/'examples.json',examples)
    before=adapter_fingerprint(model,'default');frozen=adapter_fingerprint(model,'executor')
    assert before==frozen
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.peft_config['default'].lora_dropout=0.;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    enable_planner(model);snapshots,trainable=trainable_snapshots(model);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=r['learning_rate'],weight_decay=0.);history=[]
    iterator=range(r['steps'])
    if progress:
        from tqdm import tqdm
        iterator=tqdm(iterator,desc='执行器SFT',unit='步',dynamic_ncols=True)
    for step in iterator:
        indices=[targets[(2*step)%len(targets)],targets[(2*step+1)%len(targets)],
                 retention[(2*step)%len(retention)],retention[(2*step+1)%len(retention)]]
        enable_planner(model);optimizer.zero_grad(set_to_none=True);losses=[]
        for i in indices:
            e=examples[i]
            with policy.llm._autocast():loss=F.cross_entropy(causal_logits(model,e['prefix'],e['target']).float(),torch.tensor(e['target'],device=model.device))
            assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm) and norm>0
        optimizer.step();record={'step':step+1,'indices':indices,'losses':losses,'gradient_norm':float(norm)};history.append(record)
        if progress:iterator.set_postfix(loss=f'{sum(losses)/len(losses):.5f}')
        else:print(json.dumps(record),flush=True)
    final=out/'final_adapter';model.save_pretrained(final,selected_adapters=['default']);tok.save_pretrained(final)
    changes=changed_tensors(model,snapshots);original=prefix_forward(model,examples[0]['prefix'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'):restored=prefix_forward(model,examples[0]['prefix'])
    delta=float((original-restored).abs().max());assert delta<=1e-5 and adapter_fingerprint(model,'executor')==frozen
    write_json(out/'result.json',{'status':'COMPLETE','steps':r['steps'],'exposures':4*r['steps'],'target_exposures':2*r['steps'],'retention_exposures':2*r['steps'],
      'initial_executor_fingerprint':before,'final_executor_fingerprint':adapter_fingerprint(model,'default'),'frozen_control_fingerprint':frozen,
      'reload_max_abs_difference':delta,'parameter_changes':changes,'trainable_parameters':trainable,'history':history,'adapter_sha256':file_hash(final/'adapter_model.safetensors')})


def context(method):
    r=json.loads((ROOT/'request.json').read_text());adapter=F01 if method=='BEFORE' else str(ROOT/'training/final_adapter')
    base,executor=load_policies(r['framework'],GATE,adapter)
    base.model.load_adapter(BODY,adapter_name='body',is_trainable=False,local_files_only=True);base.model.set_adapter('default');base.model.requires_grad_(False)
    policy=RootGatePolicy(base.llm,BODY);_,metadata=load_adapter_metadata(Path(BODY),r['framework']['model'],base.tokenizer)
    policy.body.identity={**policy.body.identity,**metadata,'active_adapters':'["body"]'}
    executor.llm.identity['role']='frozen_F01_control' if method=='BEFORE' else 'frozen_executor_contract_sft'
    rows=load_root_manifest(r['config']['root_manifest']);tests={t['root_id']:t for t in read_jsonl(r['config']['test_manifest'])}
    return r,policy,executor,rows,tests


def evaluate(method):
    r,policy,executor,rows,tests=context(method);before=fingerprints(policy);cases=[]
    for i,selected in enumerate(r['selection']):
        row=rows[selected['root_id']];d=json.loads(Path(row['fixture_path']).read_text());plan=allocation(d['contract'],d,'parent_aggregate')
        fixed=FixedPlanner(policy,{'N0':plan})
        for rep in range(selected['repetitions']):
            c=run_case(row,tests,r['framework'],r['config'],fixed,executor,ROOT/'evaluation'/method/row['root_id']/f'repeat_{rep:02d}',r['eval_seed']+i*100+rep,reset_seed_after_root_plan=True)
            assert c['status']=='COMPLETE'
            checks=local_checks(c,row,tests,r['framework']);write_json(Path(c['run_dir'])/'execution_checks.json',checks)
            cases.append({**c,'scope':selected['scope'],'execution_checks':checks})
    assert before==fingerprints(policy)
    write_json(ROOT/'evaluation'/method/'result.json',{'status':'COMPLETE','cases':cases,'frozen':before})


def report():
    arms={a:json.loads((ROOT/'evaluation'/a/'result.json').read_text()) for a in ['BEFORE','AFTER']}
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    summary={};by_root=[]
    for selected in json.loads((ROOT/'request.json').read_text())['selection']:
        item=deepcopy(selected)
        for arm in arms:
            cs=[c for c in arms[arm]['cases'] if c['root_id']==selected['root_id']]
            item[arm]={'runs':len(cs),'root_success':sum(c['final_scores']['private']['status']=='PASS' for c in cs),
              'local_private':sum(any(e['visibility']=='PRIVATE' and e['status']=='PASS' for e in c['execution_checks']['local']) for c in cs),
              'parent_called':sum(c['execution_checks']['parent_called'] for c in cs)}
        by_root.append(item)
    for scope in ['FIT','TRANSFER','RETENTION']:
        summary[scope]={a:sum(x[a]['root_success'] for x in by_root if x['scope']==scope) for a in arms}
    proceed=all(summary[s]['AFTER']>summary[s]['BEFORE'] for s in ['FIT','TRANSFER']) and summary['RETENTION']['AFTER']>=summary['RETENTION']['BEFORE']
    result={'status':'COMPLETE','fixed_plan_cases':40,'by_root':by_root,'summary':summary,'proceed_to_values':proceed}
    write_json(ROOT/'result.json',result);print(json.dumps(result))


def values(method):
    from transformers import set_seed
    assert json.loads((ROOT/'result.json').read_text())['proceed_to_values']
    r,policy,executor,rows,tests=context(method);before=fingerprints(policy);cases=[]
    for i,selected in enumerate([x for x in r['selection'] if x['scope']=='FIT']):
        row=rows[selected['root_id']];base=ROOT/'values'/method/row['root_id'];base.mkdir(parents=True,exist_ok=True)
        runtime,_=runtime_for(row,tests,r['framework'],None,base/'decision');gate=policy.score_modes(runtime.build_observation());write_json(base/'gate.json',gate)
        for rep in range(4):
            set_seed(r['plan_seed']+i*100+rep);candidate=policy.sample_split(gate);write_json(base/f'candidate_{rep:02d}.json',candidate)
            for arm in ['ATOMIC','SPLIT']:
                c=run_case(row,tests,r['framework'],r['config'],policy,executor,base/arm/f'repeat_{rep:02d}',r['value_seed']+i*100+rep,
                           initial=(gate,policy.atomic(gate) if arm=='ATOMIC' else candidate))
                assert c['status']=='COMPLETE';cases.append({**c,'arm':arm})
    assert fingerprints(policy)==before
    write_json(ROOT/'values'/method/'result.json',{'status':'COMPLETE','cases':cases,'frozen':before})

def report_values():
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text())
    arms={a:json.loads((ROOT/'values'/a/'result.json').read_text()) for a in ['BEFORE','AFTER']}
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    rows=[]
    for selected in [x for x in r['selection'] if x['scope']=='FIT']:
        rid=selected['root_id'];item={'root_id':rid}
        for rep in range(4):
            a=json.loads((ROOT/'values/BEFORE'/rid/f'candidate_{rep:02d}.json').read_text())
            b=json.loads((ROOT/'values/AFTER'/rid/f'candidate_{rep:02d}.json').read_text())
            assert a['body_output_ids']==b['body_output_ids'] and a['raw_action']==b['raw_action']
        for method,records in arms.items():
            item[method]={}
            for arm in ['ATOMIC','SPLIT']:
                cs=sorted([c for c in records['cases'] if c['root_id']==rid and c['arm']==arm],key=lambda c:c['seed']);assert len(cs)==4
                item[method][arm]={'success':sum(c['final_scores']['private']['status']=='PASS' for c in cs),'runs':4,
                  'outcomes':[c['final_scores']['private']['status'] for c in cs],
                  'first_registered':sum(c['root_first_plan_registered'] for c in cs),'rejections':[c['protocol_rejections'] for c in cs]}
            item[method]['delta']=(item[method]['SPLIT']['success']-item[method]['ATOMIC']['success'])/4
        item['p_split']=json.loads((ROOT/'values/AFTER'/rid/'gate.json').read_text())['mode_probabilities']['SPLIT']
        rows.append(item)
    result['fresh_values']={'status':'COMPLETE','cases':32,'paired_bodies_identical':8,'by_root':rows}
    write_json(ROOT/'result.json',result);print(json.dumps(result['fresh_values']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','evaluate','report','values','report_values']);p.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE');args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='train':train()
    elif args.command=='evaluate':evaluate(args.method)
    elif args.command=='values':values(args.method)
    elif args.command=='report_values':report_values()
    else:report()
