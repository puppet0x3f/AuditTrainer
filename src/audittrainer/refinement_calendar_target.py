"""Calendar best-plan SFT positive control, with gate/body target ablations."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import yaml

if __name__ == '__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _path=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_calendar_target.yaml'
    _reservation=acquire(Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json',
                         'train' if sys.argv[1]=='train' else 'collect')

import torch
import torch.nn.functional as F
from transformers import set_seed

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum import plans
from .refinement_curriculum_rl import conditional_probe, plan_key
from .refinement_first_credit import outcome_records, metrics
from .refinement_generalization import guarded_json
from .refinement_parent_validation import presentation_identity
from .refinement_policy import MODES, active_adapter, adapter_fingerprint, load_policies, mode_log_probs
from .refinement_pr1 import load_config, causal_logits, enable_planner
from .refinement_rollout import run_case, case_summary, costs
from .train_refinement import encode_demo, representation_loss
from .train_sft import prefix_forward, changed_tensors, trainable_snapshots
from .verifier import canonical


def target_weights(method):
    return {'GATE_SFT':(1.,0.),'BODY_SFT':(0.,1.),'JOINT_SFT':(1.,1.)}[method]


def prepare(study):
    config,framework=load_config(study['source_config']); out=Path(study['output_dir']); out.mkdir(parents=True,exist_ok=True)
    rows=load_root_manifest(config['root_manifest']); row=rows[study['root_id']]
    if row['split']!='TRAIN': raise ValueError('TRAIN positive control only')
    fixed=json.loads(Path(study['source_fixed_plan']).read_text())
    values={v:r['summary']['final_successes']/r['summary']['runs'] for v,r in fixed['results'][row['root_id']].items()}
    if [v for v,q in values.items() if q==max(values.values())]!=[study['preferred_variant']]:
        raise ValueError('Expected the unique prior calibrated best plan')
    action=fixed['fixed_plans'][row['root_id']][study['preferred_variant']]
    saved_gate=json.loads((Path(config['output_dir'])/'groups'/row['root_id']/'gate.json').read_text())
    tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    data=out/'data'; data.mkdir(exist_ok=True)
    for name,item in [('roots',row),('tests',tests[row['root_id']])]:
        path=data/f'{name}.jsonl'; content=canonical(item)+'\n'
        if path.exists() and path.read_text()!=content: raise ValueError('Changed manifest')
        path.write_text(content)
    demo={'sample_id':'calendar_prior_best_plan','root_id':row['root_id'],'family_id':row['family_id'],
          'split':'TRAIN','target_action':action,'messages':[{}, {'content':canonical({'observation':saved_gate['observation']})}]}
    guarded_json(out/'target.json',demo)
    prior=json.loads(Path('runs/refinement/B3_fixed_plan/request.json').read_text())
    if prior['presentation']!=presentation_identity(): raise ValueError('Changed frozen summary')
    config.update(root_manifest=str(data/'roots.jsonl'),test_manifest=str(data/'tests.jsonl'),
                  root_limit=1,seed=study['seed'])
    guarded_json(out/'request.json',{'study':study,'config':config,'framework':framework,'training_target':demo,
        'prior_plan_returns':values,'source_sha256':file_hash(study['source_fixed_plan']),
        'presentation':presentation_identity(),
        'adapters':{k:file_hash(Path(config[k])/'adapter_model.safetensors') for k in ('initial_adapter','executor_adapter')},
        'planned':{'new_training_continuations':0,'reused_calibration_continuations':36,
                   'optimizer_steps':len(study['methods'])*study['optimizer_steps'],
                   'conditional_probes':(1+len(study['methods']))*study['conditional_split_samples'],
                   'natural_evaluations':(1+len(study['methods']))*len(study['evaluation_seeds'])*study['natural_repetitions']},
        'scope':'One known TRAIN state; explicit best-plan imitation, not reward-weighted sampled SFT or B3. Fixed public target has no behavior-sampling likelihood. Ablations remove target loss terms, not separate parameter heads. Evaluation never selects checkpoints.'})
    guarded_json(out/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000,'train':22000}})
    return config,framework,demo


def train(study,method):
    config,framework,demo=prepare(study); wg,wb=target_weights(method)
    output=Path(study['output_dir'])/method; output.mkdir(exist_ok=False)
    planner,executor=load_policies(framework,config['initial_adapter'],config['executor_adapter'])
    model,tok=planner.model,planner.tokenizer
    model.load_adapter(config['initial_adapter'],adapter_name='reference',is_trainable=False,local_files_only=True)
    frozen={n:adapter_fingerprint(model,n) for n in ('reference','executor')}
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
    model.peft_config['default'].lora_dropout=0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.config.use_cache=False
    example=encode_demo(demo,tok,framework); write_json(output/'encoded_target.json',example)
    saved_gate=json.loads((Path(config['output_dir'])/'groups'/study['root_id']/'gate.json').read_text())
    if example['prefix']!=saved_gate['prefix_ids']:raise ValueError('Training prompt differs from the decision state')
    retention=read_jsonl(Path(config['adaptation']['output_dir'])/'examples.jsonl')
    if any(r['split']!='TRAIN' for r in retention): raise ValueError('Non-TRAIN retention')
    settings=config['update']; scale=settings['objective_scale']; beta=settings['kl_weight']; eta=settings['sft_weight']
    enable_planner(model); before,trainable=trainable_snapshots(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=0)
    with active_adapter(model,'reference'),torch.no_grad(),planner.llm._autocast():
        ref_gate=mode_log_probs(model,example['prefix'],example['mode_ids'],MODES,tok.pad_token_id)[0].detach()
        ref_body=F.log_softmax(causal_logits(model,example['body_prefix'],example['body_target']).float(),dim=-1).detach()
    labels=torch.tensor(example['body_target'],device=model.device)
    history=[]; started=time.monotonic(); set_seed(study['seed'])
    for step in range(study['optimizer_steps']):
        optimizer.zero_grad(set_to_none=True); enable_planner(model)
        with planner.llm._autocast():
            gate=mode_log_probs(model,example['prefix'],example['mode_ids'],MODES,tok.pad_token_id)[0]
            gate_ce=-gate[MODES.index('SPLIT')]
            gate_kl=(gate.exp()*(gate-ref_gate)).sum()
            loss=scale*(wg*gate_ce+beta*gate_kl)
        loss.backward()
        stat={'step':step+1,'p_split_before_update':float(gate[1].detach().exp()),
              'gate_ce':float(gate_ce.detach()),'gate_kl':float(gate_kl.detach())}
        with planner.llm._autocast():
            body=F.log_softmax(causal_logits(model,example['body_prefix'],example['body_target']).float(),dim=-1)
            body_ce=-body.gather(1,labels[:,None]).mean()
            body_kl=(body.exp()*(body-ref_body)).sum()
            loss=scale*(wb*body_ce+beta*ref_gate[1].exp()*body_kl)
        loss.backward()
        stat.update(body_ce=float(body_ce.detach()),body_kl=float(body_kl.detach()))
        del gate,gate_ce,gate_kl,body,body_ce,body_kl,loss
        with planner.llm._autocast():
            retain,_,_=representation_loss(model,retention[step%len(retention)],tok.pad_token_id)
        (scale*eta*retain).backward(); stat['retention_loss']=float(retain.detach()); del retain
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings['max_grad_norm'])
        if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradients')
        optimizer.step(); stat['gradient_norm']=float(norm); history.append(stat)
        print(json.dumps({'method':method,**stat}),flush=True)
    final=output/'final_adapter'; model.save_pretrained(final,selected_adapters=['default']); tok.save_pretrained(final)
    after={n:adapter_fingerprint(model,n) for n in frozen}
    if after!=frozen: raise ValueError('Changed frozen model')
    old=prefix_forward(model,example['prefix'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'): restored=prefix_forward(model,example['prefix'])
    delta=float((old-restored).abs().max())
    if delta>1e-5: raise ValueError('Saved adapter failed reload')
    report={'status':'COMPLETE','method':method,'scope':'SINGLE_TRAIN_STATE_POSITIVE_CONTROL',
        'optimizer_steps':len(history),'previous_optimizer_steps':0,'rl_updates':0,
        'target_weights':{'gate':wg,'body':wb},'settings':settings,'seed':study['seed'],
        'history':history,'trainable_parameters':trainable,'parameter_changes':changed_tensors(model,before),
        'frozen_before':frozen,'frozen_after':after,'reload_max_abs_difference':delta,
        'adapter':str(final),'adapter_sha256':file_hash(final/'adapter_model.safetensors'),
        'adapter_config_sha256':file_hash(final/'adapter_config.json'),
        'train_seconds':time.monotonic()-started,
        'objective':'scale*(gate_weight*CE(SPLIT)+body_weight*mean_CE(best_plan_body)+beta*gate_KL+beta*p0_split*sum_body_KL+eta*common_retention_SFT); all branches share KL and retention.'}
    write_json(output/'training_report.json',report)


def choice_records(cases,definition):
    variants={plan_key(a,definition):v for v,a in plans(definition).items()}
    records=outcome_records(cases)
    for rec in records:
        events=[e for e in read_jsonl(Path(rec['run_dir'])/'runtime/trace.jsonl')
                if e['node_id']=='N0' and e['event_kind']=='POLICY' and e['observation_before']['phase']=='NEW']
        first=events[0]
        rec['first_mode']=(first.get('model_input') or {}).get('refinement',{}).get('selected_mode','UNPARSED')
        def label(e):
            if e is None:return 'NO_ACCEPTED_PLAN'
            if e['host_result']['code']!='PLAN_REGISTERED':return 'REJECTED'
            return variants.get(plan_key(e['parsed_action'],definition),'OTHER_REGISTERED')
        rec['first_choice']=label(first)
        rec['accepted_choice']=label(next((e for e in events if e['host_result']['code']=='PLAN_REGISTERED'),None))
    return records


def evaluate(study,name):
    config,framework,demo=prepare(study); output=Path(study['output_dir'])/'evaluation'/name
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():return
    adapter=config['initial_adapter']
    if name!='B0':
        trained=json.loads((Path(study['output_dir'])/name/'training_report.json').read_text())
        if trained['status']!='COMPLETE' or trained['optimizer_steps']!=study['optimizer_steps']:raise ValueError('Incomplete training')
        adapter=trained['adapter']
        if trained['adapter_sha256']!=file_hash(Path(adapter)/'adapter_model.safetensors'):raise ValueError('Changed adapter')
    planner,executor=load_policies(framework,adapter,config['executor_adapter'])
    before={n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}
    rows=load_root_manifest(config['root_manifest']); tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    probeconfig=deepcopy(config)
    probeconfig['evaluation']={'seed':study['probe_seed'],'focus_root':study['root_id'],
                              'conditional_split_samples':study['conditional_split_samples']}
    probe=conditional_probe(probeconfig,framework,planner,executor,output,rows,tests)
    gate=json.loads((output/'conditional_probe/gate.json').read_text())
    cases=[]; blocks={}; row=rows[study['root_id']]; definition=json.loads(Path(row['fixture_path']).read_text())
    for seed in study['evaluation_seeds']:
        block=[]
        for rep in range(study['natural_repetitions']):
            c=run_case(row,tests,framework,config,planner,executor,output/'natural'/str(seed)/f'repeat_{rep:02d}',seed+rep)
            if c['status']!='COMPLETE':raise ValueError('Incomplete execution')
            block.append(c)
        cases.extend(block); records=choice_records(block,definition)
        blocks[str(seed)]={'metrics':metrics(records),'records':records,'summary':case_summary(block),
            'first_choice':dict(Counter(r['first_choice'] for r in records)),
            'first_mode':dict(Counter(r['first_mode'] for r in records))}
    after={n:adapter_fingerprint(planner.model,n) for n in before}
    if before!=after:raise ValueError('Changed evaluation weights')
    result={'status':'COMPLETE','model':name,'adapter':adapter,'adapter_sha256':file_hash(Path(adapter)/'adapter_model.safetensors'),
        'p_split':gate['mode_probabilities']['SPLIT'],'conditional_probe':probe,'blocks':blocks,
        'summary':case_summary(cases),'frozen_before':before,'frozen_after':after,
        'optimizer_updates_from_evaluation':0,'actual_model_cost':dict(costs(planner,executor))}
    write_json(output/'result.json',result)
    print(json.dumps({'model':name,'p_split':result['p_split'],'probe':probe['reviewed_variants'],
                      'passes':result['summary']['final_successes']}),flush=True)


def report(study):
    out=Path(study['output_dir']); request=json.loads((out/'request.json').read_text())
    evaluations={m:json.loads((out/'evaluation'/m/'result.json').read_text()) for m in ['B0',*study['methods']]}
    updates={m:json.loads((out/m/'training_report.json').read_text()) for m in study['methods']}
    paired={}
    for m,e in evaluations.items():
        if e['status']!='COMPLETE' or e['frozen_before']!=e['frozen_after']:raise ValueError('Incomplete model')
        if m=='B0':continue
        paired[m]={}
        for seed in study['evaluation_seeds']:
            a=evaluations['B0']['blocks'][str(seed)]['records']; b=e['blocks'][str(seed)]['records']
            if len(a)!=study['natural_repetitions'] or {r['seed'] for r in a}!={r['seed'] for r in b}:raise ValueError('Unpaired')
            paired[m][str(seed)]={'final_success':paired_rows(a,b,{study['root_id']}),
                'first_good_plan':paired_rows([{**r,'success':r['first_choice']==study['preferred_variant']} for r in a],
                                             [{**r,'success':r['first_choice']==study['preferred_variant']} for r in b],{study['root_id']})}
    b,j=evaluations['B0'],evaluations['JOINT_SFT']
    criterion=(j['p_split']>b['p_split'] and j['conditional_probe']['reviewed_variants'].get(study['preferred_variant'],0)>
               b['conditional_probe']['reviewed_variants'].get(study['preferred_variant'],0) and
               all(p[k]['wins']>p[k]['losses'] for p in paired['JOINT_SFT'].values() for k in ('final_success','first_good_plan')))
    result={'status':'COMPLETE','request':request,'updates':updates,'evaluations':evaluations,
            'paired_vs_B0':paired,'learning_to_execution_criterion_met':criterion,
            'optimizer_steps':sum(u['optimizer_steps'] for u in updates.values()),
            'natural_evaluations':sum(e['summary']['runs'] for e in evaluations.values()),
            'conditional_probes':sum(e['conditional_probe']['samples'] for e in evaluations.values())}
    if any(result[k]!=request['planned'][k] for k in ('optimizer_steps','natural_evaluations','conditional_probes')):raise ValueError('Incomplete budget')
    write_json(out/'result.json',result);print(json.dumps({'criterion':criterion,'paired':paired}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_calendar_target.yaml')
    parser.add_argument('--model',choices=['B0','GATE_SFT','BODY_SFT','JOINT_SFT'])
    args=parser.parse_args();study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':prepare(study)
    elif args.command=='train':train(study,args.model)
    elif args.command=='evaluate':evaluate(study,args.model)
    else:report(study)


if __name__=='__main__':main()
