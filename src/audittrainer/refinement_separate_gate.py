"""Root-only gate adaptation with frozen MULTI_BODY planning continuations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import yaml

DEFAULT='configs/refinement_separate_gate.yaml'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'sft','calibrate','natural'}:
    from .gpu_resource import acquire
    path=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else DEFAULT
    study=yaml.safe_load(Path(path).read_text())
    _reservation=acquire(Path(study['output_dir'])/'gpu_resources.json','collect' if sys.argv[1]=='natural' else 'train')

from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_policy import RefinementPolicy,load_policies,adapter_fingerprint,active_adapter,mode_log_probs,load_adapter_metadata
from .refinement_rollout import run_case

class RootGatePolicy(RefinementPolicy):
    """Only N0 mode scores use the trainable copy; every body uses frozen body."""
    def __init__(self,llm,body_path):
        super().__init__(llm)
        self.body=RefinementPolicy(llm,adapter_name='body')
        self.body.cost=self.cost
        self.identity={**self.identity,'policy':'root_gate_frozen_body',
            'root_gate_adapter':'default','body_and_internal_adapter':'body',
            'body_adapter':str(body_path),'body_sha256':file_hash(Path(body_path)/'adapter_model.safetensors')}
    def score_modes(self,observation):
        if observation['node_id']!='N0':return self.body.score_modes(observation)
        return super().score_modes(observation)
    def sample_split(self,gate):
        return self.body.sample_split(gate)


def prepare(study):
    source=Path(study['source_dir']);previous=json.loads((source/'request.json').read_text())
    if json.loads((source/'result.json').read_text())['status']!='COMPLETE':raise ValueError('Incomplete previous values')
    if previous['planner_adapter']!=study['body_adapter']:raise ValueError('Values must use the frozen body policy')
    root=Path(study['output_dir']);root.mkdir(parents=True,exist_ok=True)
    guarded_json(root/'request.json',{'study':study,'framework':previous['framework'],'config':previous['config'],
        'value_roots':previous['study']['value_roots'],'natural_seed':previous['study']['natural_seed'],
        'body_sha256':file_hash(Path(study['body_adapter'])/'adapter_model.safetensors'),
        'sft_objective':'Balanced two-state mode cross entropy; both states once per optimizer step; no body loss.',
        'pass_rule':'At final fixed-budget checkpoint, positive root P(SPLIT) rises >=0.05 and negative root falls >=0.05; frozen checks pass.',
        'calibration_objective':'Mean expected-success gate objective using all six previous TRAIN means, plus mode KL to original MULTI_BODY; zero directional term within 0.1.',
        'continuation':'SPLIT bodies and all non-root choices use the original frozen MULTI_BODY. F01 and execution presentation unchanged.',
        'scope':'TRAIN learnability positive control; fit is not generalization. No validation selection.'})
    guarded_json(root/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000,'train':22000}})


def load(study,adapter):
    root=Path(study['output_dir']);r=json.loads((root/'request.json').read_text())
    base,executor=load_policies(r['framework'],adapter,r['config']['executor_adapter'])
    base.model.load_adapter(study['body_adapter'],adapter_name='body',is_trainable=False,local_files_only=True)
    base.model.set_adapter('default');base.model.requires_grad_(False)
    policy=RootGatePolicy(base.llm,study['body_adapter'])
    _,metadata=load_adapter_metadata(Path(study['body_adapter']),r['framework']['model'],base.tokenizer)
    policy.body.identity={**policy.body.identity,**metadata,'active_adapters':'["body"]','role':'frozen_body_and_internal_policy'}
    return root,r,policy,executor


def fingerprints(policy):return {n:adapter_fingerprint(policy.model,n) for n in ['default','body','executor']}


def frozen_checks(study,policy,directory):
    """Eight paired generations against cached original output; internal modes too."""
    from transformers import set_seed
    states=read_jsonl('data/refinement/body_transfer/states.jsonl')
    selected={rid+'_N0' for rid in study['sft_roots']} | {
        'train_course_calendar_matrix_04_N0_p1_compute','train_course_chunks_chunks_04_N0_p1_compute'}
    rows=[]
    for i,state in enumerate(states):
        if state['sample_id'] not in selected:continue
        obs=json.loads(state['messages'][1]['content'])['observation'];gate=policy.score_modes(obs)
        oldbase=Path('runs/refinement/body_transfer/evaluation/MULTI_BODY/conditional')/state['sample_id']
        oldgate=json.loads((oldbase/'gate.json').read_text())
        assert gate['prefix_ids']==oldgate['prefix_ids']
        if state['node_id']!='N0':assert gate['mode_probabilities']==oldgate['mode_probabilities']
        for rep in range(2):
            seed=250600713+i*100+rep;set_seed(seed);candidate=policy.sample_split(gate)
            original=json.loads((oldbase/f'sample_{rep:02d}/candidate.json').read_text())
            same=candidate['body_output_ids']==original['body_output_ids'] and candidate['raw_action']==original['raw_action']
            if not same:raise ValueError('Frozen body paired generation changed')
            folder=directory/state['sample_id']/f'sample_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'candidate.json',candidate)
            rows.append({'sample_id':state['sample_id'],'seed':seed,'tokens_identical':same,'internal_modes_identical':state['node_id']!='N0'})
    assert len(rows)==8
    write_json(directory/'result.json',{'status':'COMPLETE','paired_generations':rows})
    return rows


def train(study,stage):
    import torch
    from transformers import set_seed
    from .refinement_pr1 import enable_planner
    from .train_sft import trainable_snapshots,changed_tensors,prefix_forward
    root=Path(study['output_dir']);source=Path(study['source_dir'])
    if stage=='CALIBRATED':
        sft=json.loads((root/'SFT/result.json').read_text())
        if not sft['passed']:raise ValueError('Do not calibrate before the directional SFT control passes')
        initial=str(root/'SFT/final_adapter')
    else:initial=study['body_adapter']
    root,r,policy,executor=load(study,initial);model=policy.model;tok=policy.tokenizer
    output=root/stage;output.mkdir(exist_ok=False);set_seed(study['seed']+(stage=='CALIBRATED'))
    before=fingerprints(policy)
    expected=json.loads((source/'natural/BEFORE/result.json').read_text())['frozen']
    assert before['body']==expected['default'] and before['executor']==expected['executor']
    if stage=='SFT':assert before['default']==before['body']
    all_gates={rid:json.loads((source/'values'/rid/'gate.json').read_text()) for rid in r['value_roots']}
    roots=study['sft_roots'] if stage=='SFT' else r['value_roots']
    gates=[all_gates[rid] for rid in roots]
    # Reusable values: every initial plan registered, so no root retry policy
    # intervened in these measurements; every later policy stays frozen here.
    values=[json.loads((source/'values'/rid/'result.json').read_text()) for rid in roots]
    assert all(c['root_first_plan_registered'] for v in values for cs in v['cases'].values() for c in cs)
    assert all(c['split']=='TRAIN' for v in values for cs in v['cases'].values() for c in cs)
    for gate in gates:
        _,encoded,_=policy.prepare(gate['observation']);assert encoded['prefix']==gate['prefix_ids']
    initial_probs={rid:policy.score_modes(all_gates[rid]['observation'])['mode_probabilities']['SPLIT'] for rid in r['value_roots']}
    distinct=sum(a!=b for a,b in zip(gates[0]['prefix_ids'],gates[1]['prefix_ids']))+abs(len(gates[0]['prefix_ids'])-len(gates[1]['prefix_ids']))
    assert distinct>0
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.peft_config['default'].lora_dropout=0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    enable_planner(model);snapshot,trainable=trainable_snapshots(model)
    params=[p for p in model.parameters() if p.requires_grad]
    lr=study['sft_learning_rate'] if stage=='SFT' else study['calibration_learning_rate']
    steps=study['sft_steps'] if stage=='SFT' else study['calibration_steps']
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.);history=[]
    guarded_json(output/'request.json',{'stage':stage,'initial_adapter':initial,'roots':roots,'steps':steps,'lr':lr,
        'labels':study['sft_labels'] if stage=='SFT' else None,'initial_probs':initial_probs,
        'prefix_lengths':[len(g['prefix_ids']) for g in gates],'distinct_prefix_positions':distinct,
        'body_frozen':before['body'],'executor_frozen':before['executor']})
    for step in range(steps):
        enable_planner(model);optimizer.zero_grad(set_to_none=True);total=0.;probs=[];losses=[]
        for i,gate in enumerate(gates):
            modes=list(gate['mode_log_probs'])
            with policy.llm._autocast():
                lp,_=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],modes,tok.pad_token_id)
                if stage=='SFT':loss=-lp[modes.index(study['sft_labels'][i])]
                else:
                    delta=values[i]['delta'];delta=delta if abs(delta)>study['close_margin'] else 0.
                    ref=torch.tensor([gate['mode_log_probs'][m] for m in modes],device=model.device)
                    loss=-lp[modes.index('SPLIT')].exp()*delta+study['kl_weight']*(lp.exp()*(lp-ref)).sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite gate objective')
            probs.append(float(lp[modes.index('SPLIT')].detach().exp()));losses.append(float(loss.detach()))
            (loss/len(gates)).backward();total+=float(loss.detach())/len(gates)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Invalid gate gradient')
        optimizer.step();record={'step':step+1,'objective':total,'per_state_loss':losses,'p_split_before_step':probs,'gradient_norm':float(norm)}
        history.append(record);print(json.dumps(record),flush=True)
    final=output/'final_adapter';model.save_pretrained(final,selected_adapters=['default']);tok.save_pretrained(final)
    changes=changed_tensors(model,snapshot)
    original=prefix_forward(model,gates[0]['prefix_ids'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'):restored=prefix_forward(model,gates[0]['prefix_ids'])
    reload_delta=float((original-restored).abs().max());assert reload_delta<=1e-5
    final_probs={rid:policy.score_modes(all_gates[rid]['observation'])['mode_probabilities']['SPLIT'] for rid in r['value_roots']}
    after=fingerprints(policy);assert before['body']==after['body'] and before['executor']==after['executor']
    probes=frozen_checks(study,policy,output/'frozen_checks')
    a,b=study['sft_roots'];margin=study['minimum_direction_change']
    passed=final_probs[a]-initial_probs[a]>=margin and initial_probs[b]-final_probs[b]>=margin if stage=='SFT' else None
    result={'status':'COMPLETE','stage':stage,'passed':passed,'initial_probs':initial_probs,'final_probs':final_probs,
        'steps':steps,'history':history,'frozen_before':before,'frozen_after':after,'reload_max_abs_difference':reload_delta,
        'parameter_changes':changes,'trainable_parameters':trainable,'body_checks':probes,
        'adapter_sha256':file_hash(final/'adapter_model.safetensors')}
    write_json(output/'result.json',result);print(json.dumps({'stage':stage,'passed':passed,'initial_probs':initial_probs,'final_probs':final_probs}),flush=True)


def natural(study,shard):
    source=Path(study['source_dir']);root=Path(study['output_dir'])
    if json.loads((root/'CALIBRATED/result.json').read_text())['status']!='COMPLETE':raise ValueError('Calibration missing')
    root,r,policy,executor=load(study,str(root/'CALIBRATED/final_adapter'));before=fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    selected=[rows[rid] for rid in r['value_roots']]+[row for row in rows.values() if row['split']=='VALIDATION']
    cases=[];gates=[]
    states=read_jsonl('data/refinement/body_transfer/states.jsonl')
    for i,row in enumerate(selected):
        if i%2!=shard:continue
        state=next(s for s in states if s['root_id']==row['root_id'] and s['node_id']=='N0')
        obs=json.loads(state['messages'][1]['content'])['observation'];gate=policy.score_modes(obs)
        gates.append({'root_id':row['root_id'],'split':row['split'],'p_split':gate['mode_probabilities']['SPLIT']})
        for rep in range(4 if row['split']=='TRAIN' else 2):
            c=run_case(row,tests,r['framework'],r['config'],policy,executor,root/'natural'/row['root_id']/f'repeat_{rep:02d}',r['natural_seed']+i*100+rep)
            if c['status']!='COMPLETE':raise ValueError('Infrastructure failure')
            assert c['split']=='TRAIN' or c['reward'] is None
            cases.append(c)
    assert before==fingerprints(policy)
    write_json(root/'natural'/f'shard_{shard}.json',{'status':'COMPLETE','cases':cases,'gates':gates,'frozen':before})


def report(study):
    root=Path(study['output_dir']);sft=json.loads((root/'SFT/result.json').read_text())
    result={'status':'COMPLETE','sft':sft,'calibration_run':sft['passed']}
    if sft['passed']:
        result['calibration']=json.loads((root/'CALIBRATED/result.json').read_text())
        pieces=[json.loads((root/'natural'/f'shard_{i}.json').read_text()) for i in range(2)]
        cases=[c for p in pieces for c in p['cases']];assert len(cases)==48
        baseline=json.loads((Path(study['source_dir'])/'natural/BEFORE/result.json').read_text())
        assert pieces[0]['frozen']==pieces[1]['frozen']
        assert pieces[0]['frozen']['body']==baseline['frozen']['default'] and pieces[0]['frozen']['executor']==baseline['frozen']['executor']
        result['natural']={};result['root_choices']={}
        for label,records in [('BEFORE',baseline['cases']),('AFTER',cases)]:
            choices=[]
            for c in records:
                event=next(e for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl') if e['event_kind']=='POLICY' and e['node_id']=='N0')
                choices.append({'root_id':c['root_id'],'seed':c['seed'],'split':c['split'],
                    'mode':(event.get('parsed_action') or {}).get('args',{}).get('mode'),'success':c['final_scores']['private']['status']=='PASS'})
            result['root_choices'][label]=choices
        for split in ['TRAIN','VALIDATION']:
            a={(c['root_id'],c['seed']):c for c in result['root_choices']['BEFORE'] if c['split']==split}
            b={(c['root_id'],c['seed']):c for c in result['root_choices']['AFTER'] if c['split']==split}
            assert a.keys()==b.keys()
            result['natural'][split]={'runs_per_model':len(a),'before':sum(c['success'] for c in a.values()),'after':sum(c['success'] for c in b.values()),
                'wins':sum(b[k]['success']>a[k]['success'] for k in a),'losses':sum(b[k]['success']<a[k]['success'] for k in a),
                'split_before':sum(c['mode']=='SPLIT' for c in a.values()),'split_after':sum(c['mode']=='SPLIT' for c in b.values())}
        result['gates']=[g for p in pieces for g in p['gates']]
    write_json(root/'result.json',result)
    print(json.dumps({'status':'COMPLETE','sft_passed':sft['passed'],'initial_probs':sft['initial_probs'],'sft_probs':sft['final_probs'],'natural':result.get('natural')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','sft','calibrate','natural','report'])
    parser.add_argument('--config',default=DEFAULT);parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args();study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':prepare(study)
    elif args.command in {'sft','calibrate'}:train(study,'SFT' if args.command=='sft' else 'CALIBRATED')
    elif args.command=='natural':natural(study,args.shard)
    else:report(study)
