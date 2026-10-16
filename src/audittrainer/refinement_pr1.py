"""One finite root-planning RL pilot: shared data, B3/B4 credit, held-out evaluation."""
# Allocation must precede imports that may initialize CUDA. Library users and
# experiments without gpu_resources.json keep their existing device selection.
if __name__ == '__main__':
    from .gpu_resource import reserve_cli_gpu
    _gpu_reservation = reserve_cli_gpu()

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from transformers import set_seed
import yaml

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_policy import (MODES, active_adapter, adapter_fingerprint, load_policies,
                                mode_log_probs, two_level_advantages)
from .refinement_rollout import collect, costs, run_case, case_summary, split_candidate_count
from .train_refinement import representation_loss
from .train_sft import changed_tensors, prefix_forward, trainable_snapshots


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    if config['stage']!='PR1' or config['split']!='TRAIN' or config['repair']!='disabled':
        raise ValueError('This finite PR1 only trains root plans on TRAIN without repair')
    s=config['candidate_sampling']
    if tuple(s[k] for k in ('atomic','temperature','top_p','top_k'))!=(1,1.,1.,0):
        raise ValueError('PR1 requires one ATOMIC and unwarped conditional sampling')
    counts=[s['split_bodies'], *s.get('split_bodies_by_root', {}).values()]
    repeats=s['continuations_per_candidate']; baselines=config.get('independent_baseline_repetitions',1)
    if any(type(n) is not int or n<2 for n in counts) or any(type(n) is not int or n<1 for n in (repeats,baselines)):
        raise ValueError('Invalid candidate or continuation counts')
    if s.get('candidate_seed_stride',3)<max(counts) or s.get('continuation_seed_stride',2)<repeats or config.get('baseline_seed_stride',1)<baselines:
        raise ValueError('Seed strides must separate roots')
    framework=yaml.safe_load(Path(config['framework_config']).read_text())
    framework['runtime'].update(config.get('runtime_overrides', {}))
    framework['protocol_version']=config['protocol_version']
    framework['runtime']['require_obligation_coverage']=True
    framework['inference']['seed']=config['seed']
    if 'parent_composition_summary' in config:
        framework['parent_composition_summary']=config['parent_composition_summary']
    return config,framework


def enable_planner(model):
    model.set_adapter('default')
    for name,p in model.named_parameters(): p.requires_grad_('lora_' in name and '.default.' in name)
    model.train()


def clipped_objective(logp, old_logp, advantage, epsilon):
    ratio=(logp-old_logp).exp()
    return torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)


def credit_terms(group, method, baseline):
    """A complete state, including invalid/duplicate samples; no reward filtering."""
    p=group['mode_probabilities']['SPLIT']; qa,*qs=group['q_values']
    if method=='B4':
        a=group['advantages']
        return [a['mode'][m] for m in MODES],a['body']
    if method!='B3': raise ValueError(method)
    # The independent natural suffix is not a mean containing any candidate's
    # own reward. B3 clips each plan's gate contribution separately below.
    return [qa-baseline,sum(qs)/len(qs)-baseline],[q-baseline for q in qs]


def mode_surrogate(current,old,mode_advantages,body_advantages,method,epsilon):
    atomic=clipped_objective(current[0],old[0],mode_advantages[0],epsilon)
    if method=='B3':
        advantages=torch.as_tensor(body_advantages,device=current.device,dtype=current.dtype)
        split=clipped_objective(current[1],old[1],advantages,epsilon).mean()
    else:
        split=clipped_objective(current[1],old[1],mode_advantages[1],epsilon)
    return old[0].exp()*atomic+old[1].exp()*split


def causal_logits(model,prefix,target):
    ids=torch.tensor([prefix+target],device=model.device)
    positions=torch.arange(len(prefix)-1,len(prefix)+len(target)-1,device=model.device)
    return model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=positions,use_cache=False).logits[0]


def conditional_token_terms(logits,reference_logits,target,old_logps,advantage,epsilon):
    current=F.log_softmax(logits.float(),dim=-1)
    reference=F.log_softmax(reference_logits.float(),dim=-1)
    labels=torch.tensor(target,device=logits.device)
    chosen=current.gather(1,labels[:,None]).squeeze(1)
    # In this single-round pilot the fixed reference is also the behavior
    # policy. Rescore it with the same full-prefix kernel used by the actor;
    # BF16 cached decoding can otherwise clip tokens before the first update.
    old=(reference.gather(1,labels[:,None]).squeeze(1) if old_logps is None
         else torch.as_tensor(old_logps,device=logits.device,dtype=torch.float32))
    objective=clipped_objective(chosen,old,advantage,epsilon).sum()
    kl=(current.exp()*(current-reference)).sum()
    return objective,kl,chosen


def reward_gradient_norm(objective, model):
    """Check the reward term itself, independently of KL/SFT gradients."""
    gradients=torch.autograd.grad(objective,[p for p in model.parameters() if p.requires_grad],
                                  retain_graph=True,allow_unused=True)
    norm=torch.stack([g.detach().float().norm() for g in gradients if g is not None]).norm()
    if not torch.isfinite(norm) or norm<=0:
        raise RuntimeError('Reward contrast has no finite nonzero adapter gradient')
    return float(norm)


def support_probe(config,framework,planner,executor):
    output=Path(config['adaptation']['output_dir']).parent/'probe'; output.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path('runs/refinement/PR0/groups').glob('*/gate.json'))
    selected=[]; seen=set()
    roots=load_root_manifest('data/manifests/roots.jsonl')
    for i,path in enumerate(paths):
        rid=path.parent.name; family=roots[rid]['family_id']
        if family in seen: continue
        seen.add(family); selected.append((i,path))
    before_valid=after_valid=0; gate_ps=[]; targets=[]
    for i,path in selected:
        saved=json.loads(path.read_text()); folder=output/path.parent.name; folder.mkdir(exist_ok=True)
        gate=planner.score_modes(saved['observation']); write_json(folder/'gate.json',gate)
        gate_ps.append(gate['mode_probabilities']['SPLIT'])
        for j in range(3):
            set_seed(config['seed']+200000+i*3+j)
            candidate=planner.sample_split(gate); write_json(folder/f'candidate_{j+1}.json',candidate)
            previous=json.loads((path.parent/f'candidate_{j+1}.json').read_text())
            before_valid+=previous['parse_error'] is None; after_valid+=candidate['parse_error'] is None
            targets.append(len(candidate['body_output_ids']))
        print(json.dumps({'format_probe':path.parent.name,'p_split':gate_ps[-1]}),flush=True)
    result={'status':'COMPLETE','roots':len(selected),'split_candidates':len(targets),
            'before_parse_valid':before_valid,'after_parse_valid':after_valid,'mean_p_split':sum(gate_ps)/len(gate_ps),
            'comparison':'same PR0 public observations and candidate seeds, unwarped sampling, no retries or repairs',
            'actual_model_cost':dict(costs(planner,executor))}
    write_json(output/'result.json',result); return result


def baseline_paths(config, root_id):
    folder=Path(config['output_dir'])/'independent_baseline'/root_id
    count=config.get('independent_baseline_repetitions',1)
    return [folder] if count==1 else [folder/f'repeat_{i:02d}' for i in range(count)]


def load_baselines(config, root_id):
    rows=[json.loads((p/'case_result.json').read_text()) for p in baseline_paths(config,root_id)]
    if any(r['root_id']!=root_id or r['split']!='TRAIN' or r['status']!='COMPLETE' for r in rows):
        raise ValueError('Invalid independent baseline')
    return rows


def credited_return(case, rule):
    """Assign a saved TRAIN outcome to its initial planning action."""
    if rule not in {'full_return', 'first_registered'}:
        raise ValueError(f'Unknown reward credit rule: {rule}')
    if case['split'] != 'TRAIN' or case['status'] != 'COMPLETE' or case['reward'] is None:
        raise ValueError('Credit assignment requires complete TRAIN feedback')
    if rule == 'first_registered' and not case['root_first_plan_registered']:
        return 0.
    return float(case['reward']['reward'])


def load_training_groups(config):
    rule = config.get('reward_credit', 'full_return')
    if rule not in {'full_return', 'first_registered'}:
        raise ValueError(f'Unknown reward credit rule: {rule}')
    roots=load_root_manifest(config['root_manifest'])
    groups=[]
    for rid,row in sorted(roots.items()):
        if row['split']!='TRAIN': continue
        path=Path(config['output_dir'])/'groups'/rid/'group_result.json'
        group=json.loads(path.read_text())
        count=split_candidate_count(config,rid)+1
        repeats=config['candidate_sampling']['continuations_per_candidate']
        if (group['status']!='COMPLETE' or group['split']!='TRAIN' or group['root_id']!=rid
                or group['candidate_count']!=count or group['continuations']!=count*repeats
                or len(group['case_paths'])!=count or any(len(p)!=repeats for p in group['case_paths'])):
            raise ValueError('Unfinished/non-TRAIN group')
        gate=json.loads(Path(group['gate_path']).read_text())
        candidates=[json.loads((path.parent/f'candidate_{i}.json').read_text()) for i in range(count)]
        baselines=load_baselines(config,rid)
        if any(len(c['body_output_ids'])!=len(c['body_old_log_probs']) for c in candidates): raise ValueError('Bad saved likelihoods')
        baseline=sum(credited_return(b,rule) for b in baselines)/len(baselines)
        if rule == 'first_registered':
            original_q=group['q_values']
            cases=[[json.loads((Path(p)/'case_result.json').read_text()) for p in paths] for paths in group['case_paths']]
            assigned_q=[sum(credited_return(c,rule) for c in row)/len(row) for row in cases]
            group['q_values']=assigned_q
            group['advantages']=two_level_advantages(group['mode_probabilities']['SPLIT'],assigned_q[0],assigned_q[1:])
            group['credit_assignment']={'rule':rule,'original_q_values':original_q,'assigned_q_values':assigned_q,
                'original_baseline':sum(b['reward']['reward'] for b in baselines)/len(baselines),
                'assigned_baseline':baseline,
                'initially_rejected_candidate_continuations':sum(not c['root_first_plan_registered'] for row in cases for c in row),
                'initially_rejected_baseline_continuations':sum(not c['root_first_plan_registered'] for c in baselines)}
        groups.append((group,gate,candidates,baseline))
    if len(groups)!=config['root_limit']: raise ValueError('Missing training states')
    return groups


def success_imitation_targets(group, gate):
    """Success-filtered MLE targets, correcting forced mode/candidate counts."""
    if group['split'] != 'TRAIN' or group['status'] != 'COMPLETE':
        raise ValueError('Success imitation requires complete TRAIN feedback')
    counts = []
    rates = []
    for paths in group['case_paths']:
        cases = [json.loads((Path(p)/'case_result.json').read_text()) for p in paths]
        if any(c['status'] != 'COMPLETE' or c['reward'] is None for c in cases):
            raise ValueError('Incomplete success-filtering feedback')
        successes = sum(c['root_first_plan_registered'] and c['reward']['y'] == 1
                        and c['reward']['v'] == 0 for c in cases)
        counts.append(successes); rates.append(successes/len(cases))
    p = gate['mode_probabilities']['SPLIT']; k = len(rates)-1
    mass = [(1-p)*rates[0], *[p*q/k for q in rates[1:]]]
    total = sum(mass)
    return {'success_counts': counts, 'success_mass': total,
            'weights': [x/total if total else 0. for x in mass]}


def train(config,framework,method):
    if method not in {'B3','B4','SUCCESS_SFT'} or not config['update']['enabled']: raise ValueError('Select an enabled planning update')
    if config.get('reward_credit','full_return') != 'full_return' and method != 'B3':
        raise ValueError('First-plan credit is only enabled for the B3 ablation')
    groups=load_training_groups(config)
    imitation = {g['root_id']: success_imitation_targets(g, gate) for g,gate,_,_ in groups} if method == 'SUCCESS_SFT' else {}
    checksum=file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')
    requests=list(Path(config['output_dir']).glob('collect_*_shard*_request.json'))
    if not requests or any(json.loads(p.read_text())['adapter_sha256']!=checksum for p in requests):
        raise ValueError('The behavior adapter must be the unchanged sampling policy')
    output=Path(config['output_dir'])/method; output.mkdir(parents=True,exist_ok=False)
    (output/'scoring_probabilities').mkdir()
    write_json(output/'request.json',{'config':config,'method':method,'sampling_rounds':1,
        'data_sha256':{g['root_id']:file_hash(Path(g['gate_path']).parent/'group_result.json') for g,_,_,_ in groups}})
    planner,executor=load_policies(framework,config['initial_adapter'],config['executor_adapter'])
    model,tokenizer=planner.model,planner.tokenizer
    reference_path=config.get('reference_adapter',config['initial_adapter'])
    model.load_adapter(reference_path,adapter_name='reference',is_trainable=False,local_files_only=True)
    behavior_name='reference'
    if Path(reference_path).resolve()!=Path(config['initial_adapter']).resolve():
        behavior_name='behavior'
        model.load_adapter(config['initial_adapter'],adapter_name=behavior_name,is_trainable=False,local_files_only=True)
    frozen={n:adapter_fingerprint(model,n) for n in dict.fromkeys(('reference','executor',behavior_name))}
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.0
    model.peft_config['default'].lora_dropout=0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.config.use_cache=False; enable_planner(model)
    before,trainable=trainable_snapshots(model)
    settings=config['update']; accumulation=settings['gradient_accumulation_groups']
    if len(groups)%accumulation: raise ValueError('Use complete state accumulation windows')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=0)
    if config.get('optimizer_state'):
        optimizer.load_state_dict(torch.load(config['optimizer_state'],map_location=model.device,weights_only=True))
    previous_steps={int(s['step']) for s in optimizer.state.values()}
    if len(previous_steps)>1: raise ValueError('Optimizer parameters have inconsistent step counts')
    previous_steps=next(iter(previous_steps),0)
    demos=read_jsonl(Path(config['adaptation']['output_dir'])/'examples.jsonl')
    if any(e['split']!='TRAIN' for e in demos): raise ValueError('Non-TRAIN retention example')
    eps,beta,eta,scale=(settings[k] for k in ('clip_epsilon','kl_weight','sft_weight','objective_scale'))
    started=time.monotonic(); history=[]; steps=0; max_initial_delta=0.; max_cached_delta=0.
    old_body_scores={}; reference_modes={}; reward_gradients={}
    optimizer.zero_grad(set_to_none=True)
    set_seed(config['seed'])
    for epoch in range(settings['epochs']):
        order=list(range(len(groups))); random.Random(config['seed']+epoch).shuffle(order)
        for position,index in enumerate(order):
            group,gate,candidates,baseline=groups[index]
            mode_a,body_a=credit_terms(group,'B3' if method == 'SUCCESS_SFT' else method,baseline)
            positive = imitation[group['root_id']]['weights'] if imitation else None
            device=model.device
            old=torch.tensor([gate['mode_log_probs'][m] for m in MODES],device=device)
            weights=old.exp().detach()
            if index not in reference_modes:
                with active_adapter(model,'reference'),torch.no_grad(),planner.llm._autocast():
                    reference_modes[index]=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],MODES,tokenizer.pad_token_id)[0].detach()
            enable_planner(model)
            with planner.llm._autocast():
                current,_=mode_log_probs(model,gate['prefix_ids'],gate['mode_token_ids'],MODES,tokenizer.pad_token_id)
                if positive is not None:
                    targets = current.new_tensor([positive[0], sum(positive[1:])])
                    mode_objective = (targets*current).sum()
                else:
                    mode_objective=mode_surrogate(current,old,mode_a,body_a,method,eps)
                mode_kl=(current.exp()*(current-reference_modes[index])).sum()
                gate_loss=scale*(-settings.get('mode_reward_weight',1.)*mode_objective+beta*mode_kl)/accumulation
            if settings.get('check_reward_gradients') and 'mode' not in reward_gradients and mode_a[0]!=mode_a[1]:
                reward_gradients['mode']={'root_id':group['root_id'],'norm':reward_gradient_norm(mode_objective,model)}
            if steps==0:
                delta=float((current.detach()-old).abs().max()); max_initial_delta=max(max_initial_delta,delta)
                if delta>.05: raise RuntimeError(f'Initial gate probabilities differ: {delta}')
            gate_loss.backward()
            stats={'epoch':epoch,'root_id':group['root_id'],'mode_advantages':mode_a,'body_advantages':body_a,
                   'gate_loss':float(gate_loss.detach()),'body_loss':0.,'body_kl':0.}
            for candidate_index,(candidate,advantage) in enumerate(zip(candidates[1:],body_a),start=1):
                prefix,target=candidate['body_input_ids'],candidate['body_output_ids']
                if not target: raise ValueError('Missing sampled body; invalid actions must still retain tokens')
                with active_adapter(model,'reference'),torch.no_grad(),planner.llm._autocast():
                    reference=causal_logits(model,prefix,target).detach()
                score_key=(index,candidate_index)
                if score_key not in old_body_scores:
                    with active_adapter(model,behavior_name),torch.no_grad(),planner.llm._autocast():
                        behavior=(reference if behavior_name=='reference' else causal_logits(model,prefix,target))
                        old_body_scores[score_key]=F.log_softmax(behavior.float(),dim=-1).gather(
                            1,torch.tensor(target,device=device)[:,None]).squeeze(1).detach()
                        del behavior
                old_scoring=old_body_scores[score_key]
                enable_planner(model)
                with planner.llm._autocast():
                    logits=causal_logits(model,prefix,target)
                    objective,kl,chosen=conditional_token_terms(logits,reference,target,old_scoring,advantage,eps)
                    if positive is not None:
                        # Use the existing representation-SFT convention: mode
                        # cross entropy plus mean body token cross entropy.
                        objective = positive[candidate_index]*chosen.mean()
                        loss=scale*(-objective+beta*weights[1]/len(body_a)*kl)/accumulation
                    else:
                        loss=scale*weights[1]/len(body_a)*(-objective+beta*kl)/accumulation
                has_signal = positive[candidate_index] > 0 if positive is not None else advantage != 0
                if settings.get('check_reward_gradients') and 'body' not in reward_gradients and has_signal:
                    reward_gradients['body']={'root_id':group['root_id'],'candidate':candidate_index,
                                              'norm':reward_gradient_norm(objective,model)}
                if epoch==0:
                    cached_delta=float((old_scoring-torch.tensor(candidate['body_old_log_probs'],device=device)).abs().max())
                    max_cached_delta=max(max_cached_delta,cached_delta)
                    write_json(output/'scoring_probabilities'/f"{group['root_id']}_{candidate_index}.json",
                        {'source_candidate':str(Path(group['gate_path']).parent/f'candidate_{candidate_index}.json'),
                         'old_teacher_forced_log_probs':old_scoring.cpu().tolist(),
                         'max_cached_log_prob_difference':cached_delta})
                if steps==0:
                    delta=float((chosen.detach()-old_scoring).abs().max())
                    max_initial_delta=max(max_initial_delta,delta)
                    if delta>1e-4: raise RuntimeError(f'Initial scoring probabilities differ: {delta}')
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RL loss')
                loss.backward(); stats['body_loss']+=float(loss.detach()); stats['body_kl']+=float(kl.detach())
                del reference,logits,objective,kl,chosen,loss,old_scoring
            enable_planner(model)
            with planner.llm._autocast():
                retention_index=config.get('retention_offset',0)+epoch*len(groups)+position
                retention,_,_=representation_loss(model,demos[retention_index%len(demos)],tokenizer.pad_token_id)
                regularizer=scale*eta*retention/accumulation
            regularizer.backward(); stats['sft_loss']=float(regularizer.detach())
            if (position+1)%accumulation==0:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings['max_grad_norm'])
                if not torch.isfinite(norm) or norm<=0: raise RuntimeError('Missing/nonfinite gradient')
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps+=1
                stats.update(optimizer_step=steps,gradient_norm=float(norm))
            history.append(stats); print(json.dumps(stats),flush=True)
    changes=changed_tensors(model,before)
    final=output/'final_adapter'; model.save_pretrained(final,selected_adapters=['default']); tokenizer.save_pretrained(final)
    torch.save(optimizer.state_dict(),output/'optimizer.pt')
    after={n:adapter_fingerprint(model,n) for n in frozen}
    if frozen!=after: raise RuntimeError('Fixed reference/executor changed')
    # Reload the written adapter into the same immutable NF4 base and compare.
    prefix=groups[0][1]['prefix_ids']; before_logits=prefix_forward(model,prefix)
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'): restored=prefix_forward(model,prefix)
    delta=float((restored-before_logits).abs().max())
    if delta>1e-5: raise RuntimeError(f'Saved RL adapter does not reload: {delta}')
    report={'status':'COMPLETE','stage':'PR1_ROOT_RL','method':method,'scope':'TRAIN_PILOT',
            'sampling_rounds':1,'epochs':settings['epochs'],'root_states':len(groups),'candidates':sum(len(cs) for _,_,cs,_ in groups),
            'states_with_mode_value_difference':sum(g['advantages']['q_atomic']!=g['advantages']['q_split'] for g,_,_,_ in groups),
            'states_with_conditional_value_difference':sum(len(set(g['q_values'][1:]))>1 for g,_,_,_ in groups),
            'candidate_continuations':sum(g['continuations'] for g,_,_,_ in groups),
            'independent_baseline_continuations':len(groups)*config.get('independent_baseline_repetitions',1),
            'optimizer_steps':steps,'previous_optimizer_steps':previous_steps,'cumulative_optimizer_steps':previous_steps+steps,
            'reward_gradient_checks':reward_gradients,'trainable_parameters':trainable,'parameter_changes':changes,
            'frozen_before':frozen,'frozen_after':after,'max_initial_log_prob_difference':max_initial_delta,
            'max_cached_vs_scoring_log_prob_difference':max_cached_delta,
            'body_probability_convention':'Frozen current-round behavior teacher-forced denominator, same scoring path as actor; original cached-generation probabilities retained. KL uses the separately configured fixed reference.',
            'behavior_adapter':config['initial_adapter'],'behavior_sha256':checksum,
            'reference_adapter':reference_path,'reference_sha256':file_hash(Path(reference_path)/'adapter_model.safetensors'),
            'reload_max_abs_difference':delta,'history':history,'train_seconds':time.monotonic()-started,
            'adapter':str(final),'adapter_sha256':file_hash(final/'adapter_model.safetensors'),
            'adapter_config_sha256':file_hash(final/'adapter_config.json'),
            'objective':'mode KL exact; body full-vocabulary KL on old sampled prefixes; body token sum; shared global scale',
            'mode_reward_weight':settings.get('mode_reward_weight',1.),
            'seed':config['seed']}
    if imitation:
        report.update(stage='PR1_ROOT_SUCCESS_SFT', success_imitation_targets=imitation,
            objective='Success-filtered weighted mode CE plus mean body-token CE; original mode probabilities correct stratified candidate sampling. Same full-vocabulary KL on all sampled prefixes and same retention SFT as B3. No advantages or clipping in the imitation term.')
    report['reward_credit']=config.get('reward_credit','full_return')
    if report['reward_credit'] == 'first_registered':
        report['credit_assignment']={g['root_id']:g['credit_assignment'] for g,_,_,_ in groups}
    write_json(output/'training_report.json',report); return report


def evaluate(config,framework,planner,executor,method):
    rows=sorted(load_root_manifest(config['root_manifest']).values(),key=lambda r:r['root_id'])
    tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    f=deepcopy(framework); f['allow_validation']=True
    output=Path(config['output_dir'])/'evaluation'/method; records=[]
    for index,row in enumerate(rows):
        repetitions=config['evaluation']['validation_repetitions' if row['split']=='VALIDATION' else 'train_repetitions']
        for rep in range(repetitions):
            records.append(run_case(row,tests,f,config,planner,executor,output/row['root_id']/f'repeat_{rep}',
                                    config['seed']+700000+index*2+rep))
    if any(r['status']!='COMPLETE' for r in records): raise RuntimeError('Incomplete evaluation infrastructure')
    result={'status':'COMPLETE','method':method,'TRAIN':case_summary([r for r in records if r['split']=='TRAIN']),
            'VALIDATION':case_summary([r for r in records if r['split']=='VALIDATION']),
            'selection':'One natural branch; no candidate search, fixed seeds, validation never provides optimizer rewards'}
    write_json(output/'result.json',result); return result


def summarize(config):
    output=Path(config['output_dir']); groups=load_training_groups(config)
    atomic,split=[],[]; valid=registered=body_differences=split_better=0
    baselines=[]; generation=Counter()
    for group,gate,candidates,baseline in groups:
        generation.update(group['candidate_generation_cost'])
        valid+=sum(c['parse_error'] is None for c in candidates[1:])
        split_better+=group['advantages']['q_split']>group['advantages']['q_atomic']
        values=[]
        for index,paths in enumerate(group['case_paths']):
            records=[json.loads((Path(p)/'case_result.json').read_text()) for p in paths]
            (atomic if index==0 else split).extend(records)
            if index and all(r['root_first_plan_registered'] for r in records):
                registered+=1; values.append(group['q_values'][index])
        body_differences+=len(set(values))>1
        baselines.extend(load_baselines(config,group['root_id']))
    adaptation=json.loads((Path(config['adaptation']['output_dir'])/'training_report.json').read_text())
    probe=json.loads((Path(config['adaptation']['output_dir']).parent/'probe/result.json').read_text())
    if adaptation['status']!='COMPLETE' or probe['status']!='COMPLETE': raise ValueError('Unfinished support')
    evaluations={m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in ('B0','B3','B4')}
    updates={m:json.loads((output/m/'training_report.json').read_text()) for m in ('B3','B4')}
    for m,r in updates.items():
        expected=config['update']['epochs']*len(groups)//config['update']['gradient_accumulation_groups']
        if r['status']!='COMPLETE' or r['optimizer_steps']!=expected or r['frozen_before']!=r['frozen_after']:
            raise ValueError(f'Unfinished/faulty RL update: {m}')
    for m,r in evaluations.items():
        if r['status']!='COMPLETE' or r['TRAIN']['runs']!=10 or r['VALIDATION']['runs']!=8:
            raise ValueError(f'Missing natural evaluation: {m}')
        for p in (output/'evaluation'/m).glob('pr1_validation_*/repeat_*/case_result.json'):
            case=json.loads(p.read_text())
            if case['reward'] is not None or case['split']!='VALIDATION': raise ValueError('Validation reward leakage')
    workers=[]
    for p in sorted(output.glob('*_request.json')):
        completed=p.with_name(p.name.replace('_request.json','_complete.json'))
        r=json.loads(completed.read_text())
        if r['status']!='COMPLETE' or r['before']!=r['after']: raise ValueError(f'Unfinished/unfrozen worker {p}')
        workers.append({'request':str(p),'completion':str(completed),'actual_model_cost':r['actual_model_cost']})
    result={'status':'COMPLETE','stage':'PR1','scope':'ONE_ROUND_TRAIN_PILOT_WITH_FAMILY_DISJOINT_VALIDATION',
            'support_sft':adaptation,'format_probe':probe,
            'pilot':{'roots':len(groups),'candidates':len(groups)*4,'continuations':len(atomic)+len(split),
                     'atomic':case_summary(atomic),'split':case_summary(split),'independent_baseline':case_summary(baselines),
                     'split_parse_valid':valid,'split_initially_registered':registered,'split_candidates':len(groups)*3,
                     'split_successes_with_initial_registration':sum(r['root_first_plan_registered'] and r['reward']['y'] for r in split),
                     'split_successes_after_initial_rejection':sum(not r['root_first_plan_registered'] and r['reward']['y'] for r in split),
                     'states_split_average_better':split_better,'states_registered_split_body_differences':body_differences,
                     'mean_p_split':sum(g['mode_probabilities']['SPLIT'] for g,_,_,_ in groups)/len(groups),
                     'candidate_generation_cost':dict(generation)},
            'updates':updates,'evaluation':evaluations,'frozen_workers':workers,
            'comparability':'Identical B0 candidates and independent baseline shared by B3/B4; same optimizer schedule and eval seeds. Common sampling cost charged equally; only advantage construction differs.',
            'groups':[g for g,_,_,_ in groups]}
    write_json(output/'result.json',result)
    print(json.dumps({'status':'COMPLETE','pilot_atomic':result['pilot']['atomic']['final_successes'],
                      'pilot_split':result['pilot']['split']['final_successes'],
                      'validation':{m:r['VALIDATION']['final_successes'] for m,r in evaluations.items()}}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['support','collect','train','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_pr1.yaml')
    parser.add_argument('--method',choices=['B0','B3','B4','SUCCESS_SFT'],default='B0')
    parser.add_argument('--shards',type=int,default=1); parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args(); config,framework=load_config(args.config)
    if args.command=='report': summarize(config); return
    if args.command=='train': train(config,framework,args.method); return
    adapter=config['initial_adapter'] if args.method=='B0' or args.command=='collect' else str(Path(config['output_dir'])/args.method/'final_adapter')
    planner,executor=load_policies(framework,adapter,config['executor_adapter'])
    before={n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    job=f'{args.command}_{args.method}_shard{args.shard_index}'
    request={'config':config,'adapter':adapter,'adapter_sha256':file_hash(Path(adapter)/'adapter_model.safetensors'),
             'fingerprints_before':before,'shards':args.shards,'shard_index':args.shard_index}
    request_path=output/f'{job}_request.json'
    if request_path.exists() and json.loads(request_path.read_text())!=request: raise ValueError('Changed resumed job')
    write_json(request_path,request)
    if args.command=='support': support_probe(config,framework,planner,executor)
    elif args.command=='collect':
        rows=sorted((r for r in load_root_manifest(config['root_manifest']).values() if r['split']=='TRAIN'),key=lambda r:r['root_id'])
        tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
        collect(rows,tests,framework,config,planner,executor,args.shard_index,args.shards)
        for i,row in enumerate(rows):
            if i%args.shards==args.shard_index:
                for repetition, folder in enumerate(baseline_paths(config,row['root_id'])):
                    run_case(row,tests,framework,config,planner,executor,folder,
                             config['seed']+config.get('baseline_seed_offset',900000)
                             +i*config.get('baseline_seed_stride',1)+repetition)
    else: evaluate(config,framework,planner,executor,args.method)
    after={n:adapter_fingerprint(planner.model,n) for n in before}
    if before!=after: raise RuntimeError('Collection/evaluation changed weights')
    write_json(output/f'{job}_complete.json',{'status':'COMPLETE','before':before,'after':after,
                                           'actual_model_cost':dict(costs(planner,executor)),'rl_optimizer_steps':0})


if __name__=='__main__': main()
