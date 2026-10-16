"""B3: candidate breadth versus repeated return estimates at equal rollout budget."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

import yaml

from .dataset import file_hash, load_root_manifest
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum_rl import evaluate, evaluation_summary
from .refinement_multiround import natural_allocations, prepare as prepare_diagnostics, training_signals
from .refinement_pr1 import load_baselines, load_config, load_training_groups, train
from .refinement_rollout import split_candidate_count

METHODS = ('B0', 'B3_wide', 'B3_repeat')


def arm_config(config, arm):
    result=deepcopy(config)
    result.update(output_dir=str(Path(config['output_dir'])/'arms'/arm),
                  root_manifest=str(Path(config['output_dir'])/'training_roots.jsonl'),
                  root_limit=len(config['training_roots']), independent_baseline_repetitions=1 if arm=='wide' else 4,
                  baseline_seed_stride=8)
    result['candidate_sampling'].update(continuations_per_candidate=2 if arm=='wide' else 4,
                                       candidate_seed_stride=32,continuation_seed_stride=8)
    if arm=='wide': result['candidate_sampling']['split_bodies_by_root']=config['wide_split_counts']
    if arm=='audit':
        result['candidate_sampling']['continuation_seed_offset']=600000
        result['baseline_seed_offset']=1800000
        result['update']['enabled']=False
    return result


def prepare(config):
    prepare_diagnostics(config)
    output=Path(config['output_dir']); rows=load_root_manifest(config['root_manifest'])
    chosen=config['training_roots']
    if len(set(chosen))!=6 or any(rows[r]['split']!='TRAIN' for r in chosen):
        raise ValueError('Expected the six predeclared TRAIN roots')
    text=''.join(json.dumps(rows[r],sort_keys=True)+'\n' for r in sorted(chosen))
    path=output/'training_roots.jsonl'
    if path.exists() and path.read_text()!=text: raise ValueError('Changed training tasks')
    path.write_text(text)
    for arm in ('wide','repeat','audit'):
        current=arm_config(config,arm); folder=Path(current['output_dir']); folder.mkdir(parents=True,exist_ok=True)
        path=folder/'config.yaml'
        if path.exists() and yaml.safe_load(path.read_text())!=current: raise ValueError('Changed arm')
        path.write_text(yaml.safe_dump(current,sort_keys=False))
        load_config(path)
        budget=sum((1+split_candidate_count(current,r))*current['candidate_sampling']['continuations_per_candidate']
                   +current['independent_baseline_repetitions'] for r in chosen)
        if budget!=120: raise ValueError('Expected 120 complete trajectories per arm/audit')
        if arm!='audit':
            link=output/f'B3_{arm}'/'final_adapter'; link.parent.mkdir(exist_ok=True)
            target=(folder/'B3/final_adapter').resolve()
            if link.is_symlink() or link.exists():
                if link.resolve()!=target: raise ValueError('Changed evaluation checkpoint')
            else: link.symlink_to(target,target_is_directory=True)


def prepare_audit(config):
    """Freeze the actual sampled repeat-arm actions; audit rewards never train."""
    source=arm_config(config,'repeat'); target=arm_config(config,'audit')
    # Only frozen sampled actions are needed; source rewards may still be running.
    for rid in config['training_roots']:
        a=Path(source['output_dir'])/'groups'/rid; b=Path(target['output_dir'])/'groups'/rid
        b.mkdir(parents=True,exist_ok=True)
        for name in ['gate.json',*[f'candidate_{i}.json' for i in range(4)]]:
            content=(a/name).read_bytes()
            if (b/name).exists() and (b/name).read_bytes()!=content: raise ValueError('Changed audit action')
            (b/name).write_bytes(content)
        # Audit reuses sampled actions and their likelihoods; it generates no new candidates.
        write_json(b/'generation_cost.json',{})


def estimation_summary(config):
    arms={name:{g['root_id']:(g,gate,candidates,b) for g,gate,candidates,b in load_training_groups(arm_config(config,name))}
          for name in ('wide','repeat','audit')}
    records=[]; baselines=[]
    for rid in config['training_roots']:
        audit,gate,actions,reference_b=arms['audit'][rid]
        baselines.append({'root_id':rid,'audit':reference_b,
                          **{m:arms[m][rid][3] for m in ('wide','repeat')}})
        for i in range(4):
            if any(arms[m][rid][2][i]['parsed_action']!=actions[i]['parsed_action']
                   or arms[m][rid][2][i]['body_output_ids']!=actions[i]['body_output_ids']
                   or arms[m][rid][1]['observation_sha256']!=gate['observation_sha256'] for m in ('wide','repeat')):
                raise ValueError('Independent value comparison requires the same sampled action/state')
            row={'root_id':rid,'candidate':i,'audit_q':audit['q_values'][i],
                 'audit_advantage':audit['q_values'][i]-reference_b}
            for m in ('wide','repeat'):
                g,_,_,b=arms[m][rid]
                row[m]={'q':g['q_values'][i],'advantage':g['q_values'][i]-b}
            records.append(row)
    sign=lambda x: (x>0)-(x<0)
    summary={m:{
        'q_mse_vs_independent_audit':mean((r[m]['q']-r['audit_q'])**2 for r in records),
        'baseline_mse_vs_independent_audit':mean((r[m]-r['audit'])**2 for r in baselines),
        'advantage_mse_vs_independent_audit':mean((r[m]['advantage']-r['audit_advantage'])**2 for r in records),
        'advantage_sign_disagreements_including_zero':sum(sign(r[m]['advantage'])!=sign(r['audit_advantage']) for r in records),
    } for m in ('wide','repeat')}
    return {'matched_states':len(baselines),'matched_actions':len(records),'summary':summary,
            'records':records,'baselines':baselines,
            'meaning':'Same first three sampled SPLIT actions plus ATOMIC; audit uses independent four-repeat Q and four-repeat natural baseline. Audit is noisy, not ground truth; errors are correlated within each of six roots. Extra wide candidates only enter its training, not this paired estimator diagnostic.'}


def recovery_counts(cases):
    return {'success_with_first_plan_registered':sum(c['reward']['y'] and c['root_first_plan_registered'] for c in cases),
            'success_after_first_plan_rejection':sum(c['reward']['y'] and not c['root_first_plan_registered'] for c in cases)}


def report(config):
    output=Path(config['output_dir']); updates={}; signals={}; workers={}; case_summaries={}
    from .refinement_rollout import case_summary
    for arm in ('wide','repeat','audit'):
        current=arm_config(config,arm); folder=Path(current['output_dir'])
        groups=load_training_groups(current)
        cases=[json.loads((Path(p)/'case_result.json').read_text()) for g,_,_,_ in groups for ps in g['case_paths'] for p in ps]
        cases.extend(c for rid in config['training_roots'] for c in load_baselines(current,rid))
        if len(cases)!=120 or any(c['status']!='COMPLETE' for c in cases): raise ValueError('Incomplete collection')
        case_summaries[arm]={**case_summary(cases),**recovery_counts(cases)}
        workers[arm]=[json.loads(p.read_text()) for p in folder.glob('collect_B3_shard*_complete.json')]
        requests=[json.loads(p.read_text()) for p in folder.glob('collect_B3_shard*_request.json')]
        if (not workers[arm] or len(requests)!=len(workers[arm])
                or len(requests)!=requests[0]['shards']
                or any(w['before']!=w['after'] or w['status']!='COMPLETE' for w in workers[arm])
                or any(q['adapter_sha256']!=file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors') for q in requests)):
            raise ValueError('Unfinished/unfrozen collection')
        if arm=='audit': continue
        u=json.loads((folder/'B3/training_report.json').read_text())
        if (u['status']!='COMPLETE' or u['optimizer_steps']!=10 or u['previous_optimizer_steps']!=50
                or u['cumulative_optimizer_steps']!=60 or u['frozen_before']!=u['frozen_after']
                or u['candidate_continuations']+u['independent_baseline_continuations']!=120):
            raise ValueError('Unmatched update budget')
        updates[f'B3_{arm}']=u
        signals[arm]=training_signals(current,{'rounds':[{'round':1,'output_dir':str(folder)}]})
    evaluations={m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in METHODS}
    allocations={m:natural_allocations(config,m) for m in METHODS}
    all_roots=set(load_root_manifest(config['root_manifest'])); trained=set(config['training_roots'])
    paired={name:paired_rows(allocations['B3_wide']['records'],allocations['B3_repeat']['records'],ids)
            for name,ids in {'all':all_roots,'updated_six':trained,'remaining_six':all_roots-trained}.items()}
    recovery={m:recovery_counts([json.loads(p.read_text()) for p in (output/'evaluation'/m).glob('natural/*/repeat_*/case_result.json')]) for m in METHODS}
    result={'status':'COMPLETE','scope':'B3_EQUAL_ROLLOUT_BUDGET_RETURN_ESTIMATION_PILOT',
            'updates':updates,'collections':case_summaries,'frozen_workers':workers,
            'estimation':estimation_summary(config),'training_signals':signals,
            **evaluation_summary(config,evaluations,METHODS),'natural_allocations':allocations,
            'paired_repeat_vs_wide':paired,'natural_recovery':recovery,
            'data_use':'Both continue the same rotating B3 weights and Adam step 50, fixed common SFT KL reference. Six TRAIN roots chosen from prior TRAIN calibration/collection; no new SFT phase. Same 120 training-rollout and ten-update budget per arm. Independent 120-rollout value audit and final natural evaluation never train. One training seed; all twelve evaluation roots are TRAIN.'}
    write_json(output/'result.json',result)
    print(json.dumps({'status':'COMPLETE','estimation':result['estimation']['summary'],
                      'passes':{m:e['natural']['final_successes'] for m,e in evaluations.items()},'paired':paired}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-audit','train','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_variance.yaml')
    parser.add_argument('--arm',choices=['wide','repeat'],default='repeat')
    parser.add_argument('--method',choices=METHODS,default='B0')
    args=parser.parse_args(); config,framework=load_config(args.config)
    prepare(config)
    if args.command=='prepare-audit': prepare_audit(config)
    elif args.command=='train':
        current,fw=load_config(Path(config['output_dir'])/'arms'/args.arm/'config.yaml')
        train(current,fw,'B3')
    elif args.command=='evaluate': evaluate(config,framework,args.method)
    elif args.command=='report': report(config)


if __name__=='__main__': main()
