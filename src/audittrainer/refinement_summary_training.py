"""Shared fresh feedback, successful-plan SFT versus B3, with frozen parent summary."""
import argparse
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_summary_training.yaml'
    _gpu_reservation = acquire(Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json',
                               'train' if sys.argv[1]=='train' else 'collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_first_credit import evaluate_checkpoint
from .refinement_generalization import guarded_json
from .refinement_multiround import training_signals
from .refinement_parent_validation import presentation_identity
from .refinement_pr1 import load_config, load_training_groups, success_imitation_targets, train
from .refinement_success_sft import batch_config, models, METHODS
from .verifier import canonical


def prepare(study):
    output=Path(study['output_dir']); output.mkdir(parents=True,exist_ok=True)
    base,_=load_config(study['base_config'])
    rows=load_root_manifest(base['root_manifest'])
    tests={r['root_id']:r for r in read_jsonl(base['test_manifest'])}
    if any(rows[r]['split']!='TRAIN' for r in study['root_ids']): raise ValueError('TRAIN roots required')
    data=output/'data'; data.mkdir(exist_ok=True)
    for name,source in [('roots',rows),('tests',tests)]:
        path=data/f'{name}.jsonl'; content=''.join(canonical(source[r])+'\n' for r in sorted(study['root_ids']))
        if path.exists() and path.read_text()!=content: raise ValueError('Changed TRAIN manifest')
        path.write_text(content)
    for seed in study['batch_seeds']:
        config=batch_config(study,seed)
        config.update(parent_composition_summary=study['parent_composition_summary'],reward_credit='full_return')
        folder=Path(config['output_dir']); folder.mkdir(exist_ok=True); path=folder/'config.yaml'
        if path.exists() and yaml.safe_load(path.read_text())!=config: raise ValueError('Changed batch')
        path.write_text(yaml.safe_dump(config,sort_keys=False))
    prior=json.loads(Path('runs/refinement/B3_parent_validation/request.json').read_text())
    if presentation_identity()!=prior['presentation']: raise ValueError('Validated summary changed')
    guarded_json(output/'request.json',{'study':study,'presentation':presentation_identity(),
        'adapters':{k:file_hash(Path(base[k])/'adapter_model.safetensors') for k in ('initial_adapter','executor_adapter')},
        'manifests':{name:file_hash(data/f'{name}.jsonl') for name in ('roots','tests')},
        'planned':{'fresh_training_continuations':240,'optimizer_steps':80,'natural_evaluations':480},
        'scope':'Three existing TRAIN roots; fresh execution seeds, not held-out task generalization. No old returns or optimizer states reused.'})
    guarded_json(output/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000,'train':22000}})


def batch_path(study,seed):
    if seed not in study['batch_seeds']: raise ValueError('Unknown batch')
    return Path(study['output_dir'])/f'batch_{seed}'/'config.yaml'


def signals(study):
    prepare(study); results={}
    for seed in study['batch_seeds']:
        config,_=load_config(batch_path(study,seed)); folder=Path(config['output_dir'])
        for shard in range(3):
            worker=json.loads((folder/f'collect_B0_shard{shard}_complete.json').read_text())
            request=json.loads((folder/f'collect_B0_shard{shard}_request.json').read_text())
            if worker['before']!=worker['after'] or request['config']!=config:
                raise ValueError('Collection conditions changed')
        result=training_signals(config,{'rounds':[{'round':0,'output_dir':config['output_dir']}]})
        result['imitation']={g['root_id']:success_imitation_targets(g,gate) for g,gate,_,_ in load_training_groups(config)}
        result['ready']=(result['registered_oracle_backed_plan_value_difference']>0
                         and all(v['success_mass']>0 for v in result['imitation'].values()))
        results[str(seed)]=result
    result={'status':'COMPLETE','ready':all(r['ready'] for r in results.values()),'batches':results}
    guarded_json(Path(study['output_dir'])/'training_signals.json',result)
    print(json.dumps(result),flush=True)
    return result


def evaluate(study,name,seed):
    prepare(study)
    if seed not in study['evaluation_seeds']: raise ValueError('Unregistered evaluation seed')
    config,framework=load_config(batch_path(study,study['batch_seeds'][0]))
    settings={**study,'output_dir':str(Path(study['output_dir'])/f'evaluation_seed_{seed}'),'evaluation_seed':seed}
    # Evaluate only a finished fixed-budget checkpoint; other branches may
    # still be training. Evaluation outcomes never change the update schedule.
    if name=='B0':
        selected={'adapter':config['initial_adapter'],
                  'adapter_sha256':file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')}
    else:
        method,batch=name.rsplit('_',1)
        if method not in METHODS: raise ValueError('Unknown method')
        update=json.loads((batch_path(study,int(batch)).parent/method/'training_report.json').read_text())
        if (update['status']!='COMPLETE' or update['optimizer_steps']!=study['optimizer_steps']
                or update['previous_optimizer_steps']!=0 or update['frozen_before']!=update['frozen_after']
                or update['adapter_sha256']!=file_hash(Path(update['adapter'])/'adapter_model.safetensors')):
            raise ValueError('Unfinished or changed checkpoint')
        selected={k:update[k] for k in ('adapter','adapter_sha256')}
    return evaluate_checkpoint(settings,name,selected,config,framework)


def repeated_benefit(paired):
    return len(paired)==4 and all(p['final_success']['wins']>p['final_success']['losses'] for p in paired.values())


def report(study):
    prepare(study); output=Path(study['output_dir']); selected=models(study)
    evaluations={}; paired={}; roots=set(study['root_ids'])
    for seed in study['evaluation_seeds']:
        current={m:json.loads((output/f'evaluation_seed_{seed}'/'evaluation'/m/'result.json').read_text()) for m in selected}
        expected={(rid,seed+i*1000+j) for i,rid in enumerate(sorted(roots)) for j in range(study['natural_repetitions'])}
        for e in current.values():
            if (e['status']!='COMPLETE' or e['frozen_before']!=e['frozen_after']
                    or e['optimizer_updates_from_evaluation']!=0 or len(e['records'])!=48
                    or {(r['root_id'],r['seed']) for r in e['records']}!=expected):
                raise ValueError('Incomplete or unpaired evaluation')
        for batch in study['batch_seeds']:
            a,b=[current[f'{m}_{batch}']['records'] for m in METHODS]
            paired[f'{batch}/{seed}']={'final_success':paired_rows(a,b,roots),
                'first_plan_success':paired_rows([{**r,'success':r['first_plan_success']} for r in a],
                                                 [{**r,'success':r['first_plan_success']} for r in b],roots),
                'by_root':{rid:paired_rows(a,b,{rid}) for rid in sorted(roots)}}
        evaluations[str(seed)]=current
    updates={f'{m}_{seed}':json.loads((batch_path(study,seed).parent/m/'training_report.json').read_text())
             for seed in study['batch_seeds'] for m in METHODS}
    for seed in study['batch_seeds']:
        a,b=[json.loads((batch_path(study,seed).parent/m/'request.json').read_text()) for m in METHODS]
        if a['data_sha256']!=b['data_sha256'] or a['config']!=b['config']: raise ValueError('Not shared training data')
    result={'status':'COMPLETE','scope':'TRAIN_LEARNABILITY_WITH_FROZEN_PARENT_SUMMARY',
        'training_signals':json.loads((output/'training_signals.json').read_text()),
        'fresh_training_continuations':sum(u['candidate_continuations']+u['independent_baseline_continuations']
                                         for m,u in updates.items() if m.startswith('B3_')),
        'optimizer_steps':sum(u['optimizer_steps'] for u in updates.values()),
        'natural_evaluations':sum(e['metrics']['runs'] for group in evaluations.values() for e in group.values()),
        'updates':updates,'evaluations':evaluations,'paired_B3_vs_SUCCESS_SFT':paired,
        'supports_task_expansion':repeated_benefit(paired)}
    if (result['fresh_training_continuations'],result['optimizer_steps'],result['natural_evaluations'])!=(240,80,480):
        raise ValueError('Fixed budget incomplete')
    write_json(output/'result.json',result)
    print(json.dumps({'paired':paired,'supports_task_expansion':result['supports_task_expansion']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','signals','train','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_summary_training.yaml')
    parser.add_argument('--batch',type=int); parser.add_argument('--evaluation-seed',type=int)
    parser.add_argument('--model'); parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args(); study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare': prepare(study)
    elif args.command=='signals': signals(study)
    elif args.command=='train':
        prepare(study)
        if not json.loads((Path(study['output_dir'])/'training_signals.json').read_text())['ready']:
            raise ValueError('No usable allocation signal; do not update')
        config,framework=load_config(batch_path(study,args.batch)); train(config,framework,args.method)
    elif args.command=='evaluate': evaluate(study,args.model,args.evaluation_seed)
    else: report(study)


if __name__=='__main__': main()
