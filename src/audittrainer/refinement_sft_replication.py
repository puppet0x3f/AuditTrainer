"""Complete the SFT controls on the existing first-credit evaluation seeds."""
import argparse
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_sft_replication.yaml'
    _resources = Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json'
    _gpu_reservation = acquire(_resources, 'collect')

from .dataset import file_hash, load_root_manifest
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_first_credit import evaluate_checkpoint, evaluation_seed, training_config
from .refinement_generalization import guarded_json
from .refinement_mode_weight import link
from .refinement_pr1 import load_config


def evaluation_setup(settings):
    original = yaml.safe_load(Path(settings['source_study']).read_text())
    evaluation = {**original, 'output_dir':settings['output_dir']}
    path = Path(training_config(original,original['batch_seeds'][0])['output_dir'])/'config.yaml'
    config, framework = load_config(path)
    return original, evaluation, config, framework


def models(settings):
    original,_,_,_ = evaluation_setup(settings)
    study = yaml.safe_load(Path(original['source_study']).read_text())
    selected = json.loads((Path(study['output_dir'])/'evaluation_checkpoints.json').read_text())
    expected = {'B0', *[f'{method}_{seed}' for seed in original['batch_seeds'] for method in ('SUCCESS_SFT','B3')]}
    if set(selected) != expected:
        raise ValueError('Expected the existing common SFT, two success SFT and two original B3 checkpoints')
    for row in selected.values():
        if file_hash(Path(row['adapter'])/'adapter_model.safetensors') != row['adapter_sha256']:
            raise ValueError('Changed comparison checkpoint')
    return selected


def prepare(settings):
    original,evaluation,config,_ = evaluation_setup(settings)
    output = Path(settings['output_dir']); output.mkdir(parents=True,exist_ok=True)
    source = Path(original['output_dir'])
    previous = json.loads((source/'result.json').read_text())
    if previous['status']!='COMPLETE' or json.loads((source/'request.json').read_text())['settings']!=original:
        raise ValueError('Expected the completed unchanged first-credit evaluation')
    checkpoints = models(settings); reused = {}
    for seed in original['batch_seeds']:
        name = f'B3_{seed}'; path = source/'evaluation'/name
        result = json.loads((path/'result.json').read_text())
        if result['adapter_sha256']!=checkpoints[name]['adapter_sha256']:
            raise ValueError('Reused B3 evaluation has a different checkpoint')
        link(path,output/'evaluation'/name)
        reused[name] = file_hash(path/'result.json')
    guarded_json(output/'evaluation_checkpoints.json',checkpoints)
    guarded_json(output/'request.json',{'settings':settings,'evaluation_settings':evaluation,
        'execution_config':config,'reused_evaluation_sha256':reused,
        'new_models':['B0',*[f'SUCCESS_SFT_{seed}' for seed in original['batch_seeds']]],
        'planned':{'new_training_continuations':0,'new_optimizer_steps':0,'new_natural_evaluations':144,'reused_natural_evaluations':96},
        'decision':'Continue original B3 research only if its final success count exceeds the paired success-SFT model in BOTH batches. First-accepted success and retries explain outcomes; they do not replace the primary criterion.'})
    guarded_json(output/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    return checkpoints


def evaluate(settings, name):
    _,evaluation,config,framework = evaluation_setup(settings)
    selected = models(settings)
    if name not in selected or name.startswith('B3_'):
        raise ValueError('Only the three missing SFT controls may be newly evaluated')
    return evaluate_checkpoint(evaluation,name,selected[name],config,framework)


def compare(left, right, roots):
    if len(left)!=len(right) or {(r['root_id'],r['seed']) for r in left}!={(r['root_id'],r['seed']) for r in right}:
        raise ValueError('Unpaired evaluation records')
    return {'final_success':paired_rows(left,right,roots),
        'first_plan_success':paired_rows([{**r,'success':r['first_plan_success']} for r in left],
                                        [{**r,'success':r['first_plan_success']} for r in right],roots)}


def continued_support(paired):
    return all(p['B3_vs_SUCCESS_SFT']['final_success']['right_passes']>
               p['B3_vs_SUCCESS_SFT']['final_success']['left_passes'] for p in paired.values())


def report(settings):
    original,evaluation,config,_ = evaluation_setup(settings)
    output = Path(settings['output_dir']); selected = models(settings)
    request = json.loads((output/'request.json').read_text())
    if request['execution_config']!=config or request['evaluation_settings']!=evaluation:
        raise ValueError('Changed execution conditions')
    evaluations = {m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in selected}
    roots = set(load_root_manifest(config['root_manifest']))
    expected = {(rid,evaluation_seed(evaluation,i,j)) for i,rid in enumerate(sorted(roots)) for j in range(evaluation['natural_repetitions'])}
    for name,e in evaluations.items():
        if (e['status']!='COMPLETE' or e['frozen_before']!=e['frozen_after'] or e['optimizer_updates_from_evaluation']!=0
                or e['adapter_sha256']!=selected[name]['adapter_sha256'] or e['natural']['statuses']!={'COMPLETE':len(expected)}
                or len(e['records'])!=len(expected) or {(r['root_id'],r['seed']) for r in e['records']}!=expected):
            raise ValueError('Unfinished, changed or unpaired evaluation')
    if len({e['frozen_before']['executor'] for e in evaluations.values()})!=1:
        raise ValueError('Different frozen executors')
    for name,sha in request['reused_evaluation_sha256'].items():
        if file_hash(output/'evaluation'/name/'result.json')!=sha:
            raise ValueError('Changed reused B3 results')
    paired = {}
    for seed in original['batch_seeds']:
        b0,sft,b3 = [evaluations[m]['records'] for m in ('B0',f'SUCCESS_SFT_{seed}',f'B3_{seed}')]
        paired[str(seed)] = {'B3_vs_SUCCESS_SFT':compare(sft,b3,roots),
            'B3_vs_B0':compare(b0,b3,roots),'SUCCESS_SFT_vs_B0':compare(b0,sft,roots)}
    result = {'status':'COMPLETE','scope':'SFT_CONTROLS_ON_THE_FIRST_CREDIT_EVALUATION_SEEDS',
        'evaluations':evaluations,'paired_by_batch':paired,'supports_continued_b3':continued_support(paired),
        'new_training_continuations':0,'new_optimizer_steps':0,
        'new_natural_evaluations':sum(evaluations[m]['metrics']['runs'] for m in request['new_models']),
        'reused_natural_evaluations':sum(evaluations[m]['metrics']['runs'] for m in request['reused_evaluation_sha256']),
        'data_use':'Three missing SFT controls evaluated once. Same TRAIN roots, seeds, execution budgets, reward and frozen executor as the reused original B3 evaluations. No training or reward change. This tests replication across execution seeds, not held-out-task generalization.'}
    write_json(output/'result.json',result)
    print(json.dumps({'metrics':{m:e['metrics'] for m,e in evaluations.items()},'paired':paired,
                      'supports_continued_b3':result['supports_continued_b3']}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','report'))
    parser.add_argument('--config',default='configs/refinement_sft_replication.yaml')
    parser.add_argument('--model',default='B0')
    args = parser.parse_args(); settings = yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare': print(json.dumps(prepare(settings)),flush=True)
    elif args.command=='evaluate': evaluate(settings,args.model)
    else: report(settings)


if __name__ == '__main__':
    main()
