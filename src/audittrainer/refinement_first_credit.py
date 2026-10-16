"""Matched B3 ablation: give rejected initial plans zero execution credit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_first_credit.yaml'
    _resources = Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json'
    _gpu_reservation = acquire(_resources, 'collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_generalization import guarded_json
from .refinement_mode_weight import link
from .refinement_policy import adapter_fingerprint, load_policies
from .refinement_pr1 import load_config, load_training_groups
from .refinement_rollout import case_summary, costs, run_case


def source_study(settings):
    return yaml.safe_load(Path(settings['source_study']).read_text())


def training_config(settings, seed):
    if seed not in settings['batch_seeds']:
        raise ValueError('Unregistered training batch')
    source = Path(source_study(settings)['output_dir'])/f'batch_{seed}'
    original, _ = load_config(source/'config.yaml')
    config = deepcopy(original)
    config.update(output_dir=str(Path(settings['output_dir'])/f'batch_{seed}'), reward_credit='first_registered')
    if original['update']['epochs'] != settings['optimizer_steps'] or 'optimizer_state' in original:
        raise ValueError('Expected the original fresh 20-step comparison')
    return config


def prepare(settings):
    source = source_study(settings); output = Path(settings['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((Path(source['output_dir'])/'result.json').read_text())
    if previous['status'] != 'COMPLETE' or settings['batch_seeds'] != source['batch_seeds']:
        raise ValueError('Expected the two complete shared-feedback batches')
    relabeling = {}
    for seed in settings['batch_seeds']:
        original = Path(source['output_dir'])/f'batch_{seed}'
        config = training_config(settings, seed); folder = Path(config['output_dir'])
        folder.mkdir(exist_ok=True)
        path = folder/'config.yaml'
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError('Changed first-plan credit experiment')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        for name in ('groups', 'independent_baseline'):
            link(original/name, folder/name)
        for request in original.glob('collect_B3_shard*_request.json'):
            link(request, folder/request.name)
        relabeling[str(seed)] = {g['root_id']:g['credit_assignment'] for g,_,_,_ in load_training_groups(config)}
    guarded_json(output/'credit_relabeling.json', relabeling)
    guarded_json(output/'request.json', {'settings':settings,
        'source_result_sha256':file_hash(Path(source['output_dir'])/'result.json'),
        'original_checkpoints':{str(seed):previous['updates'][f'B3_{seed}']['adapter_sha256'] for seed in settings['batch_seeds']},
        'planned':{'new_training_continuations':0,'reused_training_continuations':240,
                   'new_optimizer_steps':40,'natural_evaluations':192},
        'credit_rule':'Accepted initial plan: original full return. Rejected initial plan: zero. Apply identically to candidates and independent baselines; retain every sample.',
        'decision':'Adopt only if BOTH batches increase first-accepted-and-finally-successful runs and neither batch loses overall successes. Root planning retries are diagnostic.'})
    guarded_json(output/'gpu_resources.json', {'devices':['0','1','2'], 'min_free_mib':{'collect':16000,'train':22000}})
    return relabeling


def models(settings):
    source = Path(source_study(settings)['output_dir']); result = {}
    for seed in settings['batch_seeds']:
        reports = {}
        for prefix, folder in [('B3',source/f'batch_{seed}'), ('B3_FIRST',Path(training_config(settings,seed)['output_dir']))]:
            report = json.loads((folder/'B3/training_report.json').read_text())
            if (report['status'] != 'COMPLETE' or report['optimizer_steps'] != settings['optimizer_steps']
                    or report['previous_optimizer_steps'] != 0 or report['frozen_before'] != report['frozen_after']
                    or report['reload_max_abs_difference'] != 0):
                raise ValueError('Incomplete or unmatched checkpoint')
            if report.get('reward_credit','full_return') != ('first_registered' if prefix=='B3_FIRST' else 'full_return'):
                raise ValueError('Wrong checkpoint credit rule')
            if file_hash(Path(report['adapter'])/'adapter_model.safetensors') != report['adapter_sha256']:
                raise ValueError('Changed checkpoint')
            result[f'{prefix}_{seed}'] = {k:report[k] for k in ('adapter','adapter_sha256')}
            reports[prefix] = report
        if any(reports['B3'][k] != reports['B3_FIRST'][k] for k in ('behavior_sha256','reference_sha256')):
            raise ValueError('Different initial/reference policies')
    guarded_json(Path(settings['output_dir'])/'evaluation_checkpoints.json', result)
    return result


def evaluation_seed(settings, root_index, repetition):
    return settings['evaluation_seed']+root_index*1000+repetition


def outcome_records(cases):
    records = []
    for case in cases:
        events = read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
        attempts = [e for e in events if e['event_kind']=='POLICY' and e['node_id']=='N0'
                    and e['observation_before']['phase']=='NEW']
        if not attempts:
            raise ValueError('Missing initial root planning event')
        success = case['final_scores']['private']['status']=='PASS'
        registered = case['root_first_plan_registered']
        records.append({'root_id':case['root_id'],'seed':case['seed'],'success':int(success),
            'first_plan_success':int(registered and success),'first_plan_registered':registered,
            'root_planning_retries':len(attempts)-1,'run_dir':case['run_dir']})
    return records


def metrics(records):
    return {'runs':len(records),'final_successes':sum(r['success'] for r in records),
        'first_plan_successes':sum(r['first_plan_success'] for r in records),
        'first_plan_registered':sum(r['first_plan_registered'] for r in records),
        'root_planning_retries':sum(r['root_planning_retries'] for r in records),
        'runs_with_root_retries':sum(r['root_planning_retries']>0 for r in records),
        'successes_after_initial_rejection':sum(r['success'] and not r['first_plan_registered'] for r in records)}


def evaluate(settings, name):
    selected = models(settings)[name]
    config, framework = load_config(Path(training_config(settings,settings['batch_seeds'][0])['output_dir'])/'config.yaml')
    return evaluate_checkpoint(settings,name,selected,config,framework)


def evaluate_checkpoint(settings, name, selected, config, framework):
    """Run the same natural evaluation for an explicitly selected checkpoint."""
    output = Path(settings['output_dir'])/'evaluation'/name; output.mkdir(parents=True,exist_ok=True)
    guarded_json(output/'request.json', {'settings':settings,'model':name,**selected})
    if (output/'result.json').exists():
        return json.loads((output/'result.json').read_text())
    planner, executor = load_policies(framework,selected['adapter'],config['executor_adapter'])
    before = {n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}
    rows = load_root_manifest(config['root_manifest'])
    tests = {r['root_id']:r for r in read_jsonl(config['test_manifest'])}; cases = []
    for index,(rid,row) in enumerate(sorted(rows.items())):
        for rep in range(settings['natural_repetitions']):
            cases.append(run_case(row,tests,framework,config,planner,executor,
                output/'natural'/rid/f'repeat_{rep:02d}',evaluation_seed(settings,index,rep)))
    after = {n:adapter_fingerprint(planner.model,n) for n in before}
    if before != after or any(c['status']!='COMPLETE' for c in cases):
        raise ValueError('Unfinished evaluation or changed weights')
    records = outcome_records(cases)
    result = {'status':'COMPLETE','model':name,**selected,'natural':case_summary(cases),
        'metrics':metrics(records),'by_root':{rid:metrics([r for r in records if r['root_id']==rid]) for rid in rows},
        'records':records,'frozen_before':before,'frozen_after':after,
        'optimizer_updates_from_evaluation':0,'actual_model_cost':dict(costs(planner,executor))}
    write_json(output/'result.json',result)
    print(json.dumps({'model':name,**result['metrics']}),flush=True)
    return result


def adoption(paired):
    return all(p['first_plan_success']['right_passes']>p['first_plan_success']['left_passes']
               and p['final_success']['right_passes']>=p['final_success']['left_passes'] for p in paired.values())


def report(settings):
    selected = models(settings); output = Path(settings['output_dir']); source = source_study(settings)
    evaluations = {m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in selected}
    roots = set(source['root_ids']); expected = {(rid,evaluation_seed(settings,i,j))
        for i,rid in enumerate(sorted(roots)) for j in range(settings['natural_repetitions'])}
    for e in evaluations.values():
        if (e['status']!='COMPLETE' or e['frozen_before']!=e['frozen_after']
                or e['optimizer_updates_from_evaluation']!=0 or e['metrics']['runs']!=len(expected)
                or {(r['root_id'],r['seed']) for r in e['records']}!=expected):
            raise ValueError('Unpaired or incomplete evaluation')
    updates = {}; paired = {}
    for seed in settings['batch_seeds']:
        original = Path(source['output_dir'])/f'batch_{seed}'
        current = Path(training_config(settings,seed)['output_dir'])
        left = json.loads((original/'B3/request.json').read_text()); right = json.loads((current/'B3/request.json').read_text())
        if left['data_sha256']!=right['data_sha256']:
            raise ValueError('Changed execution data')
        if {k:v for k,v in right['config'].items() if k not in ('output_dir','reward_credit')} != {k:v for k,v in left['config'].items() if k!='output_dir'}:
            raise ValueError('Training changed more than credit assignment')
        u = json.loads((current/'B3/training_report.json').read_text())
        old = json.loads((original/'B3/training_report.json').read_text())
        if [(x['epoch'],x['root_id']) for x in u['history']] != [(x['epoch'],x['root_id']) for x in old['history']]:
            raise ValueError('Changed accumulation/retention sequence')
        updates[str(seed)] = u
        a,b = [evaluations[f'{m}_{seed}']['records'] for m in ('B3','B3_FIRST')]
        paired[str(seed)] = {'final_success':paired_rows(a,b,roots),
            'first_plan_success':paired_rows([{**r,'success':r['first_plan_success']} for r in a],
                                            [{**r,'success':r['first_plan_success']} for r in b],roots)}
    result = {'status':'COMPLETE','scope':'FIRST_REGISTERED_CREDIT_ON_TWO_EXISTING_TRAIN_BATCHES',
        'updates':updates,'evaluations':evaluations,'paired_by_batch':paired,'adopt_first_plan_credit':adoption(paired),
        'credit_relabeling':json.loads((output/'credit_relabeling.json').read_text()),
        'new_training_continuations':0,'reused_training_continuations':240,
        'new_optimizer_steps':sum(u['optimizer_steps'] for u in updates.values()),
        'natural_evaluations':sum(e['metrics']['runs'] for e in evaluations.values()),
        'data_use':'Same saved TRAIN candidates and independent baselines; only return assignment changes. Original B3 checkpoints reused; all four natural evaluations use new paired seeds. Evaluation never updates weights. No new-task or recursive-training claim.'}
    write_json(output/'result.json',result)
    print(json.dumps({'paired':paired,'adopt':result['adopt_first_plan_credit']}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','report'))
    parser.add_argument('--config',default='configs/refinement_first_credit.yaml')
    parser.add_argument('--model',default='B3_91000713')
    args = parser.parse_args(); settings = yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare': print(json.dumps(prepare(settings)),flush=True)
    elif args.command=='evaluate': evaluate(settings,args.model)
    else: report(settings)


if __name__ == '__main__':
    main()
