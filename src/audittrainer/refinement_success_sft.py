"""Small shared-feedback comparison of successful-plan SFT and B3."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_success_sft.yaml'
    _resources = Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json'
    _gpu_reservation = acquire(_resources, 'collect') if _resources.exists() else None

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum import plans, prepare as prepare_calibration, repeated_contrasts
from .refinement_curriculum_rl import conditional_probe, plan_key
from .refinement_generalization import guarded_json
from .refinement_multiround import training_signals
from .refinement_policy import adapter_fingerprint, load_policies
from .refinement_pr1 import load_config, load_training_groups, success_imitation_targets
from .refinement_rollout import case_summary, costs, run_case
from .verifier import canonical

METHODS = ('SUCCESS_SFT', 'B3')


def batch_config(study, seed):
    if seed not in study['batch_seeds']:
        raise ValueError('Unregistered collection seed')
    base, _ = load_config(study['base_config'])
    config = deepcopy(base)
    for key in ('fixed_control_dir', 'fixed_control_config', 'transfers', 'evaluation',
                'optimizer_state', 'round_root_ids', 'sampling_rounds', 'reference_adapter'):
        config.pop(key, None)
    output = Path(study['output_dir'])
    config.update(output_dir=str(output/f'batch_{seed}'), seed=seed, root_limit=len(study['root_ids']),
                  root_manifest=str(output/'data/roots.jsonl'), test_manifest=str(output/'data/tests.jsonl'),
                  independent_baseline_repetitions=study['baseline_repetitions'], baseline_seed_stride=8,
                  retention_offset=0)
    config['candidate_sampling'].update(split_bodies=study['split_candidates'],
        continuations_per_candidate=study['continuations_per_candidate'],
        candidate_seed_stride=32, continuation_seed_stride=8)
    config['update'].update(epochs=study['optimizer_steps'], gradient_accumulation_groups=len(study['root_ids']),
                            mode_reward_weight=1.0)
    return config


def prepare(study):
    output = Path(study['output_dir']); output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(study['calibration_source']).read_text())
    reference = {r['root_id']: r for r in source['reference']}
    selected = {}
    for rid in study['root_ids']:
        row = reference[rid]
        values = {name: sum(v['returns'])/len(v['returns']) for name,v in row['variants'].items()}
        best = {name for name, value in values.items() if value == max(values.values())}
        if best != set(study['preferred_variants'][rid]) or not repeated_contrasts(row['variants']):
            raise ValueError('Selection must match prior repeated TRAIN contrasts')
        selected[rid] = {'returns': {k:v['returns'] for k,v in row['variants'].items()},
                         'preferred_variants': sorted(best), 'repeated_contrasts': repeated_contrasts(row['variants'])}
    base, framework = load_config(study['base_config'])
    roots = load_root_manifest(base['root_manifest'])
    tests = {r['root_id']:r for r in read_jsonl(base['test_manifest'])}
    if any(roots[rid]['split'] != 'TRAIN' for rid in study['root_ids']):
        raise ValueError('This is a TRAIN learnability experiment')
    data = output/'data'; data.mkdir(exist_ok=True)
    for name, values in [('roots', [roots[r] for r in sorted(study['root_ids'])]),
                         ('tests', [tests[r] for r in sorted(study['root_ids'])])]:
        path = data/f'{name}.jsonl'; content = ''.join(canonical(r)+'\n' for r in values)
        if path.exists() and path.read_text() != content:
            raise ValueError('Changed selected TRAIN manifest')
        path.write_text(content)
    for seed in study['batch_seeds']:
        config = batch_config(study, seed); folder = Path(config['output_dir']); folder.mkdir(exist_ok=True)
        path = folder/'config.yaml'
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError('Changed batch configuration')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    calibration = batch_config(study, study['batch_seeds'][0])
    calibration.update(seed=study['calibration_seed'], output_dir=str(output/'calibration'),
        source_root_manifest=base['root_manifest'], source_test_manifest=base['test_manifest'],
        root_ids=study['root_ids'], reference_repetitions=study['calibration_repetitions'],
        collect_policy_samples=False, selection_reason='Three preselected TRAIN states with repeated authored allocation contrasts. Fresh replication is retained in full and never optimized.')
    calibration['update']['enabled'] = False
    prepare_calibration(calibration, framework)
    path = output/'calibration/config.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != calibration:
        raise ValueError('Changed calibration configuration')
    path.write_text(yaml.safe_dump(calibration, sort_keys=False))
    request = {'study': study, 'prior_calibration_sha256': file_hash(study['calibration_source']),
        'selected_calibration': selected, 'manifests': {str(p): file_hash(p) for p in (data/'roots.jsonl', data/'tests.jsonl')},
        'initial_adapter_sha256': file_hash(Path(base['initial_adapter'])/'adapter_model.safetensors'),
        'executor_adapter_sha256': file_hash(Path(base['executor_adapter'])/'adapter_model.safetensors'),
        'planned': {'fresh_calibration': 48, 'shared_training_continuations': 240,
                    'optimizer_updates': 80, 'natural_evaluations': 240, 'conditional_plan_probes': 360}}
    guarded_json(output/'request.json', request)
    guarded_json(output/'gpu_resources.json', {'devices': ['0','1','2'], 'min_free_mib': {'collect':16000, 'train':22000}})
    return request


def models(study):
    base, _ = load_config(study['base_config'])
    result = {'B0': {'adapter': base['initial_adapter'],
                     'adapter_sha256': file_hash(Path(base['initial_adapter'])/'adapter_model.safetensors')}}
    for seed in study['batch_seeds']:
        for method in METHODS:
            path = Path(batch_config(study, seed)['output_dir'])/method/'training_report.json'
            report = json.loads(path.read_text())
            if (report['status'] != 'COMPLETE' or report['optimizer_steps'] != study['optimizer_steps']
                    or report['previous_optimizer_steps'] != 0 or report['frozen_before'] != report['frozen_after']):
                raise ValueError('Incomplete or nonfresh training branch')
            if report['adapter_sha256'] != file_hash(Path(report['adapter'])/'adapter_model.safetensors'):
                raise ValueError('Changed final adapter')
            result[f'{method}_{seed}'] = {k:report[k] for k in ('adapter','adapter_sha256')}
    guarded_json(Path(study['output_dir'])/'evaluation_checkpoints.json', result)
    return result


def natural_seed(study, root_index, repetition):
    return study['evaluation_seed']+1000000+root_index*1000+repetition


def evaluate(study, model_name):
    selected = models(study)[model_name]
    config, framework = load_config(Path(batch_config(study, study['batch_seeds'][0])['output_dir'])/'config.yaml')
    output = Path(study['output_dir'])/'evaluation'/model_name; output.mkdir(parents=True, exist_ok=True)
    guarded_json(output/'request.json', {'study': study, 'model':model_name, **selected})
    if (output/'result.json').exists():
        return json.loads((output/'result.json').read_text())
    planner, executor = load_policies(framework, selected['adapter'], config['executor_adapter'])
    before = {n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}
    rows = load_root_manifest(config['root_manifest'])
    tests = {r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    probes = {}; cases = []
    for index, (rid,row) in enumerate(sorted(rows.items())):
        probe_config = deepcopy(config)
        probe_config['evaluation'] = {'seed': study['evaluation_seed']+index*1000,
            'focus_root':rid, 'conditional_split_samples':study['conditional_split_samples']}
        folder = output/'probes'/rid; folder.mkdir(parents=True, exist_ok=True)
        probe = conditional_probe(probe_config, framework, planner, executor, folder, rows, tests)
        gate = json.loads((folder/'conditional_probe/gate.json').read_text())
        preferred = study['preferred_variants'][rid]
        count = sum(r['reviewed_variant'] in preferred and r['registered'] for r in probe['records'])
        p = gate['mode_probabilities']['SPLIT']
        probes[rid] = {**probe, 'p_split':p, 'preferred_conditional_draws':count,
            'estimated_preferred_initial_mass': (1-p)*('atomic' in preferred)+p*count/probe['samples']}
        for repetition in range(study['natural_repetitions']):
            cases.append(run_case(row, tests, framework, config, planner, executor,
                output/'natural'/rid/f'repeat_{repetition:02d}', natural_seed(study,index,repetition)))
    after = {n:adapter_fingerprint(planner.model,n) for n in before}
    if before != after or any(c['status'] != 'COMPLETE' for c in cases):
        raise ValueError('Incomplete evaluation or changed weights')
    result = {'status':'COMPLETE', 'model':model_name, **selected,
        'natural':case_summary(cases), 'by_root':{rid:case_summary([c for c in cases if c['root_id']==rid]) for rid in rows},
        'probes':probes, 'frozen_before':before, 'frozen_after':after,
        'optimizer_updates_from_evaluation':0, 'actual_model_cost':dict(costs(planner,executor))}
    write_json(output/'result.json', result)
    print(json.dumps({'model':model_name, 'passes':result['natural']['final_successes']}), flush=True)
    return result


def outcome_records(study, model):
    rows = load_root_manifest(Path(study['output_dir'])/'data/roots.jsonl')
    definitions = {rid:json.loads(Path(row['fixture_path']).read_text()) for rid,row in rows.items()}
    result = []
    for path in sorted((Path(study['output_dir'])/'evaluation'/model/'natural').glob('*/repeat_*/case_result.json')):
        case = json.loads(path.read_text()); rid = case['root_id']
        events = [e for e in read_jsonl(path.parent/'runtime/trace.jsonl') if e['node_id']=='N0']
        accepted = next((e for e in events if e['host_result']['code']=='PLAN_REGISTERED'), None)
        action = accepted['parsed_action'] if accepted else None
        variants = {plan_key(a,definitions[rid]):name for name,a in plans(definitions[rid]).items()}
        variant = variants.get(plan_key(action,definitions[rid])) if action else None
        result.append({'root_id':rid, 'seed':case['seed'], 'success':case['reward']['y'],
            'registered_mode':action['args']['mode'] if action else None, 'reviewed_variant':variant,
            'root_first_plan_registered':case['root_first_plan_registered'],
            'initial_preferred':bool(case['root_first_plan_registered'] and variant in study['preferred_variants'][rid]),
            'stop_reason':case['stop_reason'], 'run_dir':case['run_dir']})
    expected = {(rid,natural_seed(study,i,j)) for i,rid in enumerate(sorted(rows)) for j in range(study['natural_repetitions'])}
    if len(result)!=len(expected) or {(r['root_id'],r['seed']) for r in result}!=expected:
        raise ValueError('Missing or duplicated evaluation slots')
    return result


def report(study):
    selected = models(study); output = Path(study['output_dir'])
    evaluations = {m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in selected}
    records = {m:outcome_records(study,m) for m in selected}
    updates = {}; data = {}; pairs = {}
    for seed in study['batch_seeds']:
        config = batch_config(study,seed); folder=Path(config['output_dir'])
        groups = load_training_groups(config)
        data[str(seed)] = {'continuations':sum(g['continuations'] for g,_,_,_ in groups)+len(groups)*study['baseline_repetitions'],
            'success_imitation_targets':{g['root_id']:success_imitation_targets(g,gate) for g,gate,_,_ in groups},
            'signals':training_signals(config, {'rounds':[{'round':1,'output_dir':str(folder)}]})}
        hashes=[]
        for method in METHODS:
            name=f'{method}_{seed}'
            updates[name]=json.loads((folder/method/'training_report.json').read_text())
            hashes.append(json.loads((folder/method/'request.json').read_text())['data_sha256'])
        if hashes[0]!=hashes[1]:
            raise ValueError('The paired methods did not share execution feedback')
        sft,b3=[f'{m}_{seed}' for m in METHODS]
        roots=set(study['root_ids'])
        pairs[str(seed)]={'SFT_vs_B0':paired_rows(records['B0'],records[sft],roots),
                         'B3_vs_B0':paired_rows(records['B0'],records[b3],roots),
                         'B3_vs_SFT':paired_rows(records[sft],records[b3],roots)}
    calibration = {rid:json.loads((output/'calibration/reference'/rid/'result.json').read_text()) for rid in study['root_ids']}
    if any(e['status']!='COMPLETE' or e['frozen_before']!=e['frozen_after'] or e['natural']['runs']!=48 for e in evaluations.values()):
        raise ValueError('Incomplete or unfrozen evaluation')
    result={'status':'COMPLETE','scope':'TWO_INDEPENDENT_SHARED_FEEDBACK_BATCHES_ON_THREE_SELECTED_TRAIN_STATES',
        'calibration':calibration, 'data':data, 'updates':updates, 'evaluations':evaluations,
        'outcomes':records, 'paired_by_batch':pairs,
        'mean_preferred_initial_mass':{m:sum(p['estimated_preferred_initial_mass'] for p in e['probes'].values())/3 for m,e in evaluations.items()},
        'root_modes':{m:dict(Counter(r['registered_mode'] or 'UNREGISTERED' for r in rows)) for m,rows in records.items()},
        'new_calibration_continuations':sum(v['summary']['runs'] for row in calibration.values() for v in row['variants'].values()),
        'new_shared_training_continuations':sum(x['continuations'] for x in data.values()),
        'new_optimizer_steps':sum(u['optimizer_steps'] for u in updates.values()),
        'natural_evaluations':sum(e['natural']['runs'] for e in evaluations.values()),
        'conditional_plan_probes':sum(p['samples'] for e in evaluations.values() for p in e['probes'].values()),
        'data_use':'Selected TRAIN states only. Separate fresh authored calibration is not optimized. Each independent raw batch feeds both methods, including failed and duplicate candidates; only registered successful continuations contribute imitation targets. Natural evaluation and plan probes use fresh paired seeds and never update weights. Shared B0 is evaluated once. This is learnability, not held-out generalization.'}
    write_json(output/'result.json',result)
    print(json.dumps({'passes':{m:e['natural']['final_successes'] for m,e in evaluations.items()},
                      'preferred_mass':result['mean_preferred_initial_mass'],'paired':pairs}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_success_sft.yaml')
    parser.add_argument('--model',default='B0')
    args=parser.parse_args(); study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':
        print(json.dumps(prepare(study)),flush=True)
    elif args.command=='evaluate':
        evaluate(study,args.model)
    else:
        report(study)


if __name__=='__main__':
    main()
