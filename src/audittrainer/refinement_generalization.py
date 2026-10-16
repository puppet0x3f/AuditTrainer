"""Three fresh training seeds, equal rollout/update budgets, unseen compositions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
from statistics import mean, stdev
import subprocess
import sys

import yaml

# Natural evaluation needs the same GPU capacity as collection. Reserve before
# importing model libraries, and keep the reservation until the process exits.
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _config_path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_generalization.yaml'
    _resources = Path(yaml.safe_load(Path(_config_path).read_text())['output_dir'])/'gpu_resources.json'
    _gpu_reservation = acquire(_resources, 'collect') if _resources.exists() else None

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum import plans
from .refinement_curriculum_rl import plan_key
from .refinement_heldout_tasks import build
from .refinement_multiround import round_config, run, training_signals
from .refinement_policy import adapter_fingerprint, load_policies
from .refinement_pr1 import load_config
from .refinement_rollout import case_summary, costs, run_case, split_candidate_count

ARMS = ('original', 'optimized')


def arm_config(config, arm, seed):
    if arm not in ARMS or seed not in config['training_seeds']:
        raise ValueError('Unknown arm/training seed')
    base, _ = load_config(config['base_config'])
    result = deepcopy(base)
    for key in ('fixed_control_dir', 'fixed_control_config', 'transfers', 'evaluation', 'optimizer_state'):
        result.pop(key, None)
    result.update(output_dir=str(Path(config['output_dir'])/f'seed_{seed}'/arm), seed=seed,
                  sampling_rounds=config['sampling_rounds'], independent_baseline_repetitions=1 if arm == 'original' else 4,
                  baseline_seed_stride=8, retention_offset=0)
    result['update'].update(epochs=config['epochs_per_round'], mode_reward_weight=1.0)
    result['candidate_sampling'].update(continuations_per_candidate=2 if arm == 'original' else 4,
                                        candidate_seed_stride=32, continuation_seed_stride=8)
    if arm == 'original':
        roots = base['round_root_ids'][0]
        result['round_root_ids'] = [roots[:] for _ in range(result['sampling_rounds'])]
        result['round_split_counts'] = [dict(zip(roots, counts, strict=True)) for counts in config['original_split_schedule']]
    else:
        result['round_root_ids'] = base['round_root_ids'][:result['sampling_rounds']]
    return result


def budget(config):
    candidate = baseline = 0
    for index, roots in enumerate(config['round_root_ids']):
        current = round_config(config, 'B3', index)
        candidate += sum((1+split_candidate_count(current, rid))*current['candidate_sampling']['continuations_per_candidate'] for rid in roots)
        baseline += len(roots)*current['independent_baseline_repetitions']
    return {'candidate_continuations': candidate, 'independent_baseline_continuations': baseline,
            'total_continuations': candidate+baseline,
            'optimizer_steps': config['sampling_rounds']*config['root_limit']*config['update']['epochs']//config['update']['gradient_accumulation_groups']}


def guarded_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Changed frozen request: {path}')
        return
    write_json(path, value)


def prepare(config):
    output = Path(config['output_dir']); output.mkdir(parents=True, exist_ok=True)
    task_checks = build(config['heldout_dir'])
    budgets = {}
    for seed in config['training_seeds']:
        for arm in ARMS:
            current = arm_config(config, arm, seed)
            folder = Path(current['output_dir']); folder.mkdir(parents=True, exist_ok=True)
            path = folder/'config.yaml'
            if path.exists() and yaml.safe_load(path.read_text()) != current:
                raise ValueError('Changed training configuration')
            path.write_text(yaml.safe_dump(current, sort_keys=False)); load_config(path)
            budgets[arm] = budget(current)
    if {b['total_continuations'] for b in budgets.values()} != {240} or {b['optimizer_steps'] for b in budgets.values()} != {40}:
        raise ValueError('Expected 240 rollouts and 40 updates per seed/arm')
    base, _ = load_config(config['base_config'])
    files = [base['root_manifest'], base['test_manifest'],
             str(Path(config['heldout_dir'])/'roots.jsonl'), str(Path(config['heldout_dir'])/'tests.jsonl')]
    request = {'config': config, 'budgets': budgets, 'task_checks': task_checks,
               'manifests': {p: file_hash(p) for p in files},
               'initial_adapter_sha256': file_hash(Path(base['initial_adapter'])/'adapter_model.safetensors'),
               'executor_adapter_sha256': file_hash(Path(base['executor_adapter'])/'adapter_model.safetensors')}
    guarded_json(output/'request.json', request)
    return request


def checkpoints(config):
    """All six training runs must finish before any held-out model evaluation."""
    output = Path(config['output_dir']); base, _ = load_config(config['base_config'])
    result = {'SFT': {'adapter': base['initial_adapter'], 'adapter_sha256': file_hash(Path(base['initial_adapter'])/'adapter_model.safetensors')}}
    for seed in config['training_seeds']:
        for arm in ARMS:
            current = arm_config(config, arm, seed)
            update = json.loads((Path(current['output_dir'])/'B3/training_report.json').read_text())
            expected = budget(current)
            if (update['status'] != 'COMPLETE' or any(update[k] != expected[k] for k in
                    ('candidate_continuations', 'independent_baseline_continuations', 'optimizer_steps'))
                    or update['adapter_sha256'] != file_hash(Path(update['adapter'])/'adapter_model.safetensors')):
                raise ValueError('Incomplete or changed training run')
            result[f'{arm}_{seed}'] = {k: update[k] for k in ('adapter', 'adapter_sha256')}
    guarded_json(output/'evaluation_checkpoints.json', result)
    return result


def evaluation_seed(config, root_index, repetition):
    return config['evaluation_seed']+1000*root_index+repetition


def evaluate(config, model):
    selected = checkpoints(config)[model]
    base, framework = load_config(config['base_config']); framework['allow_validation'] = True
    folder = Path(config['output_dir'])/'evaluation'/model; folder.mkdir(parents=True, exist_ok=True)
    guarded_json(folder/'request.json', {'model': model, **selected, 'evaluation_seed': config['evaluation_seed'],
                 'repetitions': config['evaluation_repetitions']})
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    rows = load_root_manifest(Path(config['heldout_dir'])/'roots.jsonl')
    tests = {r['root_id']: r for r in read_jsonl(Path(config['heldout_dir'])/'tests.jsonl')}
    if len(rows) != 12 or any(r['split'] != 'VALIDATION' for r in rows.values()):
        raise ValueError('Expected twelve frozen VALIDATION compositions')
    planner, executor = load_policies(framework, selected['adapter'], base['executor_adapter'])
    before = {n: adapter_fingerprint(planner.model, n) for n in ('default', 'executor')}
    cases = []
    for index, (rid, row) in enumerate(sorted(rows.items())):
        for repetition in range(config['evaluation_repetitions']):
            cases.append(run_case(row, tests, framework, base, planner, executor,
                folder/'natural'/rid/f'repeat_{repetition:02d}', evaluation_seed(config, index, repetition)))
    if any(c['status'] != 'COMPLETE' or c['reward'] is not None for c in cases):
        raise ValueError('Incomplete held-out evaluation or unexpected training reward')
    after = {n: adapter_fingerprint(planner.model, n) for n in before}
    if before != after:
        raise RuntimeError('Evaluation changed weights')
    result = {'status': 'COMPLETE', 'model': model, **selected, 'natural': case_summary(cases),
        'by_root': {rid: case_summary([c for c in cases if c['root_id'] == rid]) for rid in rows},
        'frozen_before': before, 'frozen_after': after, 'optimizer_updates_from_evaluation': 0,
        'actual_model_cost': dict(costs(planner, executor))}
    write_json(folder/'result.json', result)
    print(json.dumps({'model': model, 'passes': result['natural']['final_successes'], 'runs': len(cases)}), flush=True)
    return result


def outcome_records(config, model):
    definitions = {rid: json.loads(Path(row['fixture_path']).read_text())
                   for rid, row in load_root_manifest(Path(config['heldout_dir'])/'roots.jsonl').items()}
    records = []
    for path in sorted((Path(config['output_dir'])/'evaluation'/model/'natural').glob('*/repeat_*/case_result.json')):
        case = json.loads(path.read_text()); definition = definitions[case['root_id']]
        events = read_jsonl(path.parent/'runtime/trace.jsonl')
        event = next((e for e in events if e['node_id'] == 'N0' and e['host_result']['code'] == 'PLAN_REGISTERED'), None)
        action = event['parsed_action'] if event else None
        variants = {plan_key(a, definition): name for name, a in plans(definition).items()}
        records.append({'root_id': case['root_id'], 'seed': case['seed'],
            'success': int(case['final_scores']['private']['status'] == 'PASS'),
            'registered_mode': action['args']['mode'] if action else None,
            'reviewed_variant': variants.get(plan_key(action, definition)) if action else None,
            'root_first_plan_registered': case['root_first_plan_registered'],
            'stop_reason': case['stop_reason'], 'run_dir': case['run_dir']})
    expected = {(rid, evaluation_seed(config, i, j)) for i, rid in enumerate(sorted(definitions))
                for j in range(config['evaluation_repetitions'])}
    if len(records) != len(expected) or {(r['root_id'], r['seed']) for r in records} != expected:
        raise ValueError('Missing or duplicated paired evaluation slots')
    return records


def report(config):
    models = checkpoints(config); output = Path(config['output_dir'])
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in models}
    records = {m: outcome_records(config, m) for m in models}
    if any(e['status'] != 'COMPLETE' or e['natural']['runs'] != 60 or e['frozen_before'] != e['frozen_after'] for e in evaluations.values()):
        raise ValueError('Incomplete or unfrozen evaluation')
    roots = set(load_root_manifest(Path(config['heldout_dir'])/'roots.jsonl'))
    paired = {}; signals = {}; updates = {}
    for seed in config['training_seeds']:
        original, optimized = [f'{a}_{seed}' for a in ARMS]
        paired[str(seed)] = {'original_vs_SFT': paired_rows(records['SFT'], records[original], roots),
                            'optimized_vs_SFT': paired_rows(records['SFT'], records[optimized], roots),
                            'optimized_vs_original': paired_rows(records[original], records[optimized], roots)}
        for arm in ARMS:
            current = arm_config(config, arm, seed); model = f'{arm}_{seed}'
            updates[model] = json.loads((Path(current['output_dir'])/'B3/training_report.json').read_text())
            signals[model] = training_signals(current, updates[model])
    totals = {a: [evaluations[f'{a}_{s}']['natural']['final_successes'] for s in config['training_seeds']] for a in ARMS}
    result = {'status': 'COMPLETE', 'scope': 'THREE_TRAINING_SEEDS_UNSEEN_COMPONENT_COMPOSITIONS',
              'evaluations': evaluations, 'paired_by_training_seed': paired, 'training_signals': signals, 'updates': updates,
              'passes_across_training_seeds': {a: {'values': values, 'mean': mean(values), 'sample_sd': stdev(values)} for a, values in totals.items()},
              'root_modes': {m: dict(Counter(r['registered_mode'] or 'UNREGISTERED' for r in rows)) for m, rows in records.items()},
              'outcomes': records, 'new_training_rollouts': sum(u['candidate_continuations']+u['independent_baseline_continuations'] for u in updates.values()),
              'new_optimizer_steps': sum(u['optimizer_steps'] for u in updates.values()), 'heldout_rollouts': sum(e['natural']['runs'] for e in evaluations.values()),
              'data_use': '12 predeclared novel component pairings, 2 visible and 4 private fresh inputs each. All six RL models trained from common SFT before held-out model evaluation. VALIDATION supplies no optimizer reward. One shared SFT evaluation is reused in paired comparisons, not counted as three independent baselines. Original B3 uses increased candidate breadth to match optimized rollout budget; compute/token costs need not match.'}
    interruptions = [json.loads(p.read_text()) for p in sorted(output.glob('interruption_*.json'))]
    result['interruptions'] = interruptions
    result['executed_optimizer_steps_including_interrupted'] = result['new_optimizer_steps']+sum(r['optimizer_steps'] for r in interruptions)
    result['extra_incomplete_training_continuations'] = sum(r.get('interrupted_incomplete_continuations', 0) for r in interruptions)
    write_json(output/'result.json', result)
    print(json.dumps({'passes': result['passes_across_training_seeds'], 'paired': paired}), flush=True)
    return result


def run_experiment(config, config_path, devices):
    prepare(config)
    output = Path(config['output_dir'])
    def stage(jobs):
        queue = Queue()
        for name, arguments in jobs:
            queue.put((name, arguments))
        def worker(device):
            while True:
                try:
                    name, arguments = queue.get_nowait()
                except Empty:
                    return
                print(json.dumps({'starting': name, 'device': device}), flush=True)
                with (output/f'{name}.log').open('a') as log:
                    subprocess.run([sys.executable, '-m', 'audittrainer.refinement_generalization',
                                    *arguments, '--config', config_path], stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': device}, check=True)
                print(json.dumps({'completed': name, 'device': device}), flush=True)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, device) for device in devices]
            for future in futures:
                future.result()
    stage([(f'train_{arm}_{seed}', ['train', '--arm', arm, '--seed', str(seed)])
           for seed in config['training_seeds'] for arm in ARMS])
    selected = checkpoints(config)
    stage([(f'evaluate_{model}', ['evaluate', '--model', model]) for model in selected])
    report(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'report', 'run'])
    parser.add_argument('--config', default='configs/refinement_generalization.yaml')
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--seed', type=int)
    parser.add_argument('--model', default='SFT')
    parser.add_argument('--devices', default='2,0')
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'prepare':
        print(json.dumps(prepare(config)), flush=True)
    elif args.command == 'train':
        current, _ = load_config(Path(arm_config(config, args.arm, args.seed)['output_dir'])/'config.yaml')
        run(current, 'B3')
    elif args.command == 'evaluate':
        evaluate(config, args.model)
    elif args.command == 'report':
        report(config)
    else:
        run_experiment(config, args.config, args.devices.split(','))


if __name__ == '__main__':
    main()
