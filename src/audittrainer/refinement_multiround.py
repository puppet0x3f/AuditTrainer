"""Five finite rounds of fresh root-planning RL with a fixed SFT KL reference."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_curriculum import partition_values, plans
from .refinement_curriculum_rl import METHODS, evaluate, evaluation_summary, plan_key
from .refinement_pr1 import load_config, load_training_groups


def round_config(config, method, index):
    if method not in METHODS[1:] or not 0 <= index < config['sampling_rounds']:
        raise ValueError('Invalid method/round')
    result = deepcopy(config)
    root = Path(config['output_dir'])/method
    previous = root/f'round_{index:02d}'/method
    result.update(output_dir=str(root/f'round_{index+1:02d}'),
        initial_adapter=config['initial_adapter'] if index == 0 else str(previous/'final_adapter'),
        reference_adapter=config['initial_adapter'],
        seed=config['seed']+index*config['round_seed_stride'],
        retention_offset=index*config['update']['epochs']*config['root_limit'])
    if index:
        result['optimizer_state'] = str(previous/'optimizer.pt')
    if 'round_root_ids' in config:
        result['root_manifest'] = str(Path(result['output_dir'])/'roots.jsonl')
    if 'round_split_counts' in config:
        result['candidate_sampling']['split_bodies_by_root'] = config['round_split_counts'][index]
    return result


def prepare(config):
    output = Path(config['output_dir']); output.mkdir(parents=True, exist_ok=True)
    request = output/'request.json'
    if request.exists() and json.loads(request.read_text()) != config:
        raise ValueError('Changed multiround experiment')
    write_json(request, config)
    for name in ('groups', 'independent_baseline'):
        target = (Path(config['diagnostic_sampling_dir'])/name).resolve()
        link = output/name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        elif link.resolve() != target:
            raise ValueError('Changed fixed diagnostic pool')


def run(config, method, collection_devices=None):
    prepare(config)
    steps_per_round = config['update']['epochs']*config['root_limit']//config['update']['gradient_accumulation_groups']
    fixed_hash = file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')
    rounds = []
    for index in range(config['sampling_rounds']):
        current = round_config(config, method, index)
        output = Path(current['output_dir']); output.mkdir(parents=True, exist_ok=True)
        if 'round_root_ids' in config:
            selected = config['round_root_ids'][index]
            rows = load_root_manifest(config['root_manifest'])
            if len(selected) != config['root_limit'] or len(set(selected)) != len(selected):
                raise ValueError('Each round must preserve the fixed state budget')
            content = ''.join(json.dumps(rows[rid], sort_keys=True)+'\n' for rid in sorted(selected))
            manifest = Path(current['root_manifest'])
            if manifest.exists() and manifest.read_text() != content:
                raise ValueError('Changed round task selection')
            manifest.write_text(content)
        path = output/'config.yaml'
        if path.exists() and yaml.safe_load(path.read_text()) != current:
            raise ValueError('Changed resumed round')
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        shards = len(collection_devices) if collection_devices else 1
        for command, completions in (
            ('collect', [output/f'collect_{method}_shard{i}_complete.json' for i in range(shards)]),
            ('train', [output/method/'training_report.json']),
        ):
            if all(p.exists() for p in completions):
                if any(json.loads(p.read_text())['status'] != 'COMPLETE' for p in completions):
                    raise ValueError(f'Incomplete stage: {completions}')
                continue
            print(json.dumps({'method': method, 'round': index+1, 'command': command}), flush=True)
            base = [sys.executable, '-m', 'audittrainer.refinement_pr1', command,
                    '--config', str(path), '--method', method]
            if command == 'collect' and collection_devices:
                with ExitStack() as stack:
                    jobs = []
                    for shard, device in enumerate(collection_devices):
                        if completions[shard].exists():
                            if json.loads(completions[shard].read_text())['status'] != 'COMPLETE':
                                raise ValueError('Incomplete collection shard')
                            continue
                        log = stack.enter_context((output/f'collect_shard{shard}.log').open('a'))
                        jobs.append(subprocess.Popen(base+['--shards', str(shards), '--shard-index', str(shard)],
                            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)}))
                    codes = [job.wait() for job in jobs]
                    if any(codes):
                        raise RuntimeError(f'Collection worker failed: {codes}')
            else:
                with (output/f'{command}.log').open('a') as log:
                    subprocess.run(base, stdout=log, stderr=subprocess.STDOUT, check=True)
        trained = json.loads((output/method/'training_report.json').read_text())
        if (trained['optimizer_steps'] != steps_per_round
                or trained['cumulative_optimizer_steps'] != (index+1)*steps_per_round
                or trained['reference_sha256'] != fixed_hash
                or trained['behavior_sha256'] != file_hash(Path(current['initial_adapter'])/'adapter_model.safetensors')
                or trained['frozen_before'] != trained['frozen_after']):
            raise ValueError('Round failed update/behavior/reference continuity checks')
        groups = load_training_groups(current)
        rounds.append({'round': index+1, 'seed': current['seed'], 'output_dir': str(output),
            'candidate_continuations': sum(len(paths) for g, _, _, _ in groups for paths in g['case_paths']),
            'independent_baseline_continuations': len(groups)*current.get('independent_baseline_repetitions', 1),
            'optimizer_steps': trained['optimizer_steps'], 'cumulative_optimizer_steps': trained['cumulative_optimizer_steps'],
            'training_report': str(output/method/'training_report.json'),
            'behavior_sha256': trained['behavior_sha256'], 'adapter_sha256': trained['adapter_sha256'],
            'reference_sha256': trained['reference_sha256'], 'reward_gradient_checks': trained['reward_gradient_checks'],
            'groups': [{'root_id': g['root_id'], 'q_values': g['q_values'], 'p_split': g['mode_probabilities']['SPLIT'],
                        'baseline': b} for g, _, _, b in groups]})
        print(json.dumps({'method': method, 'round_complete': index+1,
                          'cumulative_steps': trained['cumulative_optimizer_steps'],
                          'reward_gradients': trained['reward_gradient_checks']}), flush=True)
    root = Path(config['output_dir'])/method
    adapter = root/'final_adapter'
    target = (Path(rounds[-1]['output_dir'])/method/'final_adapter').resolve()
    if not adapter.exists():
        adapter.symlink_to(target, target_is_directory=True)
    elif adapter.resolve() != target:
        raise ValueError('Final adapter is not the last fixed-budget round')
    result = {'status': 'COMPLETE', 'method': method, 'sampling_rounds': len(rounds),
        'optimizer_steps': sum(r['optimizer_steps'] for r in rounds), 'rounds': rounds,
        'candidate_continuations': sum(r['candidate_continuations'] for r in rounds),
        'independent_baseline_continuations': sum(r['independent_baseline_continuations'] for r in rounds),
        'adapter': str(adapter), 'adapter_sha256': file_hash(adapter/'adapter_model.safetensors')}
    write_json(root/'training_report.json', result)


def natural_allocations(config, method):
    rows = load_root_manifest(config['root_manifest']); records = []
    definitions = {rid: json.loads(Path(row['fixture_path']).read_text()) for rid, row in rows.items()}
    for path in sorted((Path(config['output_dir'])/'evaluation'/method/'natural').glob('*/repeat_*/case_result.json')):
        case = json.loads(path.read_text()); definition = definitions[case['root_id']]
        variants = {plan_key(a, definition): name for name, a in plans(definition).items()}
        events = read_jsonl(path.parent/'runtime'/'trace.jsonl')
        event = next((e for e in events if e['node_id'] == 'N0' and e['host_result']['code'] == 'PLAN_REGISTERED'), None)
        variant = variants.get(plan_key(event['parsed_action'], definition)) if event else None
        records.append({'root_id': case['root_id'], 'seed': case['seed'], 'reviewed_variant': variant,
            'registered_mode': event['parsed_action']['args']['mode'] if event else None,
            'success': case['reward']['y'], 'protocol_failure': case['stop_reason'] == 'PROTOCOL_RETRIES_EXHAUSTED'})
    return {'by_root': {rid: dict(Counter(r['reviewed_variant'] or 'UNMATCHED' for r in records if r['root_id'] == rid)) for rid in rows},
            'records': records}


def training_signals(config, update):
    """Count distinct registered plans with exact reviewed child contracts."""
    definitions = {rid: json.loads(Path(row['fixture_path']).read_text())
                   for rid, row in load_root_manifest(config['root_manifest']).items()}
    records = []
    for round_ in update['rounds']:
        current, _ = load_config(Path(round_['output_dir'])/'config.yaml')
        for group, _, candidates, _ in load_training_groups(current):
            definition = definitions[group['root_id']]
            variants = {plan_key(a, definition): name for name, a in plans(definition).items()}
            draws = []; eligible = []
            for index, candidate in enumerate(candidates[1:], 1):
                registered = all(json.loads((Path(path)/'case_result.json').read_text())['root_first_plan_registered']
                                 for path in group['case_paths'][index])
                key = plan_key(candidate['parsed_action'], definition)
                variant = variants.get(key)
                q = group['q_values'][index]
                draws.append({'candidate': index, 'registered': registered, 'oracle_backed': key is not None,
                              'reviewed_variant': variant, 'q': q})
                if registered and key is not None:
                    eligible.append((key, q))
            values = partition_values(eligible)
            records.append({'round': round_['round'], 'root_id': group['root_id'], 'draws': draws,
                'mode_value_difference': group['advantages']['q_atomic'] != group['advantages']['q_split'],
                'body_reward_difference': len(set(group['q_values'][1:])) > 1,
                'registered_oracle_backed_plan_values': values,
                'registered_oracle_backed_plan_value_difference': len(set(values.values())) > 1})
    return {'root_state_batches': len(records),
        **{key: sum(r[key] for r in records) for key in
           ('mode_value_difference', 'body_reward_difference', 'registered_oracle_backed_plan_value_difference')},
        'registered_split_draws': sum(d['registered'] for r in records for d in r['draws']),
        'registered_oracle_backed_split_draws': sum(d['registered'] and d['oracle_backed'] for r in records for d in r['draws']),
        'registered_reviewed_split_draws': sum(bool(d['registered'] and d['reviewed_variant']) for r in records for d in r['draws']),
        'records': records}


def report(config):
    output = Path(config['output_dir'])
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in METHODS}
    updates = {m: json.loads((output/m/'training_report.json').read_text()) for m in METHODS[1:]}
    expected_steps = config['sampling_rounds']*config['update']['epochs']*config['root_limit']//config['update']['gradient_accumulation_groups']
    for update in updates.values():
        if (update['status'] != 'COMPLETE' or update['optimizer_steps'] != expected_steps
                or update['sampling_rounds'] != config['sampling_rounds']):
            raise ValueError('The fixed multiround budget is not complete')
    interrupted = {m: sum(max((json.loads(line).get('optimizer_step', 0)
        for line in log.read_text().splitlines() if line.startswith('{')), default=0)
        for log in (output/m).glob('round_*/train_*_interrupted.log')) for m in METHODS[1:]}
    result = {'status': 'COMPLETE', 'scope': 'FIVE_ROUND_ROOT_RL_TRAIN_DIAGNOSTIC', 'updates': updates,
        **evaluation_summary(config, evaluations),
        'natural_allocations': {m: natural_allocations(config, m) for m in METHODS},
        'training_signals': {m: training_signals(config, updates[m]) for m in METHODS[1:]},
        'training_trajectories': sum(t['candidate_continuations']+t['independent_baseline_continuations'] for t in updates.values()),
        'interrupted_optimizer_steps': interrupted,
        'executed_optimizer_steps_including_interrupted': sum(t['optimizer_steps'] for t in updates.values())+sum(interrupted.values()),
        'data_use': 'Each method samples its own current behavior with paired round seeds. All raw samples retained. Fixed common SFT KL reference; continuous Adam state. Evaluation uses only final-budget checkpoints and fresh paired seeds; no evaluation reward is optimized.'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'paired': result['paired_natural_vs_B0'], 'changes': result['changes']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'evaluate', 'report'])
    parser.add_argument('--config', default='configs/refinement_multiround.yaml')
    parser.add_argument('--method', choices=METHODS, default='B0')
    args = parser.parse_args(); config, framework = load_config(args.config)
    if args.command == 'run':
        if args.method == 'B0': raise ValueError('B0 remains the fixed initial policy')
        run(config, args.method)
    elif args.command == 'evaluate':
        prepare(config); evaluate(config, framework, args.method)
    else:
        report(config)


if __name__ == '__main__':
    main()
