"""Matched-budget B3 comparison: three fixed roots versus twelve rotating roots."""
import argparse
import json
from pathlib import Path

from .dataset import file_hash, load_root_manifest
from .evaluate import write_json
from .refinement_curriculum_rl import evaluate, evaluation_summary
from .refinement_multiround import natural_allocations, prepare as prepare_rounds, run, training_signals
from .refinement_pr1 import load_config

METHODS = ('B0', 'B3_fixed', 'B3_rotate')


def prepare(config):
    """Reuse the completed control; both policies start from the same SFT model."""
    control, _ = load_config(config['fixed_control_config'])
    for key in ('initial_adapter', 'executor_adapter', 'seed', 'root_limit', 'repair',
                'runtime_overrides', 'candidate_sampling', 'reward', 'adaptation',
                'update', 'sampling_rounds', 'round_seed_stride'):
        if config[key] != control[key]:
            raise ValueError(f'Coverage comparison changed {key}')
    roots = load_root_manifest(config['root_manifest'])
    schedule = config['round_root_ids']
    if (len(roots) != 12 or set(r for batch in schedule for r in batch) != set(roots)
            or len(schedule) != config['sampling_rounds']
            or any(len(set(batch)) != config['root_limit'] for batch in schedule)):
        raise ValueError('Expected twelve roots in five three-state rounds')
    if any(row['split'] != 'TRAIN' for row in roots.values()):
        raise ValueError('Coverage pool must remain TRAIN')
    fixed = json.loads((Path(config['fixed_control_dir'])/'training_report.json').read_text())
    if fixed['status'] != 'COMPLETE' or fixed['optimizer_steps'] != 50:
        raise ValueError('The fixed-root control is unfinished')
    if fixed['adapter_sha256'] != file_hash(Path(fixed['adapter'])/'adapter_model.safetensors'):
        raise ValueError('Fixed-root control weights changed')
    prepare_rounds(config)
    output = Path(config['output_dir'])
    for method, target in [('B3_fixed', Path(fixed['adapter'])), ('B3_rotate', output/'B3'/'final_adapter')]:
        folder = output/method; folder.mkdir(exist_ok=True)
        link = folder/'final_adapter'
        if link.is_symlink() or link.exists():
            if link.resolve() != target.resolve():
                raise ValueError('Changed evaluation checkpoint')
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)


def paired_rows(left, right, root_ids):
    a = {(r['root_id'], r['seed']): r['success'] for r in left if r['root_id'] in root_ids}
    b = {(r['root_id'], r['seed']): r['success'] for r in right if r['root_id'] in root_ids}
    if a.keys() != b.keys() or not a:
        raise ValueError('Missing paired outcomes')
    return {'runs': len(a), 'left_passes': sum(a.values()), 'right_passes': sum(b.values()),
            'wins': sum(b[k] > a[k] for k in a), 'losses': sum(b[k] < a[k] for k in a),
            'ties': sum(b[k] == a[k] for k in a)}


def report(config):
    prepare(config)
    output = Path(config['output_dir'])
    calibration = json.loads((output/'calibration'/'result.json').read_text())
    if calibration['status'] != 'COMPLETE' or calibration['reference_summary']['runs'] != 192:
        raise ValueError('Incomplete twelve-root execution calibration')
    updates = {
        'B3_fixed': json.loads((Path(config['fixed_control_dir'])/'training_report.json').read_text()),
        'B3_rotate': json.loads((output/'B3'/'training_report.json').read_text())}
    for update in updates.values():
        if (update['status'] != 'COMPLETE' or update['optimizer_steps'] != 50
                or update['candidate_continuations'] != 120 or update['independent_baseline_continuations'] != 15):
            raise ValueError('The matched training budget is incomplete')
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in METHODS}
    allocations = {m: natural_allocations(config, m) for m in METHODS}
    all_roots = set(load_root_manifest(config['root_manifest']))
    control, _ = load_config(config['fixed_control_config'])
    anchors = set(load_root_manifest(control['root_manifest']))
    partitions = {'all': all_roots, 'original_three': anchors, 'additional_nine': all_roots-anchors}
    paired = {name: paired_rows(allocations['B3_fixed']['records'], allocations['B3_rotate']['records'], ids)
              for name, ids in partitions.items()}
    signals = {'B3_fixed': training_signals(control, updates['B3_fixed']),
               'B3_rotate': training_signals(config, updates['B3_rotate'])}
    result = {'status': 'COMPLETE', 'scope': 'B3_TRAIN_COVERAGE_MATCHED_BUDGET',
        'updates': updates, 'calibration': calibration,
        **evaluation_summary(config, evaluations, METHODS),
        'natural_allocations': allocations, 'training_signals': signals,
        'paired_rotate_vs_fixed': paired,
        'paired_rotate_vs_fixed_by_root': {rid: paired_rows(allocations['B3_fixed']['records'],
            allocations['B3_rotate']['records'], {rid}) for rid in sorted(all_roots)},
        'unique_rl_roots': {m: len({g['root_id'] for r in u['rounds'] for g in r['groups']}) for m,u in updates.items()},
        'new_training_trajectories': 135, 'reused_control_training_trajectories': 135,
        'data_use': 'Twelve predeclared TRAIN compositions; all authored calibration outcomes retained, not optimized. Rotating B3 starts from the common SFT checkpoint. Fixed B3 reuses the existing matched-budget five-round training chain. Same fresh final-evaluation root/seed pairs for all models; evaluation rewards never trained. This is one training seed and not held-out task generalization.'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'rotate_vs_fixed': paired,
        'passes': {m:e['natural']['final_successes'] for m,e in evaluations.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate', 'report'])
    parser.add_argument('--config', default='configs/refinement_coverage.yaml')
    parser.add_argument('--method', choices=METHODS, default='B0')
    parser.add_argument('--collection-gpus', nargs='+', type=int,
                        help='Optional physical GPU per existing root shard; statistical budget is unchanged')
    args = parser.parse_args(); config, framework = load_config(args.config)
    if args.command == 'report':
        report(config); return
    prepare(config)
    if args.command == 'run':
        calibration = json.loads((Path(config['output_dir'])/'calibration'/'result.json').read_text())
        if calibration['status'] != 'COMPLETE':
            raise ValueError('Finish TRAIN execution calibration first')
        run(config, 'B3', args.collection_gpus)
    elif args.command == 'evaluate':
        evaluate(config, framework, args.method)


if __name__ == '__main__':
    main()
