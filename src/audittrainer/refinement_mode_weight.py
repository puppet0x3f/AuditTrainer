"""One mode-reward-weight ablation, reusing the matched completed weight-1 control."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .dataset import file_hash, load_root_manifest
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum_rl import evaluate, evaluation_summary
from .refinement_multiround import natural_allocations, prepare as prepare_diagnostics
from .refinement_pr1 import load_config, train


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise ValueError(f'Changed experiment input: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def prepare(settings):
    control, framework = load_config(settings['control_config'])
    source = Path(control['output_dir'])
    previous = json.loads((source/'B3/training_report.json').read_text())
    if previous['status'] != 'COMPLETE' or previous['cumulative_optimizer_steps'] != 60:
        raise ValueError('Expected the completed repeat control at Adam step 60')
    output = Path(settings['output_dir'])
    current = deepcopy(control)
    current['output_dir'] = str(output/'training')
    current['update']['mode_reward_weight'] = settings['mode_reward_weight']
    folder = Path(current['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'config.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != current:
        raise ValueError('Changed weight experiment')
    path.write_text(yaml.safe_dump(current, sort_keys=False))
    for name in ('groups', 'independent_baseline'):
        link(source/name, folder/name)
    for request in source.glob('collect_B3_shard*_request.json'):
        link(request, folder/request.name)
    evaluation, _ = load_config(settings['evaluation_config'])
    evaluation.update(output_dir=str(output), initial_adapter=previous['adapter'])
    # Same held-fixed evaluation slots as the completed control; no new search.
    prepare_diagnostics(evaluation)
    link(settings['control_evaluation'], output/'evaluation/B0')
    link(folder/'B3/final_adapter', output/'B3_mode4/final_adapter')
    write_json(output/'design.json', {'status': 'PREDECLARED', 'settings': settings,
        'control_training_report': str(source/'B3/training_report.json'),
        'training_config': current, 'evaluation_config': evaluation,
        'control_weight': 1., 'treatment_weight': settings['mode_reward_weight'],
        'new_training_rollouts': 0, 'reused_training_rollouts': 120,
        'new_optimizer_steps': 10, 'new_natural_evaluations': 60, 'new_conditional_probes': 12,
        'decision': 'Compare final success and paired outcomes. Do not adopt weight 4 merely for a larger split frequency.',
        'data_use': 'Both branches start at the original sampling weights and Adam step 50, and use identical saved TRAIN candidates, rewards, baseline means and retention slots. The weight-1 checkpoint and evaluation are reused. No extra update on the step-60 actor using stale step-50 trajectories.'})
    return current, framework, evaluation


def report(settings, current, evaluation):
    output = Path(settings['output_dir'])
    control, _ = load_config(settings['control_config'])
    reports = {name: json.loads(path.read_text()) for name, path in {
        'B0': Path(control['output_dir'])/'B3/training_report.json',
        'B3_mode4': Path(current['output_dir'])/'B3/training_report.json'}.items()}
    for name, row in reports.items():
        if (row['status'] != 'COMPLETE' or row['previous_optimizer_steps'] != 50
                or row['optimizer_steps'] != 10 or row['cumulative_optimizer_steps'] != 60
                or row['frozen_before'] != row['frozen_after']
                or row['adapter_sha256'] != file_hash(Path(row['adapter'])/'adapter_model.safetensors')):
            raise ValueError(f'Unmatched or unfinished update: {name}')
    if reports['B0']['behavior_sha256'] != reports['B3_mode4']['behavior_sha256']:
        raise ValueError('Different behavior checkpoints')
    left = json.loads((Path(control['output_dir'])/'B3/request.json').read_text())
    right = json.loads((Path(current['output_dir'])/'B3/request.json').read_text())
    if left['data_sha256'] != right['data_sha256']:
        raise ValueError('Different candidate/reward data')
    if [(h['epoch'], h['root_id']) for h in reports['B0']['history']] != [(h['epoch'], h['root_id']) for h in reports['B3_mode4']['history']]:
        raise ValueError('Different accumulation/retention order')
    methods = ('B0', 'B3_mode4')
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in methods}
    allocations = {m: natural_allocations(evaluation, m) for m in methods}
    roots = set(load_root_manifest(evaluation['root_manifest']))
    paired = paired_rows(allocations['B0']['records'], allocations['B3_mode4']['records'], roots)
    result = {'status': 'COMPLETE', 'scope': 'SINGLE_VARIABLE_B3_MODE_REWARD_WEIGHT_TRAIN_PILOT',
              'weights': {'B0': 1., 'B3_mode4': settings['mode_reward_weight']},
              'updates': reports, **evaluation_summary(evaluation, evaluations, methods),
              'paired': paired, 'natural_allocations': allocations,
              'new_optimizer_steps': 10, 'new_training_rollouts': 0,
              'data_use': 'One training seed, existing TRAIN roots and fixed diagnostic evaluation slots. Weight-1 results reused. This is not unseen-task generalization or an independent confirmation set.'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'paired': paired, 'changes': result['changes']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'evaluate', 'report'))
    parser.add_argument('--config', default='configs/refinement_mode_weight.yaml')
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    current, framework, evaluation = prepare(settings)
    if args.command == 'train':
        train(current, framework, 'B3')
    elif args.command == 'evaluate':
        evaluate(evaluation, framework, 'B3_mode4')
    elif args.command == 'report':
        report(settings, current, evaluation)


if __name__ == '__main__':
    main()
