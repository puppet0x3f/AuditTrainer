"""A5 same-root factorial data and finite matched training schedules."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .dataset import EventReview, export_sft, file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .training_data import ReferencePolicy, execute, load_task
from .verifier import canonical, digest


GROUPS = {'F00': (False, False), 'F10': (True, False), 'F01': (False, True), 'F11': (True, True)}


def collect_functional():
    config = yaml.safe_load(Path('configs/data_pilot.yaml').read_text())
    framework = yaml.safe_load(Path(config['framework_config']).read_text())
    roots = load_root_manifest(config['root_manifest'])
    directory = Path('data/raw_traces/a5_functional')
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for row in roots.values():
        if row['split'] != 'TRAIN':
            continue
        d, private, ref = load_task(row, config)
        policy = ReferencePolicy(d, ref, plan='atomic', style='plain')
        target = directory / row['root_id']
        request = {'trajectory_source': 'authored_functional_reference', 'plan': 'atomic',
                   'style': 'plain', 'model_identity': policy.identity,
                   'selection': 'whole-program visible and private functional checks; no code responsibility filter'}
        result = execute(d, private, policy, target, framework, request)
        if result['functional_status'] != 'PASS' or not all(e['status'] == 'PASS' for e in result['offline_evidence']):
            raise ValueError(f"Functional reference failed: {row['root_id']}")
        # Basic ATOMIC protocol supervision is shared with the controls; code
        # remains FUNCTION_ONLY. No responsibility-violating target is injected.
        plan = next(e for e in read_jsonl(target / 'trace.jsonl') if e.get('parsed_action', {}).get('action') == 'plan')
        review = EventReview(event_id=plan['event_id'], root_id=row['root_id'], node_id='N0',
            action_hash=digest(plan['parsed_action']), contract_hash=digest(plan['observation_before']['incoming_contract']),
            verdict='SUPPORTED', label_origin='reviewed_reference',
            rationale='Basic atomic protocol: retain all incoming obligations in self; no learned decomposition target.',
            meta={'review_method': 'a5_basic_atomic_protocol', 'human_annotation': False})
        (target / 'supervision.jsonl').write_text(review.model_dump_json() + '\n')
        results.append(result)
        print(f"functional {len(results)} {row['root_id']}: PASS", flush=True)
    Path('runs/a5').mkdir(parents=True, exist_ok=True)
    write_json('runs/a5/functional_collection.json', {'status': 'PASS', 'results': results})
    return export_sft(directory, 'data/sft/a5_functional.jsonl', root_manifest=config['root_manifest'],
        config=framework, max_length=8192, allowed_splits=('TRAIN',), allow_function_only=True)


def replacement(row, d_on, i_on, ordinary):
    name = row['target_action']['action']
    if name == 'plan' and not d_on:
        return ordinary[(row['root_id'], 'plan')]
    if name == 'write_code' and not i_on:
        return ordinary[(row['root_id'], 'write_code')]
    # All arms receive the exact same consume and control actions. Merely knowing
    # artifact/version syntax must not create a protocol disadvantage for F00.
    return row


def prepare_groups():
    source = read_jsonl('data/sft/train.jsonl')
    base_manifest = json.loads(Path('data/sft/train.manifest.json').read_text())
    functional = read_jsonl('data/sft/a5_functional.jsonl')
    function_manifest = json.loads(Path('data/sft/a5_functional.manifest.json').read_text())
    ordinary = {(r['root_id'], r['target_action']['action']): r for r in functional}
    slots = read_jsonl('runs/a4/sampling_schedule.jsonl')
    assert len(slots) == len(source) == 2240
    assert {r['root_id'] for r in source} == {r['root_id'] for r in functional}
    output = Path('data/sft/a5')
    output.mkdir(parents=True, exist_ok=False)
    configs = Path('configs/a5')
    configs.mkdir(parents=True, exist_ok=False)
    comparisons = {}
    for group, (d_on, i_on) in GROUPS.items():
        pool, index = [], {}
        for row in source:
            selected = replacement(row, d_on, i_on, ordinary)
            if selected['sample_id'] not in index:
                index[selected['sample_id']] = len(pool)
                pool.append(selected)
        schedule = []
        for slot in slots:
            original = source[slot['index']]
            selected = replacement(original, d_on, i_on, ordinary)
            schedule.append({**slot, 'index': index[selected['sample_id']], 'sample_id': selected['sample_id'],
                             'source_sample_id': original['sample_id'], 'actual_bucket': selected['bucket'],
                             'replaced': selected['sample_id'] != original['sample_id']})
        data_path, schedule_path = output / f'{group}.jsonl', output / f'{group}.schedule.jsonl'
        data_path.write_text(''.join(canonical(r) + '\n' for r in pool))
        schedule_path.write_text(''.join(canonical(r) + '\n' for r in schedule))
        manifest = {**deepcopy(base_manifest), 'dataset_version': 'a5-factorial-real-events-v1',
            'samples': len(pool), 'allow_function_only': not i_on, 'output_sha256': file_hash(data_path),
            'raw_sources': base_manifest['raw_sources'] + function_manifest['raw_sources'],
            'by_bucket': dict(Counter(r['bucket'] for r in pool)),
            'by_scope': dict(Counter(r['supervision_scope'] for r in pool)),
            'factorial': {'group': group, 'decomposition': d_on, 'implementation': i_on,
                          'source_data_sha256': base_manifest['output_sha256'],
                          'functional_data_sha256': function_manifest['output_sha256'],
                          'common_schedule_sha256': file_hash('runs/a4/sampling_schedule.jsonl')},
            'rejected': 0, 'rejection_reasons': {}, 'selection': 'subset/replacement of already exported real events'}
        write_json(data_path.with_suffix('.manifest.json'), manifest)
        config = {'extends': '../sft_main.yaml', 'stage': 'A5',
                  'data': {'path': str(data_path), 'allow_function_only': not i_on,
                           'schedule_path': str(schedule_path), 'matched_slot_budget': len(slots)},
                  'training': {'output_dir': f'runs/a5/train/{group}_seed42', 'activation_offload': True}}
        (configs / f'{group}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        if group in {'F00', 'F11'}:
            pilot = {'extends': f'{group}.yaml', 'training': {'max_steps': 5,
                       'output_dir': f'runs/a5/pilot/{group}_seed42'}}
            if group == 'F11':
                # Include the longest actually scheduled action in the memory
                # pilot; a warmup-only prefix does not exercise long consumption.
                pilot_slots = deepcopy(schedule[:80])
                longest = max(schedule, key=lambda s: pool[s['index']]['meta']['prefix_tokens'] + pool[s['index']]['meta']['target_tokens'])
                pilot_slots[-1] = {**longest, 'optimizer_step': 5, 'phase': 'pilot_longest_scheduled'}
                pilot_path = output / 'F11.pilot.schedule.jsonl'
                pilot_path.write_text(''.join(canonical(s) + '\n' for s in pilot_slots))
                pilot['data'] = {'schedule_path': str(pilot_path)}
            (configs / f'{group}_pilot.yaml').write_text(yaml.safe_dump(pilot, sort_keys=False))
        comparisons[group] = {'unique_pool_examples': len(pool), 'slots': len(schedule), 'optimizer_steps': 140,
            'roots': len({s['root_id'] for s in schedule}),
            'unique_sampled_examples': len({s['sample_id'] for s in schedule}),
            'replaced_slots': sum(s['replaced'] for s in schedule),
            'target_tokens': sum(pool[s['index']]['meta']['target_tokens'] for s in schedule),
            'prefix_tokens': sum(pool[s['index']]['meta']['prefix_tokens'] for s in schedule),
            'actual_sampled_actions': dict(Counter(pool[s['index']]['target_action']['action'] for s in schedule)),
            'actual_sampled_buckets': dict(Counter(pool[s['index']]['bucket'] for s in schedule))}
    report = {'status': 'COMPLETE', 'seed': 42, 'scope': 'A5 single-seed development factorial',
              'groups': comparisons, 'matched': ['root per slot', 'curriculum slot', 'updates', 'base', 'LoRA', 'seed'],
              'not_exactly_matched': ['target tokens', 'prefix tokens', 'distinct example exposure', 'wall time'],
              'functional_control': 'new atomic whole-program authored trajectories; functional selection only',
              'interpretation': 'decomposition demonstration coverage and modular fulfillment coverage; not label-only ablation',
              'common_protocol': 'consume and control targets identical in all arms',
              'pilot_initialization': 'fresh public base', 'formal_initialization': 'fresh public base; no pilot updates inherited'}
    write_json('runs/a5/data_comparison.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect-functional', 'prepare-groups'])
    args = parser.parse_args()
    if args.command == 'collect-functional':
        print(json.dumps(collect_functional(), default=str)[:1000])
    else:
        prepare_groups()


if __name__ == '__main__':
    main()
