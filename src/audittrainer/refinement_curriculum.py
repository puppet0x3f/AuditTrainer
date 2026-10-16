"""Finite TRAIN calibration of verified root decompositions and raw policy support.

Authored probes are never passed off as policy samples. Only the separate raw
groups contain conditional body probabilities usable by the existing RL loader.
"""
if __name__ == '__main__':
    from .gpu_resource import reserve_cli_gpu
    _gpu_reservation = reserve_cli_gpu()

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from .a6 import reassignment_plan
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_policy import adapter_fingerprint, load_policies
from .refinement_pr1 import load_config, load_training_groups
from .refinement_rollout import collect, costs, run_case, runtime_for, case_summary
from .schema import PolicyOutput, TokenUsage
from .training_data import plan_args
from .training_tasks import FAMILIES, contract, definition as old_definition, oracle_a, oracle_b
from .verifier import canonical

VARIANTS = ('atomic', 'pipeline', 'fused', 'parent_aggregate')


def transfer_definition(first, second, parameter, *, split='TRAIN', input_cases=None, aggregate_goal=None):
    """Compose familiar components with compatible intermediate values."""
    a, b = FAMILIES[first], FAMILIES[second]
    rid = f'{split.lower()}_transfer_{first}_{second}_{parameter:02d}'
    goal = aggregate_goal or b['aggregate'].format(p=parameter, p15=parameter*15)
    inputs = a['cases'] if input_cases is None else input_cases
    root = contract('solve', 'text', {'parse': 'Parse the JSON input.', 'prepare': a['prepare'],
        'aggregate': goal, 'format': "Return compact JSON with separators=(',', ':') and no newline."},
        ['parse', 'prepare', 'aggregate', 'format'], 'Input text encodes '+a['input'], 'str')
    local = {
        'prepare': contract('prepare', 'items', {'prepare': a['prepare']}, ['prepare'], a['input']),
        'aggregate': contract('aggregate', 'prepared', {'aggregate': goal}, ['aggregate'], 'The exact output of: '+a['prepare']),
        'compute': contract('compute', 'items', {'prepare': a['prepare'], 'aggregate': goal}, ['prepare', 'aggregate'], a['input'])}
    suites, private = [], []
    for visibility, indices in [('VISIBLE', [0, 1]), ('PRIVATE', [2, 3, 4, 5])]:
        for key, spec in [('root', root), *local.items()]:
            cases = []
            for i in indices:
                items = deepcopy(inputs[i]); prepared = oracle_a(first, items)
                answer = oracle_b(second, prepared, parameter)
                args = [json.dumps(items)] if key == 'root' else [prepared if key == 'aggregate' else items]
                expected = json.dumps(answer, separators=(',', ':')) if key == 'root' else prepared if key == 'prepare' else answer
                cases.append({'case_id': f'{key}_{i}', 'args': args, 'expected': expected})
            suite = {'suite_id': f'{rid}_{key}_{visibility.lower()}', 'check_kind': 'ROOT' if key == 'root' else 'LOCAL',
                'visibility': visibility, 'oracle_origin': 'PR1C composition of independently checked A4 oracles',
                'covered_obligations': [o['id'] for o in spec['ensures']], 'cases': cases}
            if key != 'root': suite['contract_key'] = key
            (suites if visibility == 'VISIBLE' else private).append(suite)
    mapping = first in {'ledger', 'inventory', 'graph'}
    result_description = 'dict' if mapping else 'result'
    labels = ("The account/SKU/node labels in the summary refer to the input dictionary keys; numerical balances/units refer to its values. "
              if mapping else "The column sums in the summary denote the numbers in the prepared list, and column indices denote their positions. "
              if second == 'matrix' and first != second else '')
    task = {'root_id': rid, 'family_id': f'pr1c_transfer_{first}_{second}_v1', 'split': split,
        'entrypoint': 'solve', 'visible_suite_ref': f'{rid}_root_visible', 'private_suite_ref': f'{rid}_root_private',
        'spec': f"Input is JSON encoding {a['input']}. {a['prepare']} Then apply this summary to that resulting {result_description}: {goal} "
                + labels +
                "Return compact JSON without a newline. Implement solve(text: str) -> str. "
                'Optional reviewed subcontracts (no implementation): '+canonical(local)}
    reference = {'prepare': old_definition(first, parameter)[2]['prepare'],
                 'aggregate': old_definition(second, parameter)[2]['aggregate'],
                 'component_lineage': [first, second], 'parameter': parameter}
    return {'task': task, 'contract': root, 'local_contracts': local, 'suites': suites}, {'suites': private}, reference


def build_transfer(config):
    directory = Path(config['root_manifest']).parent
    directory.mkdir(parents=True, exist_ok=False)
    rows, tests = [], []
    for first, second, parameter in config['transfers']:
        d, private, reference = transfer_definition(first, second, parameter)
        task = d['task']; folder = directory/task['root_id']; folder.mkdir()
        write_json(folder/'task.json', d); write_json(folder/'private_tests.json', private)
        write_json(folder/'reference.json', reference)
        rows.append({**{k: task[k] for k in ('root_id', 'family_id', 'split')},
            'source': 'pr1c_train_component_transfer_v1', 'original_split': 'TRAIN', 'io_type': 'solve_text',
            'fixture_path': str(folder/'task.json'), 'source_hash': file_hash(folder/'task.json'),
            'reference_path': str(folder/'reference.json'), 'reference_hash': file_hash(folder/'reference.json')})
        tests.append({'root_id': task['root_id'], 'path': str(folder/'private_tests.json'),
                      'sha256': file_hash(folder/'private_tests.json')})
    for key, values in [('root_manifest', rows), ('test_manifest', tests)]:
        Path(config[key]).write_text(''.join(canonical(r)+'\n' for r in values))


def plans(definition):
    root = definition['contract']
    result = {name: {'action': 'plan', 'args': plan_args(root, definition, name, True)}
              for name in VARIANTS[:3]}
    result['parent_aggregate'] = {'action': 'plan', 'args': reassignment_plan(root, definition)}
    return deepcopy(result)


def prepare(config, framework):
    directory = Path(config['root_manifest']).parent
    directory.mkdir(parents=True, exist_ok=True)
    roots = load_root_manifest(config.get('source_root_manifest', 'data/manifests/roots.jsonl'))
    tests = {r['root_id']: r for r in read_jsonl(config.get('source_test_manifest', 'data/manifests/tests.jsonl'))}
    selected = [roots[rid] for rid in sorted(config['root_ids'])]
    if len(selected) != config['root_limit'] or any(r['split'] != 'TRAIN' for r in selected):
        raise ValueError('Calibration must contain exactly the fixed TRAIN roots')
    records = []
    for row in selected:
        d = json.loads(Path(row['fixture_path']).read_text())
        for variant, action in plans(d).items():
            # Exercise the real host and exact local-suite bindings before GPU use.
            output = PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(),
                                  model_identity={'backend': 'authored_calibration_plan_check'})
            policy = SimpleNamespace(act=lambda observation, o=output: o)
            runtime, _ = runtime_for(row, tests, framework, policy,
                Path(config['output_dir'])/'plan_checks'/row['root_id']/variant)
            event = runtime.step()
            if event.host_result.code != 'PLAN_REGISTERED':
                raise ValueError(f'{row["root_id"]}/{variant}: {event.host_result}')
            bindings = {}
            for child in runtime.contexts.values():
                if child.node.node_id == 'N0':
                    continue
                available = runtime.verifier.available_suites(row['root_id'], child.node.incoming_contract)
                if available == ['NO_APPLICABLE_SUITE']:
                    raise ValueError('Reviewed calibration child has no visible local oracle')
                bindings[child.node.node_id] = available
            records.append({'root_id': row['root_id'], 'variant': variant, 'action': action,
                            'source': 'AUTHORED_CALIBRATION_ONLY', 'local_suites': bindings})
    for key, values in [('root_manifest', selected), ('test_manifest', [tests[r['root_id']] for r in selected])]:
        path = Path(config[key]); content = ''.join(canonical(r)+'\n' for r in values)
        if path.exists() and sorted(read_jsonl(path), key=lambda r: r['root_id']) != values:
            raise ValueError('Changed calibration selection')
        path.write_text(content)
    write_json(directory/'selection.json', {
        'status': 'FIXED_BEFORE_NEW_ROLLOUTS', 'split': 'TRAIN', 'roots': [r['root_id'] for r in selected],
        'selection_reason': config.get('selection_reason', 'PR0 exposed body differences in inventory/ledger/matrix; include both parameters 3 and 4 in each family, retaining every outcome. This is deliberately selected TRAIN calibration, not a held-out benchmark.'),
        'task_sources_fixed_before_sampling': True, 'runtime_overrides': config['runtime_overrides'],
        'curriculum': 'Only root may split; existing max_depth=1 forces atomic leaves. All other budgets and fixed F01 execution are unchanged.',
        'reference_variants': list(VARIANTS), 'reference_repetitions': 4,
        'repeat_blocks': 'First two seeds diagnose; last two independently check the same directional contrast. No selection between blocks.',
        'planned_reference_runs': len(selected)*16,
        'planned_raw_runs': len(selected)*8 if config.get('collect_policy_samples', True) else 0,
        'planned_independent_baselines': len(selected) if config.get('collect_policy_samples', True) else 0,
        'readiness': 'Report execution support, repeated contrasts between oracle-backed authored plans, and separately value differences between registered raw plans. No reward for node count; no manufactured likelihood for authored plans.',
        'plan_checks': records})
    print(json.dumps({'prepared_roots': len(selected), 'registered_reference_plans': len(records)}), flush=True)


def reference_collect(config, framework, planner, executor, rows, tests, shard, shards):
    for index, row in enumerate(rows):
        if index % shards != shard:
            continue
        base = Path(config['output_dir'])/'reference'/row['root_id']
        if (base/'result.json').exists():
            continue
        base.mkdir(parents=True, exist_ok=True)
        gate_path = base/'gate.json'
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
        else:
            template, _ = runtime_for(row, tests, framework, None, base/'decision')
            gate = planner.score_modes(template.build_observation())
            write_json(gate_path, gate)
        definition = json.loads(Path(row['fixture_path']).read_text())
        outcomes = {}
        for variant, action in plans(definition).items():
            candidate = planner._candidate(gate, action['args']['mode'], canonical(action),
                [], [], [], False, 'AUTHORED_CALIBRATION_ONLY_NOT_POLICY_SAMPLE')
            candidate['gate_old_log_prob'] = None
            candidate['logical_usage'] = {'input_tokens': 0, 'output_tokens': 0}
            write_json(base/f'{variant}.json', candidate)
            cases = [run_case(row, tests, framework, config, planner, executor,
                        base/variant/f'repeat_{rep}', config['seed']+500000+index*4+rep,
                        initial=(gate, candidate)) for rep in range(config['reference_repetitions'])]
            if any(r['status'] != 'COMPLETE' or not r['root_first_plan_registered'] for r in cases):
                raise ValueError('Incomplete or rejected authored calibration plan')
            outcomes[variant] = {'returns': [r['reward']['reward'] for r in cases],
                                 'case_paths': [r['run_dir'] for r in cases], 'summary': case_summary(cases)}
        write_json(base/'result.json', {'root_id': row['root_id'], 'status': 'COMPLETE', 'variants': outcomes})


def repeated_contrasts(values):
    """A signed paired difference must recur in both predeclared seed blocks."""
    result = []
    for a, b in combinations(VARIANTS[1:], 2):
        x, y = values[a]['returns'], values[b]['returns']
        gaps = [sum(x[start:start+2])/2-sum(y[start:start+2])/2 for start in (0, 2)]
        if gaps[0]*gaps[1] > 0:
            result.append({'a': a, 'b': b, 'block_gaps': gaps,
                           'mean_gap': (sum(x)-sum(y))/4})
    return result


def prepare_support(config_path, source_paths):
    """Teach complete legal plan forms, including unsuccessful variants, by SFT."""
    from .train_refinement import load_pr0_config, select_adaptation_rows
    config, _ = load_pr0_config(config_path)
    data = Path(config['root_manifest']).parent
    data.mkdir(parents=True, exist_ok=False)
    roots, tests, demonstrations = [], [], []
    for path in source_paths:
        source, _ = load_config(path)
        for row in load_root_manifest(source['root_manifest']).values():
            roots.append(row)
            definition = json.loads(Path(row['fixture_path']).read_text())
            gate_path = Path(source['output_dir'])/'reference'/row['root_id']/'gate.json'
            gate = json.loads(gate_path.read_text())
            for variant, action in plans(definition).items():
                # Equal exposure for all three SPLIT forms, without consulting Q.
                for copy in range(1 if variant == 'atomic' else 2):
                    demonstrations.append({
                        'sample_id': f"pr1c_{row['root_id']}_{variant}_{copy}",
                        'root_id': row['root_id'], 'family_id': row['family_id'], 'split': 'TRAIN',
                        'messages': gate['model_input']['messages'], 'target_action': action,
                        'source': 'AUTHORED_STRUCTURAL_SFT_SUPPORT', 'variant': variant,
                        'public_snapshot': str(gate_path), 'explicit_duplicate': copy > 0})
        tests.extend(read_jsonl(source['test_manifest']))
    if len({r['root_id'] for r in roots}) != config['root_limit'] or len(roots) != config['root_limit']:
        raise ValueError('Unexpected or duplicated support roots')
    for key, values in [('root_manifest', roots), ('test_manifest', tests)]:
        Path(config[key]).write_text(''.join(canonical(r)+'\n' for r in sorted(values, key=lambda r: r['root_id'])))
    source = Path(config['adaptation']['source'])
    source.write_text(''.join(canonical(r)+'\n' for r in demonstrations))
    manifest = {'protocol_version': config['protocol_version'], 'output_sha256': file_hash(source),
        'split': 'TRAIN', 'source_rows': len(demonstrations), 'unique_prompt_target_pairs': len(roots)*4,
        'source_configs': source_paths,
        'selection': 'All four authored legal plans for all nine calibration roots, irrespective of execution result. Two explicit copies per SPLIT form; ATOMIC repeats in the existing balanced selector. No implementation targets, scores or private tests enter SFT.'}
    write_json(source.with_suffix('.manifest.json'), manifest)
    selected, _ = select_adaptation_rows(config)
    manifest.update(selected_exposures=len(selected), by_mode=dict(Counter(r['target_action']['args']['mode'] for r in selected)))
    write_json(source.with_suffix('.manifest.json'), manifest)
    print(json.dumps(manifest), flush=True)


def partition_values(samples):
    """Duplicate draws of one responsibility partition are not distinct plans."""
    total, count = Counter(), Counter()
    for partition, q in samples:
        total[partition] += q
        count[partition] += 1
    return {key: total[key]/count[key] for key in count}


def raw_results(config):
    rows = load_root_manifest(config['root_manifest'])
    raw, all_raw = [], []
    for group, gate, candidates, baseline in load_training_groups(config):
        rid = group['root_id']
        eligible = []; details = []
        for i, paths in enumerate(group['case_paths']):
            cases = [json.loads((Path(p)/'case_result.json').read_text()) for p in paths]
            all_raw.extend(cases)
            registered = all(c['root_first_plan_registered'] for c in cases)
            # Match sampled child contracts to public, reviewed local suites exactly.
            definition = json.loads(Path(rows[rid]['fixture_path']).read_text())
            action = candidates[i]['parsed_action']
            oracle_backed = bool(action) and all(child['contract'] in definition['local_contracts'].values()
                                                   for child in action['args']['children'])
            partition = None
            if i and registered and oracle_backed:
                partition = canonical({'children': sorted(next(k for k, contract in definition['local_contracts'].items()
                    if child['contract'] == contract) for child in action['args']['children']),
                    'retained': sorted(action['args']['retained_responsibilities'])})
                eligible.append((partition, group['q_values'][i]))
            details.append({'index': i, 'registered': registered, 'oracle_backed': oracle_backed,
                            'parse_valid': candidates[i]['parse_error'] is None,
                            'partition': partition, 'q': group['q_values'][i]})
        values = partition_values(eligible)
        raw.append({'root_id': rid, 'q_values': group['q_values'], 'candidates': details,
                    'oracle_backed_partition_q': values,
                    'oracle_backed_split_value_difference': len(set(values.values())) > 1,
                    'p_split': gate['mode_probabilities']['SPLIT'], 'independent_baseline': baseline})
    return raw, all_raw


def summarize(config):
    output = Path(config['output_dir']); rows = load_root_manifest(config['root_manifest'])
    raw, all_raw = raw_results(config) if config.get('collect_policy_samples', True) else ([], [])
    references, all_reference = [], []
    for rid in sorted(rows):
        reference = json.loads((output/'reference'/rid/'result.json').read_text())
        reference['repeated_split_contrasts'] = repeated_contrasts(reference['variants'])
        references.append(reference)
        for entry in reference['variants'].values():
            all_reference.extend(json.loads((Path(p)/'case_result.json').read_text()) for p in entry['case_paths'])
    workers = [json.loads(p.read_text()) for p in sorted(output.glob('collect_*_complete.json'))]
    if not workers or any(w['before'] != w['after'] or w['status'] != 'COMPLETE' for w in workers):
        raise ValueError('Missing or changed frozen workers')
    expected_shards = workers[0]['shards']
    if sorted(w['shard'] for w in workers) != list(range(expected_shards)):
        raise ValueError('Missing worker shard')
    expected_raw = len(rows)*8 if config.get('collect_policy_samples', True) else 0
    if len(all_reference) != len(rows)*16 or len(all_raw) != expected_raw:
        raise ValueError('Incomplete fixed calibration batch')
    physical = Counter()
    for w in workers:
        physical.update(w['actual_model_cost'])
    result = {'status': 'COMPLETE', 'scope': 'TRAIN_ONLY_ROOT_CURRICULUM_CALIBRATION',
              'optimizer_steps': 0, 'reference': references, 'raw': raw,
              'reference_summary': case_summary(all_reference), 'raw_summary': case_summary(all_raw),
              'roots_atomic_at_least_three_of_four': sum(r['variants']['atomic']['summary']['final_successes'] >= 3 for r in references),
              'roots_with_repeated_legal_split_contrast': sum(bool(r['repeated_split_contrasts']) for r in references),
              'roots_with_repeated_successful_split': sum(any(sum(v['returns'][:2]) > 0 and sum(v['returns'][2:]) > 0
                  for name, v in r['variants'].items() if name != 'atomic') for r in references),
              'roots_with_raw_oracle_backed_split_value_difference': sum(r['oracle_backed_split_value_difference'] for r in raw),
              'roots_with_raw_split_average_above_atomic': sum(sum(r['q_values'][1:])/3 > r['q_values'][0] for r in raw),
              'actual_model_cost': dict(physical), 'workers': workers,
              'data_boundary': 'Only TRAIN roots; authored plans are diagnostic, separate from raw policy groups; no VALIDATION or sealed task access.'}
    write_json(output/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k.startswith('roots_') or k == 'status'}), flush=True)


def supported_report(config):
    """Check fresh raw support; never pool pre-SFT behavior with post-SFT data."""
    output = Path(config['output_dir'])
    groups = load_training_groups(config)
    raw, cases = raw_results(config)
    adaptation = json.loads((Path(config['adaptation']['output_dir'])/'training_report.json').read_text())
    checksum = file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')
    if adaptation['status'] != 'COMPLETE' or adaptation['adapter_sha256'] != checksum:
        raise ValueError('Incomplete or different structural SFT checkpoint')
    requests = [json.loads(p.read_text()) for p in sorted(output.glob('collect_B0_shard*_request.json'))]
    if not requests or sorted(r['shard_index'] for r in requests) != list(range(requests[0]['shards'])):
        raise ValueError('Missing fresh collection shard')
    workers = []
    for request in requests:
        if request['config'] != config or request['adapter_sha256'] != checksum:
            raise ValueError('Do not mix sampling policies or conditions')
        worker = json.loads((output/f"collect_B0_shard{request['shard_index']}_complete.json").read_text())
        if worker['status'] != 'COMPLETE' or worker['before'] != worker['after']:
            raise ValueError('Unfinished or changed collection worker')
        if worker['before']['executor'] != adaptation['executor']['after']:
            raise ValueError('Executor differs from the fixed SFT executor')
        workers.append(worker)
    for _, _, candidates, _ in groups:
        if any(c['body_source'] != 'MODEL_CONDITIONAL_BODY' for c in candidates[1:]):
            raise ValueError('Authored probes cannot enter the raw RL pool')
    baselines = [json.loads((output/'independent_baseline'/r['root_id']/'case_result.json').read_text()) for r in raw]
    body_states = sum(r['oracle_backed_split_value_difference'] for r in raw)
    mode_states = sum(sum(r['q_values'][1:])/3 > r['q_values'][0] for r in raw)
    details = [c for r in raw for c in r['candidates'][1:]]
    physical = Counter()
    for worker in workers:
        physical.update(worker['actual_model_cost'])
    result = {'status': 'COMPLETE', 'scope': 'TRAIN_ONLY_ROOT_CURRICULUM_SUPPORT',
        'readiness': 'READY' if body_states and mode_states else 'SIGNAL_INSUFFICIENT',
        'root_states': len(groups), 'raw_candidates': len(groups)*4,
        'candidate_continuations': len(cases), 'raw': raw, 'raw_summary': case_summary(cases),
        'independent_baseline': case_summary(baselines),
        'split_parse_valid': sum(c['parse_valid'] for c in details),
        'split_registered': sum(c['registered'] for c in details),
        'split_registered_and_oracle_backed': sum(c['registered'] and c['oracle_backed'] for c in details),
        'raw_oracle_backed_body_signal_states': body_states, 'raw_positive_split_mode_states': mode_states,
        'rl_optimizer_steps': 0, 'support_sft_optimizer_steps': adaptation['optimizer_steps'],
        'actual_model_cost': dict(physical), 'workers': workers,
        'calibration_sources': config['calibration_sources'],
        'provenance': 'Fresh raw sampling only, from the common structural SFT checkpoint. All invalid and duplicate outcomes retained with original probabilities. Authored calibration is separate; no VALIDATION or sealed task access.'}
    write_json(output/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in {'raw', 'workers', 'actual_model_cost', 'provenance'}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'prepare-transfer', 'prepare-support', 'collect', 'report', 'ready'])
    parser.add_argument('--config', default='configs/refinement_curriculum.yaml')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--transfer-config', default='configs/refinement_curriculum_transfer.yaml')
    args = parser.parse_args()
    if args.command == 'prepare-support':
        prepare_support(args.config, ['configs/refinement_curriculum.yaml', args.transfer_config]); return
    config, framework = load_config(args.config)
    if config['update']['enabled'] or framework['runtime']['max_depth'] != 1:
        raise ValueError('This calibration freezes weights and uses the root-only curriculum')
    if args.command == 'ready':
        supported_report(config); return
    if args.command == 'prepare-transfer':
        build_transfer(config); prepare(config, framework); return
    if args.command == 'prepare':
        prepare(config, framework); return
    if args.command == 'report':
        summarize(config); return
    rows = sorted(load_root_manifest(config['root_manifest']).values(), key=lambda r: r['root_id'])
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    planner, executor = load_policies(framework, config['initial_adapter'], config['executor_adapter'])
    before = {name: adapter_fingerprint(planner.model, name) for name in ('default', 'executor')}
    output = Path(config['output_dir']); job = f'collect_{args.shard_index}'
    request = {'config': config, 'shard': args.shard_index, 'shards': args.shards, 'before': before,
               'adapter_sha256': file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')}
    request_path = output/f'{job}_request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Changed resumed calibration configuration')
    write_json(request_path, request)
    reference_collect(config, framework, planner, executor, rows, tests, args.shard_index, args.shards)
    if config.get('collect_policy_samples', True):
        collect(rows, tests, framework, config, planner, executor, args.shard_index, args.shards)
        for i, row in enumerate(rows):
            if i % args.shards == args.shard_index:
                run_case(row, tests, framework, config, planner, executor,
                    output/'independent_baseline'/row['root_id'], config['seed']+900000+i)
    after = {name: adapter_fingerprint(planner.model, name) for name in before}
    if before != after:
        raise RuntimeError('Calibration changed weights')
    write_json(output/f'{job}_complete.json', {'status': 'COMPLETE', 'before': before, 'after': after,
        'shard': args.shard_index, 'shards': args.shards, 'actual_model_cost': dict(costs(planner, executor))})


if __name__ == '__main__':
    main()
