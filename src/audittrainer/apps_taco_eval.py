"""Paired APPS evaluation of the TACO-extended executor against DIVERSE."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

from . import apps_direct_rejudge as rejudge, apps_direct_weight_compare as direct
from . import apps_diverse_sft as diverse, apps_fresh_select as fresh
from . import apps_multitask as previous, apps_scale, apps_taco_sft as taco
from .apps_passk_select import _score
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox


def _context():
    from .refinement_policy import load_policies

    request = json.loads((taco.ROOT/'request.json').read_text())
    base, executor = load_policies(request['framework'], request['adapters']['internal_gate'],
                                   str(taco.ROOT/'training/final_adapter'))
    base.model.load_adapter(request['adapters']['body'], adapter_name='body',
                            is_trainable=False, local_files_only=True)
    base.model.load_adapter(request['adapters']['root_gate'], adapter_name='root_gate',
                            is_trainable=False, local_files_only=True)
    base.model.set_adapter('default')
    base.model.requires_grad_(False)
    return previous.CurrentPlanner(base), executor


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .policy import LLMPolicy
    from .refinement_rollout import run_case

    if not 0 <= shard < len(taco.GPUS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    taco.check(trained=True)
    done = taco.ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed TACO evaluation shard exists: {done}')
    request = json.loads((taco.ROOT/'request.json').read_text())
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    all_fresh = fresh._rows()
    fresh_indices = {row['root_id']: i for i, row in enumerate(all_fresh)}
    fresh_rows = rejudge._rows()[shard::len(taco.GPUS)]
    fresh_tests = fresh._tests()
    mixed_rows = [(i, row) for i, row in enumerate(apps_scale._validation_rows())
                  if i % len(taco.GPUS) == shard]
    mixed_tests = {row['root_id']: row for row in read_jsonl(previous.DATA/'tests.jsonl')}
    retention_rows = [(i, selection, row) for i, (selection, row) in
                      enumerate(apps_scale._retention_rows()) if i % len(taco.GPUS) == shard]
    retention_tests = {row['root_id']: row for row in
                       read_jsonl(apps_scale.CORRECTION_DATA/'tests.jsonl')}
    sandbox = Sandbox(ExecutionLimits(**request['framework']['execution']))
    reservation = acquire(taco.ROOT/'gpu_resources_eval.json', 'collect')
    try:
        signals = {row['root_id']: {'DIRECT': [], 'ATOMIC': []} for row in fresh_rows}
        with tqdm(total=len(fresh_rows)*6+len(mixed_rows)+len(retention_rows),
                  desc=f'TACO扩量评测 GPU {taco.GPUS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            llm = LLMPolicy.from_pretrained(
                request['framework'], adapter_path=taco.ROOT/'training/final_adapter')
            adapter_sha = file_hash(taco.ROOT/'training/final_adapter/adapter_model.safetensors')
            if llm.identity['adapter_weights_sha256'] != adapter_sha:
                raise ValueError('Direct model differs from completed TACO SFT')
            for row in fresh_rows:
                rid = row['root_id']
                sample = rejudge._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                for repetition in range(3):
                    seed = fresh.SEED_BASE+100*fresh_indices[rid]+repetition
                    directory = taco.ROOT/'evaluation/direct'/rid/f'seed_{seed}'
                    case = direct._direct_case(row, fresh_tests, request['framework'], llm,
                                               'TACO', directory, seed)
                    if case['status'] != 'COMPLETE':
                        raise RuntimeError(f'Incomplete TACO direct candidate: {rid}/{repetition}')
                    signals[rid]['DIRECT'].append(
                        _score(directory/'case_result.json', rid, sample, sandbox))
                    bar.update(1)
            del llm
            gc.collect()
            import torch
            torch.cuda.empty_cache()

            planner, executor = _context()
            for row in fresh_rows:
                rid = row['root_id']
                sample = rejudge._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                for repetition in range(3):
                    seed = fresh.SEED_BASE+100*fresh_indices[rid]+repetition
                    directory = taco.ROOT/'evaluation/atomic'/rid/f'seed_{seed}'
                    case = run_case(row, fresh_tests, old['framework'], old['config'],
                                    previous._ForcedRoot(planner, 'ATOMIC'), executor,
                                    directory, seed, reset_seed_after_root_plan=True)
                    if case['status'] != 'COMPLETE' or case['registered_plan_modes'] != {'ATOMIC': 1}:
                        raise RuntimeError(f'Incomplete TACO ATOMIC candidate: {rid}/{repetition}')
                    signals[rid]['ATOMIC'].append(
                        _score(directory/'case_result.json', rid, sample, sandbox))
                    bar.update(1)
            for index, row in mixed_rows:
                rid = row['root_id']
                case = run_case(row, mixed_tests, old['framework'], old['config'],
                                planner, executor, taco.ROOT/'evaluation/mixed'/rid,
                                json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete TACO mixed case: {rid}')
                bar.update(1)
            for index, selection, row in retention_rows:
                rid = row['root_id']
                definition = json.loads(Path(row['fixture_path']).read_text())
                policy = apps_scale._retention_planner(planner, selection, definition)
                case = run_case(row, retention_tests, old['framework'], old['config'],
                                policy, executor, taco.ROOT/'evaluation/retention'/rid,
                                json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']
                                +100000+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete TACO retention case: {rid}')
                bar.update(1)
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(taco.ROOT/'request.json'),
                          'fresh_signals': signals,
                          'mixed_roots': [row['root_id'] for _, row in mixed_rows],
                          'retention_roots': [row['root_id'] for _, _, row in retention_rows]})
    finally:
        reservation.close()


def report():
    from tqdm import tqdm

    taco.check(trained=True)
    if (taco.ROOT/'result.json').exists():
        raise FileExistsError(f'TACO SFT report exists: {taco.ROOT/"result.json"}')
    request_sha = file_hash(taco.ROOT/'request.json')
    baseline_result = json.loads((diverse.ROOT/'result.json').read_text())
    if baseline_result['request_sha256'] != file_hash(diverse.ROOT/'request.json'):
        raise ValueError('Frozen DIVERSE baseline changed')
    old_signals = {rid: signals for i in range(3) for rid, signals in json.loads(
        (diverse.ROOT/f'shard_{i}.json').read_text())['fresh_signals'].items()}
    shards = [json.loads((taco.ROOT/f'shard_{i}.json').read_text())
              for i in range(len(taco.GPUS))]
    new_signals = {}
    for i, shard in enumerate(shards):
        if (shard['status'] != 'COMPLETE' or shard['shard'] != i
                or shard['request_sha256'] != request_sha
                or len(shard['fresh_signals']) != 13
                or set(new_signals) & set(shard['fresh_signals'])):
            raise ValueError(f'Incomplete TACO evaluation shard: {i}')
        new_signals.update(shard['fresh_signals'])
    roots = {row['root_id'] for row in rejudge._rows()}
    if set(old_signals) != roots or set(new_signals) != roots:
        raise ValueError('Missing corrected APPS fresh roots')
    totals = {mode: {arm: {str(k): Counter() for k in (1, 3)}
                     for arm in ('DIVERSE', 'TACO')} for mode in ('DIRECT', 'ATOMIC')}
    paired = {mode: Counter() for mode in ('DIRECT', 'ATOMIC')}
    adapter_sha = file_hash(taco.ROOT/'training/final_adapter/adapter_model.safetensors')
    fresh_indices = {row['root_id']: i for i, row in enumerate(fresh._rows())}
    for rid in tqdm(sorted(roots), desc='TACO/APPS配对汇总', unit='题', dynamic_ncols=True):
        for mode in ('DIRECT', 'ATOMIC'):
            selected = {}
            for arm, all_signals in (('DIVERSE', old_signals), ('TACO', new_signals)):
                signals = all_signals[rid][mode]
                if len(signals) != 3:
                    raise ValueError(f'Missing {mode} candidates: {arm}/{rid}')
                if arm == 'TACO':
                    for repetition, signal in enumerate(signals):
                        case = json.loads(Path(signal['case_path']).read_text())
                        identity = (case['model_identity'] if mode == 'DIRECT'
                                    else case['model_identity']['executor'])
                        if (case['root_id'] != rid or case['status'] != 'COMPLETE'
                                or case['seed'] != fresh.SEED_BASE+100*fresh_indices[rid]+repetition
                                or identity['adapter_weights_sha256'] != adapter_sha):
                            raise ValueError(f'TACO model or seed changed: {mode}/{rid}/{repetition}')
                counts, selected[arm] = diverse._counts(signals)
                for k in (1, 3):
                    totals[mode][arm][str(k)]['roots'] += 1
                    totals[mode][arm][str(k)].update(counts[str(k)])
            before, after = selected['DIVERSE'], selected['TACO']
            paired[mode]['win' if after and not before else
                         'loss' if before and not after else 'tie'] += 1
    mixed_rows = apps_scale._validation_rows()
    retention_rows = apps_scale._retention_rows()
    if ({rid for shard in shards for rid in shard['mixed_roots']}
            != {row['root_id'] for row in mixed_rows}
            or {rid for shard in shards for rid in shard['retention_roots']}
            != {row['root_id'] for _, row in retention_rows}):
        raise ValueError('Missing mixed or retention evaluation roots')
    mixed = {arm: Counter() for arm in ('DIVERSE', 'TACO')}
    mixed_paired = Counter()
    for row in mixed_rows:
        rid = row['root_id']
        before = json.loads((diverse.ROOT/'evaluation/mixed'/rid/'case_result.json').read_text())
        after = json.loads((taco.ROOT/'evaluation/mixed'/rid/'case_result.json').read_text())
        a, b = apps_scale._valid_private(before), apps_scale._valid_private(after)
        for arm, passed in (('DIVERSE', a), ('TACO', b)):
            mixed[arm]['roots'] += 1
            mixed[arm]['strict'] += passed
        mixed_paired['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
    retention = {arm: Counter() for arm in ('DIVERSE', 'TACO')}
    for _, row in retention_rows:
        rid = row['root_id']
        for arm, parent in (('DIVERSE', diverse.ROOT), ('TACO', taco.ROOT)):
            case = json.loads((parent/'evaluation/retention'/rid/'case_result.json').read_text())
            retention[arm]['roots'] += 1
            retention[arm]['private'] += case['final_scores']['private']['status'] == 'PASS'
    trained = json.loads((taco.ROOT/'training/result.json').read_text())
    result = {'status': 'COMPLETE', 'request_sha256': request_sha,
              'baseline_result_sha256': file_hash(diverse.ROOT/'result.json'),
              'fresh_roots': len(roots),
              'fresh': {mode: {arm: {k: dict(v) for k, v in counts.items()}
                               for arm, counts in arms.items()}
                        for mode, arms in totals.items()},
              'paired_TACO_vs_DIVERSE_k3': {mode: dict(value)
                                           for mode, value in paired.items()},
              'mixed': {arm: dict(value) for arm, value in mixed.items()},
              'mixed_paired_TACO_vs_DIVERSE': dict(mixed_paired),
              'retention': {arm: dict(value) for arm, value in retention.items()},
              'training': {key: trained[key] for key in ('status', 'steps',
                           'target_exposures', 'retention_exposures', 'adapter_sha256')}}
    write_json(taco.ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('evaluate', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        evaluate(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
