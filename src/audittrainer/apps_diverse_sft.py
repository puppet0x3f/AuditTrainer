"""Larger, diverse APPS executor SFT with frozen direct and ATOMIC controls.

Preparation verifies new APPS teachers on CPU. Training uses the existing
single-GPU executor loop; evaluation shards run independently on GPUs 0–2.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile

from . import apps_competition_sft as competition, apps_direct_rejudge as rejudge
from . import apps_direct_weight_compare as direct, apps_fresh_select as fresh
from . import apps_multitask as previous, apps_scale
from .apps_passk_select import _score, choose
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .prompts import PromptOverflow
from .sandbox import ExecutionLimits, Sandbox
from .verifier import canonical


DATA = Path('data/refinement/apps_diverse_sft')
ROOT = Path('runs/refinement/apps_diverse_sft')
NEW_QUOTAS = {'interview': 558, 'introductory': 3}
TOTAL_TARGETS = 1444
STEPS = TOTAL_TARGETS // 2
SEED = 20260927
GPUS = ('0', '1', '2')
_WORKER_FRAMEWORK = None
_WORKER_TOKENIZER = None


def _rank(record):
    return hashlib.sha256(f'{SEED}:{record["id"]}'.encode()).hexdigest()


def _old_data():
    old = read_jsonl(apps_scale.DATA/'LARGE/training.jsonl')
    competition_targets = read_jsonl(competition.DATA/'new_competition_targets.jsonl')
    old_targets = [row for row in old if row['kind'] == 'target']
    retention = [row for row in old if row['kind'] == 'retention']
    if (len(old_targets) != 768 or len(competition_targets) != 115
            or len(retention) != 177):
        raise ValueError('Previously verified APPS targets or retention data changed')
    return old_targets, competition_targets, retention


def _record_sources():
    return (read_jsonl(previous.DATA/'train_records.jsonl')
            + read_jsonl(apps_scale.DATA/'LARGE/extra_records.jsonl')
            + read_jsonl(competition.DATA/'new_competition_records.jsonl'))


def _excluded():
    records = _record_sources() + read_jsonl(previous.DATA/'validation_records.jsonl')
    records += read_jsonl(fresh.DATA/'selected_records.jsonl')
    return ({int(row['id']) for row in records},
            {row['url'] for row in records if row.get('url')})


def _pool():
    excluded_ids, excluded_urls = _excluded()
    buckets = {name: [] for name in NEW_QUOTAS}
    for record in read_jsonl(previous.SOURCE):
        difficulty = record.get('difficulty')
        if (difficulty not in buckets or int(record['id']) in excluded_ids
                or (record.get('url') and record['url'] in excluded_urls)
                or not isinstance(record.get('question'), str)
                or len(record['question']) > 9000):
            continue
        buckets[difficulty].append(record)
    for difficulty, values in buckets.items():
        values.sort(key=_rank)
        if len(values) < NEW_QUOTAS[difficulty]:
            raise ValueError(f'Insufficient unused APPS {difficulty} records')
    return buckets


def _init_worker(framework):
    from transformers import AutoTokenizer

    global _WORKER_FRAMEWORK, _WORKER_TOKENIZER
    _WORKER_FRAMEWORK = framework
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        framework['model']['local_path'], local_files_only=True)


def _verify_worker(record):
    with tempfile.TemporaryDirectory(prefix='apps_diverse_teacher_') as scratch:
        try:
            return previous._target(record, 'TRAIN', _WORKER_FRAMEWORK,
                                    _WORKER_TOKENIZER, Path(scratch))
        except (ValueError, KeyError, TypeError, AttributeError,
                PromptOverflow, json.JSONDecodeError):
            return None


def _schedule(targets, retention):
    ordered = sorted(targets, key=lambda row: hashlib.sha256(
        f'{SEED}:{row["sample_id"]}'.encode()).hexdigest())
    return ordered + retention


def _jsonl_line(row):
    # dataset.read_jsonl uses str.splitlines(), so escape Unicode line separators.
    return json.dumps(row, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False)+'\n'


def _paths():
    return {
        'source': previous.SOURCE,
        'scale_request': apps_scale.ROOT/'request.json',
        'scale_result': apps_scale.ROOT/'result.json',
        'old_training': apps_scale.DATA/'LARGE/training.jsonl',
        'competition_request': competition.ROOT/'request.json',
        'competition_result': competition.ROOT/'result.json',
        'competition_targets': competition.DATA/'new_competition_targets.jsonl',
        'old_train_records': previous.DATA/'train_records.jsonl',
        'old_extra_records': apps_scale.DATA/'LARGE/extra_records.jsonl',
        'competition_records': competition.DATA/'new_competition_records.jsonl',
        'old_validation_records': previous.DATA/'validation_records.jsonl',
        'fresh_heldout_records': fresh.DATA/'selected_records.jsonl',
        'fresh_rejudge_result': rejudge.ROOT/'result.json',
        'new_records': DATA/'new_records.jsonl',
        'new_targets': DATA/'new_targets.jsonl',
        'training': DATA/'training.jsonl',
    }


def prepare():
    from tqdm import tqdm

    if DATA.exists() or ROOT.exists():
        raise FileExistsError('Diverse APPS SFT already prepared')
    apps_scale.check(trained=True)
    competition.check(trained=True)
    rejudge.check()
    if file_hash(previous.SOURCE) != previous.SOURCE_SHA256:
        raise ValueError('Pinned APPS train source changed')
    old_targets, competition_targets, retention = _old_data()
    buckets = _pool()
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    framework = scale['framework']
    # Check adapter/tokenizer agreement once before starting parallel CPU work.
    previous._tokenizer(framework, scale['adapters'])
    selected, targets, attempts = [], [], Counter()
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(framework,)) as workers, tqdm(
            total=sum(NEW_QUOTAS.values()), desc='APPS多任务参考解验证',
            unit='题', dynamic_ncols=True) as bar:
        for difficulty, candidates in buckets.items():
            needed = NEW_QUOTAS[difficulty]
            for start in range(0, len(candidates), 48):
                batch = candidates[start:start+48]
                futures = {workers.submit(_verify_worker, row): i for i, row in enumerate(batch)}
                verified = {}
                for future in as_completed(futures):
                    index = futures[future]
                    verified[index] = future.result()
                    attempts[difficulty] += 1
                for index in range(len(batch)):
                    target = verified[index]
                    if target is not None and sum(row['difficulty'] == difficulty for row in selected) < needed:
                        selected.append(batch[index])
                        targets.append(target)
                        bar.update(1)
                bar.set_postfix(difficulty=difficulty, attempted=attempts[difficulty])
                if sum(row['difficulty'] == difficulty for row in selected) == needed:
                    break
            else:
                raise ValueError(f'Only {sum(row["difficulty"] == difficulty for row in selected)} '
                                 f'verified {difficulty} problems; need {needed}')
    all_targets = old_targets + competition_targets + targets
    if (len(all_targets) != TOTAL_TARGETS or len({row['root_id'] for row in all_targets}) != TOTAL_TARGETS):
        raise ValueError(f'Large APPS SFT lacks {TOTAL_TARGETS} distinct verified targets')
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    (DATA/'new_records.jsonl').write_text(''.join(_jsonl_line(row) for row in selected))
    (DATA/'new_targets.jsonl').write_text(''.join(_jsonl_line(row) for row in targets))
    (DATA/'training.jsonl').write_text(''.join(
        _jsonl_line(row) for row in _schedule(all_targets, retention)))
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'framework': framework,
        'initial_executor': scale['initial_executor'],
        'adapters': scale['adapters'], 'adapter_sha256': scale['adapter_sha256'],
        'steps': STEPS, 'learning_rate': scale['learning_rate'],
        'seed': scale['training_seed'], 'schedule_seed': SEED,
        'new_quotas': NEW_QUOTAS, 'new_attempts': dict(attempts),
        'old_targets': len(old_targets), 'competition_targets': len(competition_targets),
        'target_examples': TOTAL_TARGETS, 'retention_examples': len(retention),
        'fresh_benchmark_roots': 39, 'mixed_validation_roots': 48, 'retention_roots': 10,
        'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS),
        'source_sha256': {name: file_hash(path) for name, path in _paths().items()},
        'design': f'One verified APPS target per problem, {TOTAL_TARGETS} distinct targets once, '
                  'two targets plus two fixed-cycle retention examples per update. '
                  'Same initial executor, learning rate, action format, and frozen planner as LARGE.',
    })
    write_json(ROOT/'gpu_resources_train.json',
               {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}})
    write_json(ROOT/'gpu_resources_eval.json',
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check(*, trained=False):
    request = json.loads((ROOT/'request.json').read_text())
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    if (request['status'] != 'PREPARED' or request['framework'] != scale['framework']
            or request['initial_executor'] != scale['initial_executor']
            or request['adapters'] != scale['adapters']
            or request['adapter_sha256'] != scale['adapter_sha256']
            or request['steps'] != STEPS or request['learning_rate'] != scale['learning_rate']
            or request['seed'] != scale['training_seed'] or request['schedule_seed'] != SEED
            or request['new_quotas'] != NEW_QUOTAS
            or request['old_targets'] != 768 or request['competition_targets'] != 115
            or request['target_examples'] != TOTAL_TARGETS
            or request['retention_examples'] != 177
            or request['fresh_benchmark_roots'] != 39
            or request['mixed_validation_roots'] != 48
            or request['retention_roots'] != 10
            or request['train_gpu'] != GPUS[0]
            or request['evaluation_gpus'] != list(GPUS)
            or request['source_sha256'] != {name: file_hash(path) for name, path in _paths().items()}
            or json.loads((ROOT/'gpu_resources_train.json').read_text()) !=
               {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}}
            or json.loads((ROOT/'gpu_resources_eval.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('Frozen diverse APPS SFT design or launch configuration changed')
    old_targets, competition_targets, retention = _old_data()
    records = read_jsonl(DATA/'new_records.jsonl')
    new_targets = read_jsonl(DATA/'new_targets.jsonl')
    training = read_jsonl(DATA/'training.jsonl')
    ids, urls = _excluded()
    new_ids = [int(row['id']) for row in records]
    all_targets = old_targets + competition_targets + new_targets
    if (len(records) != sum(NEW_QUOTAS.values()) or len(records) != len(new_targets)
            or len(set(new_ids)) != len(new_ids) or set(new_ids) & ids
            or any(row.get('url') in urls for row in records if row.get('url'))
            or Counter(row['difficulty'] for row in records) != NEW_QUOTAS
            or [row['root_id'] for row in new_targets] !=
               [f'apps_train_{rid:04d}' for rid in new_ids]
            or len(all_targets) != TOTAL_TARGETS
            or len({row['root_id'] for row in all_targets}) != TOTAL_TARGETS
            or any(row['kind'] != 'target' or row['split'] != 'TRAIN'
                   or row['prompt_tokens'] + row['target_tokens'] > 8192
                   or row['target_tokens'] > 3072 for row in new_targets)
            or training != _schedule(all_targets, retention)):
        raise ValueError('Verified targets, held-out separation, or training order changed')
    if trained:
        result = json.loads((ROOT/'training/result.json').read_text())
        adapter = ROOT/'training/final_adapter/adapter_model.safetensors'
        if (result['status'] != 'COMPLETE' or result['steps'] != STEPS
                or result['target_exposures'] != TOTAL_TARGETS
                or result['retention_exposures'] != TOTAL_TARGETS
                or result['adapter_sha256'] != file_hash(adapter)):
            raise ValueError('Diverse APPS executor training is incomplete or changed')
    ready = {'status': 'READY', 'distinct_targets': TOTAL_TARGETS,
             'new_verified': dict(NEW_QUOTAS), 'new_attempts': request['new_attempts'],
             'steps': STEPS, 'retention_examples': len(retention),
             'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS),
             'fresh_roots': 39, 'mixed_roots': 48, 'retention_roots': 10,
             'trained': trained}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def train():
    from .gpu_resource import acquire
    from .refinement_executor_sft import train as train_executor

    check()
    reservation = acquire(ROOT/'gpu_resources_train.json', 'train')
    try:
        train_executor(ROOT, DATA, json.loads((ROOT/'request.json').read_text())['initial_executor'],
                       progress=True)
    finally:
        reservation.close()


def _context():
    from .refinement_policy import load_policies

    request = json.loads((ROOT/'request.json').read_text())
    base, executor = load_policies(request['framework'], request['adapters']['internal_gate'],
                                   str(ROOT/'training/final_adapter'))
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

    if not 0 <= shard < len(GPUS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check(trained=True)
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed diverse evaluation shard already exists: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    all_fresh = fresh._rows()
    fresh_indices = {row['root_id']: i for i, row in enumerate(all_fresh)}
    fresh_rows = rejudge._rows()[shard::len(GPUS)]
    fresh_tests = fresh._tests()
    mixed_rows = [(i, row) for i, row in enumerate(apps_scale._validation_rows())
                  if i % len(GPUS) == shard]
    mixed_tests = {row['root_id']: row for row in read_jsonl(previous.DATA/'tests.jsonl')}
    retention_rows = [(i, selection, row) for i, (selection, row) in
                      enumerate(apps_scale._retention_rows()) if i % len(GPUS) == shard]
    retention_tests = {row['root_id']: row for row in
                       read_jsonl(apps_scale.CORRECTION_DATA/'tests.jsonl')}
    sandbox = Sandbox(ExecutionLimits(**request['framework']['execution']))
    reservation = acquire(ROOT/'gpu_resources_eval.json', 'collect')
    try:
        signals = {row['root_id']: {'DIRECT': [], 'ATOMIC': []} for row in fresh_rows}
        with tqdm(total=len(fresh_rows)*6+len(mixed_rows)+len(retention_rows),
                  desc=f'APPS扩量评测 GPU {GPUS[shard]}', unit='轨迹', position=shard,
                  dynamic_ncols=True) as bar:
            # Use the same single-adapter model path as the earlier direct arms.
            llm = LLMPolicy.from_pretrained(
                request['framework'], adapter_path=ROOT/'training/final_adapter')
            if llm.identity['adapter_weights_sha256'] != file_hash(
                    ROOT/'training/final_adapter/adapter_model.safetensors'):
                raise ValueError('Direct model differs from completed diverse SFT')
            for row in fresh_rows:
                rid = row['root_id']
                sample = rejudge._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                for repetition in range(3):
                    seed = fresh.SEED_BASE+100*fresh_indices[rid]+repetition
                    directory = ROOT/'evaluation/direct'/rid/f'seed_{seed}'
                    case = direct._direct_case(row, fresh_tests, request['framework'], llm,
                                               'DIVERSE', directory, seed)
                    if case['status'] != 'COMPLETE':
                        raise RuntimeError(f'Incomplete diverse direct candidate: {rid}/{repetition}')
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
                    directory = ROOT/'evaluation/atomic'/rid/f'seed_{seed}'
                    case = run_case(row, fresh_tests, old['framework'], old['config'],
                                    previous._ForcedRoot(planner, 'ATOMIC'), executor,
                                    directory, seed, reset_seed_after_root_plan=True)
                    if case['status'] != 'COMPLETE' or case['registered_plan_modes'] != {'ATOMIC': 1}:
                        raise RuntimeError(f'Incomplete diverse ATOMIC candidate: {rid}/{repetition}')
                    signals[rid]['ATOMIC'].append(
                        _score(directory/'case_result.json', rid, sample, sandbox))
                    bar.update(1)
            for index, row in mixed_rows:
                rid = row['root_id']
                case = run_case(row, mixed_tests, old['framework'], old['config'],
                                planner, executor, ROOT/'evaluation/mixed'/rid,
                                json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete diverse mixed case: {rid}')
                bar.update(1)
            for index, selection, row in retention_rows:
                rid = row['root_id']
                definition = json.loads(Path(row['fixture_path']).read_text())
                policy = apps_scale._retention_planner(planner, selection, definition)
                case = run_case(row, retention_tests, old['framework'], old['config'],
                                policy, executor, ROOT/'evaluation/retention'/rid,
                                json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']
                                +100000+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete diverse retention case: {rid}')
                bar.update(1)
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'),
                          'fresh_signals': signals,
                          'mixed_roots': [row['root_id'] for _, row in mixed_rows],
                          'retention_roots': [row['root_id'] for _, _, row in retention_rows]})
    finally:
        reservation.close()


def _counts(signals):
    cases = [json.loads(Path(signal['case_path']).read_text()) for signal in signals]
    private = [apps_scale._valid_private(case) for case in cases]
    strict = [passed and bool(signal['sample_pass'])
              for passed, signal in zip(private, signals)]
    result = {}
    for k in (1, 3):
        chosen = choose(signals, k, True)
        result[str(k)] = {'strict': int(strict[chosen]), 'private': int(private[chosen]),
                          'oracle_strict': int(any(strict[:k])),
                          'sample_pass_candidates': sum(bool(s['sample_pass']) for s in signals[:k])}
    return result, strict[choose(signals, 3, True)]


def report():
    from tqdm import tqdm

    check(trained=True)
    if (ROOT/'result.json').exists():
        raise FileExistsError(f'Diverse SFT report already exists: {ROOT/"result.json"}')
    shards = [json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(len(GPUS))]
    fresh_new = {}
    for i, shard in enumerate(shards):
        if (shard['status'] != 'COMPLETE' or shard['shard'] != i
                or shard['request_sha256'] != file_hash(ROOT/'request.json')
                or len(shard['fresh_signals']) != 13
                or set(fresh_new) & set(shard['fresh_signals'])):
            raise ValueError(f'Incomplete diverse evaluation shard: {i}')
        fresh_new.update(shard['fresh_signals'])
    rows = rejudge._rows()
    if set(fresh_new) != {row['root_id'] for row in rows}:
        raise ValueError('Missing diverse fresh APPS roots')
    prior_direct = {rid: item for i in range(3) for rid, item in json.loads(
        (rejudge.ROOT/f'shard_{i}.json').read_text())['by_root'].items()}
    old_fresh = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    competition_fresh = {rid: values for i in range(3) for rid, values in json.loads(
        (competition.ROOT/f'shard_{i}.json').read_text())['fresh_signals'].items()}
    sandbox = Sandbox(ExecutionLimits(**json.loads((ROOT/'request.json').read_text())['framework']['execution']))
    totals = {mode: {arm: {str(k): Counter() for k in (1, 3)} for arm in arms}
              for mode, arms in {'DIRECT': ('BASE', 'LARGE', 'COMPETITION', 'DIVERSE'),
                                 'ATOMIC': ('LARGE', 'COMPETITION', 'DIVERSE')}.items()}
    paired = {mode: {arm: Counter() for arm in ('BASE', 'LARGE', 'COMPETITION')
                     if arm in totals[mode]} for mode in totals}
    adapter_sha = file_hash(ROOT/'training/final_adapter/adapter_model.safetensors')
    with tqdm(total=len(rows)*6, desc='既有ATOMIC样例复评', unit='候选',
              dynamic_ncols=True) as bar:
        for index, row in enumerate(rows):
            rid = row['root_id']
            sample = rejudge._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
            direct_signals = {arm: prior_direct[rid]['signals'][arm]
                              for arm in ('BASE', 'LARGE', 'COMPETITION')}
            direct_signals['DIVERSE'] = fresh_new[rid]['DIRECT']
            atomic_signals = {
                'LARGE': [_score(Path(signal['case_path']), rid, sample, sandbox)
                          for signal in old_fresh[rid]['candidate_signals']],
                'COMPETITION': [_score(Path(signal['case_path']), rid, sample, sandbox)
                                for signal in competition_fresh[rid]],
                'DIVERSE': fresh_new[rid]['ATOMIC'],
            }
            bar.update(6)
            for mode, arms in (('DIRECT', direct_signals), ('ATOMIC', atomic_signals)):
                selected = {}
                for arm, signals in arms.items():
                    if len(signals) != 3:
                        raise ValueError(f'Missing {mode} candidates: {arm}/{rid}')
                    if arm == 'DIVERSE':
                        for repetition, signal in enumerate(signals):
                            case = json.loads(Path(signal['case_path']).read_text())
                            original_index = next(i for i, r in enumerate(fresh._rows())
                                                  if r['root_id'] == rid)
                            identity = (case['model_identity'] if mode == 'DIRECT'
                                        else case['model_identity']['executor'])
                            if (case['root_id'] != rid or case['status'] != 'COMPLETE'
                                    or case['seed'] != fresh.SEED_BASE+100*original_index+repetition
                                    or identity['adapter_weights_sha256'] != adapter_sha):
                                raise ValueError(f'Diverse model or seed changed: {mode}/{rid}/{repetition}')
                    counts, selected[arm] = _counts(signals)
                    for k in (1, 3):
                        totals[mode][arm][str(k)]['roots'] += 1
                        totals[mode][arm][str(k)].update(counts[str(k)])
                for arm in paired[mode]:
                    a, b = selected[arm], selected['DIVERSE']
                    paired[mode][arm]['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
    mixed_rows = apps_scale._validation_rows()
    retention_rows = apps_scale._retention_rows()
    if ({rid for shard in shards for rid in shard['mixed_roots']}
            != {row['root_id'] for row in mixed_rows}
            or {rid for shard in shards for rid in shard['retention_roots']}
            != {row['root_id'] for _, row in retention_rows}):
        raise ValueError('Missing mixed or retention evaluation roots')
    previous_result = json.loads((competition.ROOT/'result.json').read_text())
    mixed = {arm: dict(previous_result['mixed'][arm]) for arm in ('LARGE', 'COMPETITION')}
    retention = {arm: dict(previous_result['retention'][arm]) for arm in ('LARGE', 'COMPETITION')}
    mixed['DIVERSE'] = {'roots': len(mixed_rows), 'strict': sum(
        apps_scale._valid_private(json.loads((ROOT/'evaluation/mixed'/row['root_id']/'case_result.json').read_text()))
        for row in mixed_rows)}
    retention['DIVERSE'] = {'roots': len(retention_rows), 'private': sum(
        json.loads((ROOT/'evaluation/retention'/row['root_id']/'case_result.json').read_text())
        ['final_scores']['private']['status'] == 'PASS' for _, row in retention_rows)}
    trained = json.loads((ROOT/'training/result.json').read_text())
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'fresh_roots': len(rows),
              'fresh': {mode: {arm: {k: dict(v) for k, v in counts.items()}
                              for arm, counts in arms.items()} for mode, arms in totals.items()},
              'paired_DIVERSE_vs_baselines_k3': {mode: {arm: dict(v) for arm, v in arms.items()}
                                                 for mode, arms in paired.items()},
              'mixed': mixed, 'retention': retention,
              'training': {key: trained[key] for key in ('status', 'steps', 'target_exposures',
                                                         'retention_exposures', 'adapter_sha256')}}
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'train', 'evaluate', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check(trained=args.trained)
    elif args.command == 'train':
        train()
    elif args.command == 'evaluate':
        evaluate(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
