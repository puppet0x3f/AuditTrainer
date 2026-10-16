"""Competition-enriched APPS executor SFT against the frozen LARGE run.

prepare/check are CPU-only. The user launches train and three GPU evaluation
shards after reviewing the prepared data and configuration.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from . import apps_fresh_select as fresh, apps_multitask as previous, apps_scale
from .apps_passk_select import choose
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .prompts import PromptOverflow
from .verifier import canonical


DATA = Path('data/refinement/apps_competition_sft')
ROOT = Path('runs/refinement/apps_competition_sft')
STEPS = 384
SEED = 20260926
SLOTS = {'competition': 256, 'interview': 432, 'introductory': 80}
GPUS = ('0', '1', '2')


def _rank(row):
    return hashlib.sha256(f'{SEED}:{row["sample_id"]}'.encode()).hexdigest()


def _old_targets():
    rows = read_jsonl(apps_scale.DATA/'LARGE/training.jsonl')
    targets = [row for row in rows if row['kind'] == 'target']
    retention = [row for row in rows if row['kind'] == 'retention']
    records = (read_jsonl(previous.DATA/'train_records.jsonl')
               + read_jsonl(apps_scale.DATA/'LARGE/extra_records.jsonl'))
    difficulties = {f'apps_train_{int(row["id"]):04d}': row['difficulty'] for row in records}
    if len(targets) != 768 or len(retention) != 177 or len(difficulties) != 768:
        raise ValueError('Frozen LARGE training examples are incomplete')
    if {row['root_id'] for row in targets} != set(difficulties):
        raise ValueError('Frozen LARGE targets differ from source records')
    return targets, retention, difficulties


def _schedule(old_targets, new_targets, retention, difficulties):
    buckets = {difficulty: [] for difficulty in SLOTS}
    for row in old_targets + new_targets:
        buckets[difficulties.get(row['root_id'], 'competition')].append(row)
    if (len(buckets['competition']) < 192
            or len(buckets['interview']) < SLOTS['interview']
            or len(buckets['introductory']) < SLOTS['introductory']):
        raise ValueError('Insufficient distinct verified APPS problems for the training mix')
    competition = sorted(buckets['competition'], key=_rank)
    competition = [competition[i % len(competition)] for i in range(SLOTS['competition'])]
    other = [row for difficulty in ('interview', 'introductory')
             for row in sorted(buckets[difficulty], key=_rank)[:SLOTS[difficulty]]]
    other.sort(key=_rank)
    competition_iter, other_iter = iter(competition), iter(other)
    targets = []
    for step in range(STEPS):
        if step % 3 == 2:
            targets.extend((next(other_iter), next(other_iter)))
        else:
            targets.extend((next(competition_iter), next(other_iter)))
    return targets + retention


def _paths():
    return {
        'source': previous.SOURCE,
        'large_request': apps_scale.ROOT/'request.json',
        'large_result': apps_scale.ROOT/'result.json',
        'large_training': apps_scale.DATA/'LARGE/training.jsonl',
        'old_train_records': previous.DATA/'train_records.jsonl',
        'extra_records': apps_scale.DATA/'LARGE/extra_records.jsonl',
        'fresh_selection': fresh.DATA/'selection.json',
        'fresh_result': fresh.ROOT/'result.json',
        'new_records': DATA/'new_competition_records.jsonl',
        'new_targets': DATA/'new_competition_targets.jsonl',
        'training': DATA/'training.jsonl',
    }


def prepare():
    from tqdm import tqdm

    if DATA.exists() or ROOT.exists():
        raise FileExistsError('Competition SFT is already prepared')
    apps_scale.check(trained=True)
    fresh.check()
    if file_hash(previous.SOURCE) != previous.SOURCE_SHA256:
        raise ValueError('Pinned APPS train source changed')
    old_targets, retention, difficulties = _old_targets()
    heldout = read_jsonl(fresh.DATA/'selected_records.jsonl')
    excluded_ids = {int(rid.removeprefix('apps_train_')) for rid in difficulties}
    excluded_ids |= {int(row['id']) for row in read_jsonl(previous.DATA/'validation_records.jsonl') + heldout}
    excluded_urls = {row['url'] for row in (
        read_jsonl(previous.DATA/'train_records.jsonl')
        + read_jsonl(previous.DATA/'validation_records.jsonl')
        + read_jsonl(apps_scale.DATA/'LARGE/extra_records.jsonl') + heldout) if row.get('url')}
    candidates = [row for row in read_jsonl(previous.SOURCE)
                  if row.get('difficulty') == 'competition'
                  and int(row['id']) not in excluded_ids
                  and (not row.get('url') or row['url'] not in excluded_urls)
                  and isinstance(row.get('question'), str)
                  and len(row['question']) <= 9000]
    candidates.sort(key=lambda row: hashlib.sha256(f'{SEED}:{row["id"]}'.encode()).hexdigest())
    tokenizer = previous._tokenizer(json.loads((apps_scale.ROOT/'request.json').read_text())['framework'],
                                    json.loads((apps_scale.ROOT/'request.json').read_text())['adapters'])
    framework = json.loads((apps_scale.ROOT/'request.json').read_text())['framework']
    new_records, new_targets = [], []
    with tempfile.TemporaryDirectory(prefix='apps_competition_teacher_') as temp:
        for record in tqdm(candidates, desc='APPS 新竞赛题参考解验证', unit='题', dynamic_ncols=True):
            try:
                with tempfile.TemporaryDirectory(dir=temp) as one:
                    target = previous._target(record, 'TRAIN', framework, tokenizer, Path(one))
            except (ValueError, KeyError, TypeError, AttributeError, PromptOverflow, json.JSONDecodeError):
                target = None
            if target is not None:
                new_records.append(record)
                new_targets.append(target)
    if len(new_targets) < 64:
        raise ValueError(f'Only {len(new_targets)} new verified competition problems; do not train on repeats')
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    (DATA/'new_competition_records.jsonl').write_text(''.join(canonical(x)+'\n' for x in new_records))
    (DATA/'new_competition_targets.jsonl').write_text(''.join(canonical(x)+'\n' for x in new_targets))
    training = _schedule(old_targets, new_targets, retention, difficulties)
    (DATA/'training.jsonl').write_text(''.join(canonical(x)+'\n' for x in training))
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'framework': scale['framework'], 'initial_executor': scale['initial_executor'],
        'adapters': scale['adapters'], 'adapter_sha256': scale['adapter_sha256'],
        'steps': STEPS, 'seed': scale['training_seed'], 'learning_rate': scale['learning_rate'],
        'schedule_seed': SEED, 'slots': SLOTS, 'new_verified_competition': len(new_targets),
        'unique_competition': len(old_targets) - 640 + len(new_targets),
        'retention_examples': len(retention), 'candidates_attempted': len(candidates),
        'source_sha256': {name: file_hash(path) for name, path in _paths().items()},
        'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS), 'fresh_roots': 48,
        'fresh_candidates_per_root': 3,
        'design': 'Same 384 updates, initial executor, learning rate, seed, and retention schedule as LARGE; '
                  'one competition target in two of every three updates. Freeze all planner adapters '
                  'and the 48 fresh APPS roots. One final checkpoint, no selection on validation.',
    })
    write_json(ROOT/'gpu_resources_train.json', {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}})
    write_json(ROOT/'gpu_resources_eval.json', {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check(*, trained=False):
    apps_scale.check(trained=True)
    fresh.check()
    request = json.loads((ROOT/'request.json').read_text())
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    if (request['status'] != 'PREPARED' or request['framework'] != scale['framework']
            or request['initial_executor'] != scale['initial_executor']
            or request['adapters'] != scale['adapters']
            or request['adapter_sha256'] != scale['adapter_sha256']
            or request['steps'] != STEPS or request['seed'] != scale['training_seed']
            or request['learning_rate'] != scale['learning_rate']
            or request['schedule_seed'] != SEED or request['slots'] != SLOTS
            or request['train_gpu'] != GPUS[0] or request['evaluation_gpus'] != list(GPUS)
            or request['fresh_roots'] != 48 or request['fresh_candidates_per_root'] != 3
            or request['source_sha256'] != {name: file_hash(path) for name, path in _paths().items()}
            or json.loads((ROOT/'gpu_resources_train.json').read_text()) !=
               {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}}
            or json.loads((ROOT/'gpu_resources_eval.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('Competition SFT design or launch configuration changed')
    old_targets, retention, difficulties = _old_targets()
    new_records = read_jsonl(DATA/'new_competition_records.jsonl')
    new_targets = read_jsonl(DATA/'new_competition_targets.jsonl')
    training = read_jsonl(DATA/'training.jsonl')
    fresh_ids = {int(row['id']) for row in read_jsonl(fresh.DATA/'selected_records.jsonl')}
    old_validation = {int(row['id']) for row in read_jsonl(previous.DATA/'validation_records.jsonl')}
    new_ids = [int(row['id']) for row in new_records]
    if (len(new_records) != len(new_targets) or len(new_targets) != request['new_verified_competition']
            or len(new_targets) < 64 or len(set(new_ids)) != len(new_ids)
            or set(new_ids) & (fresh_ids | old_validation
                               | {int(rid.removeprefix('apps_train_')) for rid in difficulties})
            or any(row['difficulty'] != 'competition' for row in new_records)
            or [row['root_id'] for row in new_targets] != [f'apps_train_{rid:04d}' for rid in new_ids]
            or request['unique_competition'] != 128 + len(new_targets)
            or request['retention_examples'] != len(retention)
            or training != _schedule(old_targets, new_targets, retention, difficulties)):
        raise ValueError('Training mix, teacher records, or held-out separation changed')
    labels = {**difficulties, **{row['root_id']: 'competition' for row in new_targets}}
    targets = training[:2*STEPS]
    counts = Counter(labels[row['root_id']] for row in targets)
    if (dict(counts) != SLOTS or any(
            [labels[targets[2*i+j]['root_id']] == 'competition' for j in range(2)]
            != ([False, False] if i % 3 == 2 else [True, False]) for i in range(STEPS))):
        raise ValueError('Expected the frozen competition/noncompetition update schedule')
    for row in new_targets:
        if (row['kind'] != 'target' or row['split'] != 'TRAIN'
                or row['prompt_tokens'] + row['target_tokens'] > 8192
                or row['target_tokens'] > 3072):
            raise ValueError(f'Invalid verified target: {row["sample_id"]}')
    if trained:
        result = json.loads((ROOT/'training/result.json').read_text())
        adapter = ROOT/'training/final_adapter/adapter_model.safetensors'
        if (result['status'] != 'COMPLETE' or result['steps'] != STEPS
                or result['target_exposures'] != 2*STEPS
                or result['retention_exposures'] != 2*STEPS
                or result['adapter_sha256'] != file_hash(adapter)):
            raise ValueError('Competition SFT training is incomplete or changed')
    ready = {'status': 'READY', 'new_verified_competition': len(new_targets),
             'unique_competition': 128+len(new_targets), 'target_slots': dict(counts),
             'retention_examples': len(retention), 'steps': STEPS,
             'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS),
             'fresh_rollouts_per_gpu': 48, 'mixed_and_retention_per_gpu': (20, 19, 19),
             'trained': trained}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def train():
    from .gpu_resource import acquire
    from .refinement_executor_sft import train as train_executor

    check()
    reservation = acquire(ROOT/'gpu_resources_train.json', 'train')
    try:
        train_executor(ROOT, DATA, json.loads((ROOT/'request.json').read_text())['initial_executor'], progress=True)
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
    from .apps_passk_select import _score
    from .gpu_resource import acquire
    from .refinement_rollout import run_case
    from .sandbox import ExecutionLimits, Sandbox

    if not 0 <= shard < len(GPUS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check(trained=True)
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed evaluation shard already exists: {done}')
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    fresh_rows, fresh_tests = fresh._rows(), fresh._tests()
    mixed_rows = apps_scale._validation_rows()
    mixed_tests = {row['root_id']: row for row in read_jsonl(previous.DATA/'tests.jsonl')}
    retention_rows = apps_scale._retention_rows()
    retention_tests = {row['root_id']: row for row in read_jsonl(apps_scale.CORRECTION_DATA/'tests.jsonl')}
    fresh_jobs = [(i, row, repeat) for i, row in enumerate(fresh_rows) if i % 3 == shard
                  for repeat in range(3)]
    mixed_jobs = [(i, row) for i, row in enumerate(mixed_rows) if i % 3 == shard]
    retention_jobs = [(i, selection, row) for i, (selection, row) in enumerate(retention_rows)
                      if i % 3 == shard]
    reservation = acquire(ROOT/'gpu_resources_eval.json', 'collect')
    try:
        planner, executor = _context()
        sandbox = Sandbox(ExecutionLimits(**old['framework']['execution']))
        signals = {}
        with tqdm(total=len(fresh_jobs)+len(mixed_jobs)+len(retention_jobs),
                  desc=f'APPS竞赛SFT GPU {GPUS[shard]}', unit='轨迹', position=shard,
                  dynamic_ncols=True) as bar:
            for index, row, repeat in fresh_jobs:
                rid = row['root_id']
                seed = fresh.SEED_BASE + 100*index + repeat
                directory = ROOT/'evaluation/fresh'/rid/f'seed_{seed}'
                case = run_case(row, fresh_tests, old['framework'], old['config'],
                                previous._ForcedRoot(planner, 'ATOMIC'), executor, directory, seed,
                                reset_seed_after_root_plan=True)
                if case['status'] != 'COMPLETE' or case['registered_plan_modes'] != {'ATOMIC': 1}:
                    raise RuntimeError(f'Incomplete fresh ATOMIC rollout: {rid}/{repeat}')
                sample = fresh._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                signals.setdefault(rid, []).append(_score(directory/'case_result.json', rid, sample, sandbox))
                bar.update(1)
            for index, row in mixed_jobs:
                rid = row['root_id']
                case = run_case(row, mixed_tests, old['framework'], old['config'], planner, executor,
                                ROOT/'evaluation/mixed'/rid, json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete mixed rollout: {rid}')
                bar.update(1)
            for index, selection, row in retention_jobs:
                rid = row['root_id']
                definition = json.loads(Path(row['fixture_path']).read_text())
                policy = apps_scale._retention_planner(planner, selection, definition)
                case = run_case(row, retention_tests, old['framework'], old['config'], policy, executor,
                                ROOT/'evaluation/retention'/rid,
                                json.loads((apps_scale.ROOT/'request.json').read_text())['eval_seed']+100000+index)
                if case['status'] != 'COMPLETE':
                    raise RuntimeError(f'Incomplete retention rollout: {rid}')
                bar.update(1)
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'),
                          'fresh_signals': signals, 'mixed_roots': [row['root_id'] for _, row in mixed_jobs],
                          'retention_roots': [row['root_id'] for _, _, row in retention_jobs]})
    finally:
        reservation.close()


def report():
    check(trained=True)
    shards = [json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(3)]
    if any(row['status'] != 'COMPLETE' or row['shard'] != i
           or row['request_sha256'] != file_hash(ROOT/'request.json')
           for i, row in enumerate(shards)):
        raise ValueError('Incomplete or mismatched evaluation shards')
    fresh_rows = fresh._rows()
    old_fresh = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    new_fresh = {rid: signals for shard in shards for rid, signals in shard['fresh_signals'].items()}
    if set(new_fresh) != {row['root_id'] for row in fresh_rows}:
        raise ValueError('Missing fresh APPS roots')
    fresh_counts = {arm: {str(k): Counter() for k in (1, 3)} for arm in ('LARGE', 'COMPETITION')}
    paired = Counter()
    adapter_sha = file_hash(ROOT/'training/final_adapter/adapter_model.safetensors')
    for index, row in enumerate(fresh_rows):
        rid = row['root_id']
        group = old_fresh[rid]['group']
        signals_by_arm = {'LARGE': old_fresh[rid]['candidate_signals'],
                          'COMPETITION': new_fresh[rid]}
        outcomes = {}
        for arm, signals in signals_by_arm.items():
            if len(signals) != 3:
                raise ValueError(f'Missing candidates: {arm}/{rid}')
            cases = [json.loads(Path(signal['case_path']).read_text()) for signal in signals]
            if arm == 'COMPETITION' and any(
                    case['seed'] != fresh.SEED_BASE+100*index+repeat
                    or case['model_identity']['executor']['adapter_weights_sha256'] != adapter_sha
                    or case['registered_plan_modes'] != {'ATOMIC': 1}
                    for repeat, case in enumerate(cases)):
                raise ValueError(f'Candidate model, seed, or plan changed: {rid}')
            strict = [apps_scale._valid_private(case)
                      and (group == 'no_sample' or bool(signal['sample_pass']))
                      for case, signal in zip(cases, signals)]
            private = [apps_scale._valid_private(case) for case in cases]
            outcomes[arm] = strict
            for k in (1, 3):
                chosen = choose(signals, k, group == 'sample')
                cost = [case['actual_model_cost'] for case in cases[:k]]
                counts = fresh_counts[arm][str(k)]
                counts['roots'] += 1
                counts['strict'] += strict[chosen]
                counts['private'] += private[chosen]
                counts['oracle_strict'] += any(strict[:k])
                counts['input_tokens'] += sum(item['input_tokens'] for item in cost)
                counts['output_tokens'] += sum(item['output_tokens'] for item in cost)
                counts['generation_calls'] += sum(item['generation_calls'] for item in cost)
        old_choice = choose(signals_by_arm['LARGE'], 3, group == 'sample')
        new_choice = choose(signals_by_arm['COMPETITION'], 3, group == 'sample')
        a, b = outcomes['LARGE'][old_choice], outcomes['COMPETITION'][new_choice]
        paired['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
    mixed_rows = apps_scale._validation_rows()
    mixed_roots = {rid for shard in shards for rid in shard['mixed_roots']}
    retention_rows = apps_scale._retention_rows()
    retention_roots = {rid for shard in shards for rid in shard['retention_roots']}
    if (mixed_roots != {row['root_id'] for row in mixed_rows}
            or retention_roots != {row['root_id'] for _, row in retention_rows}):
        raise ValueError('Missing mixed or retention roots')
    mixed = {arm: Counter() for arm in ('LARGE', 'COMPETITION')}
    mixed_by_difficulty = {arm: {difficulty: Counter() for difficulty in SLOTS}
                           for arm in ('LARGE', 'COMPETITION')}
    for row in mixed_rows:
        rid = row['root_id']
        difficulty = json.loads(Path(row['raw_path']).read_text())['difficulty']
        old_case = json.loads(apps_scale._old_case('LARGE', rid).read_text())
        new_case = json.loads((ROOT/'evaluation/mixed'/rid/'case_result.json').read_text())
        for arm, case in (('LARGE', old_case), ('COMPETITION', new_case)):
            passed = apps_scale._valid_private(case)
            mixed[arm]['roots'] += 1
            mixed[arm]['strict'] += passed
            mixed_by_difficulty[arm][difficulty]['roots'] += 1
            mixed_by_difficulty[arm][difficulty]['strict'] += passed
    retention = {arm: Counter() for arm in ('LARGE', 'COMPETITION')}
    for _, row in retention_rows:
        rid = row['root_id']
        old_case = json.loads(apps_scale._case('LARGE', 'RETENTION', rid).read_text())
        new_case = json.loads((ROOT/'evaluation/retention'/rid/'case_result.json').read_text())
        for arm, case in (('LARGE', old_case), ('COMPETITION', new_case)):
            retention[arm]['roots'] += 1
            retention[arm]['private'] += case['final_scores']['private']['status'] == 'PASS'
    trained = json.loads((ROOT/'training/result.json').read_text())
    result = {'status': 'COMPLETE', 'fresh': {arm: {k: dict(v) for k, v in ks.items()}
                                             for arm, ks in fresh_counts.items()},
              'fresh_paired_k3': dict(paired), 'mixed': {k: dict(v) for k, v in mixed.items()},
              'mixed_by_difficulty': {arm: {d: dict(v) for d, v in groups.items()}
                                      for arm, groups in mixed_by_difficulty.items()},
              'retention': {k: dict(v) for k, v in retention.items()},
              'training': {key: trained[key] for key in ('status', 'steps', 'target_exposures',
                                                          'retention_exposures', 'adapter_sha256')}}
    write_json(ROOT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'training'}, ensure_ascii=False, indent=2), flush=True)


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
