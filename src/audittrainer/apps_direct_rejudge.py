"""Rejudge saved direct APPS candidates on tasks with unambiguous exact outputs.

The original trajectories and reports remain frozen. This replay recognizes
plural Examples headings, executes the first public example for every retained
task, and selects among the same three saved candidates per model.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from . import apps_direct_weight_compare as direct
from . import apps_fresh_select as fresh, apps_fresh_system_compare as earlier
from . import apps_passk_select, apps_scale
from .dataset import file_hash
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox


ROOT = Path('runs/refinement/apps_direct_rejudge')
ARMS = ('BASE', 'LARGE', 'COMPETITION')
SHARDS = 3
# These statements accept multiple valid outputs, a numerical tolerance, or
# noncanonical capitalization. Their exact-string private scores are unsuitable
# for the headline metric until problem-specific judges are available.
EXCLUDED = {
    'apps_train_2020': 'multiple valid necklaces',
    'apps_train_2022': 'multiple valid matrices',
    'apps_train_2051': 'multiple optimal schedules',
    'apps_train_2111': 'multiple valid skill allocations',
    'apps_train_2197': 'multiple valid palindromes',
    'apps_train_2203': 'multiple valid permutations',
    'apps_train_2273': 'case-insensitive YES/NO output',
    'apps_train_2301': 'floating-point tolerance',
    'apps_train_2330': 'multiple valid trees',
}


def _sources():
    paths = {
        'fresh_selection': fresh.DATA/'selection.json',
        'fresh_roots': fresh.DATA/'roots.jsonl',
        'fresh_tests': fresh.DATA/'tests.jsonl',
        'large_direct_result': earlier.ROOT/'result.json',
        'direct_request': direct.ROOT/'request.json',
        'direct_result': direct.ROOT/'result.json',
    }
    paths.update({f'direct_shard_{i}': direct.ROOT/f'shard_{i}.json' for i in range(SHARDS)})
    return paths


def _sample(spec):
    # Keep the historical parser and its frozen source checks unchanged. Some
    # plural blocks put the next Input immediately after an Output without a
    # blank line; insert a delimiter so only the first pair is selected.
    if '-----Examples-----' not in spec:
        return fresh._sample(spec)
    normalized = spec.replace('-----Examples-----', '-----Example-----')
    normalized = re.sub(r'(?m)^Input[ \t]*$', '\nInput', normalized)
    return fresh._sample(normalized)


def _rows():
    return [row for row in fresh._rows() if row['root_id'] not in EXCLUDED]


def _saved():
    large = json.loads((earlier.ROOT/'result.json').read_text())['by_root']
    shards = [json.loads((direct.ROOT/f'shard_{i}.json').read_text()) for i in range(SHARDS)]
    current = {rid: value for shard in shards for rid, value in shard['by_root'].items()}
    return large, current


def _signals(rid, arm, large, current):
    return large[rid]['signals']['DIRECT'] if arm == 'LARGE' else current[rid]['signals'][arm]


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'APPS direct rejudge already exists: {ROOT}')
    prior = json.loads((direct.ROOT/'result.json').read_text())
    if prior['status'] != 'COMPLETE':
        raise ValueError('Completed direct weight comparison required')
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arms': list(ARMS), 'original_roots': 48,
        'included_roots': 48-len(EXCLUDED), 'excluded': EXCLUDED,
        'candidates_per_root': 3, 'source_sha256': {k: file_hash(v) for k, v in _sources().items()},
        'public_example': 'first statement example, singular or plural heading',
        'metric': 'exact private and first public example on unambiguous-output tasks',
        'cpu_shards': SHARDS, 'gpu_devices': [], 'training_steps': 0,
    })
    return check()


def check():
    direct.check()
    request = json.loads((ROOT/'request.json').read_text())
    direct_result = json.loads((direct.ROOT/'result.json').read_text())
    rows = fresh._rows()
    if (request['status'] != 'PREPARED' or request['arms'] != list(ARMS)
            or request['original_roots'] != 48 or request['included_roots'] != 39
            or request['excluded'] != EXCLUDED or request['candidates_per_root'] != 3
            or request['source_sha256'] != {k: file_hash(v) for k, v in _sources().items()}
            or request['public_example'] != 'first statement example, singular or plural heading'
            or request['metric'] != 'exact private and first public example on unambiguous-output tasks'
            or request['cpu_shards'] != SHARDS or request['gpu_devices'] != []
            or request['training_steps'] != 0 or direct_result['status'] != 'COMPLETE'
            or direct_result['request_sha256'] != file_hash(direct.ROOT/'request.json')
            or len(rows) != 48 or not set(EXCLUDED) <= {row['root_id'] for row in rows}):
        raise ValueError('Saved trajectories or rejudge design changed')
    large, current = _saved()
    if set(large) != set(current) or set(large) != {row['root_id'] for row in rows}:
        raise ValueError('Saved direct candidate roots differ')
    old_missing = 0
    framework = earlier._framework()
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    if sandbox.bwrap_version == 'UNAVAILABLE':
        raise RuntimeError('Bubblewrap sandbox is unavailable')
    for index, row in enumerate(rows):
        rid = row['root_id']
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        sample, old_sample = _sample(spec), fresh._sample(spec)
        if sample is None or (old_sample is not None and sample != old_sample):
            raise ValueError(f'Public example missing or changed: {rid}')
        old_missing += old_sample is None
        for arm in ARMS:
            signals = _signals(rid, arm, large, current)
            if len(signals) != 3:
                raise ValueError(f'Expected three saved candidates: {arm}/{rid}')
            expected_adapter = None if arm == 'BASE' else json.loads(
                (direct.ROOT/'request.json').read_text())['source_sha256'][
                    'large_adapter' if arm == 'LARGE' else 'competition_adapter']
            for repetition, signal in enumerate(signals):
                case = json.loads(Path(signal['case_path']).read_text())
                if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                        or case['seed'] != fresh.SEED_BASE+100*index+repetition
                        or case['model_calls'] != 1
                        or case['model_identity'].get('adapter_weights_sha256') != expected_adapter):
                    raise ValueError(f'Saved direct candidate differs: {arm}/{rid}/{repetition}')
    if old_missing != 16:
        raise ValueError('Expected exactly 16 formerly missed plural Examples headings')
    ready = {'status': 'READY', 'original_roots': 48, 'included_roots': len(_rows()),
             'excluded_roots': len(EXCLUDED), 'newly_recognized_sample_roots': old_missing,
             'saved_candidates': 48*3*len(ARMS), 'public_replays': len(_rows())*3*len(ARMS),
             'public_replays_per_cpu_shard': len(_rows())*3*len(ARMS)//SHARDS,
             'cpu_shards': SHARDS, 'gpu_devices': [], 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def score_shard(shard):
    from tqdm import tqdm

    if not 0 <= shard < SHARDS:
        raise ValueError('Expected CPU shard 0, 1, or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed rejudge shard already exists: {done}')
    large, current = _saved()
    rows = _rows()[shard::SHARDS]
    sandbox = Sandbox(ExecutionLimits(**earlier._framework()['execution']))
    scored = {}
    with tqdm(total=len(rows)*3*len(ARMS), desc=f'APPS样例复评 CPU {shard}',
              unit='候选', position=shard, dynamic_ncols=True) as bar:
        for row in rows:
            rid = row['root_id']
            spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
            sample = _sample(spec)
            scored[rid] = {'signals': {}}
            for arm in ARMS:
                values = []
                for old_signal in _signals(rid, arm, large, current):
                    values.append(apps_passk_select._score(
                        Path(old_signal['case_path']), rid, sample, sandbox))
                    bar.update(1)
                scored[rid]['signals'][arm] = values
    write_json(done, {'status': 'COMPLETE', 'shard': shard,
                      'request_sha256': file_hash(ROOT/'request.json'), 'by_root': scored})


def report():
    check()
    if (ROOT/'result.json').exists():
        raise FileExistsError(f'Completed rejudge already exists: {ROOT/"result.json"}')
    shards = [json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(SHARDS)]
    scored = {}
    for i, shard in enumerate(shards):
        if (shard['status'] != 'COMPLETE' or shard['shard'] != i
                or shard['request_sha256'] != file_hash(ROOT/'request.json')
                or len(shard['by_root']) != 13 or set(scored) & set(shard['by_root'])):
            raise ValueError(f'Incomplete rejudge shard: {i}')
        scored.update(shard['by_root'])
    rows = _rows()
    if set(scored) != {row['root_id'] for row in rows}:
        raise ValueError('Missing rejudged roots')
    large, current = _saved()
    summary = {arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS}
    previous = {arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS}
    paired = {arm: Counter() for arm in ('BASE', 'COMPETITION')}
    for row in rows:
        rid = row['root_id']
        outcomes = {}
        for arm in ARMS:
            old_signals = _signals(rid, arm, large, current)
            new_signals = scored[rid]['signals'][arm]
            if (len(new_signals) != 3 or [s['case_path'] for s in new_signals]
                    != [s['case_path'] for s in old_signals]):
                raise ValueError(f'Rejudged candidate order differs: {arm}/{rid}')
            cases = [json.loads(Path(s['case_path']).read_text()) for s in new_signals]
            private = [apps_scale._valid_private(case) for case in cases]
            strict = [passed and bool(signal['sample_pass'])
                      for passed, signal in zip(private, new_signals)]
            outcomes[arm] = strict
            for k in (1, 3):
                chosen = apps_passk_select.choose(new_signals, k, True)
                old_choice = apps_passk_select.choose(old_signals, k, large[rid]['group'] == 'sample')
                old_strict = private[old_choice] and (large[rid]['group'] == 'no_sample'
                                                      or bool(old_signals[old_choice]['sample_pass']))
                count = summary[arm][str(k)]
                count['roots'] += 1
                count['strict'] += strict[chosen]
                count['private'] += private[chosen]
                count['oracle_strict'] += any(strict[:k])
                count['sample_pass_candidates'] += sum(bool(s['sample_pass']) for s in new_signals[:k])
                previous[arm][str(k)]['strict'] += old_strict
        old_choice = apps_passk_select.choose(scored[rid]['signals']['LARGE'], 3, True)
        for arm in ('BASE', 'COMPETITION'):
            chosen = apps_passk_select.choose(scored[rid]['signals'][arm], 3, True)
            a, b = outcomes['LARGE'][old_choice], outcomes[arm][chosen]
            paired[arm]['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'included_roots': len(rows), 'excluded': EXCLUDED,
              'summary': {arm: {k: dict(v) for k, v in ks.items()} for arm, ks in summary.items()},
              'previous_on_same_roots': {arm: {k: dict(v) for k, v in ks.items()}
                                         for arm, ks in previous.items()},
              'paired_vs_LARGE_k3': {arm: dict(v) for arm, v in paired.items()},
              'metric': json.loads((ROOT/'request.json').read_text())['metric']}
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'score', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'score':
        score_shard(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
