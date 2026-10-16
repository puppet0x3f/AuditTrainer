"""Replay frozen APPS SFT teachers on their statement examples using CPU workers."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

from . import apps, apps_direct_rejudge as rejudge, apps_diverse_sft as diverse
from .dataset import read_jsonl
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox


ROOT = Path('runs/refinement/apps_teacher_sample_audit')
_SANDBOX = None


def _worker_init(execution):
    global _SANDBOX
    _SANDBOX = Sandbox(ExecutionLimits(**execution))
    if _SANDBOX.bwrap_version == 'UNAVAILABLE':
        raise RuntimeError('Bubblewrap sandbox is unavailable')


def _run(job):
    result = _SANDBOX.run(job['source'], 'solve', [job['sample']['input']])
    actual = result.value if result.status == 'RETURNED' else None
    expected = job['reference_expected']
    shown = job['sample']['expected']
    return {
        'root_id': job['root_id'], 'difficulty': job['difficulty'],
        'reference_alignment': job['reference_alignment'],
        'execution_status': result.status, 'execution_reason': result.reason,
        'exact_reference': isinstance(actual, str) and actual.strip() == expected.strip(),
        'token_reference': isinstance(actual, str) and actual.split() == expected.split(),
        'legacy_reference': apps.compare_stdio(actual, expected),
        'exact_sample': isinstance(actual, str) and actual.strip() == shown.strip(),
        'token_sample': isinstance(actual, str) and actual.split() == shown.split(),
        'actual_excerpt': actual[:500] if isinstance(actual, str) else None,
        'reference_excerpt': expected[:500], 'sample_excerpt': shown[:500],
        'actual_sha256': hashlib.sha256(actual.encode()).hexdigest()
                         if isinstance(actual, str) else None,
    }


def run(workers=3):
    from tqdm import tqdm

    if (ROOT/'result.json').exists():
        raise FileExistsError(f'Audit result already exists: {ROOT/"result.json"}')
    diverse.check(trained=True)
    request = json.loads((diverse.ROOT/'request.json').read_text())
    records = diverse._record_sources() + read_jsonl(diverse.DATA/'new_records.jsonl')
    lookup = {f'apps_train_{int(row["id"]):04d}': row for row in records}
    targets = [row for row in read_jsonl(diverse.DATA/'training.jsonl')
               if row['kind'] == 'target']
    if len(targets) != 1444 or len(lookup) != len(targets):
        raise ValueError('Training targets and source records do not match')
    jobs, unavailable = [], []
    for target in targets:
        root_id = target['root_id']
        record = lookup[root_id]
        sample = rejudge._sample(record['question'])
        if sample is None:
            unavailable.append({'root_id': root_id, 'difficulty': record['difficulty'],
                                'reason': 'NO_PARSED_SAMPLE'})
            continue
        io = record['input_output']
        io = json.loads(io) if isinstance(io, str) else io
        matching = [(apps.stdin_text(left), apps.stdin_text(right))
                    for left, right in zip(io['inputs'], io['outputs'])
                    if apps.stdin_text(left).split() == sample['input'].split()]
        if not matching:
            unavailable.append({'root_id': root_id, 'difficulty': record['difficulty'],
                                'reason': 'SAMPLE_INPUT_NOT_IN_ORIGINAL_TESTS'})
            continue
        # The original IO tests already validated this teacher using APPS's
        # legacy comparator. This replay identifies stricter discrepancies.
        expected = matching[0][1]
        alignment = ('AGREES' if all(value.strip() == sample['expected'].strip()
                                     for _, value in matching) else 'DISAGREES')
        jobs.append({'root_id': root_id, 'difficulty': record['difficulty'],
                     'source': target['target_action']['args']['source'],
                     'sample': sample, 'reference_expected': expected,
                     'reference_alignment': alignment})
    results = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init,
                             initargs=(request['framework']['execution'],)) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='APPS教师公开样例复核', unit='题', dynamic_ncols=True):
            results.append(future.result())
    results.sort(key=lambda row: row['root_id'])
    counts = Counter()
    by_difficulty = {name: Counter() for name in ('introductory', 'interview', 'competition')}
    for row in results:
        for bucket in (counts, by_difficulty[row['difficulty']]):
            bucket['tested'] += 1
            bucket['alignment_'+row['reference_alignment']] += 1
            bucket['execution_'+row['execution_status']] += 1
            bucket['exact_reference'] += row['exact_reference']
            bucket['token_reference'] += row['token_reference']
            bucket['legacy_reference'] += row['legacy_reference']
            bucket['exact_sample'] += row['exact_sample']
            bucket['trusted_exact'] += (row['reference_alignment'] == 'AGREES'
                                        and row['exact_sample'])
            bucket['trusted_token'] += (row['reference_alignment'] == 'AGREES'
                                        and row['token_sample'])
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {'status': 'COMPLETE', 'teachers': len(targets),
              'unavailable': unavailable, 'counts': dict(counts),
              'by_difficulty': {key: dict(value) for key, value in by_difficulty.items()},
              'results': results,
              'interpretation': 'Only AGREES samples have independently aligned statement and APPS labels; '
                                'an exact mismatch alone need not imply an incorrect program on multi-output tasks.'}
    write_json(ROOT/'result.json', report)
    print(json.dumps({'teachers': len(targets), 'unavailable': len(unavailable),
                      'counts': report['counts'], 'by_difficulty': report['by_difficulty']},
                     ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('At least one CPU worker is required')
    run(args.workers)


if __name__ == '__main__':
    main()
