"""Fresh APPS competition transfer test for public-example candidate selection.

The source task split and selector are frozen before inference. Each task has at
least two original APPS I/O cases held out from the public statement example.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import apps, apps_multitask, apps_passk_select, apps_scale
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox
from .verifier import canonical, digest


DATA = Path('data/refinement/apps_fresh_select')
ROOT = Path('runs/refinement/apps_fresh_select')
GPU_IDS = ('0', '1', '2')
QUOTAS = {'sample': 32, 'no_sample': 16}
CANDIDATES = 3
RANK_SEED = 20260924
SEED_BASE = 717000713
EXCLUSION_FILES = (apps_multitask.DATA/'train_records.jsonl',
                   apps_multitask.DATA/'validation_records.jsonl',
                   apps_scale.DATA/'LARGE/extra_records.jsonl')


def _same_input(left, right):
    return left.split() == right.split()


def _shown_in_statement(text, spec):
    return ' '.join(text.split()) in ' '.join(spec.split())


def _sample(spec):
    example = apps_passk_select.public_example(spec, stop_at_blank_output=True)
    if example is None:
        return None
    # Statement formatting often inserts blank lines between input lines.
    return {'input': '\n'.join(line.strip() for line in example['input'].splitlines()
                               if line.strip()) + '\n', 'expected': example['expected']}


def _excluded():
    ids, urls = set(), set()
    for path in EXCLUSION_FILES:
        for record in read_jsonl(path):
            ids.add(int(record['id']))
            if record.get('url'):
                urls.add(record['url'])
    return ids, urls


def _candidate(record, excluded_ids, excluded_urls):
    if (record.get('difficulty') != 'competition' or int(record.get('id', -1)) in excluded_ids
            or record.get('url') in excluded_urls or len(record.get('question', '')) > 6000
            or not apps_multitask._solutions(record)):
        return None
    try:
        io = record['input_output']
        io = json.loads(io) if isinstance(io, str) else io
        if io.get('fn_name') is not None:
            return None
        inputs = [apps.stdin_text(value) for value in io['inputs']]
        outputs = [apps.stdin_text(value) for value in io['outputs']]
        if len(inputs) < 3 or len(inputs) != len(outputs):
            return None
        sample = _sample(record['question'])
        matched = [index for index, value in enumerate(inputs)
                   if sample is not None and _same_input(value, sample['input'])]
        if sample is not None and (not matched or any(outputs[i].strip() != sample['expected'].strip()
                                                       for i in matched)):
            return None
        removed = [index for index, value in enumerate(inputs)
                   if index in matched or _shown_in_statement(value, record['question'])]
        heldout = [index for index in range(len(inputs)) if index not in removed]
        if len(heldout) < 2:
            return None
        adjusted = {**record, 'input_output': json.dumps({
            'inputs': [inputs[i] for i in heldout], 'outputs': [outputs[i] for i in heldout]},
            ensure_ascii=False)}
        apps.convert_record(adjusted, original_split='train', split='VALIDATION', visible_cases=0)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return {'record': adjusted, 'original_sha256': digest(record),
            'sample_group': 'sample' if sample is not None else 'no_sample',
            'original_case_count': len(inputs), 'removed_public_indices': removed,
            'heldout_indices': heldout}


def _rank(item):
    record = item['record']
    return hashlib.sha256(f'{RANK_SEED}:{record["id"]}'.encode()).hexdigest()


def prepare():
    if DATA.exists() or ROOT.exists():
        raise FileExistsError('Fresh APPS selection already exists; refusing to overwrite')
    if file_hash(apps_multitask.SOURCE) != apps_multitask.SOURCE_SHA256:
        raise ValueError('Pinned APPS train source changed')
    excluded_ids, excluded_urls = _excluded()
    groups = {group: [] for group in QUOTAS}
    for record in read_jsonl(apps_multitask.SOURCE):
        item = _candidate(record, excluded_ids, excluded_urls)
        if item is not None:
            groups[item['sample_group']].append(item)
    if any(len(groups[group]) < count for group, count in QUOTAS.items()):
        raise ValueError(f'Insufficient fresh APPS tasks: { {k: len(v) for k, v in groups.items()} }')
    selected = [item for group, count in QUOTAS.items()
                for item in sorted(groups[group], key=_rank)[:count]]
    selected.sort(key=lambda item: int(item['record']['id']))
    if len({item['record'].get('url') for item in selected}) != len(selected):
        raise ValueError('Fresh APPS selection contains repeated origin URLs')
    DATA.mkdir(parents=True)
    selected_path = DATA/'selected_records.jsonl'
    selected_path.write_text(''.join(canonical(item['record']) + '\n' for item in selected))
    (DATA/'roots.jsonl').write_text('')
    (DATA/'tests.jsonl').write_text('')
    report = apps.import_apps(selected_path, DATA/'tasks', DATA/'roots.jsonl', DATA/'tests.jsonl',
                              original_split='train', split='VALIDATION', source=apps_multitask.SOURCE_URL,
                              visible_cases=0)
    if report['counts'] != {'imported': sum(QUOTAS.values()), 'rejected': 0}:
        raise ValueError(f'Fresh APPS import incomplete: {report["counts"]}')
    write_json(DATA/'selection.json', {'status': 'FROZEN', 'rank_seed': RANK_SEED, 'quotas': QUOTAS,
                                      'source_sha256': apps_multitask.SOURCE_SHA256,
                                      'excluded_record_files': {str(p): file_hash(p) for p in EXCLUSION_FILES},
                                      'selected': [{'root_id': f'apps_train_{int(item["record"]["id"]):04d}',
                                                    'source_record_sha256': item['original_sha256'],
                                                    'sample_group': item['sample_group'],
                                                    'original_case_count': item['original_case_count'],
                                                    'removed_public_indices': item['removed_public_indices'],
                                                    'heldout_indices': item['heldout_indices']}
                                                   for item in selected]})
    adapter = apps_scale.ROOT/'LARGE/training/final_adapter/adapter_model.safetensors'
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {'status': 'PREPARED', 'source_sha256': apps_multitask.SOURCE_SHA256,
                                    'selection_sha256': file_hash(DATA/'selection.json'),
                                    'roots_sha256': file_hash(DATA/'roots.jsonl'),
                                    'tests_sha256': file_hash(DATA/'tests.jsonl'),
                                    'adapter_sha256': file_hash(adapter), 'rank_seed': RANK_SEED,
                                    'seed_base': SEED_BASE, 'quotas': QUOTAS, 'candidates': CANDIDATES,
                                    'root_plan': 'fixed ATOMIC', 'selector': 'public example pass then earliest',
                                    'primary_metric': 'all held-out original APPS cases exact pass',
                                    'evaluation_gpus': GPU_IDS, 'training_steps': 0})
    write_json(ROOT/'gpu_resources.json', {'devices': GPU_IDS, 'min_free_mib': {'collect': 16000}})
    print(json.dumps({'status': 'PREPARED', 'roots': len(selected), 'groups': QUOTAS,
                      'heldout_cases': sum(len(item['heldout_indices']) for item in selected)},
                     ensure_ascii=False), flush=True)


def _rows():
    return sorted(load_root_manifest(DATA/'roots.jsonl').values(), key=lambda row: row['root_id'])


def _tests():
    return {row['root_id']: row for row in read_jsonl(DATA/'tests.jsonl')}


def check():
    request = json.loads((ROOT/'request.json').read_text())
    selection = json.loads((DATA/'selection.json').read_text())
    adapter = apps_scale.ROOT/'LARGE/training/final_adapter/adapter_model.safetensors'
    resources = json.loads((ROOT/'gpu_resources.json').read_text())
    if (request['status'] != 'PREPARED' or selection['status'] != 'FROZEN'
            or file_hash(apps_multitask.SOURCE) != request['source_sha256']
            or request['source_sha256'] != apps_multitask.SOURCE_SHA256
            or request['selection_sha256'] != file_hash(DATA/'selection.json')
            or request['roots_sha256'] != file_hash(DATA/'roots.jsonl')
            or request['tests_sha256'] != file_hash(DATA/'tests.jsonl')
            or request['adapter_sha256'] != file_hash(adapter)
            or request['quotas'] != QUOTAS or request['candidates'] != CANDIDATES
            or request['seed_base'] != SEED_BASE or request['evaluation_gpus'] != list(GPU_IDS)
            or resources != {'devices': list(GPU_IDS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('Frozen fresh APPS design or launch configuration changed')
    if selection['excluded_record_files'] != {str(p): file_hash(p) for p in EXCLUSION_FILES}:
        raise ValueError('Source exclusion list changed')
    rows, tests = _rows(), _tests()
    selected = {item['root_id']: item for item in selection['selected']}
    if len(rows) != 48 or len(tests) != 48 or set(selected) != {r['root_id'] for r in rows}:
        raise ValueError('Expected 48 frozen fresh roots')
    excluded_ids, excluded_urls = _excluded()
    counts, heldout_total = Counter(), 0
    for row in rows:
        rid = row['root_id']
        item = selected[rid]
        raw = json.loads(Path(row['raw_path']).read_text())
        fixture = json.loads(Path(row['fixture_path']).read_text())
        suite = json.loads(Path(tests[rid]['path']).read_text())['suites'][0]
        source_id = int(raw['id'])
        if (source_id in excluded_ids or raw.get('url') in excluded_urls
                or file_hash(row['fixture_path']) != row['source_hash']
                or file_hash(tests[rid]['path']) != tests[rid]['sha256']
                or fixture['suites'] or row['visible_case_indices']
                or len(suite['cases']) != len(item['heldout_indices'])
                or len(suite['cases']) < 2):
            raise ValueError(f'Fresh root is overlapping or incomplete: {rid}')
        sample = _sample(fixture['task']['spec'])
        if (item['sample_group'] != ('sample' if sample else 'no_sample')
                or any(_shown_in_statement(case['args'][0], fixture['task']['spec'])
                       for case in suite['cases'])):
            raise ValueError(f'Public statement input leaks into held-out cases: {rid}')
        counts[item['sample_group']] += 1
        heldout_total += len(suite['cases'])
    if dict(counts) != QUOTAS:
        raise ValueError('Fresh task strata changed')
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    if old['framework'] != scale['framework']:
        raise ValueError('Frozen LARGE model and rollout framework differ')
    decoding = old['framework']['inference']['decoding']
    if not decoding['do_sample'] or decoding['temperature'] <= 0:
        raise ValueError('Three candidate samples require stochastic decoding')
    ready = {'status': 'READY', 'roots': len(rows), 'groups': dict(counts),
             'heldout_cases': heldout_total, 'min_heldout_cases_per_root': 2,
             'candidates_per_root': CANDIDATES, 'model_rollouts': len(rows)*CANDIDATES,
             'root_plan': 'fixed ATOMIC', 'decoding': decoding,
             'selection': 'public statement sample and execution status only',
             'metric': 'exact pass on all independent original APPS cases',
             'gpu_devices': GPU_IDS, 'rollouts_per_gpu': len(rows)*CANDIDATES//len(GPU_IDS),
             'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < len(GPU_IDS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed fresh APPS shard already exists: {done}')
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    framework = old['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    rows = _rows()[shard::len(GPU_IDS)]
    tests = _tests()
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = apps_scale._context('LARGE')
        selected = {}
        all_rows = _rows()
        indices = {row['root_id']: index for index, row in enumerate(all_rows)}
        for row in tqdm(rows, desc=f'APPS新题 GPU {GPU_IDS[shard]}', unit='题', position=shard,
                        dynamic_ncols=True):
            rid = row['root_id']
            sample = _sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
            signals = []
            for repetition in range(CANDIDATES):
                seed = SEED_BASE + 100*indices[rid] + repetition
                directory = ROOT/'evaluation'/rid/f'seed_{seed}'
                policy = apps_multitask._ForcedRoot(planner, 'ATOMIC')
                run_case(row, tests, framework, old['config'], policy, executor, directory, seed,
                         reset_seed_after_root_plan=True)
                signals.append(apps_passk_select._score(directory/'case_result.json', rid,
                                                        sample, sandbox))
            selected[rid] = {'has_public_example': sample is not None, 'candidate_signals': signals,
                             'selected_index_by_k': {str(k): apps_passk_select.choose(signals, k,
                                                  sample is not None) for k in range(1, CANDIDATES+1)}}
        write_json(done, {'status': 'COMPLETE', 'shard': shard, 'request_sha256':
                          file_hash(ROOT/'request.json'), 'by_root': selected})
    finally:
        reservation.close()
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(selected),
                      'model_rollouts': len(selected)*CANDIDATES}, ensure_ascii=False), flush=True)


def report():
    selection = json.loads((DATA/'selection.json').read_text())
    strata = {item['root_id']: item['sample_group'] for item in selection['selected']}
    signals = {}
    for shard in range(len(GPU_IDS)):
        record = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (record['status'] != 'COMPLETE' or record['shard'] != shard
                or record['request_sha256'] != file_hash(ROOT/'request.json')
                or len(record['by_root']) != 16 or set(signals) & set(record['by_root'])):
            raise ValueError(f'Incomplete or mismatched fresh APPS shard: {shard}')
        signals.update(record['by_root'])
    if set(signals) != set(strata):
        raise ValueError('Missing fresh APPS roots')
    by_root = {}
    counts = {group: {'roots': 0, 'baseline_exact': 0, 'selected_k3_exact': 0,
                      'oracle_k3_exact': 0, 'baseline_legacy': 0, 'selected_k3_legacy': 0,
                      'incomplete_rollouts': 0}
              for group in ('all', 'sample', 'no_sample')}
    for rid in sorted(signals):
        item = signals[rid]
        group = strata[rid]
        choices = item['selected_index_by_k']
        if (len(item['candidate_signals']) != CANDIDATES
                or item['has_public_example'] != (group == 'sample')
                or any(choices[str(k)] != apps_passk_select.choose(item['candidate_signals'], k,
                                                                    item['has_public_example'])
                       for k in range(1, CANDIDATES+1))):
            raise ValueError(f'Invalid frozen selection: {rid}')
        cases = [json.loads(Path(signal['case_path']).read_text())
                 for signal in item['candidate_signals']]
        exact = [case['status'] == 'COMPLETE' and apps_scale._valid_private(case)
                 for case in cases]
        legacy = [case['status'] == 'COMPLETE' and case['final_scores']['private']['status'] == 'PASS'
                  for case in cases]
        chosen = choices[str(CANDIDATES)]
        for label in ('all', group):
            entry = counts[label]
            entry['roots'] += 1
            entry['baseline_exact'] += exact[0]
            entry['selected_k3_exact'] += exact[chosen]
            entry['oracle_k3_exact'] += any(exact)
            entry['baseline_legacy'] += legacy[0]
            entry['selected_k3_legacy'] += legacy[chosen]
            entry['incomplete_rollouts'] += sum(case['status'] != 'COMPLETE' for case in cases)
        by_root[rid] = {'group': group, 'candidate_signals': item['candidate_signals'],
                        'selected_index_by_k': choices, 'exact_pass': exact,
                        'legacy_pass': legacy, 'case_statuses': [case['status'] for case in cases]}
    summary = {'groups': counts, 'metric': 'all held-out original APPS I/O exact pass',
               'oracle': 'offline upper bound, never used to select',
               'root_plan': 'fixed ATOMIC', 'candidate_count': CANDIDATES}
    write_json(ROOT/'result.json', {'status': 'COMPLETE', 'request_sha256':
                                    file_hash(ROOT/'request.json'), 'summary': summary,
                                    'by_root': by_root})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'evaluate', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'evaluate':
        evaluate(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
