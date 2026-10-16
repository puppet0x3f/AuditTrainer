"""CPU audit of saved fresh APPS programs with public examples and diagnostic probes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from . import apps_fresh_old96 as old96, apps_fresh_select as fresh, apps_passk_select, apps_scale
from .artifacts import ArtifactStore
from .dataset import file_hash
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox
from .schema import Artifact


DATA = Path('data/refinement/apps_fresh_audit')
ROOT = Path('runs/refinement/apps_fresh_audit')

# These probes diagnose observed false passes. They are not an independent holdout
# for future checkpoint selection. Expected outputs follow the local APPS solutions.
PROBES = {
    'apps_train_2231': [
        {'name': 'one_conversion', 'input': '1\n1 2\nAP\n', 'expected': '1\n'},
        {'name': 'all_p_impossible', 'input': '1\n2 2\nPP\nPP\n', 'expected': 'MORTAL\n'},
    ],
    'apps_train_2273': [
        {'name': 'same_sum_different_collision', 'input': '2\n3\n0 2 1\n3\n0 0 0\n',
         'expected': 'NO\nYES\n'},
    ],
    'apps_train_2283': [
        {'name': 'both_winners_on_path',
         'input': '2\n5 1 5 1 4\n1 2\n2 3\n3 4\n4 5\n'
                  '5 1 2 1 4\n1 2\n2 3\n3 4\n4 5\n',
         'expected': 'Bob\nAlice\n'},
    ],
}


def _sources():
    return {'fresh_result': fresh.ROOT/'result.json',
            'old96_result': old96.ROOT/'result.json',
            'fresh_selection': fresh.DATA/'selection.json',
            'fresh_tests': fresh.DATA/'tests.jsonl',
            'scale_request': apps_scale.ROOT/'request.json'}


def _validate_probes(probes):
    if probes != PROBES:
        raise ValueError('Diagnostic probes changed')
    tests = fresh._tests()
    for rid, items in probes.items():
        suite = json.loads(Path(tests[rid]['path']).read_text())['suites'][0]
        heldout = {tuple(case['args'][0].split()) for case in suite['cases']}
        task = json.loads((fresh.DATA/'tasks'/rid/'task.json').read_text())['task']
        sample = fresh._sample(task['spec'])
        for item in items:
            tokens = tuple(item['input'].split())
            if tokens in heldout or (sample and tokens == tuple(sample['input'].split())):
                raise ValueError(f'Diagnostic probe repeats an original APPS input: {rid}/{item["name"]}')


def prepare():
    if DATA.exists() or ROOT.exists():
        raise FileExistsError('Fresh APPS audit already exists')
    old96.check()
    _validate_probes(PROBES)
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    write_json(DATA/'probes.json', PROBES)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'roots': 48, 'arm_candidates': {'OLD96': 1, 'LARGE': 3},
        'source_sha256': {key: file_hash(path) for key, path in _sources().items()},
        'probes_sha256': file_hash(DATA/'probes.json'),
        'public_rule': 'private exact and matched public statement example when available',
        'diagnostic_rule': 'public rule and all post-hoc probes on three identified roots',
        'cpu_only': True, 'gpu_devices': [], 'training_steps': 0,
    })
    check()


def check():
    old96.check()
    request = json.loads((ROOT/'request.json').read_text())
    probes = json.loads((DATA/'probes.json').read_text())
    fresh_result = json.loads((fresh.ROOT/'result.json').read_text())
    old_result = json.loads((old96.ROOT/'result.json').read_text())
    rows = fresh._rows()
    _validate_probes(probes)
    if (request['status'] != 'PREPARED' or request['roots'] != 48
            or request['arm_candidates'] != {'OLD96': 1, 'LARGE': 3}
            or request['source_sha256'] != {key: file_hash(path) for key, path in _sources().items()}
            or request['probes_sha256'] != file_hash(DATA/'probes.json')
            or request['public_rule'] != 'private exact and matched public statement example when available'
            or request['diagnostic_rule'] != 'public rule and all post-hoc probes on three identified roots'
            or request['cpu_only'] is not True or request['gpu_devices'] != []
            or request['training_steps'] != 0
            or fresh_result['status'] != 'COMPLETE' or old_result['status'] != 'COMPLETE'
            or set(fresh_result['by_root']) != {row['root_id'] for row in rows}
            or set(old_result['by_root']) != set(fresh_result['by_root'])):
        raise ValueError('Frozen fresh APPS audit inputs or launch configuration changed')
    for row in rows:
        rid = row['root_id']
        source = fresh_result['by_root'][rid]
        if (len(source['candidate_signals']) != 3
                or source['group'] != old_result['by_root'][rid]['group']
                or source['group'] != ('sample' if fresh._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec']) else 'no_sample')
                or not Path(old_result['by_root'][rid]['case_path']).is_file()
                or any(not Path(signal['case_path']).is_file() for signal in source['candidate_signals'])):
            raise ValueError(f'Incomplete saved candidates or public example: {rid}')
    framework = json.loads((apps_scale.ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    if sandbox.bwrap_version == 'UNAVAILABLE':
        raise RuntimeError('Bubblewrap sandbox is unavailable')
    ready = {'status': 'READY', 'roots': len(rows), 'saved_candidates': len(rows)*4,
             'public_example_checks': 32, 'diagnostic_probe_checks': 16,
             'diagnostic_roots': sorted(probes), 'cpu_only': True,
             'gpu_devices': [], 'training_steps': 0,
             'sandbox_timeout_seconds': sandbox.limits.per_case_timeout_seconds}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _probe_scores(path, items, sandbox):
    case = json.loads(Path(path).read_text())
    snapshot = case['final_program']
    bundle = ArtifactStore(Path(snapshot['path'])).bundle(Artifact.model_validate(snapshot['artifact'])) if snapshot else None
    results = []
    for item in items:
        execution = sandbox.run(bundle.sources['N0'], 'solve', [item['input']], bundle=bundle) if bundle else None
        results.append({'name': item['name'], 'status': execution.status if execution else 'NO_PROGRAM',
                        'pass': bool(execution and execution.status == 'RETURNED'
                                     and isinstance(execution.value, str)
                                     and execution.value.strip() == item['expected'].strip())})
    return results


def score():
    from tqdm import tqdm

    check()
    done = ROOT/'signals.json'
    if done.exists():
        raise FileExistsError(f'Fresh APPS audit signals already exist: {done}')
    framework = json.loads((apps_scale.ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    old_result = json.loads((old96.ROOT/'result.json').read_text())['by_root']
    fresh_result = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    old_samples, probes = {}, {}
    with tqdm(total=48, desc='APPS旧程序 CPU 复评', unit='项', dynamic_ncols=True) as bar:
        for row in fresh._rows():
            rid = row['root_id']
            sample = fresh._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
            if sample is not None:
                old_samples[rid] = apps_passk_select._score(
                    Path(old_result[rid]['case_path']), rid, sample, sandbox)
                bar.update(1)
        for rid, items in PROBES.items():
            paths = {'OLD96': old_result[rid]['case_path']}
            paths.update({f'LARGE_{i}': signal['case_path'] for i, signal in enumerate(
                fresh_result[rid]['candidate_signals'])})
            probes[rid] = {}
            for arm, path in paths.items():
                probes[rid][arm] = _probe_scores(path, items, sandbox)
                bar.update(len(items))
    write_json(done, {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
                      'old96_public_examples': old_samples, 'diagnostic_probes': probes})
    print(json.dumps({'status': 'COMPLETE', 'public_example_roots': len(old_samples),
                      'probe_roots': len(probes), 'sandbox_checks': 48}, ensure_ascii=False), flush=True)


def _outcome(private, sample, probes):
    public = private and (sample is None or sample)
    return {'private_exact': bool(private), 'private_and_public': bool(public),
            'diagnostic_pass': None if probes is None else all(item['pass'] for item in probes),
            'private_public_and_diagnostic': bool(public and (probes is None or all(item['pass'] for item in probes)))}


def report():
    check()
    result_path = ROOT/'result.json'
    if result_path.exists():
        raise FileExistsError(f'Fresh APPS audit result already exists: {result_path}')
    signals = json.loads((ROOT/'signals.json').read_text())
    if signals['status'] != 'COMPLETE' or signals['request_sha256'] != file_hash(ROOT/'request.json'):
        raise ValueError('Fresh APPS audit signals are incomplete or stale')
    fresh_result = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    old_result = json.loads((old96.ROOT/'result.json').read_text())['by_root']
    if (set(signals['old96_public_examples']) != {rid for rid, item in fresh_result.items()
                                                  if item['group'] == 'sample'}
            or set(signals['diagnostic_probes']) != set(PROBES)):
        raise ValueError('Fresh APPS audit signals do not cover the frozen design')
    for rid, arms in signals['diagnostic_probes'].items():
        if (set(arms) != {'OLD96', 'LARGE_0', 'LARGE_1', 'LARGE_2'}
                or any([item['name'] for item in arms[arm]] != [probe['name'] for probe in PROBES[rid]]
                       for arm in arms)):
            raise ValueError(f'Incomplete diagnostic probes: {rid}')
    summaries = {group: {arm: Counter({'roots': 0, 'private_exact': 0,
                                       'private_and_public': 0,
                                       'private_public_and_diagnostic': 0})
                         for arm in ('OLD96_k1', 'LARGE_k1', 'LARGE_selected_k3', 'LARGE_oracle_k3')}
                 for group in ('all', 'sample', 'no_sample')}
    by_root = {}
    for rid in sorted(fresh_result):
        fresh_item, old_item = fresh_result[rid], old_result[rid]
        sample = fresh_item['group'] == 'sample'
        diagnostic = signals['diagnostic_probes'].get(rid)
        old_signal = signals['old96_public_examples'].get(rid)
        old = _outcome(old_item['old96_exact'], old_signal['sample_pass'] if sample else None,
                       diagnostic['OLD96'] if diagnostic else None)
        large = [_outcome(passed, signal['sample_pass'] if sample else None,
                          diagnostic[f'LARGE_{i}'] if diagnostic else None)
                 for i, (passed, signal) in enumerate(zip(fresh_item['exact_pass'],
                                                          fresh_item['candidate_signals']))]
        chosen = fresh_item['selected_index_by_k']['3']
        values = {'OLD96_k1': old, 'LARGE_k1': large[0],
                  'LARGE_selected_k3': large[chosen],
                  'LARGE_oracle_k3': {key: any(item[key] for item in large)
                                      for key in ('private_exact', 'private_and_public',
                                                  'private_public_and_diagnostic')}}
        for group in ('all', fresh_item['group']):
            for arm, outcome in values.items():
                summaries[group][arm]['roots'] += 1
                for key in ('private_exact', 'private_and_public', 'private_public_and_diagnostic'):
                    summaries[group][arm][key] += outcome[key]
        by_root[rid] = {'group': fresh_item['group'], 'selected_index': chosen,
                        'old96': old, 'large_candidates': large,
                        'old96_sample_signal': old_signal, 'diagnostic_signals': diagnostic}
    summary = {group: {arm: dict(counts) for arm, counts in arms.items()}
               for group, arms in summaries.items()}
    write_json(result_path, {'status': 'COMPLETE',
                             'request_sha256': file_hash(ROOT/'request.json'),
                             'summary': summary, 'by_root': by_root,
                             'note': 'Three diagnostic roots were chosen after observing outputs; '
                                     'use for failure analysis, not checkpoint selection.'})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary},
                     ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'score', 'report'))
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'score':
        score()
    else:
        report()


if __name__ == '__main__':
    main()
