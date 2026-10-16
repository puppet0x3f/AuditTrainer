"""Bounded calendar scoring correction; preserve all historical scores and code."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import load_task, score_program, write_json
from .refinement_pr1 import load_config
from .sandbox import ExecutionLimits, Sandbox


ROOT = 'train_transfer_calendar_matrix_03'
OUTPUT = Path('runs/refinement/calendar_scoring_v2')
DATA = Path('data/refinement/calendar_scoring_v2')
CASES = [
    (['00:01', '00:02', '00:03', '00:05'], [1, 2, 3, 5], {'indices': [2], 'total': 3}),
    (['00:09', '00:01', '00:06', '00:03', '00:03'], [1, 3, 6, 9], {'indices': [1, 2, 3], 'total': 18}),
    (['00:00', '00:01', '00:02', '00:04', '00:05'], [0, 1, 2, 4, 5], {'indices': [0], 'total': 0}),
    (['00:02', '00:04', '00:05', '00:07'], [2, 4, 5, 7], {'indices': [], 'total': 0}),
]


def patched_private(original):
    result = deepcopy(original)
    for suite in result['suites']:
        if suite['visibility'] != 'PRIVATE':
            raise ValueError('Only offline private suites can be extended')
        key = suite.get('contract_key', 'root')
        for i, (items, prepared, answer) in enumerate(CASES):
            args = [json.dumps(items)] if key == 'root' else [prepared if key == 'aggregate' else items]
            expected = (json.dumps(answer, separators=(',', ':')) if key == 'root'
                        else prepared if key == 'prepare' else answer)
            suite['cases'].append({'case_id': f'{key}_score_v2_{i}', 'args': args, 'expected': expected})
        suite['oracle_origin'] += '; calendar_score_v2 independently enumerated long-input counterexamples'
    return result


def prepare(output=OUTPUT, data=DATA):
    output, data = Path(output), Path(data)
    output.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path('runs/refinement/B3_calendar_target/request.json').read_text())
    config, framework = load_config('runs/refinement/B3_summary_training/batch_130000713/config.yaml')
    row = load_root_manifest(source['config']['root_manifest'])[ROOT]
    old_test = read_jsonl(source['config']['test_manifest'])[0]
    updated = patched_private(json.loads(Path(old_test['path']).read_text()))
    test_path = data/'private_tests.json'
    write_json(test_path, updated)
    test = {'root_id': ROOT, 'path': str(test_path), 'sha256': file_hash(test_path)}
    for name, item in [('roots', row), ('tests', test)]:
        (data/f'{name}.jsonl').write_text(json.dumps(item, ensure_ascii=False)+'\n')
    config.update(root_manifest=str(data/'roots.jsonl'), test_manifest=str(data/'tests.jsonl'))
    write_json(output/'request.json', {'config': config, 'framework': framework,
        'source_test': old_test, 'new_test': test, 'cases': CASES,
        'scope': '228 saved final programs; no policy calls or updates; old results unchanged'})
    return config, framework, row, {ROOT: test}


def reference_check(row, framework, output):
    reference = json.loads(Path(row['reference_path']).read_text())
    source = ('import json\n'+reference['prepare']+'\n'+reference['aggregate']+
              '\ndef compute(items):\n    return aggregate(prepare(items))\n'
              '\ndef solve(text):\n    return json.dumps(compute(json.loads(text)), separators=(\",\",\":\"))\n')
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    records = []
    for i, (items, prepared, answer) in enumerate(CASES):
        for symbol, args, expected in [('prepare', [items], prepared), ('aggregate', [prepared], answer),
                ('compute', [items], answer), ('solve', [json.dumps(items)], json.dumps(answer, separators=(',', ':')))]:
            result = sandbox.run(source, symbol, args)
            records.append({'case': i, 'symbol': symbol, 'execution': result.model_dump(mode='json'),
                            'expected': expected, 'pass': result.status == 'RETURNED' and result.value == expected})
    write_json(Path(output)/'reference_check.json', records)
    if not all(r['pass'] for r in records):
        raise ValueError('Reference failed the independent new cases')
    return records


def historical_cases():
    fixed = json.loads(Path('runs/refinement/B3_fixed_plan/result.json').read_text())
    cases = [(f'fixed/{variant}', c) for variant, block in fixed['results'][ROOT].items() for c in block['cases']]
    for method in ['B0', 'GATE_SFT', 'BODY_SFT', 'JOINT_SFT']:
        for path in sorted((Path('runs/refinement/B3_calendar_target/evaluation')/method/'natural').glob('*/repeat_*/case_result.json')):
            cases.append((f'natural/{method}', json.loads(path.read_text())))
    if len(cases) != 228 or any(c['root_id'] != ROOT for _, c in cases):
        raise ValueError('Expected 36 calibration and 192 natural cases')
    return cases


def copy_snapshot(snapshot, destination):
    if snapshot is None:
        return None
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    for module in snapshot['bundle']['modules'].values():
        path = Path(module['module_path'])
        for relative in (path, path.with_suffix('.json')):
            target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(snapshot['path'])/relative, target)
    copied = {**snapshot, 'path': str(destination)}
    write_json(destination/'snapshot.json', copied)
    return copied


def run(output=OUTPUT, data=DATA):
    output = Path(output)
    config, framework, row, tests = prepare(output, data)
    reference = reference_check(row, framework, output)
    _, _, _, _, verifier = load_task(row, tests, framework)

    def rescore(item):
        group, case = item
        folder = output/'cases'/group/str(case['seed']); folder.mkdir(parents=True, exist_ok=True)
        path = folder/'result.json'
        if path.exists():
            return json.loads(path.read_text())
        snapshot = copy_snapshot(case['final_program'], folder/'final_program')
        scores, cost = score_program(snapshot, verifier, ROOT)
        record = {'group': group, 'seed': case['seed'], 'source_case': case['run_dir'],
            'old_success': case['final_scores']['private']['status'] == 'PASS',
            'new_success': scores['private']['status'] == 'PASS', 'scores': scores, 'cost': cost}
        write_json(path, record)
        print(json.dumps({k: record[k] for k in ('group', 'seed', 'old_success', 'new_success')}), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(rescore, historical_cases()))
    groups = defaultdict(list)
    for r in records: groups[r['group']].append(r)
    summaries = {g: {'runs': len(rs), 'old_successes': sum(r['old_success'] for r in rs),
                    'new_successes': sum(r['new_success'] for r in rs),
                    'changed_seeds': [r['seed'] for r in rs if r['old_success'] != r['new_success']]}
                 for g, rs in groups.items()}
    errors = [r for r in records if any(v['status'] == 'ERROR' for v in r['scores'].values())]
    result = {'status': 'ERROR' if errors else 'COMPLETE', 'root_id': ROOT, 'scoring_version': 'calendar_score_v2',
        'config': config, 'reference_checks': len(reference), 'cases': len(records), 'groups': summaries,
        'changed': [r for r in records if r['old_success'] != r['new_success']], 'errors': errors,
        'model_calls': 0, 'optimizer_steps': 0}
    write_json(output/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if errors: raise RuntimeError('Scoring infrastructure errors require resolution')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--data', default=str(DATA))
    args = parser.parse_args()
    run(args.output, args.data)
