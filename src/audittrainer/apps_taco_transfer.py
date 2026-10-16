"""Held-out TACO test comparison of the frozen DIVERSE and TACO executors."""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

from . import apps, apps_diverse_sft as diverse, apps_multitask as previous
from . import apps_scale, apps_taco_eval, apps_taco_sft as taco
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .verifier import digest


DATA = Path('data/refinement/apps_taco_transfer')
ROOT = Path('runs/refinement/apps_taco_transfer')
SOURCE = Path('data/taco_source')
QUOTAS = {'EASY': 50, 'MEDIUM': 50}
SEED = 20260929
EVAL_SEED = 729000713
REPETITIONS = 3
GPUS = ('0', '1', '2')


def _source_paths():
    return {
        'test_manifest': SOURCE/'test_manifest.json',
        'special_judge': SOURCE/'output_spj.jsonl',
        'train_manifest': SOURCE/'manifest.json',
        'diverse_result': diverse.ROOT/'result.json',
        'taco_result': taco.ROOT/'result.json',
        'diverse_adapter': diverse.ROOT/'training/final_adapter/adapter_model.safetensors',
        'taco_adapter': taco.ROOT/'training/final_adapter/adapter_model.safetensors',
        'selected': DATA/'selected_records.jsonl',
        'roots': DATA/'roots.jsonl',
        'tests': DATA/'tests.jsonl',
    }


def _train_keys():
    import pyarrow.parquet as pq
    from tqdm import tqdm

    urls, questions = taco._excluded()
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for entry in tqdm(manifest['files'], desc='TACO训练集去重索引', unit='分片',
                      dynamic_ncols=True):
        path = SOURCE/entry['path']
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError(f'TACO train shard changed: {path}')
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=['url', 'question']):
            for row in batch.to_pylist():
                if key := taco._url_key(row['url']):
                    urls.add(key)
                if isinstance(row['question'], str):
                    questions.add(taco._question_key(row['question']))
    return urls, questions


def _candidates():
    import pyarrow.parquet as pq

    manifest = json.loads((SOURCE/'test_manifest.json').read_text())
    entry = manifest['test_file']
    path = SOURCE/entry['path']
    spj_path = SOURCE/manifest['special_judge']['path']
    if (manifest['dataset'] != 'BAAI/TACO'
            or manifest['revision'] != json.loads((SOURCE/'manifest.json').read_text())['revision']
            or path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']
            or file_hash(spj_path) != manifest['special_judge']['sha256']):
        raise ValueError('Pinned TACO test data or special-judge list changed')
    rows = pq.read_table(path, columns=['question', 'input_output', 'difficulty',
                                        'source', 'url', 'picture_num']).to_pylist()
    spj = read_jsonl(spj_path)
    if len(rows) != 1000 or len(spj) != 1000 or any(x['idx'] != i for i, x in enumerate(spj)):
        raise ValueError('TACO test rows and special-judge indices differ')
    excluded_urls, excluded_questions = _train_keys()
    seen_urls, seen_questions = set(), set()
    counts, buckets = Counter(), {key: [] for key in QUOTAS}
    for index, (row, judge) in enumerate(zip(rows, spj)):
        counts['all'] += 1
        if judge['special_judge']:
            counts['special_judge'] += 1
            continue
        if row['difficulty'] not in buckets or row['source'] not in taco.CONTEST_SOURCES:
            counts['scope'] += 1
            continue
        if str(row.get('picture_num') or '0') != '0':
            counts['image'] += 1
            continue
        question = row['question']
        url = taco._url_key(row['url'])
        if not isinstance(question, str) or not question.strip() or len(question) > 9000 or url is None:
            counts['question_or_url'] += 1
            continue
        question_key = taco._question_key(question)
        if url in excluded_urls or question_key in excluded_questions:
            counts['training_overlap'] += 1
            continue
        if url in seen_urls or question_key in seen_questions:
            counts['duplicate_test'] += 1
            continue
        try:
            io = json.loads(row['input_output'])
            if io.get('fn_name') is not None:
                raise ValueError('function call')
            inputs, outputs = io['inputs'], io['outputs']
            if len(inputs) < 10 or len(inputs) != len(outputs):
                raise ValueError('insufficient tests')
            indices = list(range(5)) + list(range(len(inputs)-5, len(inputs)))
            selected_inputs = [apps.stdin_text(inputs[i]) for i in indices]
            selected_outputs = [apps.stdin_text(outputs[i]) for i in indices]
            if (len({tuple(value.split()) for value in selected_inputs}) < 5
                    or any(len(value) > 12000 for value in selected_inputs+selected_outputs)):
                raise ValueError('large or duplicate tests')
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            counts['invalid_tests'] += 1
            continue
        seen_urls.add(url)
        seen_questions.add(question_key)
        counts['candidate'] += 1
        buckets[row['difficulty']].append({
            'id': index, 'question': question, 'url': row['url'],
            'source': 'TACO', 'taco_source': row['source'],
            'difficulty': row['difficulty'],
            'input_output': {'inputs': selected_inputs, 'outputs': selected_outputs},
            'original_test_indices': indices,
        })
    selected = []
    for difficulty, quota in QUOTAS.items():
        pool = sorted(buckets[difficulty], key=lambda row: hashlib.sha256(
            f'{SEED}:{row["id"]}'.encode()).digest())
        if len(pool) < quota:
            raise ValueError(f'Only {len(pool)} eligible TACO test {difficulty} tasks')
        selected.extend(pool[:quota])
    selected.sort(key=lambda row: row['id'])
    return selected, dict(counts)


def _fixture(record):
    public, private = apps.convert_record(record, original_split='train',
                                          split='VALIDATION', visible_cases=1)
    root = f'taco_test_{record["id"]:04d}'
    family = 'taco_origin_' + digest(taco._url_key(record['url']))[:16]
    public['task'].update(root_id=root, family_id=family, split='EXTERNAL_SEALED',
                          visible_suite_ref=root+'_VISIBLE',
                          private_suite_ref=root+'_PRIVATE')
    for visibility, block in (('VISIBLE', public), ('PRIVATE', private)):
        for suite in block['suites']:
            suite.update(suite_id=f'{root}_{visibility}', comparison='stdio_tokens',
                         oracle_origin=f'TACO test/{record["id"]}, ten fixed original cases')
    return public, private


def prepare():
    if DATA.exists() or ROOT.exists():
        raise FileExistsError('TACO transfer evaluation already prepared')
    diverse.check(trained=True)
    taco.check(trained=True)
    selected, counts = _candidates()
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    (DATA/'selected_records.jsonl').write_text(''.join(diverse._jsonl_line(r) for r in selected))
    roots, tests = [], []
    for record in selected:
        public, private = _fixture(record)
        root = public['task']['root_id']
        folder = DATA/'fixtures'/root
        folder.mkdir(parents=True)
        write_json(folder/'task.json', public)
        write_json(folder/'private_tests.json', private)
        roots.append({'root_id': root, 'family_id': public['task']['family_id'],
                      'split': 'EXTERNAL_SEALED', 'source': 'TACO',
                      'original_split': 'test', 'source_task_id': str(record['id']),
                      'source_record_sha256': digest(record),
                      'fixture_path': str(folder/'task.json'),
                      'source_hash': file_hash(folder/'task.json'),
                      'io_type': 'stdin', 'comparison': 'stdio_tokens'})
        tests.append({'root_id': root, 'path': str(folder/'private_tests.json'),
                      'sha256': file_hash(folder/'private_tests.json')})
    (DATA/'roots.jsonl').write_text(''.join(diverse._jsonl_line(r) for r in roots))
    (DATA/'tests.jsonl').write_text(''.join(diverse._jsonl_line(r) for r in tests))
    framework = json.loads((diverse.ROOT/'request.json').read_text())['framework']
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'framework': framework,
        'selection_seed': SEED, 'eval_seed': EVAL_SEED,
        'difficulty_quotas': QUOTAS, 'candidate_counts': counts,
        'roots': len(roots), 'test_cases_per_root': 10,
        'visible_cases_per_root': 1, 'candidates_per_root': REPETITIONS,
        'arms': ['DIVERSE', 'TACO'], 'plan': 'ATOMIC',
        'training_steps': 0, 'gpu_devices': list(GPUS),
        'source_sha256': {key: file_hash(path) for key, path in _source_paths().items()},
        'design': 'Paired frozen executors and planner, identical seeds; official TACO test '
                  'EASY/MEDIUM, no known special judge or train overlap, first and last five '
                  'original cases, ordered exact-token comparison, first case visible.',
    })
    write_json(ROOT/'gpu_resources_eval.json',
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check():
    request = json.loads((ROOT/'request.json').read_text())
    if (request['status'] != 'PREPARED' or request['selection_seed'] != SEED
            or request['eval_seed'] != EVAL_SEED
            or request['difficulty_quotas'] != QUOTAS
            or request['roots'] != sum(QUOTAS.values())
            or request['test_cases_per_root'] != 10
            or request['visible_cases_per_root'] != 1
            or request['candidates_per_root'] != REPETITIONS
            or request['arms'] != ['DIVERSE', 'TACO']
            or request['plan'] != 'ATOMIC' or request['training_steps'] != 0
            or request['gpu_devices'] != list(GPUS)
            or request['framework'] != json.loads((diverse.ROOT/'request.json').read_text())['framework']
            or request['source_sha256'] != {key: file_hash(path)
                                            for key, path in _source_paths().items()}
            or json.loads((ROOT/'gpu_resources_eval.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('TACO transfer design or startup configuration changed')
    selected = read_jsonl(DATA/'selected_records.jsonl')
    roots = read_jsonl(DATA/'roots.jsonl')
    tests = read_jsonl(DATA/'tests.jsonl')
    if (len(selected) != sum(QUOTAS.values()) or len(roots) != len(selected)
            or len(tests) != len(selected)
            or Counter(r['difficulty'] for r in selected) != QUOTAS
            or len({taco._url_key(r['url']) for r in selected}) != len(selected)
            or len({taco._question_key(r['question']) for r in selected}) != len(selected)
            or [r['root_id'] for r in roots] !=
               [f'taco_test_{r["id"]:04d}' for r in selected]
            or [r['root_id'] for r in tests] != [r['root_id'] for r in roots]):
        raise ValueError('TACO transfer selected roots changed')
    for record, row, extra in zip(selected, roots, tests):
        if (file_hash(row['fixture_path']) != row['source_hash']
                or file_hash(extra['path']) != extra['sha256']):
            raise ValueError(f'TACO test fixture changed: {row["root_id"]}')
        public = json.loads(Path(row['fixture_path']).read_text())
        private = json.loads(Path(extra['path']).read_text())
        if (public['task']['split'] != 'EXTERNAL_SEALED'
                or public['task']['root_id'] != row['root_id']
                or len(public['suites'][0]['cases']) != 1
                or len(private['suites'][0]['cases']) != 9
                or public['suites'][0]['comparison'] != 'stdio_tokens'
                or private['suites'][0]['comparison'] != 'stdio_tokens'
                or row['source_record_sha256'] != digest(record)):
            raise ValueError(f'TACO test contract changed: {row["root_id"]}')
    # Exercise the same split and fixture gate that run_case uses before loading any GPU model.
    from .refinement_rollout import task_parts
    test_map = {row['root_id']: row for row in tests}
    for row in roots:
        task_parts(row, test_map, request['framework'])
    ready = {'status': 'READY', 'roots': len(roots),
             'difficulty_quotas': QUOTAS, 'arms': request['arms'],
             'candidates_per_root': REPETITIONS,
             'gpu_devices': list(GPUS), 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < len(GPUS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record['status'] == 'COMPLETE' and record['request_sha256'] == file_hash(ROOT/'request.json'):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete TACO transfer shard: {done}')
    rows = read_jsonl(DATA/'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(DATA/'tests.jsonl')}
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    request = json.loads((ROOT/'request.json').read_text())
    if old['framework'] != request['framework']:
        raise ValueError('TACO transfer runtime framework differs from frozen DIVERSE')
    selected = [(i, row) for i, row in enumerate(rows) if i % len(GPUS) == shard]
    reservation = acquire(ROOT/'gpu_resources_eval.json', 'collect')
    try:
        with tqdm(total=len(selected)*REPETITIONS*2,
                  desc=f'TACO留出 ATOMIC GPU {GPUS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            for arm, loader, expected in (
                    ('DIVERSE', diverse._context, request['source_sha256']['diverse_adapter']),
                    ('TACO', apps_taco_eval._context, request['source_sha256']['taco_adapter'])):
                planner, executor = loader()
                for index, row in selected:
                    for repetition in range(REPETITIONS):
                        seed = EVAL_SEED+100*index+repetition
                        directory = ROOT/'evaluation'/arm/row['root_id']/f'seed_{seed}'
                        case = run_case(row, tests, old['framework'], old['config'],
                                        previous._ForcedRoot(planner, 'ATOMIC'), executor,
                                        directory, seed, reset_seed_after_root_plan=True)
                        if (case['status'] != 'COMPLETE'
                                or case['registered_plan_modes'] != {'ATOMIC': 1}
                                or case['model_identity']['executor']['adapter_weights_sha256'] != expected):
                            raise RuntimeError(f'Invalid TACO test trajectory: {arm}/{row["root_id"]}/{repetition}')
                        bar.update(1)
                del planner, executor
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'),
                          'roots': [row['root_id'] for _, row in selected]})
    finally:
        reservation.close()


def report():
    from tqdm import tqdm

    check()
    if (ROOT/'result.json').exists():
        raise FileExistsError(f'TACO transfer report already exists: {ROOT/"result.json"}')
    request = json.loads((ROOT/'request.json').read_text())
    shards = [json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(len(GPUS))]
    rows = read_jsonl(DATA/'roots.jsonl')
    if (any(s['status'] != 'COMPLETE' or s['shard'] != i
            or s['request_sha256'] != file_hash(ROOT/'request.json')
            for i, s in enumerate(shards))
            or {rid for shard in shards for rid in shard['roots']}
            != {row['root_id'] for row in rows}):
        raise ValueError('TACO transfer evaluation shards incomplete')
    records = {f'taco_test_{r["id"]:04d}': r for r in read_jsonl(DATA/'selected_records.jsonl')}
    totals = {arm: Counter() for arm in request['arms']}
    by_difficulty = {difficulty: {arm: Counter() for arm in request['arms']}
                     for difficulty in QUOTAS}
    paired = Counter()
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='TACO留出配对汇总', unit='题',
                                     dynamic_ncols=True)):
        rid = row['root_id']
        outcomes = {}
        for arm in request['arms']:
            cases = [json.loads((ROOT/'evaluation'/arm/rid/
                                 f'seed_{EVAL_SEED+100*index+rep}'/'case_result.json').read_text())
                     for rep in range(REPETITIONS)]
            if any(c['status'] != 'COMPLETE' or c['root_id'] != rid
                   or c['seed'] != EVAL_SEED+100*index+rep
                   or c['registered_plan_modes'] != {'ATOMIC': 1}
                   or c['model_identity']['executor']['adapter_weights_sha256']
                      != request['source_sha256'][arm.lower()+'_adapter']
                   for rep, c in enumerate(cases)):
                raise ValueError(f'TACO transfer candidate differs: {arm}/{rid}')
            private = [c['final_scores']['private']['status'] == 'PASS' for c in cases]
            visible = [c['final_scores']['visible']['status'] == 'PASS' for c in cases]
            chosen = next((i for i, value in enumerate(visible) if value), 0)
            values = {'k1': int(private[0]), 'k3_visible_selected': int(private[chosen]),
                      'k3_oracle': int(any(private)),
                      'visible_pass_candidates': sum(visible),
                      'private_pass_candidates': sum(private),
                      'selected_seed': cases[chosen]['seed']}
            outcomes[arm] = values
            for counter in (totals[arm], by_difficulty[records[rid]['difficulty']][arm]):
                counter['roots'] += 1
                counter.update({key: value for key, value in values.items()
                                if key != 'selected_seed'})
        a, b = [outcomes[arm]['k3_visible_selected'] for arm in request['arms']]
        paired['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
        by_root[rid] = {'difficulty': records[rid]['difficulty'], **outcomes}
    interface = json.loads((ROOT/'interface/result.json').read_text())
    if interface['status'] != 'COMPLETE' or interface['request_sha256'] != file_hash(ROOT/'request.json'):
        raise ValueError('Saved direct-interface replay incomplete')
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'roots': len(rows), 'cases_per_root': 10,
              'summary': {arm: dict(value) for arm, value in totals.items()},
              'by_difficulty': {difficulty: {arm: dict(value) for arm, value in arms.items()}
                                for difficulty, arms in by_difficulty.items()},
              'paired_TACO_vs_DIVERSE_k3': dict(paired),
              'by_root': by_root,
              'interface_replay': interface['summary']}
    write_json(ROOT/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'by_root'},
                     ensure_ascii=False, indent=2), flush=True)


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
