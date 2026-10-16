"""Extend the frozen APPS executor SFT with distinct TACO contest problems."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import unquote, urlsplit

from . import apps, apps_direct_rejudge as rejudge, apps_diverse_sft as diverse
from . import apps_fresh_select as fresh, apps_multitask as previous, apps_scale
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .prompts import PromptOverflow
from .sandbox import ExecutionLimits, Sandbox


SOURCE = Path('data/taco_source')
DATA = Path('data/refinement/apps_taco_sft')
ROOT = Path('runs/refinement/apps_taco_sft')
NEW_TARGETS = 2048
OLD_TARGETS = 1444
TOTAL_TARGETS = OLD_TARGETS + NEW_TARGETS
STEPS = TOTAL_TARGETS // 2
SEED = 20260928
GPUS = ('0', '1', '2')
CONTEST_SOURCES = {'codeforces', 'codechef', 'atcoder', 'aizu',
                   'hackerearth', 'kattis', 'hackerrank', 'codejam'}
_WORKER_FRAMEWORK = None
_WORKER_TOKENIZER = None
_WORKER_SANDBOX = None


def _url_key(value):
    if not value:
        return None
    parsed = urlsplit(value.strip().lower())
    host = (parsed.hostname or '').removeprefix('www.')
    path = unquote(parsed.path).rstrip('/')
    if not host:
        return None
    if host.endswith('codeforces.com'):
        match = re.search(r'/(?:problemset/problem|contest|gym)/(\d+)/(?:problem/)?([a-z]\d*)$', path)
        if match:
            return 'codeforces:' + ':'.join(match.groups())
    if host.endswith('atcoder.jp'):
        match = re.search(r'/tasks/([^/]+)$', path)
        if match:
            return 'atcoder:' + match.group(1)
    if host.endswith('codechef.com'):
        match = re.search(r'/problems/([^/]+)$', path)
        if match:
            return 'codechef:' + match.group(1)
    return host + path


def _question_key(value):
    normalized = unicodedata.normalize('NFKC', value).casefold()
    return hashlib.sha256(re.sub(r'\W+', '', normalized).encode()).hexdigest()


def _old_data():
    rows = read_jsonl(diverse.DATA/'training.jsonl')
    targets = [row for row in rows if row['kind'] == 'target']
    retention = [row for row in rows if row['kind'] == 'retention']
    if len(targets) != OLD_TARGETS or len(retention) != 177:
        raise ValueError('Frozen diverse APPS training data changed')
    return targets, retention


def _excluded():
    records = read_jsonl(previous.SOURCE)
    return ({key for row in records if (key := _url_key(row.get('url'))) is not None},
            {_question_key(row['question']) for row in records
             if isinstance(row.get('question'), str)})


def _candidate(record, index, excluded_urls, excluded_questions, seen_urls,
               seen_questions, reasons):
    reasons['all'] += 1
    if record.get('source') not in CONTEST_SOURCES:
        reasons['noncontest_source'] += 1
        return None
    if str(record.get('picture_num') or '0') != '0':
        reasons['requires_image'] += 1
        return None
    question = record.get('question')
    if not isinstance(question, str) or not question.strip() or len(question) > 9000:
        reasons['invalid_question_length'] += 1
        return None
    url = _url_key(record.get('url'))
    question_key = _question_key(question)
    if url is None:
        reasons['no_origin_url'] += 1
        return None
    if url in excluded_urls or question_key in excluded_questions:
        reasons['overlap_APPS'] += 1
        return None
    if url in seen_urls or question_key in seen_questions:
        reasons['duplicate_TACO'] += 1
        return None
    try:
        io = json.loads(record['input_output'])
        solutions = json.loads(record['solutions'])
        if io.get('fn_name') is not None:
            reasons['function_call'] += 1
            return None
        inputs, outputs = io['inputs'], io['outputs']
        if len(inputs) < 6 or len(inputs) != len(outputs):
            reasons['insufficient_tests'] += 1
            return None
        indices = sorted({0, 1, 2, len(inputs)-3, len(inputs)-2, len(inputs)-1})
        inputs = [apps.stdin_text(inputs[i]) for i in indices]
        outputs = [apps.stdin_text(outputs[i]) for i in indices]
        if (len({tuple(value.split()) for value in inputs}) < 4
                or any(len(value) > 12000 for value in inputs + outputs)):
            reasons['large_or_duplicate_tests'] += 1
            return None
        if not isinstance(solutions, list):
            reasons['invalid_record'] += 1
            return None
        scripts = []
        for script in solutions:
            if not isinstance(script, str) or len(script) > 5500:
                continue
            try:
                ast.parse(script)
            except SyntaxError:
                continue
            scripts.append(script)
            if len(scripts) == 3:
                break
        if not scripts:
            reasons['no_short_python3_solution'] += 1
            return None
    except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        reasons['invalid_record'] += 1
        return None
    seen_urls.add(url)
    seen_questions.add(question_key)
    reasons['candidate'] += 1
    return {'id': 100000 + index, 'taco_index': index, 'question': question,
            'url': record['url'], 'source': 'TACO',
            'taco_source': record['source'], 'difficulty': record['difficulty'],
            'solutions': json.dumps(scripts, ensure_ascii=False),
            'input_output': json.dumps({'inputs': inputs, 'outputs': outputs},
                                       ensure_ascii=False)}


def _pool():
    import pyarrow.parquet as pq
    from tqdm import tqdm

    manifest = json.loads((SOURCE/'manifest.json').read_text())
    if manifest['dataset'] != 'BAAI/TACO' or len(manifest['files']) != 9:
        raise ValueError('Expected nine pinned official TACO train shards')
    excluded_urls, excluded_questions = _excluded()
    seen_urls, seen_questions = set(), set()
    candidates, reasons, index = [], Counter(), 0
    for entry in tqdm(manifest['files'], desc='TACO train 筛选', unit='分片',
                      dynamic_ncols=True):
        path = SOURCE/entry['path']
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError(f'Official TACO shard checksum differs: {path}')
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=[
                'question', 'solutions', 'input_output', 'difficulty', 'source',
                'url', 'picture_num']):
            for row in batch.to_pylist():
                candidate = _candidate(row, index, excluded_urls, excluded_questions,
                                       seen_urls, seen_questions, reasons)
                if candidate is not None:
                    candidates.append(candidate)
                index += 1
    if index != 25443:
        raise ValueError(f'TACO train has {index} rows, expected 25443')
    candidates.sort(key=lambda row: hashlib.sha256(
        f'{SEED}:{_url_key(row["url"])}'.encode()).digest())
    if len(candidates) < NEW_TARGETS:
        raise ValueError(f'Only {len(candidates)} distinct contest stdin candidates')
    return candidates, dict(reasons)


def _init_worker(framework):
    from transformers import AutoTokenizer

    global _WORKER_FRAMEWORK, _WORKER_TOKENIZER, _WORKER_SANDBOX
    _WORKER_FRAMEWORK = framework
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        framework['model']['local_path'], local_files_only=True)
    _WORKER_SANDBOX = Sandbox(ExecutionLimits(**framework['execution']))
    if _WORKER_SANDBOX.bwrap_version == 'UNAVAILABLE':
        raise RuntimeError('Bubblewrap sandbox is unavailable')


def _verify_worker(record):
    with tempfile.TemporaryDirectory(prefix='taco_teacher_') as scratch:
        try:
            target = previous._target(record, 'TRAIN', _WORKER_FRAMEWORK,
                                      _WORKER_TOKENIZER, Path(scratch))
        except (ValueError, KeyError, TypeError, AttributeError,
                PromptOverflow, json.JSONDecodeError):
            return None
    if target is None:
        return None
    source = target['target_action']['args']['source']
    io = json.loads(record['input_output'])
    for text, expected in zip(io['inputs'], io['outputs']):
        execution = _WORKER_SANDBOX.run(source, 'solve', [text])
        if (execution.status != 'RETURNED' or not isinstance(execution.value, str)
                or execution.value.split() != expected.split()):
            return None
    return target


def _paths():
    return {'taco_manifest': SOURCE/'manifest.json',
            'apps_source': previous.SOURCE,
            'diverse_request': diverse.ROOT/'request.json',
            'diverse_result': diverse.ROOT/'result.json',
            'diverse_training': diverse.DATA/'training.jsonl',
            'new_records': DATA/'new_records.jsonl',
            'new_targets': DATA/'new_targets.jsonl',
            'training': DATA/'training.jsonl'}


def prepare(workers=8):
    from tqdm import tqdm

    if DATA.exists() or ROOT.exists():
        raise FileExistsError('TACO executor SFT already prepared')
    diverse.check(trained=True)
    rejudge.check()
    candidates, rejected = _pool()
    old_targets, retention = _old_data()
    baseline = json.loads((diverse.ROOT/'request.json').read_text())
    framework = baseline['framework']
    previous._tokenizer(framework, baseline['adapters'])
    selected, targets, attempted = [], [], 0
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker,
                             initargs=(framework,)) as pool, tqdm(
            total=NEW_TARGETS, desc='TACO教师严格验证', unit='题',
            dynamic_ncols=True) as bar:
        for start in range(0, len(candidates), 64):
            batch = candidates[start:start+64]
            futures = {pool.submit(_verify_worker, row): i
                       for i, row in enumerate(batch)}
            verified = {}
            for future in as_completed(futures):
                verified[futures[future]] = future.result()
                attempted += 1
            for i, record in enumerate(batch):
                if verified[i] is not None and len(targets) < NEW_TARGETS:
                    selected.append(record)
                    targets.append(verified[i])
                    bar.update(1)
            bar.set_postfix(attempted=attempted)
            if len(targets) == NEW_TARGETS:
                break
    if len(targets) != NEW_TARGETS:
        raise ValueError(f'Only {len(targets)} strictly verified TACO teachers '
                         f'from {attempted} attempted candidates')
    all_targets = old_targets + targets
    if len({row['root_id'] for row in all_targets}) != TOTAL_TARGETS:
        raise ValueError('TACO and prior APPS target roots overlap')
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    (DATA/'new_records.jsonl').write_text(''.join(diverse._jsonl_line(row) for row in selected))
    (DATA/'new_targets.jsonl').write_text(''.join(diverse._jsonl_line(row) for row in targets))
    (DATA/'training.jsonl').write_text(''.join(
        diverse._jsonl_line(row) for row in diverse._schedule(all_targets, retention)))
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'framework': framework,
        'initial_executor': baseline['initial_executor'],
        'adapters': baseline['adapters'], 'adapter_sha256': baseline['adapter_sha256'],
        'steps': STEPS, 'learning_rate': baseline['learning_rate'],
        'seed': baseline['seed'], 'schedule_seed': diverse.SEED,
        'old_targets': OLD_TARGETS, 'new_targets': NEW_TARGETS,
        'target_examples': TOTAL_TARGETS, 'retention_examples': len(retention),
        'candidate_counts': rejected, 'verified_attempts': attempted,
        'selection_seed': SEED,
        'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS),
        'fresh_benchmark_roots': 39, 'mixed_validation_roots': 48,
        'retention_roots': 10,
        'source_sha256': {name: file_hash(path) for name, path in _paths().items()},
        'design': 'Same initial executor, action format, planner, learning rate, and one '
                  'exposure per target as DIVERSE; TACO train contest stdin teachers '
                  'strictly pass six original tests and do not overlap APPS by URL or question.',
    })
    write_json(ROOT/'gpu_resources_train.json',
               {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}})
    write_json(ROOT/'gpu_resources_eval.json',
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check(*, trained=False):
    request = json.loads((ROOT/'request.json').read_text())
    baseline = json.loads((diverse.ROOT/'request.json').read_text())
    if (request['status'] != 'PREPARED'
            or request['framework'] != baseline['framework']
            or request['initial_executor'] != baseline['initial_executor']
            or request['adapters'] != baseline['adapters']
            or request['adapter_sha256'] != baseline['adapter_sha256']
            or request['learning_rate'] != baseline['learning_rate']
            or request['seed'] != baseline['seed']
            or request['schedule_seed'] != diverse.SEED
            or request['selection_seed'] != SEED
            or request['steps'] != STEPS
            or request['old_targets'] != OLD_TARGETS
            or request['new_targets'] != NEW_TARGETS
            or request['target_examples'] != TOTAL_TARGETS
            or request['retention_examples'] != 177
            or request['fresh_benchmark_roots'] != 39
            or request['mixed_validation_roots'] != 48
            or request['retention_roots'] != 10
            or request['train_gpu'] != GPUS[0]
            or request['evaluation_gpus'] != list(GPUS)
            or request['source_sha256'] != {name: file_hash(path)
                                            for name, path in _paths().items()}
            or json.loads((ROOT/'gpu_resources_train.json').read_text()) !=
               {'devices': [GPUS[0]], 'min_free_mib': {'train': 22000}}
            or json.loads((ROOT/'gpu_resources_eval.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('Frozen TACO executor SFT design or launch configuration changed')
    records = read_jsonl(DATA/'new_records.jsonl')
    targets = read_jsonl(DATA/'new_targets.jsonl')
    training = read_jsonl(DATA/'training.jsonl')
    old_targets, retention = _old_data()
    excluded_urls, excluded_questions = _excluded()
    urls = [_url_key(row['url']) for row in records]
    questions = [_question_key(row['question']) for row in records]
    if (len(records) != NEW_TARGETS or len(targets) != NEW_TARGETS
            or len(set(urls)) != NEW_TARGETS
            or len(set(questions)) != NEW_TARGETS
            or set(urls) & excluded_urls or set(questions) & excluded_questions
            or [target['root_id'] for target in targets] !=
               [f'apps_train_{row["id"]:04d}' for row in records]
            or any(row['source'] != 'TACO' or row['taco_source'] not in CONTEST_SOURCES
                   for row in records)
            or any(row['kind'] != 'target' or row['split'] != 'TRAIN'
                   or row['prompt_tokens'] + row['target_tokens'] > 8192
                   or row['target_tokens'] > 3072 for row in targets)
            or training != diverse._schedule(old_targets+targets, retention)):
        raise ValueError('TACO training labels, overlap exclusions, or schedule changed')
    if trained:
        result = json.loads((ROOT/'training/result.json').read_text())
        adapter = ROOT/'training/final_adapter/adapter_model.safetensors'
        if (result['status'] != 'COMPLETE' or result['steps'] != STEPS
                or result['target_exposures'] != TOTAL_TARGETS
                or result['retention_exposures'] != TOTAL_TARGETS
                or result['adapter_sha256'] != file_hash(adapter)):
            raise ValueError('TACO executor training incomplete or changed')
    ready = {'status': 'READY', 'old_targets': OLD_TARGETS,
             'new_TACO_targets': NEW_TARGETS, 'total_targets': TOTAL_TARGETS,
             'steps': STEPS, 'attempts': request['verified_attempts'],
             'train_gpu': GPUS[0], 'evaluation_gpus': list(GPUS),
             'trained': trained}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def train():
    from .gpu_resource import acquire
    from .refinement_executor_sft import train as train_executor

    check()
    reservation = acquire(ROOT/'gpu_resources_train.json', 'train')
    try:
        train_executor(ROOT, DATA,
                       json.loads((ROOT/'request.json').read_text())['initial_executor'],
                       progress=True)
    finally:
        reservation.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'train'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.workers)
    elif args.command == 'check':
        check(trained=args.trained)
    else:
        train()


if __name__ == '__main__':
    main()
