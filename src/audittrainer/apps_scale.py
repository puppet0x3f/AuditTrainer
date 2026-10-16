"""Matched-step APPS executor data-scale study with fixed validation and probes.

prepare/check are CPU-only. Train and evaluation are separate user-run commands.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from . import apps_multitask as previous
from .apps import convert_record, stdin_text
from .apps_next import PROBE_INPUTS as EARLIER_INPUTS
from .apps_scale_probes import EXTRA_INPUTS, oracle, valid_output
from .artifacts import ArtifactStore
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .prompts import PromptOverflow
from .schema import Artifact
from .verifier import canonical, digest


DATA = Path('data/refinement/apps_scale')
ROOT = Path('runs/refinement/apps_scale')
OLD_DATA = Path('data/refinement/apps_multitask')
OLD_RUN = Path('runs/refinement/apps_multitask')
NEXT_DATA = Path('data/refinement/apps_next')
NEXT_RUN = Path('runs/refinement/apps_next')
CORRECTION_DATA = Path('data/refinement/contract_correction')
CORRECTION_RUN = Path('runs/refinement/contract_correction')
GPU_IDS = ['0', '1', '2']
EXTRA_QUOTAS = {'competition': 96, 'interview': 416, 'introductory': 64}
ARMS = ('SMALL', 'LARGE')
CONTROL_ARMS = ('BASE', 'OLD96', 'INTERFACE')
STEPS = 384


def _validation_rows():
    return sorted((row for row in load_root_manifest(OLD_DATA/'roots.jsonl').values()
                   if row['split'] == 'VALIDATION'), key=lambda row: row['root_id'])


def _retention_rows():
    request = json.loads((CORRECTION_RUN/'request.json').read_text())
    selections = [row for row in request['selection'] if row['group'] == 'retention']
    roots = load_root_manifest(CORRECTION_DATA/'roots.jsonl')
    return [(selection, roots[selection['root_id']]) for selection in selections]


def _old_case(arm, root_id):
    if arm == 'BASE':
        return OLD_RUN/'evaluation/BEFORE'/root_id/'NATURAL/case_result.json'
    if arm == 'OLD96':
        return OLD_RUN/'evaluation/AFTER'/root_id/'NATURAL/case_result.json'
    if arm == 'INTERFACE':
        return NEXT_RUN/'evaluation/INTERFACE'/root_id/'NATURAL/case_result.json'
    return ROOT/'evaluation'/arm/'APPS'/root_id/'NATURAL/case_result.json'


def _case(arm, scope, root_id):
    return _old_case(arm, root_id) if scope == 'APPS' and arm in CONTROL_ARMS else (
        ROOT/'evaluation'/arm/scope/root_id/('NATURAL/case_result.json' if scope == 'APPS' else 'case_result.json'))


def _source_ids(records):
    return {f"apps_train_{int(record['id']):04d}" for record in records}


def prepare():
    """Continue the pinned APPS selection stream; validate 576 new teachers."""
    from tqdm import tqdm

    previous.check(after=True)
    if file_hash(previous.SOURCE) != previous.SOURCE_SHA256:
        raise ValueError('APPS source changed')
    if DATA.exists() or ROOT.exists():
        raise FileExistsError('Scale study already exists; refusing to change frozen selection')
    old = json.loads((OLD_RUN/'request.json').read_text())
    original = read_jsonl(OLD_DATA/'training.jsonl')
    targets = [row for row in original if row['kind'] == 'target']
    retention = [row for row in original if row['kind'] == 'retention']
    if len(targets) != 192 or len(retention) != 177:
        raise ValueError('Unexpected original training set')
    tokenizer = previous._tokenizer(old['framework'], old['adapters'])
    excluded = set(load_root_manifest('data/manifests/roots.jsonl'))
    buckets = {difficulty: [] for difficulty in EXTRA_QUOTAS}
    for record in read_jsonl(previous.SOURCE):
        difficulty = record.get('difficulty')
        if difficulty not in buckets or f"apps_train_{int(record.get('id', -1)):04d}" in excluded:
            continue
        if not isinstance(record.get('question'), str) or len(record['question']) > 9000:
            continue
        try:
            convert_record(record, original_split='train', split='TRAIN', visible_cases=None)
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        buckets[difficulty].append(record)
    old_selected = read_jsonl(OLD_DATA/'train_records.jsonl') + read_jsonl(OLD_DATA/'validation_records.jsonl')
    for difficulty, values in buckets.items():
        values.sort(key=previous._rank)
        prefix = values[:old['attempts'][difficulty]]
        wanted = _source_ids(record for record in old_selected if record['difficulty'] == difficulty)
        if not wanted <= _source_ids(prefix):
            raise ValueError(f'Original {difficulty} selection is not a prefix of pinned stream')
    DATA.mkdir(parents=True); ROOT.mkdir(parents=True)
    (DATA/'SMALL').mkdir(); (DATA/'LARGE').mkdir()
    (ROOT/'SMALL').mkdir(); (ROOT/'LARGE').mkdir()
    extra_targets, extra_records = [], []
    attempts = Counter()
    already = {row['root_id'] for row in targets} | {row['root_id'] for row in _validation_rows()}
    families = {row['family_id'] for row in load_root_manifest(OLD_DATA/'roots.jsonl').values()}
    with tempfile.TemporaryDirectory(prefix='apps_scale_teacher_') as temp, tqdm(
            total=sum(EXTRA_QUOTAS.values()), desc='APPS 扩量参考解验证', unit='题', dynamic_ncols=True) as bar:
        scratch = Path(temp)
        for difficulty in sorted(EXTRA_QUOTAS):
            for record in buckets[difficulty][old['attempts'][difficulty]:]:
                if sum(row['difficulty'] == difficulty for row in extra_records) >= EXTRA_QUOTAS[difficulty]:
                    break
                attempts[difficulty] += 1
                try:
                    with tempfile.TemporaryDirectory(dir=scratch) as one:
                        target = previous._target(record, 'TRAIN', old['framework'], tokenizer, Path(one))
                except (ValueError, KeyError, TypeError, AttributeError, PromptOverflow, json.JSONDecodeError):
                    target = None
                if target is None:
                    continue
                public, _ = convert_record(record, original_split='train', split='TRAIN', visible_cases=None)
                rid, family = public['task']['root_id'], public['task']['family_id']
                if rid in already or family in families:
                    raise ValueError(f'Duplicate APPS root or family: {rid}')
                already.add(rid); families.add(family)
                extra_targets.append(target); extra_records.append(record)
                bar.update(1)
            if sum(row['difficulty'] == difficulty for row in extra_records) != EXTRA_QUOTAS[difficulty]:
                raise ValueError(f'Insufficient verified {difficulty} problems')
    (DATA/'LARGE/extra_records.jsonl').write_text(''.join(canonical(row)+'\n' for row in extra_records))
    for arm, examples in [('SMALL', targets + retention), ('LARGE', targets + extra_targets + retention)]:
        training = DATA/arm/'training.jsonl'
        training.write_text(''.join(canonical(row)+'\n' for row in examples))
        write_json(ROOT/arm/'request.json', {'framework': old['framework'], 'seed': old['seed'],
                    'steps': STEPS, 'learning_rate': old['learning_rate'], 'training_sha256': file_hash(training),
                    'target_count': sum(row['kind'] == 'target' for row in examples),
                    'retention_count': len(retention), 'initial_executor': old['adapters']['executor_before']})
        write_json(ROOT/arm/'gpu_resources.json', {'devices': [GPU_IDS[0 if arm == 'SMALL' else 1]],
                   'min_free_mib': {'train': 22000}})
    old_probes = json.loads((NEXT_DATA/'probes.json').read_text())
    union = set()
    for row in _validation_rows():
        rid = row['root_id']
        if any(json.loads(_old_case(arm, rid).read_text())['final_scores']['private']['status'] == 'PASS'
               for arm in CONTROL_ARMS):
            union.add(rid)
    if union != set(EARLIER_INPUTS) | set(EXTRA_INPUTS) or len(old_probes) != 24:
        raise ValueError('Probe roots must equal union of original-test successes')
    rows = {row['root_id']: row for row in _validation_rows()}
    probes = list(old_probes)
    for rid, texts in EXTRA_INPUTS.items():
        raw = json.loads(Path(rows[rid]['raw_path']).read_text())
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        original_inputs = {stdin_text(value).strip() for value in io['inputs']}
        if len(texts) != 3:
            raise ValueError(f'Expected three independent probes: {rid}')
        for index, text in enumerate(texts):
            if text.strip() in original_inputs:
                raise ValueError(f'Probe repeats original APPS input: {rid}/{index}')
            probes.append({'root_id': rid, 'case_id': f'{rid}_probe_{index+1}',
                           'input': text, 'expected': oracle(rid, text)})
    write_json(DATA/'probes.json', probes)
    source_hashes = {name: file_hash(path) for name, path in {
        'original_request': OLD_RUN/'request.json',
        'original_training': OLD_DATA/'training.jsonl',
        'original_result': OLD_RUN/'result.json',
        'previous_control_result': NEXT_RUN/'result.json',
        'correction_request': CORRECTION_RUN/'request.json',
        'correction_roots': CORRECTION_DATA/'roots.jsonl',
        'correction_tests': CORRECTION_DATA/'tests.jsonl',
        'apps_source': previous.SOURCE,
        'extra_records': DATA/'LARGE/extra_records.jsonl',
        'small_training': DATA/'SMALL/training.jsonl',
        'large_training': DATA/'LARGE/training.jsonl',
        'probes': DATA/'probes.json',
    }.items()}
    write_json(ROOT/'request.json', {
        'source_hashes': source_hashes, 'framework': old['framework'],
        'initial_executor': old['adapters']['executor_before'],
        'adapters': old['adapters'], 'adapter_sha256': old['adapter_sha256'],
        'validation_roots': 48, 'retention_roots': len(_retention_rows()),
        'small_targets': 192, 'large_targets': 768, 'extra_targets': 576,
        'retention_examples': 177, 'steps': STEPS, 'eval_seed': old['eval_seed'],
        'training_seed': old['seed'], 'learning_rate': old['learning_rate'],
        'extra_quotas': EXTRA_QUOTAS, 'extra_attempts': dict(attempts),
        'probe_roots': len(union), 'probe_cases': len(probes),
        'design': 'One 192-target arm repeats each APPS target four times; one 768-target arm sees each target once. Both use 384 updates, two APPS targets plus two identical-cycle retention examples per update, the same initial executor, seed and learning rate. Planner adapters and 48 validation roots are frozen; no checkpoint selection.',
    })
    write_json(ROOT/'gpu_resources.json', {'devices': GPU_IDS,
               'min_free_mib': {'collect': 16000}})
    check()


def check(*, trained=False):
    previous.check(after=True)
    request = json.loads((ROOT/'request.json').read_text())
    paths = {
        'original_request': OLD_RUN/'request.json',
        'original_training': OLD_DATA/'training.jsonl',
        'original_result': OLD_RUN/'result.json',
        'previous_control_result': NEXT_RUN/'result.json',
        'correction_request': CORRECTION_RUN/'request.json',
        'correction_roots': CORRECTION_DATA/'roots.jsonl',
        'correction_tests': CORRECTION_DATA/'tests.jsonl',
        'apps_source': previous.SOURCE,
        'extra_records': DATA/'LARGE/extra_records.jsonl',
        'small_training': DATA/'SMALL/training.jsonl',
        'large_training': DATA/'LARGE/training.jsonl',
        'probes': DATA/'probes.json',
    }
    if any(file_hash(path) != request['source_hashes'][name] for name, path in paths.items()):
        raise ValueError('Frozen scale-study input changed')
    old = json.loads((OLD_RUN/'request.json').read_text())
    if (request['source_hashes']['apps_source'] != previous.SOURCE_SHA256
            or request['framework'] != old['framework']
            or request['adapters'] != old['adapters']
            or request['adapter_sha256'] != old['adapter_sha256']
            or request['initial_executor'] != old['adapters']['executor_before']
            or request['training_seed'] != old['seed'] or request['eval_seed'] != old['eval_seed']
            or request['learning_rate'] != old['learning_rate'] or request['steps'] != STEPS
            or request['extra_quotas'] != EXTRA_QUOTAS):
        raise ValueError('Original model, adapters, split or schedule changed')
    gpu = json.loads((ROOT/'gpu_resources.json').read_text())
    if gpu != {'devices': GPU_IDS, 'min_free_mib': {'collect': 16000}}:
        raise ValueError('Expected GPU 0, 1 and 2 for evaluation')
    earlier = read_jsonl(OLD_DATA/'training.jsonl')
    targets = [row for row in earlier if row['kind'] == 'target']
    retention = [row for row in earlier if row['kind'] == 'retention']
    extra = read_jsonl(DATA/'LARGE/extra_records.jsonl')
    small = read_jsonl(DATA/'SMALL/training.jsonl')
    large = read_jsonl(DATA/'LARGE/training.jsonl')
    if (small != earlier or large[:len(targets)] != targets
            or large[len(targets)+len(extra):] != retention
            or len(extra) != 576 or len(small) != 369 or len(large) != 945
            or Counter(row['difficulty'] for row in extra) != Counter(EXTRA_QUOTAS)):
        raise ValueError('Matched train and retention schedules are inconsistent')
    extra_targets = large[len(targets):len(targets)+len(extra)]
    small_schedule = Counter(targets[(2*step+offset) % 192]['sample_id']
                             for step in range(STEPS) for offset in (0, 1))
    large_targets = targets + extra_targets
    large_schedule = Counter(large_targets[(2*step+offset) % 768]['sample_id']
                             for step in range(STEPS) for offset in (0, 1))
    if (len(small_schedule) != 192 or set(small_schedule.values()) != {4}
            or len(large_schedule) != 768 or set(large_schedule.values()) != {1}):
        raise ValueError('Target exposure budgets are not 192 x 4 versus 768 x 1')
    old_roots = load_root_manifest(OLD_DATA/'roots.jsonl')
    if (len({row['root_id'] for row in targets + extra_targets}) != 768
            or any(row['kind'] != 'target' or row['split'] != 'TRAIN' for row in extra_targets)
            or any(row['root_id'] in old_roots for row in extra_targets)):
        raise ValueError('Extra APPS target roots overlap the frozen split')
    for record, target in zip(extra, extra_targets):
        public, _ = convert_record(record, original_split='train', split='TRAIN', visible_cases=None)
        if public['task']['root_id'] != target['root_id']:
            raise ValueError('Extra target differs from its pinned APPS source record')
        if target['prompt_tokens'] + target['target_tokens'] > 8192 or target['target_tokens'] > 3072:
            raise ValueError('Extra target exceeds the frozen context or generation cap')
    for arm, n in (('SMALL', 192), ('LARGE', 768)):
        arm_request = json.loads((ROOT/arm/'request.json').read_text())
        resources = json.loads((ROOT/arm/'gpu_resources.json').read_text())
        if (arm_request['framework'] != old['framework'] or arm_request['steps'] != STEPS
                or arm_request['seed'] != old['seed'] or arm_request['learning_rate'] != old['learning_rate']
                or arm_request['target_count'] != n or arm_request['retention_count'] != 177
                or arm_request['training_sha256'] != file_hash(DATA/arm/'training.jsonl')
                or arm_request['initial_executor'] != request['initial_executor']
                or resources != {'devices': [GPU_IDS[0 if arm == 'SMALL' else 1]],
                                 'min_free_mib': {'train': 22000}}):
            raise ValueError(f'{arm} training launch differs from matched design')
        if trained:
            result = json.loads((ROOT/arm/'training/result.json').read_text())
            if (result['status'] != 'COMPLETE' or result['steps'] != STEPS
                    or result['target_exposures'] != 768 or result['retention_exposures'] != 768
                    or result['adapter_sha256'] != file_hash(ROOT/arm/'training/final_adapter/adapter_model.safetensors')):
                raise ValueError(f'{arm} training incomplete or changed')
    probes = json.loads((DATA/'probes.json').read_text())
    if (len(probes) != 48 or len({p['case_id'] for p in probes}) != 48
            or {p['root_id'] for p in probes} != set(EARLIER_INPUTS) | set(EXTRA_INPUTS)):
        raise ValueError('Independent probe set changed')
    roots = {row['root_id']: row for row in _validation_rows()}
    for probe in probes:
        if probe['expected'] != oracle(probe['root_id'], probe['input']):
            raise ValueError(f'Probe oracle changed: {probe["case_id"]}')
        raw = json.loads(Path(roots[probe['root_id']]['raw_path']).read_text())
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        if probe['input'].strip() in {stdin_text(value).strip() for value in io['inputs']}:
            raise ValueError(f'Probe repeats original input: {probe["case_id"]}')
    correction_tests = {row['root_id']: row for row in read_jsonl(CORRECTION_DATA/'tests.jsonl')}
    for _, row in _retention_rows():
        if (file_hash(row['fixture_path']) != row['source_hash']
                or file_hash(correction_tests[row['root_id']]['path']) != correction_tests[row['root_id']]['sha256']):
            raise ValueError(f'Retention task changed: {row["root_id"]}')
    if request['validation_roots'] != len(roots) or request['retention_roots'] != len(_retention_rows()):
        raise ValueError('Evaluation root count changed')
    result = {'status': 'READY', 'training_steps_per_arm': STEPS,
              'small_targets': 192, 'large_targets': 768, 'retention_exposures_per_arm': 768,
              'validation_roots': len(roots), 'retention_roots': len(_retention_rows()),
              'probe_roots': len(set(p['root_id'] for p in probes)), 'probe_cases': len(probes),
              'train_gpus': {'SMALL': '0', 'LARGE': '1'}, 'evaluation_gpus': GPU_IDS}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def train(arm):
    from .gpu_resource import acquire
    from .refinement_executor_sft import train as train_executor

    if arm not in ARMS:
        raise ValueError('Expected SMALL or LARGE training arm')
    check()
    reservation = acquire(ROOT/arm/'gpu_resources.json', 'train')
    try:
        train_executor(ROOT/arm, DATA/arm, json.loads((ROOT/'request.json').read_text())['initial_executor'], progress=True)
    finally:
        reservation.close()


def _context(arm):
    from .refinement_policy import load_policies

    request = json.loads((ROOT/'request.json').read_text())
    adapter = {'BASE': request['initial_executor'],
               'OLD96': str(OLD_RUN/'training/final_adapter'),
               'SMALL': str(ROOT/'SMALL/training/final_adapter'),
               'LARGE': str(ROOT/'LARGE/training/final_adapter')}[arm]
    base, executor = load_policies(request['framework'], request['adapters']['internal_gate'], adapter)
    base.model.load_adapter(request['adapters']['body'], adapter_name='body', is_trainable=False, local_files_only=True)
    base.model.load_adapter(request['adapters']['root_gate'], adapter_name='root_gate', is_trainable=False, local_files_only=True)
    base.model.set_adapter('default'); base.model.requires_grad_(False)
    executor.llm.identity['apps_scale_arm'] = arm
    return previous.CurrentPlanner(base), executor


def _retention_planner(base, selection, definition):
    from .refinement_contract_correction import previous as course
    from .refinement_handoff_sft import NestedPlanner
    from .refinement_responsibility_course import internal_variant

    if 'variant' in selection:
        return course.study.fixed(base, definition, selection['variant'])
    if 'first' in selection:
        return course.study.fixed(base, definition, internal_variant(selection['first']))
    return NestedPlanner(base, definition)


def evaluate(arm, shard, shards=3):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if arm not in ('BASE', 'OLD96', *ARMS) or shards != 3 or not 0 <= shard < shards:
        raise ValueError('Expected a configured arm and one of three GPU shards')
    check(trained=True)
    request = json.loads((ROOT/'request.json').read_text())
    old = json.loads((OLD_RUN/'request.json').read_text())
    apps = _validation_rows() if arm in ARMS else []
    retain = _retention_rows()
    apps_tests = {row['root_id']: row for row in read_jsonl(OLD_DATA/'tests.jsonl')}
    retain_tests = {row['root_id']: row for row in read_jsonl(CORRECTION_DATA/'tests.jsonl')}
    jobs = [('APPS', i, row, None) for i, row in enumerate(apps) if i % shards == shard]
    jobs += [('RETENTION', i, row, selection) for i, (selection, row) in enumerate(retain) if i % shards == shard]
    done = ROOT/'evaluation'/arm/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed evaluation shard already exists: {done}')
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        base, executor = _context(arm)
        for scope, index, row, selection in tqdm(jobs, desc=f'APPS规模 {arm} GPU {GPU_IDS[shard]}',
                                                  unit='题', dynamic_ncols=True):
            rid = row['root_id']
            definition = json.loads(Path(row['fixture_path']).read_text()) if selection else None
            planner = base if selection is None else _retention_planner(base, selection, definition)
            tests = apps_tests if scope == 'APPS' else retain_tests
            seed = request['eval_seed'] + index + (100000 if scope == 'RETENTION' else 0)
            case = run_case(row, tests, old['framework'], old['config'], planner, executor,
                            _case(arm, scope, rid).parent, seed)
            if case['status'] != 'COMPLETE':
                raise RuntimeError(f'Incomplete {arm}/{scope}/{rid}: {case["errors"]}')
        done.parent.mkdir(parents=True, exist_ok=True)
        write_json(done, {'status': 'COMPLETE', 'arm': arm, 'shard': shard, 'shards': shards,
                          'apps_roots': len([x for x in jobs if x[0] == 'APPS']),
                          'retention_roots': len([x for x in jobs if x[0] == 'RETENTION'])})
    finally:
        reservation.close()


def score_probes():
    from tqdm import tqdm
    from .sandbox import ExecutionLimits, Sandbox

    check(trained=True)
    output = ROOT/'probe_results.json'
    if output.exists():
        raise FileExistsError('Scale probe results already exist')
    framework = json.loads((ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    probes = json.loads((DATA/'probes.json').read_text())
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe['root_id'], []).append(probe)
    results = []
    for arm in tqdm(('BASE', 'INTERFACE', 'OLD96', *ARMS), desc='独立输入模型', unit='组', dynamic_ncols=True):
        for rid in tqdm(sorted(grouped), desc=arm, unit='题', leave=False, dynamic_ncols=True):
            case = json.loads(_old_case(arm, rid).read_text())
            snapshot = case['final_program']
            if snapshot:
                bundle = ArtifactStore(Path(snapshot['path'])).bundle(Artifact.model_validate(snapshot['artifact']))
                source = bundle.sources['N0']
            for probe in grouped[rid]:
                execution = sandbox.run(source, 'solve', [probe['input']], bundle=bundle) if snapshot else None
                passed = bool(execution and execution.status == 'RETURNED' and
                              valid_output(rid, probe['input'], execution.value, probe['expected']))
                results.append({'arm': arm, 'root_id': rid, 'case_id': probe['case_id'], 'pass': passed,
                                'status': execution.status if execution else 'NO_PROGRAM',
                                'reason': execution.reason if execution else 'NO_PROGRAM',
                                'actual': execution.value if execution else None})
    write_json(output, {'status': 'COMPLETE', 'cases': results})
    print(json.dumps({'status': 'COMPLETE', 'runs': len(results),
                      'passed': {arm: sum(r['pass'] for r in results if r['arm'] == arm)
                                 for arm in ('BASE', 'INTERFACE', 'OLD96', *ARMS)}}, ensure_ascii=False), flush=True)


def _valid_private(case):
    private = case['final_scores']['private']
    if (private['status'] != 'PASS' and case['root_id'] != 'apps_train_1311') or not private['suites']:
        return False
    for suite in private['suites']:
        evidence = json.loads(Path(suite['evidence_path']).read_text())
        for item in evidence['details']['cases']:
            if not valid_output(case['root_id'], item['args'][0], item['execution']['value'], item['expected']):
                return False
    return True


def report():
    check(trained=True)
    request = json.loads((ROOT/'request.json').read_text())
    validation = _validation_rows()
    retention = _retention_rows()
    for arm in ('BASE', 'OLD96', *ARMS):
        for shard in range(3):
            done = json.loads((ROOT/'evaluation'/arm/f'shard_{shard}.json').read_text())
            if (done['status'] != 'COMPLETE' or done['arm'] != arm or done['shard'] != shard
                    or done['shards'] != 3):
                raise ValueError(f'Incomplete evaluation: {arm}/{shard}')
    scored = json.loads((ROOT/'probe_results.json').read_text())
    if scored['status'] != 'COMPLETE' or len(scored['cases']) != 5*request['probe_cases']:
        raise ValueError('Probe scoring incomplete')
    cases = {arm: {row['root_id']: json.loads(_old_case(arm, row['root_id']).read_text())
                   for row in validation} for arm in ('BASE', 'INTERFACE', 'OLD96', *ARMS)}
    result = {'status': 'COMPLETE', 'training': {}, 'original_private_valid_pass': {},
              'independent_probes': {}, 'retention_pass': {}, 'paired': {}}
    for arm in ARMS:
        trained = json.loads((ROOT/arm/'training/result.json').read_text())
        result['training'][arm] = {key: trained[key] for key in ('status', 'steps', 'target_exposures',
                                                                    'retention_exposures', 'adapter_sha256')}
    for arm, by_root in cases.items():
        result['original_private_valid_pass'][arm] = {
            'total': sum(_valid_private(case) for case in by_root.values()),
            'by_difficulty': {difficulty: sum(_valid_private(case) for rid, case in by_root.items()
                    if json.loads(Path(next(row for row in validation if row['root_id'] == rid)['raw_path']).read_text())['difficulty'] == difficulty)
                for difficulty in ('competition', 'interview', 'introductory')},
        }
        items = [row for row in scored['cases'] if row['arm'] == arm]
        result['independent_probes'][arm] = {
            'cases_pass': sum(row['pass'] for row in items),
            'roots_all_three_pass': sum(all(row['pass'] for row in items if row['root_id'] == rid)
                                        for rid in sorted({row['root_id'] for row in items})),
        }
    for arm in ('BASE', 'OLD96', *ARMS):
        old = {row['root_id']: json.loads(_case(arm, 'RETENTION', row['root_id']).read_text())
               for _, row in retention}
        result['retention_pass'][arm] = {'total': sum(case['final_scores']['private']['status'] == 'PASS'
                                                      for case in old.values()),
                                          'by_root': {rid: case['final_scores']['private']['status'] == 'PASS'
                                                      for rid, case in old.items()}}
    for left, right in [('BASE', 'OLD96'), ('OLD96', 'SMALL'), ('OLD96', 'LARGE'), ('SMALL', 'LARGE')]:
        paired = Counter()
        for row in validation:
            a, b = _valid_private(cases[left][row['root_id']]), _valid_private(cases[right][row['root_id']])
            paired['wins' if b and not a else 'losses' if a and not b else 'ties'] += 1
        result['paired'][f'{right}_vs_{left}'] = dict(paired)
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'score-probes', 'report'])
    parser.add_argument('--arm', choices=('BASE', 'OLD96', *ARMS))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'train':
        if args.arm not in ARMS:
            parser.error('train requires --arm SMALL or LARGE')
        train(args.arm)
    elif args.command == 'evaluate':
        if args.arm is None or args.arm == 'INTERFACE':
            parser.error('evaluate requires --arm BASE, OLD96, SMALL or LARGE')
        evaluate(args.arm, args.shard, args.shards)
    elif args.command == 'score-probes':
        score_probes()
    else:
        report()


if __name__ == '__main__':
    main()
