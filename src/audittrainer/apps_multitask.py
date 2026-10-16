"""APPS train extraction, verified executor demonstrations, and held-out free rollouts.

Preparation and check are CPU-only. Train/evaluate are deliberately separate commands.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap

from .apps import convert_record, import_apps
from .artifacts import ArtifactStore, SourceError
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import score_program, write_json
from .policy import load_adapter_metadata
from .prompts import PromptOverflow
from .refinement_parent_composition import summary_prompt
from .refinement_policy import atomic_plan
from .schema import Contract, ContractSpec, PolicyOutput, RootTask, TokenUsage
from .verifier import canonical, digest

SOURCE = Path('data/apps_source/train_full.jsonl')
SOURCE_SHA256 = '45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae'
SOURCE_URL = 'https://huggingface.co/datasets/codeparrot/apps/resolve/21e74ddf8de1a21436da12e3e653065c5213e9d1/train.jsonl'
DATA = Path('data/refinement/apps_multitask')
ROOT = Path('runs/refinement/apps_multitask')
GATE_CONTRAST = Path('runs/refinement/gate_contrast/request.json')
RETENTION = Path('data/refinement/contract_correction/training.jsonl')
QUOTAS = {'TRAIN': {'interview': 128, 'introductory': 32, 'competition': 32},
          'VALIDATION': {'interview': 32, 'introductory': 8, 'competition': 8}}
SEED = 713


def _rank(record):
    return hashlib.sha256(f"{SEED}:{record['difficulty']}:{record['id']}".encode()).hexdigest()


def _solutions(record):
    try:
        sources = json.loads(record['solutions']) if isinstance(record['solutions'], str) else record['solutions']
    except (ValueError, TypeError):
        return []
    if not isinstance(sources, list):
        return []
    return sorted((s for s in sources if isinstance(s, str) and s.strip()), key=len)[:8]


def source_to_solve(script):
    """Adapt a stdin script to the contract using only in-memory input/output."""
    import ast

    tree = ast.parse(script)
    if any(isinstance(node, ast.ImportFrom) and node.module == '__future__' for node in ast.walk(tree)):
        raise ValueError('FUTURE_IMPORT')
    if any(isinstance(node, ast.Global) and {'print', 'input'} & set(node.names) for node in ast.walk(tree)):
        raise ValueError('GLOBAL_IO')
    if any(isinstance(node, ast.Attribute) and node.attr in {'stdin', 'stdout', 'stderr'} for node in ast.walk(tree)):
        raise ValueError('DIRECT_STANDARD_STREAM')
    if any(isinstance(node, ast.ImportFrom) and node.module == 'sys'
           and any(alias.name in {'stdin', 'stdout', 'stderr'} for alias in node.names) for node in ast.walk(tree)):
        raise ValueError('DIRECT_STANDARD_STREAM')
    body = textwrap.indent(script.rstrip() + '\n', '        ')
    source = (
        'def solve(text):\n'
        '    _lines = iter(text.splitlines())\n'
        '    _output = []\n'
        '    def input(_prompt=None):\n'
        '        try:\n'
        '            return next(_lines)\n'
        '        except StopIteration:\n'
        '            raise EOFError\n'
        '    def print(*values, sep=" ", end="\\n", flush=False):\n'
        '        _output.append((" " if sep is None else sep).join(map(str, values)) + ("\\n" if end is None else end))\n'
        '    def _run():\n'
        '        __name__ = "__main__"\n'
        + body +
        '    _run()\n'
        '    return "".join(_output)\n'
    )
    compile(source, '<apps_solve>', 'exec')
    return source


class _AtomicTeacher:
    identity = {'backend': 'apps_reference_observation'}

    def act(self, observation):
        action = atomic_plan(observation['incoming_contract'])
        return PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(), model_identity=self.identity)


def _observation(public, framework, path):
    from .runtime import RecursiveRuntime
    from .refinement_rollout import framework_limits

    task = RootTask.model_validate(public['task'])
    spec = ContractSpec.model_validate(public['contract'])
    from .cli import fixture_verifier
    verifier = fixture_verifier(public, framework)
    runtime = RecursiveRuntime(task, spec, _AtomicTeacher(), path, verifier=verifier,
                               limits=framework_limits(framework), protocol_version=framework['protocol_version'],
                               implementation_stage='PR0', require_obligation_coverage=True)
    event = runtime.step()
    if event.host_result.code != 'PLAN_REGISTERED':
        raise ValueError('ATOMIC_REFERENCE_PLAN_REJECTED')
    observation = runtime.build_observation()
    if observation['phase'] != 'IMPLEMENT':
        raise ValueError('EXPECTED_IMPLEMENT')
    return observation


def _teacher_passes(public, private, framework, source, path):
    from .cli import fixture_verifier

    complete = {**public, 'suites': public['suites'] + private['suites']}
    verifier = fixture_verifier(complete, framework)
    spec = ContractSpec.model_validate(public['contract'])
    contract = Contract(**spec.model_dump(), contract_id='C_N0_1', version=1)
    store = ArtifactStore(path)
    artifact = store.save_initial('N0', contract, source)
    scores, _ = score_program({'path': str(path), 'artifact': artifact.model_dump(),
                               'contract': contract.model_dump()}, verifier, public['task']['root_id'])
    return scores['private']['status'] == 'PASS' and scores['visible']['status'] in {'PASS', 'UNKNOWN'}


def _target(record, split, framework, tokenizer, scratch):
    public, private = convert_record(record, original_split='train', split=split, visible_cases=None)
    obs = _observation(public, framework, scratch/'observation')
    prompt = summary_prompt(tokenizer, obs, context_tokens=framework['runtime']['context_tokens'],
                            generation_tokens=framework['runtime']['generation_tokens'])
    for index, script in enumerate(_solutions(record)):
        if len(script) > 5500:
            continue
        try:
            source = source_to_solve(script)
            action = {'action': 'write_code', 'args': {'source': source,
                      'contract_version': obs['incoming_contract']['version'], 'declared_dependencies': []}}
            token_count = len(tokenizer.encode(canonical(action), add_special_tokens=False)) + 1
            if token_count > framework['runtime']['generation_tokens']['write_code'] or len(prompt.token_ids) + token_count > framework['runtime']['context_tokens']:
                continue
            if not _teacher_passes(public, private, framework, source, scratch/f'solution_{index}'):
                continue
            return {'sample_id': public['task']['root_id'] + '_N0', 'root_id': public['task']['root_id'],
                    'split': split, 'kind': 'target', 'observation': obs, 'target_action': action,
                    'solution_index': index, 'source_sha256': hashlib.sha256(script.encode()).hexdigest(),
                    'target_tokens': token_count, 'prompt_tokens': len(prompt.token_ids)}
        except (SyntaxError, ValueError, SourceError, OSError):
            continue
    return None


def _framework_and_adapters():
    request = json.loads(GATE_CONTRAST.read_text())
    framework = deepcopy(request['framework'])
    adapters = {'internal_gate': str(GATE_CONTRAST.parent/'training/final_adapter'),
                'root_gate': request['adapters']['gate'], 'body': request['adapters']['body'],
                'executor_before': request['adapters']['executor']}
    return framework, adapters


def _tokenizer(framework, adapters):
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(framework['model']['local_path'], local_files_only=True)
    for path in adapters.values():
        load_adapter_metadata(Path(path), framework['model'], base)
    return base


def prepare(source=SOURCE, data=DATA, root=ROOT, quotas=QUOTAS, *, expected_sha=SOURCE_SHA256):
    """Select disjoint APPS train roots; validate teacher programs before SFT."""
    from tqdm import tqdm

    source, data, root = Path(source), Path(data), Path(root)
    if expected_sha is not None and file_hash(source) != expected_sha:
        raise ValueError('APPS train source differs from pinned revision')
    if data.exists() or root.exists():
        raise FileExistsError('APPS multitask output already exists; refusing to change frozen selection')
    framework, adapters = _framework_and_adapters()
    tokenizer = _tokenizer(framework, adapters)
    excluded = set(load_root_manifest('data/manifests/roots.jsonl'))
    buckets = {difficulty: [] for difficulty in sorted({d for q in quotas.values() for d in q})}
    for record in read_jsonl(source):
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
    for values in buckets.values():
        values.sort(key=_rank)
    total = sum(sum(q.values()) for q in quotas.values())
    selected = {split: [] for split in quotas}
    targets = []
    attempts = Counter()
    data.mkdir(parents=True)
    root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='apps_teacher_') as tmp, tqdm(total=total, desc='APPS train 抽取与参考解验证', unit='题', dynamic_ncols=True) as bar:
        scratch = Path(tmp)
        for difficulty, candidates in buckets.items():
            cursor = 0
            # Fix the held-out roots first, before any model training or results.
            for split in ('VALIDATION', 'TRAIN'):
                needed = quotas[split].get(difficulty, 0)
                while len([r for r in selected[split] if r['difficulty'] == difficulty]) < needed:
                    if cursor >= len(candidates):
                        raise ValueError(f'Insufficient verified {difficulty} {split} APPS records')
                    record = candidates[cursor]
                    cursor += 1
                    attempts[difficulty] += 1
                    try:
                        with tempfile.TemporaryDirectory(dir=scratch) as one:
                            target = _target(record, split, framework, tokenizer, Path(one))
                    except (ValueError, KeyError, TypeError, AttributeError, PromptOverflow, json.JSONDecodeError):
                        target = None
                    if target is None:
                        continue
                    selected[split].append(record)
                    if split == 'TRAIN':
                        targets.append(target)
                    bar.update(1)
    root_manifest, test_manifest = data/'roots.jsonl', data/'tests.jsonl'
    root_manifest.write_text('')
    test_manifest.write_text('')
    for split in ('TRAIN', 'VALIDATION'):
        input_path = data/f'{split.lower()}_records.jsonl'
        input_path.write_text(''.join(canonical(r)+'\n' for r in selected[split]))
        import_apps(input_path, data/split.lower(), root_manifest, test_manifest,
                    original_split='train', split=split, source=SOURCE_URL, visible_cases=None)
    retention = [{**row, 'kind': 'retention'} for row in read_jsonl(RETENTION)]
    if not retention or any(r['split'] != 'TRAIN' for r in retention):
        raise ValueError('Invalid existing executor retention data')
    training = targets + retention
    (data/'training.jsonl').write_text(''.join(canonical(row)+'\n' for row in training))
    config = deepcopy(json.loads(GATE_CONTRAST.read_text())['config'])
    config.update(root_manifest=str(root_manifest), test_manifest=str(test_manifest),
                  root_limit=len(selected['VALIDATION']), split='VALIDATION',
                  initial_adapter=adapters['internal_gate'], executor_adapter=adapters['executor_before'],
                  output_dir=str(root))
    request = {'framework': framework, 'config': config, 'adapters': adapters,
               'adapter_sha256': {k: file_hash(Path(p)/'adapter_model.safetensors') for k, p in adapters.items()},
               'source': SOURCE_URL, 'source_sha256': file_hash(source), 'seed': 714000713,
               'eval_seed': 715000713, 'steps': (len(targets)+1)//2, 'learning_rate': 2e-5,
               'train_roots': len(targets), 'validation_roots': len(selected['VALIDATION']),
               'retention_examples': len(retention), 'training_sha256': file_hash(data/'training.jsonl'),
               'roots_sha256': file_hash(root_manifest), 'tests_sha256': file_hash(test_manifest),
               'attempts': dict(attempts), 'quotas': quotas,
               'objective': 'Executor-only SFT, two verified APPS root write_code targets plus two existing-contract retention examples per step; one complete APPS epoch, equal per-example token-mean CE, frozen planner.',
               'evaluation': 'Before/after on disjoint APPS train origins allocated VALIDATION. Natural, forced-root-ATOMIC, and forced-root-SPLIT, same root and seed; private original cases scored offline. No checkpoint selection.',
               'limitations': 'Only APPS stdin problems with a passing adapted solution; single-case tasks have no online visible feedback; APPS official test split is untouched.'}
    write_json(root/'request.json', request)
    write_json(root/'gpu_resources.json', {'devices': ['0', '1'],
               'min_free_mib': {'train': 22000, 'collect': 16000}})
    result = check(data=data, root=root)
    print(json.dumps({'prepared': result['counts'], 'attempts': dict(attempts)}, ensure_ascii=False), flush=True)
    return result


def check(data=DATA, root=ROOT, *, after=False):
    """CPU-only provenance, split, tokenizer, and launch-config check."""
    data, root = Path(data), Path(root)
    request = json.loads((root/'request.json').read_text())
    framework = request['framework']
    if request['source_sha256'] != SOURCE_SHA256 and data == DATA:
        raise ValueError('Pinned APPS source hash mismatch')
    if file_hash(data/'training.jsonl') != request['training_sha256']:
        raise ValueError('Training data changed')
    if file_hash(data/'roots.jsonl') != request['roots_sha256'] or file_hash(data/'tests.jsonl') != request['tests_sha256']:
        raise ValueError('Selected manifests changed')
    if framework['model']['load_in_4bit'] is not True or framework['runtime']['context_tokens'] != 8192:
        raise ValueError('Unexpected model or context configuration')
    if framework['runtime']['generation_tokens']['write_code'] != 3072:
        raise ValueError('Unexpected executor generation cap')
    if request['config']['reward']['cost_weight'] != 0:
        raise ValueError('Unexpected root reward configuration')
    if (framework.get('allow_validation') is not True or request['config']['split'] != 'VALIDATION'
            or request['config']['root_limit'] != request['validation_roots']
            or request['config']['initial_adapter'] != request['adapters']['internal_gate']
            or request['config']['executor_adapter'] != request['adapters']['executor_before']):
        raise ValueError('Evaluation launch configuration is inconsistent')
    resources = json.loads((root/'gpu_resources.json').read_text())
    if resources['devices'] != ['0', '1']:
        raise ValueError('Expected GPUs 0 and 1 only')
    for key, path in request['adapters'].items():
        if file_hash(Path(path)/'adapter_model.safetensors') != request['adapter_sha256'][key]:
            raise ValueError(f'Changed source adapter: {key}')
    tokenizer = _tokenizer(framework, request['adapters'])
    roots = load_root_manifest(data/'roots.jsonl')
    tests = {r['root_id']: r for r in read_jsonl(data/'tests.jsonl')}
    previously_used = set(load_root_manifest('data/manifests/roots.jsonl'))
    expected = request['train_roots'] + request['validation_roots']
    if len(roots) != expected or set(roots) != set(tests) or len({r['family_id'] for r in roots.values()}) != expected:
        raise ValueError('APPS selected roots/families/tests are not one-to-one')
    for row in roots.values():
        if (row['root_id'] in previously_used or row['source'] != 'APPS'
                or row['original_split'] != 'train' or row['split'] not in {'TRAIN', 'VALIDATION'}):
            raise ValueError('Wrong APPS origin/split')
        if file_hash(row['fixture_path']) != row['source_hash'] or file_hash(tests[row['root_id']]['path']) != tests[row['root_id']]['sha256']:
            raise ValueError('Task/test file hash mismatch')
        public = json.loads(Path(row['fixture_path']).read_text())
        private = json.loads(Path(tests[row['root_id']]['path']).read_text())
        raw = json.loads(Path(row['raw_path']).read_text())
        original_io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        if digest(raw) != row['source_record_sha256'] or len(original_io['inputs']) != (
                sum(len(s['cases']) for s in public['suites']) + sum(len(s['cases']) for s in private['suites'])):
            raise ValueError('Original APPS record or case partition changed')
        if any(s['visibility'] != 'PRIVATE' or not s['cases'] for s in private['suites']):
            raise ValueError('Private suite is missing or public')
        if len(public['suites']) != int(len(private['suites'][0]['cases']) + sum(len(s['cases']) for s in public['suites']) > 1):
            raise ValueError('Visible suite partition mismatch')
    examples = read_jsonl(data/'training.jsonl')
    targets = [r for r in examples if r['kind'] == 'target']
    retention = [r for r in examples if r['kind'] == 'retention']
    if len(targets) != request['train_roots'] or len(retention) != request['retention_examples'] or request['steps'] != (len(targets)+1)//2:
        raise ValueError('Training schedule does not cover all APPS targets')
    if {r['root_id'] for r in targets} != {rid for rid, row in roots.items() if row['split'] == 'TRAIN'}:
        raise ValueError('Targets and TRAIN roots differ')
    for row in examples:
        if row['split'] != 'TRAIN' or row['target_action']['action'] not in {'write_code', 'consume'}:
            raise ValueError('Invalid executor training action')
        prompt = summary_prompt(tokenizer, row['observation'], context_tokens=framework['runtime']['context_tokens'],
                                generation_tokens=framework['runtime']['generation_tokens'])
        target_tokens = len(tokenizer.encode(canonical(row['target_action']), add_special_tokens=False)) + 1
        if len(prompt.token_ids) + target_tokens > 8192 or (row['kind'] == 'target' and target_tokens > 3072):
            raise ValueError(f"Training action overflows context or generation: {row['sample_id']}")
    if after:
        adapter = root/'training/final_adapter'
        load_adapter_metadata(adapter, framework['model'], tokenizer)
    result = {'status': 'READY', 'counts': {'train': len(targets), 'validation': request['validation_roots'],
               'retention': len(retention), 'steps': request['steps']}, 'gpu_devices': resources['devices'],
              'model': framework['model']['local_path'], 'planner_frozen': True}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


class CurrentPlanner:
    """Dispatch the already-trained root, first-compute, and body adapters."""

    def __init__(self, base):
        from .refinement_internal_gate import InternalGatePolicy
        self.policy = InternalGatePolicy(base.llm)
        self.policy.identity['policy'] = 'apps_full_free_current_composite'
        self.cost = self.policy.cost
        self.identity = self.policy.identity

    def score_modes(self, observation):
        if observation['node_id'] == 'N0':
            return self.policy.root_gate.score_modes(observation)
        if observation['node_id'] == 'N0_p1_compute':
            return self.policy.score_modes(observation)
        return self.policy.body.score_modes(observation)

    def act(self, observation):
        import torch
        gate = self.score_modes(observation)
        modes = list(gate['mode_probabilities'])
        probabilities = torch.tensor([gate['mode_probabilities'][mode] for mode in modes])
        mode = modes[int(torch.multinomial(probabilities, 1))]
        candidate = self.policy.atomic(gate) if mode == 'ATOMIC' else self.policy.sample_split(gate)
        return self.policy.as_output(gate, candidate)


class _ForcedRoot:
    def __init__(self, base, mode):
        self.base, self.mode = base, mode
        self.cost, self.identity = base.cost, {**base.identity, 'forced_initial_root_mode': mode}
        self.sent = False

    def act(self, observation):
        if observation['node_id'] != 'N0' or self.sent or observation['phase'] != 'NEW':
            return self.base.act(observation)
        self.sent = True
        gate = self.base.score_modes(observation)
        candidate = self.base.policy.atomic(gate) if self.mode == 'ATOMIC' else self.base.policy.sample_split(gate)
        return self.base.policy.as_output(gate, candidate)


def _context(request, condition):
    from .refinement_policy import load_policies

    adapter = request['adapters']['executor_before'] if condition == 'BEFORE' else str(ROOT/'training/final_adapter')
    base, executor = load_policies(request['framework'], request['adapters']['internal_gate'], adapter)
    base.model.load_adapter(request['adapters']['body'], adapter_name='body', is_trainable=False, local_files_only=True)
    base.model.load_adapter(request['adapters']['root_gate'], adapter_name='root_gate', is_trainable=False, local_files_only=True)
    base.model.set_adapter('default')
    base.model.requires_grad_(False)
    return CurrentPlanner(base), executor


def train():
    check()
    from .gpu_resource import acquire
    reservation = acquire(ROOT/'gpu_resources.json', 'train')
    from .refinement_executor_sft import train as executor_train
    try:
        executor_train(ROOT, DATA, json.loads((ROOT/'request.json').read_text())['adapters']['executor_before'], progress=True)
    finally:
        reservation.close()


def evaluate(condition, shard, shards):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if condition not in {'BEFORE', 'AFTER'} or not 0 <= shard < shards:
        raise ValueError('Expected BEFORE/AFTER and a valid shard')
    check(after=condition == 'AFTER')
    request = json.loads((ROOT/'request.json').read_text())
    rows = [r for r in load_root_manifest(DATA/'roots.jsonl').values() if r['split'] == 'VALIDATION']
    rows.sort(key=lambda r: r['root_id'])
    tests = {r['root_id']: r for r in read_jsonl(DATA/'tests.jsonl')}
    selection = [(i, row, arm) for i, row in enumerate(rows) for arm in ('NATURAL', 'ATOMIC', 'SPLIT') if i % shards == shard]
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = _context(request, condition)
        out = ROOT/'evaluation'/condition
        for i, row, arm in tqdm(selection, desc=f'APPS {condition} GPU-shard {shard}', unit='运行', position=shard, dynamic_ncols=True):
            policy = planner if arm == 'NATURAL' else _ForcedRoot(planner, arm)
            case = run_case(row, tests, request['framework'], request['config'], policy, executor,
                            out/row['root_id']/arm, request['eval_seed'] + i)
            if case['status'] != 'COMPLETE':
                raise RuntimeError(f"Model or scoring infrastructure error in {case['run_dir']}: {case['errors']}")
        write_json(out/f'shard_{shard}.json', {'status': 'COMPLETE', 'condition': condition,
                   'shard': shard, 'shards': shards, 'roots': sum(i % shards == shard for i in range(len(rows))),
                   'runs': len(selection)})
    finally:
        reservation.close()


def report(shards=2):
    def strict_private_pass(case):
        """Preserve line order and internal spacing; ignore only surrounding whitespace."""
        private = case['final_scores']['private']
        if private['status'] != 'PASS' or not private['suites']:
            return False
        for suite in private['suites']:
            evidence = json.loads(Path(suite['evidence_path']).read_text())
            for item in evidence['details']['cases']:
                actual, expected = item['execution']['value'], item['expected']
                if not isinstance(actual, str) or actual.strip() != expected.strip():
                    return False
        return True

    request = json.loads((ROOT/'request.json').read_text())
    roots = [r for r in load_root_manifest(DATA/'roots.jsonl').values() if r['split'] == 'VALIDATION']
    by_difficulty = {r['root_id']: json.loads(Path(r['raw_path']).read_text())['difficulty'] for r in roots}
    by_feedback = {r['root_id']: 'VISIBLE' if json.loads(Path(r['fixture_path']).read_text())['suites'] else 'NONE'
                   for r in roots}
    result = {'status': 'COMPLETE', 'train_roots': request['train_roots'], 'validation_roots': len(roots),
              'retention_examples': request['retention_examples'], 'arms': {}, 'paired': {}}
    records = {}
    for condition in ('BEFORE', 'AFTER'):
        count = 1 if condition == 'BEFORE' else shards
        for shard in range(count):
            summary = json.loads((ROOT/'evaluation'/condition/f'shard_{shard}.json').read_text())
            if summary['status'] != 'COMPLETE' or summary['shards'] != count:
                raise ValueError('Incomplete evaluation shard')
        for arm in ('NATURAL', 'ATOMIC', 'SPLIT'):
            cases = [json.loads((ROOT/'evaluation'/condition/rid/arm/'case_result.json').read_text()) for rid in by_difficulty]
            root_modes, p_splits = [], []
            for case in cases:
                events = read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
                root_events = [e for e in events if e['node_id'] == 'N0' and
                               e['observation_before']['phase'] == 'NEW']
                accepted = next((e for e in root_events if e['host_result']['code'] == 'PLAN_REGISTERED'), None)
                root_modes.append(accepted['parsed_action']['args']['mode'] if accepted else None)
                gate = (root_events[0].get('model_input') or {}).get('refinement', {}).get('gate', {}) if root_events else {}
                if 'SPLIT' in gate.get('mode_probabilities', {}):
                    p_splits.append(gate['mode_probabilities']['SPLIT'])
            records[(condition, arm)] = {c['root_id']: c for c in cases}
            result['arms'][f'{condition}_{arm}'] = {'roots': len(cases),
                'private_pass': sum(c['final_scores']['private']['status'] == 'PASS' for c in cases),
                'private_strict_pass': sum(strict_private_pass(c) for c in cases),
                'first_private_pass': sum(c['first_scores']['private']['status'] == 'PASS' for c in cases),
                'root_plan_registered': sum(c['root_first_plan_registered'] for c in cases),
                'root_split': root_modes.count('SPLIT'),
                'root_atomic': root_modes.count('ATOMIC'),
                'root_unregistered': root_modes.count(None),
                'mean_root_p_split': sum(p_splits)/len(p_splits) if p_splits else None,
                'mean_policy_calls': sum(c['usage']['policy_calls'] for c in cases)/len(cases),
                'by_difficulty': {d: {'roots': sum(by_difficulty[c['root_id']] == d for c in cases),
                    'private_pass': sum(by_difficulty[c['root_id']] == d and c['final_scores']['private']['status'] == 'PASS' for c in cases)}
                    for d in sorted(set(by_difficulty.values()))},
                'by_online_feedback': {f: {'roots': sum(by_feedback[c['root_id']] == f for c in cases),
                    'private_pass': sum(by_feedback[c['root_id']] == f and c['final_scores']['private']['status'] == 'PASS' for c in cases)}
                    for f in ('VISIBLE', 'NONE')}}
    for arm in ('NATURAL', 'ATOMIC', 'SPLIT'):
        before, after = records[('BEFORE', arm)], records[('AFTER', arm)]
        if set(before) != set(after):
            raise ValueError('Before/after roots differ')
        result['paired'][arm] = {'wins': 0, 'losses': 0, 'ties': 0,
                                  'strict_wins': 0, 'strict_losses': 0, 'strict_ties': 0,
                                  'by_root': []}
        for rid in sorted(before):
            old = before[rid]['final_scores']['private']['status'] == 'PASS'
            new = after[rid]['final_scores']['private']['status'] == 'PASS'
            key = 'wins' if new and not old else 'losses' if old and not new else 'ties'
            result['paired'][arm][key] += 1
            old_strict, new_strict = strict_private_pass(before[rid]), strict_private_pass(after[rid])
            strict_key = 'strict_wins' if new_strict and not old_strict else 'strict_losses' if old_strict and not new_strict else 'strict_ties'
            result['paired'][arm][strict_key] += 1
            result['paired'][arm]['by_root'].append({'root_id': rid, 'difficulty': by_difficulty[rid],
                    'online_feedback': by_feedback[rid], 'before_pass': old, 'after_pass': new,
                    'before_strict_pass': old_strict, 'after_strict_pass': new_strict})
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'evaluate', 'report'])
    parser.add_argument('--condition', choices=['BEFORE', 'AFTER'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'train':
        train()
    elif args.command == 'evaluate':
        if args.condition is None:
            parser.error('--condition is required for evaluate')
        evaluate(args.condition, args.shard, args.shards)
    else:
        report(args.shards)


if __name__ == '__main__':
    main()
