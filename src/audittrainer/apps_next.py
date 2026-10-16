"""Frozen APPS controls: independent probes and an interface-only prompt.

prepare/check are CPU-only. score-probes executes saved programs in the sandbox;
evaluate loads the frozen BEFORE model. Neither command updates weights.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .prompts import public_observation
from .refinement_parent_composition import summary_prompt


SOURCE_DATA = Path('data/refinement/apps_multitask')
SOURCE_RUN = Path('runs/refinement/apps_multitask')
DATA = Path('data/refinement/apps_next')
ROOT = Path('runs/refinement/apps_next')
GPU_IDS = ['0', '1', '2']

INTERFACE_HINT = (
    '\nTask-independent implementation guide for this stdin problem: '
    'write exactly def solve(text): and use only the supplied text as input. '
    'Parse with text.split() or text.splitlines(), implement the problem-specific algorithm, '
    'and return a str containing output lines in the required order. '
    'A possible shape is:\n'
    'def solve(text):\n'
    '    tokens = iter(text.split())\n'
    '    output_lines = []\n'
    '    # parse tokens and solve the stated problem here\n'
    '    return "\\n".join(output_lines)\n'
    'Do not call host input() or print().'
)


def interface_prompt(tokenizer, observation, **kwargs):
    visible = public_observation(observation)
    if visible['phase'] == 'IMPLEMENT' and visible['node_id'] == 'N0':
        visible['root_spec'] += INTERFACE_HINT
    return summary_prompt(tokenizer, visible, **kwargs)


def _commands(*commands):
    return f'{len(commands)}\n' + '\n'.join(commands) + '\n'


# Three new full standard-input instances per selected root, authored from the
# public problem statements. These are never sent to either model.
PROBE_INPUTS = {
    'apps_train_0527': [
        '3\n2 8 18\n6\n1 1 1\n1 1 2\n1 1 3\n2 1 2\n1 1 1\n1 1 3\n',
        '4\n3 12 25 7\n6\n1 1 2\n1 2 3\n2 4 7\n1 3 4\n2 1 3\n1 1 4\n',
        '1\n72\n4\n1 1 1\n2 1 2\n2 1 1\n1 1 1\n',
    ],
    'apps_train_0676': [
        '3\n5\nzoe amy zoe amy zoe\n4\nbob alice bob alice\n1\nsolo\n',
        '1\n6\nzeta beta alpha zeta alpha beta\n',
        '2\n7\ncat dog dog cat eel cat dog\n3\nc b a\n',
    ],
    'apps_train_0688': [
        '5\n11111111\n01010101\n00001111\n11000011\n00110011\n',
        '4\n00000001\n11100000\n10111101\n01100110\n',
        '3\n00000000\n10000001\n10100101\n',
    ],
    'apps_train_0836': [
        '3\n3\n2 1 3\n3 6 2\n4\n2 4 1 8\n4 2 8 1\n3\n3 2 1\n3 5 8\n',
        '2\n1\n7\n9\n3\n2 4 1\n2 1 3\n',
        '1\n5\n5 2 10 1 4\n2 5 1 9 2\n',
    ],
    'apps_train_1311': [
        '5\n1 1\n2 1\n2 2\n5 2\n5 5\n',
        '3\n6 1\n6 3\n6 6\n',
        '2\n20 7\n20 19\n',
    ],
    'apps_train_1371': [
        '3\n4 1\n6 13 20 2\n3 7\n7 14 21\n1 6\n1\n',
        '2\n5 10\n4 11 18 25 1\n4 2\n5 12 6 19\n',
        '1\n6 3\n1 2 3 4 11 18\n',
    ],
    'apps_train_1557': [
        '4\n4\n1010\n0101\n4\n1110\n0011\n1\n0\n0\n1\n0\n1\n',
        '3\n6\n000111\n101010\n5\n00000\n00001\n3\n111\n111\n',
        '2\n2\n01\n10\n8\n10101010\n00001111\n',
    ],
    'apps_train_2617': [
        _commands('print', 'append 4', 'append -1', 'insert 1 7', 'print',
                  'reverse', 'print', 'sort', 'print'),
        _commands('append 3', 'append 3', 'append 1', 'remove 3', 'print',
                  'pop', 'print', 'insert 0 -2', 'print'),
        _commands('append 9', 'insert 0 4', 'insert 2 8', 'print',
                  'reverse', 'remove 4', 'sort', 'print'),
    ],
}


def oracle(root_id, text):
    """Independent small-input oracle; constructive 1311 uses semantic scoring."""
    tokens = iter(text.split())
    if root_id == 'apps_train_0527':
        n = int(next(tokens)); values = [int(next(tokens)) for _ in range(n)]
        out = []
        for _ in range(int(next(tokens))):
            kind, a, b = (int(next(tokens)) for _ in range(3))
            if kind == 1:
                product = math.prod(values[a-1:b])
                out.append('YES' if math.isqrt(product)**2 == product else 'NO')
            else:
                values[a-1] *= b
        return '\n'.join(out)
    if root_id == 'apps_train_0676':
        out = []
        for _ in range(int(next(tokens))):
            counts = Counter(next(tokens) for _ in range(int(next(tokens))))
            out.append(min(counts, key=lambda name: (-counts[name], name)))
        return '\n'.join(out)
    if root_id == 'apps_train_0688':
        out = []
        for _ in range(int(next(tokens))):
            bits = next(tokens)
            changes = sum(bits[i] != bits[i-1] for i in range(8))
            out.append('uniform' if changes <= 2 else 'non-uniform')
        return '\n'.join(out)
    if root_id == 'apps_train_0836':
        out = []
        for _ in range(int(next(tokens))):
            n = int(next(tokens))
            lengths = [int(next(tokens)) for _ in range(n)]
            ratings = [int(next(tokens)) for _ in range(n)]
            best = max(range(n), key=lambda i: (lengths[i]*ratings[i], ratings[i], -i))
            out.append(str(best+1))
        return '\n'.join(out)
    if root_id == 'apps_train_1311':
        return None
    if root_id == 'apps_train_1371':
        out = []
        for _ in range(int(next(tokens))):
            n, k = int(next(tokens)), int(next(tokens))
            out.append(str(sum((int(next(tokens))+k) % 7 == 0 for _ in range(n))))
        return '\n'.join(out)
    if root_id == 'apps_train_1557':
        out = []
        for _ in range(int(next(tokens))):
            n, left, right = int(next(tokens)), next(tokens), next(tokens)
            assert len(left) == len(right) == n
            out.append('YES' if left.count('1') == right.count('1') else 'NO')
        return '\n'.join(out)
    if root_id == 'apps_train_2617':
        lines = iter(text.splitlines()); values = []; out = []
        for _ in range(int(next(lines))):
            parts = next(lines).split(); command = parts[0]
            if command == 'insert': values.insert(int(parts[1]), int(parts[2]))
            elif command == 'print': out.append(str(values))
            elif command == 'remove': values.remove(int(parts[1]))
            elif command == 'append': values.append(int(parts[1]))
            elif command == 'sort': values.sort()
            elif command == 'pop': values.pop()
            elif command == 'reverse': values.reverse()
            else: raise ValueError(f'Unknown list command: {command}')
        return '\n'.join(out)
    raise ValueError(f'Unknown probe root: {root_id}')


def valid_output(root_id, text, value, expected):
    if not isinstance(value, str):
        return False
    if root_id != 'apps_train_1311':
        return value.strip() == expected.strip()
    params = iter(text.split())
    count = int(next(params))
    pairs = [(int(next(params)), int(next(params))) for _ in range(count)]
    lines = value.strip().splitlines()
    if len(lines) != count:
        return False
    for (n, k), line in zip(pairs, lines):
        try:
            items = [int(part) for part in line.split()]
        except ValueError:
            return False
        if len(items) != n or any(abs(item) != i for i, item in enumerate(items, 1)):
            return False
        total = positive = 0
        for item in items:
            total += item
            positive += total > 0
        if positive != k:
            return False
    return True


def _rows():
    rows = load_root_manifest(SOURCE_DATA/'roots.jsonl')
    return sorted((r for r in rows.values() if r['split'] == 'VALIDATION'), key=lambda r: r['root_id'])


def _valid_private_pass(case):
    private = case['final_scores']['private']
    # The APPS reference for 1311 is one arbitrary constructive solution.
    if (private['status'] != 'PASS' and case['root_id'] != 'apps_train_1311') or not private['suites']:
        return False
    for suite in private['suites']:
        evidence = json.loads(Path(suite['evidence_path']).read_text())
        for item in evidence['details']['cases']:
            if not valid_output(case['root_id'], item['args'][0], item['execution']['value'], item['expected']):
                return False
    return True


def prepare():
    from . import apps_multitask
    from .apps import stdin_text

    apps_multitask.check(after=True)
    if DATA.exists() or ROOT.exists():
        raise FileExistsError('APPS next-step outputs already exist')
    result = json.loads((SOURCE_RUN/'result.json').read_text())
    if result['status'] != 'COMPLETE':
        raise ValueError('Original APPS result is incomplete')
    selected = {r['root_id'] for r in result['paired']['NATURAL']['by_root'] if r['after_strict_pass']}
    if selected != set(PROBE_INPUTS):
        raise ValueError(f'Strict-pass selection changed: {sorted(selected)}')
    rows = {r['root_id']: r for r in _rows()}
    probes = []
    for root_id, texts in PROBE_INPUTS.items():
        if root_id not in rows or len(texts) != 3:
            raise ValueError('Probe selection is invalid')
        raw = json.loads(Path(rows[root_id]['raw_path']).read_text())
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        originals = {stdin_text(source).strip() for source in io['inputs']}
        for index, text in enumerate(texts):
            if text.strip() in originals:
                raise ValueError(f'Probe repeats an original APPS case: {root_id}/{index}')
            probes.append({'root_id': root_id, 'case_id': f'{root_id}_probe_{index+1}',
                           'input': text, 'expected': oracle(root_id, text)})
    DATA.mkdir(parents=True); ROOT.mkdir(parents=True)
    write_json(DATA/'probes.json', probes)
    source_request = json.loads((SOURCE_RUN/'request.json').read_text())
    write_json(ROOT/'request.json', {
        'source_result_sha256': file_hash(SOURCE_RUN/'result.json'),
        'source_roots_sha256': file_hash(SOURCE_DATA/'roots.jsonl'),
        'source_tests_sha256': file_hash(SOURCE_DATA/'tests.jsonl'),
        'executor_before_sha256': source_request['adapter_sha256']['executor_before'],
        'executor_after_sha256': file_hash(SOURCE_RUN/'training/final_adapter/adapter_model.safetensors'),
        'probe_sha256': file_hash(DATA/'probes.json'),
        'runner_sha256': file_hash(__file__),
        'interface_hint_sha256': hashlib.sha256(INTERFACE_HINT.encode()).hexdigest(),
        'eval_seed': source_request['eval_seed'],
        'validation_roots': len(_rows()), 'probe_roots': len(selected), 'probe_cases': len(probes),
        'design': 'Independent sandbox probes on saved BEFORE/AFTER NATURAL final programs; '
                  'zero-update BEFORE executor with a task-independent interface hint on the same 48 held-out roots. '
                  'Frozen planner, model weights, original private cases, and per-root seed.',
    })
    write_json(ROOT/'gpu_resources.json', {'devices': GPU_IDS, 'min_free_mib': {'collect': 16000}})
    check()


def check():
    from . import apps_multitask
    from .apps import stdin_text
    from .dataset import file_hash

    apps_multitask.check(after=True)
    req = json.loads((ROOT/'request.json').read_text())
    old = json.loads((SOURCE_RUN/'request.json').read_text())
    required = {
        'source_result_sha256': SOURCE_RUN/'result.json',
        'source_roots_sha256': SOURCE_DATA/'roots.jsonl',
        'source_tests_sha256': SOURCE_DATA/'tests.jsonl',
        'probe_sha256': DATA/'probes.json',
        'runner_sha256': Path(__file__),
    }
    for key, path in required.items():
        if req[key] != file_hash(path):
            raise ValueError(f'Frozen input changed: {key}')
    if (req['executor_before_sha256'] != old['adapter_sha256']['executor_before']
            or req['executor_after_sha256'] != file_hash(SOURCE_RUN/'training/final_adapter/adapter_model.safetensors')
            or req['eval_seed'] != old['eval_seed']
            or req['interface_hint_sha256'] != hashlib.sha256(INTERFACE_HINT.encode()).hexdigest()):
        raise ValueError('Adapter, seed or prompt changed')
    gpu = json.loads((ROOT/'gpu_resources.json').read_text())
    if gpu != {'devices': GPU_IDS, 'min_free_mib': {'collect': 16000}}:
        raise ValueError('Expected GPU 0/1/2 with 16 GiB free each')
    rows = _rows()
    probes = json.loads((DATA/'probes.json').read_text())
    if len(rows) != req['validation_roots'] or len(probes) != req['probe_cases']:
        raise ValueError('Root/probe counts changed')
    roots = {r['root_id']: r for r in rows}
    selected = {r['root_id'] for r in json.loads((SOURCE_RUN/'result.json').read_text())['paired']['NATURAL']['by_root']
                if r['after_strict_pass']}
    if set(PROBE_INPUTS) != selected or {p['root_id'] for p in probes} != selected or len(selected) != req['probe_roots']:
        raise ValueError('Probe roots are not the frozen strict-pass set')
    for root_id in selected:
        raw = json.loads(Path(roots[root_id]['raw_path']).read_text())
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        originals = {stdin_text(text).strip() for text in io['inputs']}
        subset = [p for p in probes if p['root_id'] == root_id]
        if len(subset) != 3 or len({p['input'] for p in subset}) != 3:
            raise ValueError(f'Invalid probe set: {root_id}')
        for probe in subset:
            if probe['input'].strip() in originals or probe['expected'] != oracle(root_id, probe['input']):
                raise ValueError(f'Probe changed or repeats source input: {probe["case_id"]}')
    for condition in ('BEFORE', 'AFTER'):
        for row in rows:
            case = json.loads((SOURCE_RUN/'evaluation'/condition/row['root_id']/'NATURAL/case_result.json').read_text())
            if case['status'] != 'COMPLETE' or case['seed'] != old['eval_seed']+rows.index(row):
                raise ValueError('Original paired evaluation missing or mismatched')
    # Check the added prompt against the actual frozen implement observations,
    # using only a local tokenizer and the public observations.
    tokenizer = apps_multitask._tokenizer(old['framework'], old['adapters'])
    maxima = 0
    for row in rows:
        trace = read_jsonl(SOURCE_RUN/'evaluation/BEFORE'/row['root_id']/'NATURAL/runtime/trace.jsonl')
        root_impl = next((e['observation_before'] for e in trace
                          if e['node_id'] == 'N0' and e['observation_before']['phase'] == 'IMPLEMENT'), None)
        if root_impl is None:
            raise ValueError(f'No root implement observation: {row["root_id"]}')
        rendered = interface_prompt(tokenizer, root_impl,
                                    context_tokens=old['framework']['runtime']['context_tokens'],
                                    generation_tokens=old['framework']['runtime']['generation_tokens'])
        maxima = max(maxima, len(rendered.token_ids)+rendered.max_new_tokens)
        if 'probe_' in rendered.record['prompt']:
            raise ValueError('Probe content entered a model prompt')
    result = {'status': 'READY', 'validation_roots': len(rows), 'probe_roots': len(selected),
              'probe_cases': len(probes), 'interface_max_context_plus_generation': maxima,
              'context_limit': old['framework']['runtime']['context_tokens'],
              'gpu_devices': GPU_IDS, 'training_steps': 0}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def score_probes():
    from tqdm import tqdm
    from .artifacts import ArtifactStore
    from .sandbox import ExecutionLimits, Sandbox
    from .schema import Artifact

    check()
    if (ROOT/'probe_results.json').exists():
        raise FileExistsError('Probe results already exist')
    old = json.loads((SOURCE_RUN/'request.json').read_text())
    sandbox = Sandbox(ExecutionLimits(**old['framework']['execution']))
    probes = json.loads((DATA/'probes.json').read_text())
    results = []
    for condition in tqdm(('BEFORE', 'AFTER'), desc='独立输入评测', unit='模型', dynamic_ncols=True):
        for root_id in tqdm(PROBE_INPUTS, desc=condition, unit='题', leave=False, dynamic_ncols=True):
            case = json.loads((SOURCE_RUN/'evaluation'/condition/root_id/'NATURAL/case_result.json').read_text())
            snapshot = case['final_program']
            if snapshot:
                store = ArtifactStore(Path(snapshot['path']))
                bundle = store.bundle(Artifact.model_validate(snapshot['artifact']))
                source = bundle.sources['N0']
            for probe in (p for p in probes if p['root_id'] == root_id):
                execution = sandbox.run(source, 'solve', [probe['input']], bundle=bundle) if snapshot else None
                passed = bool(execution and execution.status == 'RETURNED' and
                              valid_output(root_id, probe['input'], execution.value, probe['expected']))
                results.append({'condition': condition, 'root_id': root_id, 'case_id': probe['case_id'],
                                'pass': passed, 'status': execution.status if execution else 'NO_PROGRAM',
                                'reason': execution.reason if execution else 'NO_PROGRAM',
                                'actual': execution.value if execution else None})
    write_json(ROOT/'probe_results.json', {'status': 'COMPLETE', 'cases': results})
    print(json.dumps({'status': 'COMPLETE', 'before_cases': sum(r['pass'] for r in results if r['condition']=='BEFORE'),
                      'after_cases': sum(r['pass'] for r in results if r['condition']=='AFTER')}, ensure_ascii=False), flush=True)


def evaluate(shard, shards):
    from tqdm import tqdm
    from .apps_multitask import _context
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < shards or shards != len(GPU_IDS):
        raise ValueError('Expected three shards mapped to GPU 0/1/2')
    check()
    old = json.loads((SOURCE_RUN/'request.json').read_text())
    rows = _rows()
    tests = {r['root_id']: r for r in read_jsonl(SOURCE_DATA/'tests.jsonl')}
    out = ROOT/'evaluation/INTERFACE'
    if (out/f'shard_{shard}.json').exists():
        raise FileExistsError(f'Interface shard {shard} already complete')
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = _context(old, 'BEFORE')
        executor.llm.prompt_builder = interface_prompt
        executor.llm.identity = {**executor.llm.identity, 'interface_control_sha256': hashlib.sha256(INTERFACE_HINT.encode()).hexdigest()}
        for index, row in tqdm(list(enumerate(rows))[shard::shards], desc=f'接口提示 GPU {GPU_IDS[shard]}', unit='题', dynamic_ncols=True):
            case = run_case(row, tests, old['framework'], old['config'], planner, executor,
                            out/row['root_id']/'NATURAL', old['eval_seed']+index)
            if case['status'] != 'COMPLETE':
                raise RuntimeError(f'Incomplete interface control: {case["run_dir"]}: {case["errors"]}')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/f'shard_{shard}.json', {'status': 'COMPLETE', 'shard': shard, 'shards': shards,
                                              'roots': len(rows[shard::shards]),
                                              'interface_hint_sha256': hashlib.sha256(INTERFACE_HINT.encode()).hexdigest()})
    finally:
        reservation.close()


def report():
    check()
    req = json.loads((ROOT/'request.json').read_text())
    probes = json.loads((ROOT/'probe_results.json').read_text())
    if probes['status'] != 'COMPLETE' or len(probes['cases']) != 2*req['probe_cases']:
        raise ValueError('Independent probe scoring incomplete')
    for shard in range(len(GPU_IDS)):
        done = json.loads((ROOT/f'evaluation/INTERFACE/shard_{shard}.json').read_text())
        if done['status'] != 'COMPLETE' or done['shard'] != shard or done['shards'] != len(GPU_IDS) or done['interface_hint_sha256'] != req['interface_hint_sha256']:
            raise ValueError(f'Incomplete interface shard: {shard}')
    roots = _rows()
    original = {}
    for cond in ('BEFORE', 'AFTER', 'INTERFACE'):
        original[cond] = {row['root_id']: json.loads(((SOURCE_RUN/'evaluation'/cond if cond != 'INTERFACE' else ROOT/'evaluation/INTERFACE')
                           /row['root_id']/'NATURAL/case_result.json').read_text()) for row in roots}
    valid = {cond: sum(_valid_private_pass(case) for case in original[cond].values()) for cond in original}
    probe_by = {cond: [r for r in probes['cases'] if r['condition'] == cond] for cond in ('BEFORE', 'AFTER')}
    all_pass = {cond: sum(all(r['pass'] for r in probe_by[cond] if r['root_id'] == root)
                          for root in PROBE_INPUTS) for cond in probe_by}
    result = {'status': 'COMPLETE', 'validation_roots': len(roots), 'original_private_valid_pass': valid,
              'probe_cases_pass': {cond: sum(r['pass'] for r in rows_) for cond, rows_ in probe_by.items()},
              'probe_roots_all_three_pass': all_pass,
              'interface_vs_before': {'wins': 0, 'losses': 0, 'ties': 0},
              'interface_vs_after': {'wins': 0, 'losses': 0, 'ties': 0},
              'note': 'Output order and internal spaces are preserved; root 1311 is scored by its constructive constraints on both original and new inputs.'}
    for comparison in ('BEFORE', 'AFTER'):
        pair = result[f'interface_vs_{comparison.lower()}']
        for row in roots:
            interface = _valid_private_pass(original['INTERFACE'][row['root_id']])
            other = _valid_private_pass(original[comparison][row['root_id']])
            pair['wins' if interface and not other else 'losses' if other and not interface else 'ties'] += 1
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'score-probes', 'evaluate', 'report'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == 'prepare': prepare()
    elif args.command == 'check': check()
    elif args.command == 'score-probes': score_probes()
    elif args.command == 'evaluate': evaluate(args.shard, args.shards)
    else: report()


if __name__ == '__main__':
    main()
