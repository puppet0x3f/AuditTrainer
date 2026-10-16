"""Independent CPU-only probes for APPS roots newly solved in the scale study."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import apps_scale
from .apps import stdin_text
from .artifacts import ArtifactStore
from .dataset import file_hash
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox
from .schema import Artifact


SOURCE = Path('runs/refinement/apps_scale')
ROOT = Path('runs/refinement/apps_scale_followup')
ARMS = ('BASE', 'INTERFACE', 'OLD96', 'SMALL', 'LARGE')


def _segment_stress():
    values = [(17 * i) % 100 + 1 for i in range(1, 1025)]
    queries = [(1 + (29 * i) % 700, 1024 - (13 * i) % 200,
                (1, 97, 1000000000, 101)[i % 4]) for i in range(40)]
    return (f'{len(values)}\n' + ' '.join(map(str, values)) + f'\n{len(queries)}\n'
            + ''.join(f'{l} {r} {mod}\n' for l, r, mod in queries))


# Authored from the problem statements; none is an original APPS input.
PROBE_INPUTS = {
    'apps_train_0573': [
        '6\n1 1\n2 5\n3 4\n4 2\n8 1\n10 7\n',
        '5\n2 100\n3 1\n5 3\n6 10\n1 99\n',
        '4\n4 100\n100 2\n7 7\n9 4\n',
    ],
    'apps_train_0971': [
        '4\n6\n1 2 2 2 3 2\n5\n1 2 3 4 5\n7\n4 4 1 2 2 2 2\n1\n9\n',
        '3\n8\n5 5 1 1 1 5 1 1\n9\n9 8 7 6 5 4 3 2 1\n3\n2 2 3\n',
        '4\n10\n1 1 2 2 3 3 4 4 5 5\n6\n7 7 7 7 7 7\n4\n1 2 2 1\n5\n10 20 10 20 10\n',
    ],
    'apps_train_1093': [
        '5\n2 3 5 7 11\n6\n1 5 1000\n2 4 7\n5 5 11\n1 1 1\n3 5 13\n2 5 97\n',
        '8\n1 100 6 4 25 3 2 1\n6\n1 8 1000000000\n2 5 17\n4 4 5\n6 8 2\n1 1 99\n3 7 101\n',
        _segment_stress(),
    ],
    'apps_train_1510': [
        '4\nABCDEFGHIJ\nAAAABBCDEFGHIJ\nABACADAEAFAGAHAIAJ\nJIHGFEDCBA\n',
        '3\nJJJJABCDEFGHI\nAABBCCDDEEFFGGHHIIJ\nBCDEFGHIAAAAAJ\n',
        '3\nAAAAABCDEFFGHIJ\nAABBCCDDFFGGHHIIEEJ\nJABACADAEAFAGAHAI\n',
    ],
    'apps_train_2363': [
        '5\n2\n4 1000\n6\n7 7 100 5 5 5\n5\n1000 1 500 501 999\n4\n7 2 11 20\n3\n9 9 9\n',
        '4\n8\n13 3 8 2 21 5 34 1\n2\n1 2\n6\n200 202 203 205 208 209\n5\n42 42 41 43 1000\n',
        '4\n7\n100 99 98 97 96 95 94\n3\n1 500 1000\n5\n8 8 8 9 9\n4\n2 9 4 7\n',
    ],
}


def oracle(root_id, text):
    tokens = iter(text.split())
    if root_id == 'apps_train_0573':
        outputs = []
        for _ in range(int(next(tokens))):
            n, m = int(next(tokens)), int(next(tokens))
            outputs.append(str(0 if n == 1 else m if n == 2 else n + 2*m - 3))
        return '\n'.join(outputs)
    if root_id == 'apps_train_0971':
        outputs = []
        for _ in range(int(next(tokens))):
            n = int(next(tokens))
            values = [int(next(tokens)) for _ in range(n)]
            outputs.append(str(n - max(Counter(values).values())))
        return '\n'.join(outputs)
    if root_id == 'apps_train_1093':
        n = int(next(tokens))
        values = [int(next(tokens)) for _ in range(n)]
        outputs = []
        for _ in range(int(next(tokens))):
            left, right, modulus = int(next(tokens)), int(next(tokens)), int(next(tokens))
            product = 1 % modulus
            for value in values[left-1:right]:
                product = product * value % modulus
            outputs.append(str(product))
        return '\n'.join(outputs)
    if root_id == 'apps_train_1510':
        return '\n'.join(str(len(next(tokens))) for _ in range(int(next(tokens))))
    if root_id == 'apps_train_2363':
        outputs = []
        for _ in range(int(next(tokens))):
            values = sorted(int(next(tokens)) for _ in range(int(next(tokens))))
            outputs.append(str(min(b-a for a, b in zip(values, values[1:]))))
        return '\n'.join(outputs)
    raise ValueError(f'Unknown APPS root: {root_id}')


def _probes():
    return [{'root_id': root_id, 'case_id': f'{root_id}_followup_{index+1}',
             'input': text, 'expected': oracle(root_id, text)}
            for root_id, texts in PROBE_INPUTS.items()
            for index, text in enumerate(texts)]


def check():
    apps_scale.check(trained=True)
    result = json.loads((SOURCE/'result.json').read_text())
    old = json.loads((SOURCE/'probe_results.json').read_text())
    if (result['status'] != 'COMPLETE' or old['status'] != 'COMPLETE'
            or len(old['cases']) != 240):
        raise ValueError('APPS scale result or previous probes are incomplete')
    rows = {row['root_id']: row for row in apps_scale._validation_rows()}
    old_roots = {row['root_id'] for row in old['cases']}
    new_successes = {rid for rid in rows if any(apps_scale._valid_private(
        json.loads(apps_scale._old_case(arm, rid).read_text())) for arm in ('SMALL', 'LARGE'))}
    if new_successes - old_roots != set(PROBE_INPUTS):
        raise ValueError('Follow-up roots differ from previously unprobed strict successes')
    for rid, texts in PROBE_INPUTS.items():
        if len(texts) != 3 or len(set(texts)) != 3:
            raise ValueError(f'Expected three unique inputs for {rid}')
        raw = json.loads(Path(rows[rid]['raw_path']).read_text())
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        originals = {stdin_text(value).strip() for value in io['inputs']}
        for source, expected in zip(io['inputs'], io['outputs']):
            if oracle(rid, stdin_text(source)).strip() != stdin_text(expected).strip():
                raise ValueError(f'Oracle disagrees with the original APPS answer: {rid}')
        for text in texts:
            if text.strip() in originals or not oracle(rid, text):
                raise ValueError(f'Invalid or repeated independent input: {rid}')
        if rid == 'apps_train_1510':
            for text in texts:
                words = text.split()[1:]
                if any(set(word) != set('ABCDEFGHIJ') or set(word[:-1]) == set('ABCDEFGHIJ')
                       for word in words):
                    raise ValueError('Coin probes must finish on the first complete collection')
    for arm in ARMS:
        for rid in PROBE_INPUTS:
            case = json.loads(apps_scale._old_case(arm, rid).read_text())
            if case['status'] != 'COMPLETE':
                raise ValueError(f'Missing saved model rollout: {arm}/{rid}')
    framework = json.loads((SOURCE/'request.json').read_text())['framework']
    ready = {'status': 'READY', 'probe_roots': len(PROBE_INPUTS), 'probes_per_root': 3,
             'new_probe_cases': len(_probes()), 'saved_program_runs': len(ARMS)*len(_probes()),
             'arms': ARMS, 'gpu_devices': [], 'training_steps': 0,
             'sandbox_timeout_seconds': framework['execution']['per_case_timeout_seconds']}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def score():
    from tqdm import tqdm

    check()
    output = ROOT/'result.json'
    if output.exists():
        raise FileExistsError(f'Follow-up result already exists: {output}')
    framework = json.loads((SOURCE/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    probes = _probes()
    grouped = defaultdict(list)
    for probe in probes:
        grouped[probe['root_id']].append(probe)
    new_cases = []
    for arm in tqdm(ARMS, desc='APPS补充独立输入', unit='模型', dynamic_ncols=True):
        for rid in tqdm(sorted(grouped), desc=arm, unit='题', leave=False, dynamic_ncols=True):
            case = json.loads(apps_scale._old_case(arm, rid).read_text())
            snapshot = case['final_program']
            if snapshot:
                bundle = ArtifactStore(Path(snapshot['path'])).bundle(Artifact.model_validate(snapshot['artifact']))
                source = bundle.sources['N0']
            for probe in grouped[rid]:
                execution = sandbox.run(source, 'solve', [probe['input']], bundle=bundle) if snapshot else None
                passed = bool(execution and execution.status == 'RETURNED' and
                              isinstance(execution.value, str) and
                              execution.value.strip() == probe['expected'].strip())
                new_cases.append({'arm': arm, 'root_id': rid, 'case_id': probe['case_id'],
                                  'pass': passed, 'status': execution.status if execution else 'NO_PROGRAM',
                                  'reason': execution.reason if execution else 'NO_PROGRAM',
                                  'actual': execution.value if execution else None})
    old_cases = json.loads((SOURCE/'probe_results.json').read_text())['cases']
    all_cases = old_cases + new_cases
    passed_by_root = defaultdict(list)
    for item in all_cases:
        passed_by_root[(item['arm'], item['root_id'])].append(item['pass'])
    if any(len(values) != 3 for values in passed_by_root.values()):
        raise ValueError('Expected exactly three probes per model and root')
    validation = apps_scale._validation_rows()
    robust = {}
    for arm in ARMS:
        original = {row['root_id']: apps_scale._valid_private(json.loads(
                    apps_scale._old_case(arm, row['root_id']).read_text())) for row in validation}
        robust[arm] = sum(value and len(passed_by_root[(arm, rid)]) == 3
                          and all(passed_by_root[(arm, rid)]) for rid, value in original.items())
    summary = {}
    for arm in ARMS:
        values = [row for row in new_cases if row['arm'] == arm]
        combined = [row for row in all_cases if row['arm'] == arm]
        summary[arm] = {
            'new_cases_pass': sum(row['pass'] for row in values),
            'new_roots_all_three_pass': sum(all(passed_by_root[(arm, rid)]) for rid in PROBE_INPUTS),
            'combined_cases_pass': sum(row['pass'] for row in combined),
            'combined_roots_all_three_pass': sum(all(passed_by_root[(arm, rid)])
                                                 for rid in {row['root_id'] for row in combined}),
            'strict_original_and_three_probes_pass': robust[arm],
            'by_new_root': {rid: {'original_strict_pass': apps_scale._valid_private(json.loads(
                                apps_scale._old_case(arm, rid).read_text())),
                                  'new_probes_pass': sum(passed_by_root[(arm, rid)])}
                            for rid in PROBE_INPUTS},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {'status': 'COMPLETE', 'source_result_sha256': file_hash(SOURCE/'result.json'),
                        'previous_probe_sha256': file_hash(SOURCE/'probe_results.json'),
                        'new_probes': probes, 'new_cases': new_cases, 'summary': summary})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'score'))
    args = parser.parse_args(argv)
    if args.command == 'check':
        check()
    else:
        score()


if __name__ == '__main__':
    main()
