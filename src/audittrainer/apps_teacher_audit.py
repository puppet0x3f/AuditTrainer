"""Audit APPS reference solutions on the 21 frozen independent-probe roots.

The original teacher-selection function is reused exactly. All scoring is CPU-only;
the saved model programs and model weights are not changed.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from . import apps_multitask, apps_scale, apps_scale_followup
from .apps import stdin_text
from .apps_scale_probes import valid_output
from .dataset import file_hash
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox


ROOT = Path('runs/refinement/apps_teacher_audit')
SHARDS = 3


def _probes():
    original = json.loads((apps_scale.DATA/'probes.json').read_text())
    added = json.loads((apps_scale_followup.ROOT/'result.json').read_text())['new_probes']
    return original + added


def check():
    apps_scale_followup.check()
    followup = json.loads((apps_scale_followup.ROOT/'result.json').read_text())
    if (followup['status'] != 'COMPLETE'
            or followup['source_result_sha256'] != file_hash(apps_scale.ROOT/'result.json')
            or followup['previous_probe_sha256'] != file_hash(apps_scale.ROOT/'probe_results.json')):
        raise ValueError('Completed APPS follow-up results changed')
    probes = _probes()
    grouped = defaultdict(list)
    for probe in probes:
        grouped[probe['root_id']].append(probe)
    roots = {row['root_id']: row for row in apps_scale._validation_rows()}
    if (len(grouped) != 21 or len(probes) != 63 or not set(grouped) <= set(roots)
            or any(len(items) != 3 or len({item['input'] for item in items}) != 3
                   for items in grouped.values())):
        raise ValueError('Expected 21 held-out roots with three distinct probes each')
    framework = json.loads((apps_multitask.ROOT/'request.json').read_text())['framework']
    ready = {'status': 'READY', 'roots': 21, 'probes': 63,
             'selection': 'original APPS shortest-first passing reference, same prompt and token limits',
             'shards': SHARDS, 'cpu_workers': SHARDS, 'gpu_devices': [], 'training_steps': 0,
             'sandbox_timeout_seconds': framework['execution']['per_case_timeout_seconds']}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _score_case(sandbox, source, root_id, text, expected):
    execution = sandbox.run(source, 'solve', [text])
    passed = bool(execution.status == 'RETURNED' and
                  valid_output(root_id, text, execution.value, expected))
    return {'pass': passed, 'status': execution.status, 'reason': execution.reason,
            'actual': execution.value}


def score_shard(shard):
    from tqdm import tqdm

    if not 0 <= shard < SHARDS:
        raise ValueError(f'Expected shard 0..{SHARDS-1}')
    output = ROOT/f'shard_{shard}.json'
    if output.exists():
        raise FileExistsError(f'Teacher audit shard already exists: {output}')
    request = json.loads((apps_multitask.ROOT/'request.json').read_text())
    framework = request['framework']
    tokenizer = apps_multitask._tokenizer(framework, request['adapters'])
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    probes = defaultdict(list)
    for item in _probes():
        probes[item['root_id']].append(item)
    rows = {row['root_id']: row for row in apps_scale._validation_rows()}
    selected = sorted(probes)[shard::SHARDS]
    results = []
    for rid in tqdm(selected, desc=f'教师参考解审计 CPU {shard}', unit='题', dynamic_ncols=True):
        raw = json.loads(Path(rows[rid]['raw_path']).read_text())
        with tempfile.TemporaryDirectory(prefix=f'apps_teacher_{rid}_') as temp:
            target = apps_multitask._target(raw, 'VALIDATION', framework, tokenizer, Path(temp))
        if target is None:
            raise RuntimeError(f'Original teacher selection no longer finds a reference: {rid}')
        source = target['target_action']['args']['source']
        io = json.loads(raw['input_output']) if isinstance(raw['input_output'], str) else raw['input_output']
        original = [_score_case(sandbox, source, rid, stdin_text(text), stdin_text(expected))
                    for text, expected in zip(io['inputs'], io['outputs'])]
        independent = [{'case_id': probe['case_id'],
                        **_score_case(sandbox, source, rid, probe['input'], probe['expected'])}
                       for probe in probes[rid]]
        results.append({'root_id': rid, 'difficulty': raw['difficulty'],
                        'solution_index': target['solution_index'],
                        'source_sha256': target['source_sha256'],
                        'prompt_tokens': target['prompt_tokens'],
                        'target_tokens': target['target_tokens'],
                        'original_cases': original, 'independent_cases': independent,
                        'strict_original_pass': bool(original) and all(item['pass'] for item in original),
                        'independent_all_three_pass': all(item['pass'] for item in independent)})
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(output, {'status': 'COMPLETE', 'shard': shard, 'shards': SHARDS,
                        'source_result_sha256': file_hash(apps_scale.ROOT/'result.json'),
                        'followup_result_sha256': file_hash(apps_scale_followup.ROOT/'result.json'),
                        'probe_count': 3*len(selected), 'roots': results})
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(results),
                      'strict_original_pass': sum(row['strict_original_pass'] for row in results),
                      'independent_all_three_pass': sum(row['independent_all_three_pass'] for row in results)},
                     ensure_ascii=False), flush=True)


def report():
    probes = _probes()
    expected_roots = {probe['root_id'] for probe in probes}
    rows = []
    for shard in range(SHARDS):
        result = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (result['status'] != 'COMPLETE' or result['shard'] != shard or result['shards'] != SHARDS
                or result['source_result_sha256'] != file_hash(apps_scale.ROOT/'result.json')
                or result['followup_result_sha256'] != file_hash(apps_scale_followup.ROOT/'result.json')
                or len(result['roots']) != 7 or result['probe_count'] != 21):
            raise ValueError(f'Incomplete or mismatched teacher audit shard {shard}')
        rows.extend(result['roots'])
    if len({row['root_id'] for row in rows}) != 21 or {row['root_id'] for row in rows} != expected_roots:
        raise ValueError('Teacher audit roots do not match the frozen probe set')
    rows.sort(key=lambda row: row['root_id'])
    old = json.loads((apps_scale.ROOT/'probe_results.json').read_text())['cases']
    added = json.loads((apps_scale_followup.ROOT/'result.json').read_text())['new_cases']
    model_probes = defaultdict(list)
    for item in old + added:
        model_probes[(item['arm'], item['root_id'])].append(item['pass'])
    by_root = {}
    for row in rows:
        rid = row['root_id']
        models = {}
        for arm in ('OLD96', 'SMALL', 'LARGE'):
            case = json.loads(apps_scale._old_case(arm, rid).read_text())
            values = model_probes[(arm, rid)]
            if len(values) != 3:
                raise ValueError(f'Missing model probes: {arm}/{rid}')
            models[arm] = {'strict_original_pass': apps_scale._valid_private(case),
                           'independent_cases_pass': sum(values),
                           'independent_all_three_pass': all(values)}
        by_root[rid] = {'teacher': row, 'models': models}
    fragile_large = [rid for rid, item in by_root.items()
                     if item['models']['LARGE']['strict_original_pass']
                     and not item['models']['LARGE']['independent_all_three_pass']]
    teacher_robust = {row['root_id'] for row in rows
                      if row['strict_original_pass'] and row['independent_all_three_pass']}
    summary = {
        'roots': len(rows), 'independent_cases': len(probes),
        'teacher_strict_original_pass': sum(row['strict_original_pass'] for row in rows),
        'teacher_independent_cases_pass': sum(item['pass'] for row in rows
                                              for item in row['independent_cases']),
        'teacher_independent_all_three_pass': sum(row['independent_all_three_pass'] for row in rows),
        'teacher_strict_original_and_all_three_pass': len(teacher_robust),
        'large_original_pass_but_probe_fail': fragile_large,
        'teacher_robust_among_those': sorted(teacher_robust.intersection(fragile_large)),
        'selected_solution_indices': dict(Counter(row['solution_index'] for row in rows)),
    }
    write_json(ROOT/'result.json', {'status': 'COMPLETE', 'summary': summary, 'by_root': by_root})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'score-shard', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'check':
        check()
    elif args.command == 'score-shard':
        score_shard(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
