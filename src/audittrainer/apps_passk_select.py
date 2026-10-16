"""Select saved APPS pass@k programs using only public statement examples.

Selection never reads private scores or independent probes. The completed
pass@k result is opened only by report(), after all choices are saved.
"""

import argparse
import json
from pathlib import Path
import re

from . import apps_passk, apps_scale
from .apps_scale_probes import valid_output
from .artifacts import ArtifactStore
from .dataset import file_hash
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox
from .schema import Artifact


ROOT = Path('runs/refinement/apps_passk_select')
SHARDS = 3
CANDIDATES = 5


def _heading(line):
    cleaned = re.sub(r'[-=]{2,}', ' ', line.strip())
    cleaned = re.sub(r'^[\s:#-]+|[\s:#-]+$', '', cleaned).lower()
    return ' '.join(cleaned.split())


def public_example(spec, *, stop_at_blank_output=False):
    """Extract one literal input/output pair from a statement example block."""
    lines = spec.splitlines()
    in_example = False
    start = None
    for index, line in enumerate(lines):
        heading = _heading(line)
        if start is None:
            if heading in {'example', 'sample'}:
                in_example = True
            elif heading in {'sample input', 'example input'} or (in_example and heading == 'input'):
                start = index + 1
            elif heading in {'input format', 'output format', 'constraints', 'explanation', 'note'}:
                in_example = False
            continue
        if heading in {'sample output', 'example output', 'output'}:
            input_text = '\n'.join(lines[start:index]).strip()
            output_lines = []
            for next_line in lines[index+1:]:
                if stop_at_blank_output and output_lines and not next_line.strip():
                    break
                next_heading = _heading(next_line)
                if (next_heading in {'explanation', 'note', 'sample', 'example', 'constraints',
                                     'input format', 'output format', 'sample input', 'example input'}
                        or re.match(r'(?i)^note\s*:', next_line.strip())
                        or next_line.lstrip().startswith('Implement solve(')):
                    break
                if next_line.strip():
                    output_lines.append(next_line.strip())
            expected = '\n'.join(output_lines)
            return {'input': input_text + '\n', 'expected': expected} if input_text and expected else None
        if heading in {'explanation', 'note', 'constraints', 'input format', 'output format'}:
            return None
    return None


def _roots():
    directory = apps_passk.ROOT/'evaluation'
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def _case_paths(root_id):
    fresh = sorted((apps_passk.ROOT/'evaluation'/root_id).glob('seed_*/case_result.json'))
    return [apps_scale._old_case('LARGE', root_id), *fresh]


def _rows():
    return {row['root_id']: row for row in apps_scale._validation_rows()}


def _sample(row):
    task = json.loads(Path(row['fixture_path']).read_text())['task']
    return public_example(task['spec'])


def _overlaps_original(row, sample):
    if sample is None:
        return False
    suites = json.loads(Path(row['fixture_path']).with_name('private_tests.json').read_text())['suites']
    return any(sample['input'].strip() == case['args'][0].strip()
               for suite in suites for case in suite['cases'])


def check():
    prior = json.loads((apps_passk.ROOT/'result.json').read_text())
    if (prior['status'] != 'COMPLETE' or prior['summary']['roots'] != 21
            or prior['summary']['fresh_rollouts'] != 84):
        raise ValueError('Completed APPS pass@k result required')
    roots = _roots()
    if roots != sorted(prior['by_root']):
        raise ValueError('Saved program roots differ from completed pass@k result')
    rows = _rows()
    samples = {}
    for root_id in roots:
        paths = _case_paths(root_id)
        if len(paths) != CANDIDATES or any(not path.is_file() for path in paths):
            raise ValueError(f'Expected five saved candidates: {root_id}')
        if [str(path) for path in paths[1:]] != [item['case_path']
                                                  for item in prior['by_root'][root_id][1:]]:
            raise ValueError(f'Saved candidate order differs from pass@k result: {root_id}')
        samples[root_id] = _sample(rows[root_id])
    framework = json.loads((apps_scale.ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    if sandbox.bwrap_version == 'UNAVAILABLE':
        raise RuntimeError('Bubblewrap sandbox is unavailable')
    ready = {'status': 'READY', 'roots': len(roots), 'saved_candidates': len(roots)*CANDIDATES,
             'public_example_roots': sum(value is not None for value in samples.values()),
             'public_example_overlaps_original_roots': sum(
                 _overlaps_original(rows[rid], sample) for rid, sample in samples.items()),
             'without_public_example': [rid for rid in roots if samples[rid] is None],
             'selection': 'public statement example pass, returned output, program exists, then earlier seed',
             'evaluation_labels': 'private original and independent probes, report only',
             'shards': SHARDS, 'cpu_only': True, 'gpu_devices': [], 'training_steps': 0,
             'sandbox_timeout_seconds': sandbox.limits.per_case_timeout_seconds}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def choose(scores, k, has_sample):
    """Choose one index from the first k using public execution signals only."""
    if not 1 <= k <= len(scores):
        raise ValueError('Invalid candidate count')
    def rank(index):
        item = scores[index]
        if has_sample:
            return (bool(item['sample_pass']), item['sample_status'] == 'RETURNED',
                    bool(item['has_program']), -index)
        return (bool(item['has_program']), -index)
    return max(range(k), key=rank)


def _score(path, root_id, sample, sandbox):
    record = json.loads(path.read_text())
    if record['root_id'] != root_id:
        raise ValueError(f'Candidate root mismatch: {path}')
    snapshot = record['final_program']
    result = {'case_path': str(path), 'has_program': bool(snapshot),
              'sample_status': 'NO_SAMPLE' if sample is None else 'NO_PROGRAM',
              'sample_pass': None if sample is None else False}
    if sample is None or snapshot is None:
        return result
    bundle = ArtifactStore(Path(snapshot['path'])).bundle(Artifact.model_validate(snapshot['artifact']))
    execution = sandbox.run(bundle.sources['N0'], 'solve', [sample['input']], bundle=bundle)
    result['sample_status'] = execution.status
    result['sample_pass'] = bool(execution.status == 'RETURNED' and
                                 valid_output(root_id, sample['input'], execution.value, sample['expected']))
    return result


def select_shard(shard):
    from tqdm import tqdm

    if not 0 <= shard < SHARDS:
        raise ValueError('Invalid CPU shard')
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Selection shard already exists: {done}')
    rows = _rows()
    roots = _roots()[shard::SHARDS]
    framework = json.loads((apps_scale.ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    selected = {}
    for root_id in tqdm(roots, desc=f'APPS公开样例 CPU {shard}', unit='题', position=shard,
                        dynamic_ncols=True):
        sample = _sample(rows[root_id])
        scores = [_score(path, root_id, sample, sandbox) for path in _case_paths(root_id)]
        selected[root_id] = {'has_public_example': sample is not None,
                             'candidate_signals': scores,
                             'selected_index_by_k': {str(k): choose(scores, k, sample is not None)
                                                     for k in range(1, CANDIDATES+1)}}
    write_json(done, {'status': 'COMPLETE', 'shard': shard, 'source_passk_sha256':
                      file_hash(apps_passk.ROOT/'result.json'), 'by_root': selected})
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(roots),
                      'public_example_roots': sum(v['has_public_example'] for v in selected.values())},
                     ensure_ascii=False), flush=True)


def report():
    source = json.loads((apps_passk.ROOT/'result.json').read_text())
    selected = {}
    for shard in range(SHARDS):
        record = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (record['status'] != 'COMPLETE' or record['shard'] != shard
                or record['source_passk_sha256'] != file_hash(apps_passk.ROOT/'result.json')):
            raise ValueError(f'Incomplete or mismatched selection shard: {shard}')
        if set(selected) & set(record['by_root']):
            raise ValueError(f'Duplicate roots in selection shard: {shard}')
        selected.update(record['by_root'])
    if set(selected) != set(source['by_root']):
        raise ValueError('Missing selected roots')
    totals = {str(k): {'strict_original': 0, 'independent_all_three': 0,
                       'robust': 0, 'oracle_robust': 0}
              for k in range(1, CANDIDATES+1)}
    by_root = {}
    for root_id in sorted(selected):
        signals = selected[root_id]
        candidates = source['by_root'][root_id]
        if len(signals['candidate_signals']) != CANDIDATES or len(candidates) != CANDIDATES:
            raise ValueError(f'Candidate count mismatch: {root_id}')
        expected_paths = [str(path) for path in _case_paths(root_id)]
        if [item['case_path'] for item in signals['candidate_signals']] != expected_paths:
            raise ValueError(f'Saved candidate order changed: {root_id}')
        choices = {}
        for k in range(1, CANDIDATES+1):
            index = signals['selected_index_by_k'][str(k)]
            if index != choose(signals['candidate_signals'], k, signals['has_public_example']):
                raise ValueError(f'Selection rule mismatch: {root_id}/{k}')
            outcome = candidates[index]
            totals[str(k)]['strict_original'] += bool(outcome['strict_original_pass'])
            totals[str(k)]['independent_all_three'] += bool(outcome['probes_all_three_pass'])
            totals[str(k)]['robust'] += bool(outcome['robust_pass'])
            totals[str(k)]['oracle_robust'] += any(item['robust_pass'] for item in candidates[:k])
            choices[str(k)] = {'index': index, 'strict_original_pass': outcome['strict_original_pass'],
                               'independent_all_three_pass': outcome['probes_all_three_pass'],
                               'robust_pass': outcome['robust_pass']}
        by_root[root_id] = {'has_public_example': signals['has_public_example'],
                            'candidate_signals': signals['candidate_signals'], 'choices': choices}
    rows = _rows()
    summary = {'roots': len(selected),
               'public_example_roots': sum(v['has_public_example'] for v in selected.values()),
               'public_example_overlaps_original_roots': sum(
                   _overlaps_original(rows[rid], _sample(rows[rid])) for rid in selected),
               'selected_vs_oracle': totals,
               'selection_inputs': 'public statement examples and execution status only',
               'evaluation_labels': 'strict original and independent probes; loaded only in report'}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT/'result.json', {'status': 'COMPLETE', 'source_passk_sha256':
                                    file_hash(apps_passk.ROOT/'result.json'),
                                    'summary': summary, 'by_root': by_root})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'select-shard', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'check':
        check()
    elif args.command == 'select-shard':
        select_shard(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
