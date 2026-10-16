"""Oracle pass@k diagnostic for the frozen LARGE APPS executor.

The cached natural rollout is candidate one. Four fresh seeds per root keep the
same ATOMIC root plan and sample only the executor continuation. Probe answers
are used after each rollout for offline analysis, never for candidate selection.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from . import apps_multitask, apps_scale, apps_scale_followup, apps_teacher_audit
from .apps_scale_probes import valid_output
from .artifacts import ArtifactStore
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .sandbox import ExecutionLimits, Sandbox
from .schema import Artifact


ROOT = Path('runs/refinement/apps_passk')
GPU_IDS = ('0', '1', '2')
FRESH_SAMPLES = 4
SEED_BASE = 716000713


def _probe_groups():
    grouped = defaultdict(list)
    for probe in apps_teacher_audit._probes():
        grouped[probe['root_id']].append(probe)
    return grouped


def _rows():
    return apps_scale._validation_rows()


def _seed(validation_index, repetition):
    return SEED_BASE + 100*validation_index + repetition


def _source_hashes():
    return {'scale_result': file_hash(apps_scale.ROOT/'result.json'),
            'scale_probes': file_hash(apps_scale.ROOT/'probe_results.json'),
            'followup_result': file_hash(apps_scale_followup.ROOT/'result.json'),
            'teacher_audit_result': file_hash(apps_teacher_audit.ROOT/'result.json')}


def check():
    apps_teacher_audit.check()
    teacher = json.loads((apps_teacher_audit.ROOT/'result.json').read_text())
    if teacher['status'] != 'COMPLETE' or teacher['summary']['roots'] != 21:
        raise ValueError('Teacher audit is incomplete')
    groups = _probe_groups()
    rows = _rows()
    indices = {row['root_id']: index for index, row in enumerate(rows)}
    if len(groups) != 21 or any(len(probes) != 3 for probes in groups.values()):
        raise ValueError('Expected three frozen probes for each of 21 roots')
    request = json.loads((apps_scale.ROOT/'request.json').read_text())
    decoding = request['framework']['inference']['decoding']
    if not decoding['do_sample'] or decoding['temperature'] <= 0:
        raise ValueError('Executor decoding must support independent stochastic samples')
    gpu = json.loads((apps_scale.ROOT/'gpu_resources.json').read_text())
    if gpu['devices'] != list(GPU_IDS) or gpu['min_free_mib']['collect'] != 16000:
        raise ValueError('Expected evaluation GPUs 0, 1, 2 with 16 GiB free each')
    old_probes = defaultdict(list)
    for item in (json.loads((apps_scale.ROOT/'probe_results.json').read_text())['cases']
                 + json.loads((apps_scale_followup.ROOT/'result.json').read_text())['new_cases']):
        if item['arm'] == 'LARGE':
            old_probes[item['root_id']].append(item)
    for rid in groups:
        case = json.loads(apps_scale._old_case('LARGE', rid).read_text())
        if (case['status'] != 'COMPLETE' or case['seed'] != request['eval_seed'] + indices[rid]
                or case['registered_plan_modes'] != {'ATOMIC': 1} or case['node_count'] != 1
                or len(old_probes[rid]) != 3):
            raise ValueError(f'Cached first candidate is not the expected single-node ATOMIC rollout: {rid}')
    seeds = [_seed(indices[rid], repetition) for rid in groups for repetition in range(FRESH_SAMPLES)]
    if len(set(seeds)) != 84 or any(seed in {request['eval_seed']+i for i in range(48)} for seed in seeds):
        raise ValueError('Fresh seeds collide with each other or cached APPS evaluations')
    ready = {'status': 'READY', 'probe_roots': len(groups), 'probes_per_root': 3,
             'cached_candidates': len(groups), 'fresh_candidates': len(seeds),
             'pass_at_k': list(range(1, FRESH_SAMPLES+2)),
             'root_plan': 'fixed ATOMIC', 'executor': 'LARGE final adapter',
             'decoding': decoding, 'fresh_seed_base': SEED_BASE,
             'gpu_devices': GPU_IDS, 'rollouts_per_gpu': len(seeds)//len(GPU_IDS),
             'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _score_probes(case, probes, sandbox):
    snapshot = case['final_program']
    if snapshot:
        bundle = ArtifactStore(Path(snapshot['path'])).bundle(Artifact.model_validate(snapshot['artifact']))
        source = bundle.sources['N0']
    scored = []
    for probe in probes:
        execution = sandbox.run(source, 'solve', [probe['input']], bundle=bundle) if snapshot else None
        passed = bool(execution and execution.status == 'RETURNED' and
                      valid_output(case['root_id'], probe['input'], execution.value, probe['expected']))
        scored.append({'case_id': probe['case_id'], 'pass': passed,
                       'status': execution.status if execution else 'NO_PROGRAM',
                       'reason': execution.reason if execution else 'NO_PROGRAM',
                       'actual': execution.value if execution else None})
    return scored


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < len(GPU_IDS):
        raise ValueError('Expected one of three GPU shards')
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed pass@k shard already exists: {done}')
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    framework = old['framework']
    tests = {row['root_id']: row for row in read_jsonl(apps_scale.OLD_DATA/'tests.jsonl')}
    all_rows = _rows()
    by_root = {row['root_id']: (index, row) for index, row in enumerate(all_rows)}
    groups = _probe_groups()
    selected = sorted(groups)[shard::len(GPU_IDS)]
    reservation = acquire(apps_scale.ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = apps_scale._context('LARGE')
        sandbox = Sandbox(ExecutionLimits(**framework['execution']))
        results = []
        for rid in tqdm(selected, desc=f'APPS pass@k GPU {GPU_IDS[shard]}', unit='题',
                        position=shard, dynamic_ncols=True):
            index, row = by_root[rid]
            for repetition in range(FRESH_SAMPLES):
                seed = _seed(index, repetition)
                directory = ROOT/'evaluation'/rid/f'seed_{seed}'
                policy = apps_multitask._ForcedRoot(planner, 'ATOMIC')
                case = run_case(row, tests, framework, old['config'], policy, executor,
                                directory, seed, reset_seed_after_root_plan=True)
                if (case['status'] != 'COMPLETE' or case['registered_plan_modes'] != {'ATOMIC': 1}
                        or case['node_count'] != 1):
                    raise RuntimeError(f'Incomplete or changed execution structure: {rid}/{seed}')
                score_path = directory/'probe_score.json'
                if score_path.exists():
                    scored = json.loads(score_path.read_text())
                else:
                    probes = _score_probes(case, groups[rid], sandbox)
                    scored = {'root_id': rid, 'seed': seed, 'cases': probes,
                              'strict_original_pass': apps_scale._valid_private(case),
                              'probes_all_three_pass': all(item['pass'] for item in probes)}
                    write_json(score_path, scored)
                if scored['root_id'] != rid or scored['seed'] != seed or len(scored['cases']) != 3:
                    raise ValueError(f'Invalid saved probe score: {rid}/{seed}')
                results.append({'root_id': rid, 'repetition': repetition, 'seed': seed,
                                'case_path': str(directory/'case_result.json'),
                                'probe_path': str(score_path),
                                'strict_original_pass': scored['strict_original_pass'],
                                'probe_cases_pass': sum(item['pass'] for item in scored['cases']),
                                'probes_all_three_pass': scored['probes_all_three_pass'],
                                'robust_pass': scored['strict_original_pass'] and scored['probes_all_three_pass']})
        write_json(done, {'status': 'COMPLETE', 'shard': shard, 'shards': len(GPU_IDS),
                          'source_hashes': _source_hashes(), 'roots': len(selected),
                          'rollouts': results})
    finally:
        reservation.close()
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(selected),
                      'fresh_rollouts': len(results), 'robust_rollouts': sum(r['robust_pass'] for r in results)},
                     ensure_ascii=False), flush=True)


def cumulative_pass_counts(attempts):
    """Number of roots with at least one pass among the first k attempts."""
    lengths = {len(values) for values in attempts.values()}
    if len(lengths) != 1:
        raise ValueError('Every root must have the same number of candidates')
    return [sum(any(values[:k]) for values in attempts.values())
            for k in range(1, lengths.pop()+1)]


def report():
    groups = _probe_groups()
    rows = _rows()
    indices = {row['root_id']: index for index, row in enumerate(rows)}
    expected = {(rid, repetition) for rid in groups for repetition in range(FRESH_SAMPLES)}
    fresh = {}
    for shard in range(len(GPU_IDS)):
        done = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (done['status'] != 'COMPLETE' or done['shard'] != shard or done['shards'] != len(GPU_IDS)
                or done['source_hashes'] != _source_hashes() or done['roots'] != 7
                or len(done['rollouts']) != 28):
            raise ValueError(f'Incomplete or mismatched pass@k shard: {shard}')
        for item in done['rollouts']:
            key = (item['root_id'], item['repetition'])
            if (key in fresh or item['seed'] != _seed(indices[item['root_id']], item['repetition'])
                    or item['probe_cases_pass'] not in range(4)):
                raise ValueError(f'Duplicate or invalid pass@k candidate: {key}')
            fresh[key] = item
    if set(fresh) != expected:
        raise ValueError('Missing fresh pass@k candidates')
    cached_probes = defaultdict(list)
    for item in (json.loads((apps_scale.ROOT/'probe_results.json').read_text())['cases']
                 + json.loads((apps_scale_followup.ROOT/'result.json').read_text())['new_cases']):
        if item['arm'] == 'LARGE':
            cached_probes[item['root_id']].append(item)
    by_root, strict, robust = {}, {}, {}
    for rid in sorted(groups):
        cached = json.loads(apps_scale._old_case('LARGE', rid).read_text())
        baseline = {'repetition': 'cached', 'seed': cached['seed'],
                    'strict_original_pass': apps_scale._valid_private(cached),
                    'probe_cases_pass': sum(item['pass'] for item in cached_probes[rid]),
                    'probes_all_three_pass': len(cached_probes[rid]) == 3
                                                  and all(item['pass'] for item in cached_probes[rid])}
        baseline['robust_pass'] = baseline['strict_original_pass'] and baseline['probes_all_three_pass']
        attempts = [baseline] + [fresh[(rid, repetition)] for repetition in range(FRESH_SAMPLES)]
        by_root[rid] = attempts
        strict[rid] = [item['strict_original_pass'] for item in attempts]
        robust[rid] = [item['robust_pass'] for item in attempts]
    teacher = json.loads((apps_teacher_audit.ROOT/'result.json').read_text())
    teacher_robust = {rid for rid, item in teacher['by_root'].items()
                      if item['teacher']['strict_original_pass']
                      and item['teacher']['independent_all_three_pass']}
    recovered = {rid for rid, values in robust.items() if any(values)}
    summary = {'roots': len(groups), 'fresh_rollouts': len(fresh),
               'strict_original_pass_at_k': dict(enumerate(cumulative_pass_counts(strict), 1)),
               'robust_pass_at_k': dict(enumerate(cumulative_pass_counts(robust), 1)),
               'teacher_robust_roots': len(teacher_robust),
               'new_robust_roots': sorted(rid for rid in recovered if not robust[rid][0]),
               'teacher_robust_not_generated': sorted(teacher_robust-recovered),
               'metric': 'offline oracle upper bound; private and probe answers never select candidates'}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT/'result.json', {'status': 'COMPLETE', 'source_hashes': _source_hashes(),
                                    'summary': summary, 'by_root': by_root})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'evaluate', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'check':
        check()
    elif args.command == 'evaluate':
        evaluate(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
