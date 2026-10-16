"""Single-sample OLD96 control on the frozen fresh APPS competition roots."""

import argparse
from collections import Counter
import json
from pathlib import Path

from . import apps_fresh_select as fresh, apps_multitask, apps_scale
from .dataset import file_hash
from .evaluate import write_json


ROOT = Path('runs/refinement/apps_fresh_old96')
ADAPTER = apps_scale.OLD_RUN/'training/final_adapter/adapter_model.safetensors'


def _inputs():
    return {
        'fresh_request': fresh.ROOT/'request.json',
        'fresh_result': fresh.ROOT/'result.json',
        'fresh_selection': fresh.DATA/'selection.json',
        'fresh_roots': fresh.DATA/'roots.jsonl',
        'fresh_tests': fresh.DATA/'tests.jsonl',
        'scale_request': apps_scale.ROOT/'request.json',
        'old_adapter': ADAPTER,
    }


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'OLD96 control already exists: {ROOT}')
    fresh.check()
    source = json.loads((fresh.ROOT/'result.json').read_text())
    training = json.loads((apps_scale.OLD_RUN/'training/result.json').read_text())
    if (source['status'] != 'COMPLETE' or len(source['by_root']) != 48
            or training['status'] != 'COMPLETE' or training['adapter_sha256'] != file_hash(ADAPTER)):
        raise ValueError('Frozen LARGE result or OLD96 adapter is incomplete')
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arm': 'OLD96', 'roots': 48,
        'fresh_input_sha256': {key: file_hash(path) for key, path in _inputs().items()},
        'seed_base': fresh.SEED_BASE, 'root_plan': 'fixed ATOMIC',
        'candidates_per_root': 1, 'metric': 'all held-out original APPS I/O exact pass',
        'gpu_devices': list(fresh.GPU_IDS), 'training_steps': 0,
    })
    write_json(ROOT/'gpu_resources.json', {'devices': list(fresh.GPU_IDS),
                                           'min_free_mib': {'collect': 16000}})
    check()


def check():
    fresh.check()
    request = json.loads((ROOT/'request.json').read_text())
    resources = json.loads((ROOT/'gpu_resources.json').read_text())
    source = json.loads((fresh.ROOT/'result.json').read_text())
    training = json.loads((apps_scale.OLD_RUN/'training/result.json').read_text())
    rows = fresh._rows()
    if (request['status'] != 'PREPARED' or request['arm'] != 'OLD96'
            or request['roots'] != 48 or request['seed_base'] != fresh.SEED_BASE
            or request['root_plan'] != 'fixed ATOMIC' or request['candidates_per_root'] != 1
            or request['metric'] != 'all held-out original APPS I/O exact pass'
            or request['gpu_devices'] != list(fresh.GPU_IDS) or request['training_steps'] != 0
            or request['fresh_input_sha256'] != {key: file_hash(path) for key, path in _inputs().items()}
            or resources != {'devices': list(fresh.GPU_IDS), 'min_free_mib': {'collect': 16000}}
            or training['status'] != 'COMPLETE' or training['adapter_sha256'] != file_hash(ADAPTER)
            or source['status'] != 'COMPLETE'
            or source['request_sha256'] != file_hash(fresh.ROOT/'request.json')
            or set(source['by_root']) != {row['root_id'] for row in rows}):
        raise ValueError('Frozen OLD96 control design or launch configuration changed')
    ready = {'status': 'READY', 'arm': 'OLD96', 'roots': len(rows), 'model_rollouts': len(rows),
             'rollouts_per_gpu': len(rows)//len(fresh.GPU_IDS),
             'gpu_devices': list(fresh.GPU_IDS), 'root_plan': 'fixed ATOMIC',
             'seed_base': fresh.SEED_BASE, 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _case(row, index, request):
    rid = row['root_id']
    path = ROOT/'evaluation'/rid/f'seed_{request["seed_base"]+100*index}'/'case_result.json'
    return path


def _validate_case(case, rid, seed, adapter_sha256):
    executor = case['model_identity']['executor']
    if (case['root_id'] != rid or case['seed'] != seed or case['status'] != 'COMPLETE'
            or case['registered_plan_modes'] != {'ATOMIC': 1}
            or case['continuation_seed_reset_after_root_plan'] != seed
            or executor['apps_scale_arm'] != 'OLD96'
            or executor['adapter_weights_sha256'] != adapter_sha256):
        raise ValueError(f'Incomplete or mismatched OLD96 rollout: {rid}')


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < len(fresh.GPU_IDS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed OLD96 shard already exists: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    rows, tests = fresh._rows(), fresh._tests()
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = apps_scale._context('OLD96')
        completed = {}
        for index in tqdm(range(shard, len(rows), len(fresh.GPU_IDS)),
                          desc=f'APPS旧执行器 GPU {fresh.GPU_IDS[shard]}', unit='题',
                          position=shard, dynamic_ncols=True):
            row = rows[index]
            path = _case(row, index, request)
            policy = apps_multitask._ForcedRoot(planner, 'ATOMIC')
            case = run_case(row, tests, old['framework'], old['config'], policy, executor,
                            path.parent, request['seed_base']+100*index,
                            reset_seed_after_root_plan=True)
            _validate_case(case, row['root_id'], request['seed_base']+100*index,
                           request['fresh_input_sha256']['old_adapter'])
            completed[row['root_id']] = str(path)
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'), 'by_root': completed})
    finally:
        reservation.close()
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(completed)},
                     ensure_ascii=False), flush=True)


def report():
    check()
    if (ROOT/'result.json').exists():
        raise FileExistsError('OLD96 control report already exists')
    request = json.loads((ROOT/'request.json').read_text())
    source = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    rows = fresh._rows()
    paths = {}
    for shard in range(len(fresh.GPU_IDS)):
        item = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (item['status'] != 'COMPLETE' or item['shard'] != shard
                or item['request_sha256'] != file_hash(ROOT/'request.json')
                or len(item['by_root']) != 16 or set(paths) & set(item['by_root'])):
            raise ValueError(f'Incomplete OLD96 shard: {shard}')
        paths.update(item['by_root'])
    if set(paths) != set(source):
        raise ValueError('OLD96 and LARGE roots differ')
    groups = {name: Counter() for name in ('all', 'sample', 'no_sample')}
    outcomes = {}
    for index, row in enumerate(rows):
        rid = row['root_id']
        path = _case(row, index, request)
        if paths[rid] != str(path):
            raise ValueError(f'OLD96 rollout path differs: {rid}')
        case = json.loads(path.read_text())
        _validate_case(case, rid, request['seed_base']+100*index,
                       request['fresh_input_sha256']['old_adapter'])
        old_exact = apps_scale._valid_private(case)
        large_exact = source[rid]['exact_pass'][0]
        old_legacy = case['final_scores']['private']['status'] == 'PASS'
        large_legacy = source[rid]['legacy_pass'][0]
        for group in ('all', source[rid]['group']):
            counts = groups[group]
            counts['roots'] += 1
            counts['old96_exact'] += old_exact
            counts['large_exact'] += large_exact
            counts['old96_legacy'] += old_legacy
            counts['large_legacy'] += large_legacy
            counts['old96_wins' if old_exact and not large_exact else
                   'large_wins' if large_exact and not old_exact else 'ties'] += 1
            counts['old96_programs'] += case['final_program'] is not None
        outcomes[rid] = {'group': source[rid]['group'], 'case_path': str(path),
                         'old96_exact': old_exact, 'large_exact': large_exact,
                         'old96_legacy': old_legacy, 'large_legacy': large_legacy}
    summary = {name: dict(counts) for name, counts in groups.items()}
    write_json(ROOT/'result.json', {'status': 'COMPLETE',
                                    'request_sha256': file_hash(ROOT/'request.json'),
                                    'summary': summary, 'by_root': outcomes})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary},
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
