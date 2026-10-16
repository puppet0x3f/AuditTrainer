"""CPU replay of saved TACO direct programs with one fixed stdio adapter."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import apps_taco_direct_baseline as baseline, apps_taco_transfer as transfer
from .artifacts import ArtifactStore
from .dataset import file_hash, read_jsonl
from .evaluate import load_task, score_program, write_json
from .schema import Contract


ROOT = Path('runs/refinement/apps_taco_io_replay')
ARMS = baseline.ARMS
BRIDGE = '''
def solve(text):
    import io as _bridge_io
    import sys as _bridge_sys
    _old_stdin, _old_stdout = _bridge_sys.stdin, _bridge_sys.stdout
    _new_stdin = _bridge_io.TextIOWrapper(
        _bridge_io.BytesIO(text.encode('utf-8')), encoding='utf-8')
    _output_bytes = _bridge_io.BytesIO()
    _new_stdout = _bridge_io.TextIOWrapper(
        _output_bytes, encoding='utf-8', write_through=True)
    try:
        _bridge_sys.stdin, _bridge_sys.stdout = _new_stdin, _new_stdout
        _answer = _original_solve(text)
        _new_stdout.flush()
        _captured = _output_bytes.getvalue().decode('utf-8', errors='replace')
    finally:
        _bridge_sys.stdin, _bridge_sys.stdout = _old_stdin, _old_stdout
    if isinstance(_answer, str):
        return _answer if _answer or not _captured else _captured
    return _captured if _captured else _answer
'''


def bridge_source(source):
    tree = ast.parse(source)
    solves = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'solve']
    if len(solves) != 1:
        raise ValueError('Expected one nonrecursive top-level solve function')
    # A common direct-code pattern defines a local helper called ``solve`` and
    # invokes that helper from the public function.  A plain ast.walk treats
    # that call as recursion of the public function, although renaming the
    # top-level function leaves the local binding unchanged.  Find bindings in
    # the top-level function's lexical scope before checking calls.
    root = solves[0]

    class _SolveBindingVisitor(ast.NodeVisitor):
        def __init__(self):
            self.bound = any(argument.arg == 'solve'
                             for argument in (*root.args.posonlyargs,
                                               *root.args.args, *root.args.kwonlyargs))

        def visit_FunctionDef(self, node):
            if node is root:
                self.generic_visit(node)
            elif node.name == 'solve':
                self.bound = True

        def visit_AsyncFunctionDef(self, node):
            if node is root:
                self.generic_visit(node)
            elif node.name == 'solve':
                self.bound = True

        def visit_Lambda(self, node):
            return

        def visit_Name(self, node):
            if node.id == 'solve' and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.bound = True

    root_binds_solve = _SolveBindingVisitor()
    root_binds_solve.visit(root)

    class _RecursiveCallVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.generic_visit(node)

        def visit_Call(self, node):
            if (not root_binds_solve.bound and isinstance(node.func, ast.Name)
                    and node.func.id == 'solve'):
                raise ValueError('Expected one nonrecursive top-level solve function')
            self.generic_visit(node)

    _RecursiveCallVisitor().visit(root)
    solves[0].name = '_original_solve'
    adapted = ast.unparse(tree)+'\n'+BRIDGE
    compile(adapted, '<taco_stdio_bridge>', 'exec')
    return adapted


def _sources():
    return {
        'direct_request': baseline.ROOT/'request.json',
        'direct_result': baseline.ROOT/'result.json',
        'transfer_result': transfer.ROOT/'result.json',
        'roots': transfer.DATA/'roots.jsonl',
        'tests': transfer.DATA/'tests.jsonl',
        'selected': transfer.DATA/'selected_records.jsonl',
    }


def _original_path(arm, rid, seed):
    return baseline.ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def _replay_path(arm, rid, seed):
    return ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def _scores(case):
    visible = case['final_scores']['visible']['status'] == 'PASS'
    private = case['final_scores']['private']['status'] == 'PASS'
    return {'strict': int(visible and private), 'visible': int(visible),
            'private': int(private),
            'program': int(case['final_program'] is not None),
            'candidate_exception': int(case['final_scores']['visible']['reason'] ==
                                       'CANDIDATE_EXCEPTION')}


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'TACO interface replay already prepared: {ROOT}')
    baseline.check()
    source = json.loads((baseline.ROOT/'result.json').read_text())
    if source['status'] != 'COMPLETE' or source['roots'] != 100:
        raise ValueError('Direct TACO result is incomplete')
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arms': list(ARMS), 'roots': 100,
        'candidates_per_root': 1, 'cpu_shards': 3, 'gpu_devices': [],
        'seed_base': transfer.EVAL_SEED,
        'framework': json.loads((baseline.ROOT/'request.json').read_text())['framework'],
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'bridge_sha256': hashlib.sha256(BRIDGE.encode()).hexdigest(),
        'bridge': 'same wrapper for every valid source: virtual stdin and captured stdout; '
                  'preserve a nonempty string return, otherwise use captured output',
        'metric': 'visible and nine private TACO cases both pass',
        'training_steps': 0, 'model_calls': 0,
    })
    return check()


def check():
    baseline.check()
    request = json.loads((ROOT/'request.json').read_text())
    source = json.loads((baseline.ROOT/'result.json').read_text())
    expected_sources = {name: file_hash(path) for name, path in _sources().items()}
    framework = json.loads((baseline.ROOT/'request.json').read_text())['framework']
    if (request['status'] != 'PREPARED' or request['arms'] != list(ARMS)
            or request['roots'] != 100 or request['candidates_per_root'] != 1
            or request['cpu_shards'] != 3 or request['gpu_devices'] != []
            or request['seed_base'] != transfer.EVAL_SEED
            or request['framework'] != framework
            or request['source_sha256'] != expected_sources
            or request['bridge_sha256'] != hashlib.sha256(BRIDGE.encode()).hexdigest()
            or request['metric'] != 'visible and nine private TACO cases both pass'
            or request['training_steps'] != 0 or request['model_calls'] != 0
            or source['status'] != 'COMPLETE' or source['roots'] != 100
            or source['request_sha256'] != expected_sources['direct_request']):
        raise ValueError('TACO interface replay design or sources changed')
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    programs = 0
    for index, row in enumerate(rows):
        rid = row['root_id']
        seed = request['seed_base']+100*index
        for arm in ARMS:
            case = json.loads(_original_path(arm, rid, seed).read_text())
            if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                    or case['seed'] != seed or case['arm'] != arm
                    or _scores(case)['strict'] != source['by_root'][rid]['DIRECT'][arm]['strict']):
                raise ValueError(f'Saved direct candidate differs: {arm}/{rid}')
            if snapshot := case['final_program']:
                path = Path(snapshot['path'])/snapshot['artifact']['module_path']
                if file_hash(path) != snapshot['artifact']['source_hash']:
                    raise ValueError(f'Saved direct source changed: {arm}/{rid}')
                bridge_source(path.read_text())
                programs += 1
    ready = {'status': 'READY', 'roots': len(rows), 'candidates': len(rows)*len(ARMS),
             'programs_to_rescore': programs, 'cpu_shards': 3, 'gpu_devices': [],
             'model_calls': 0, 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def replay(shard):
    from tqdm import tqdm

    if not 0 <= shard < len(ARMS):
        raise ValueError('Expected CPU shard 0, 1, or 2')
    check()
    arm = ARMS[shard]
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        result = json.loads(done.read_text())
        if (result['status'] == 'COMPLETE' and result['arm'] == arm
                and result['request_sha256'] == file_hash(ROOT/'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete TACO replay shard: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA/'tests.jsonl')}
    with tqdm(rows, desc=f'TACO接口复评 {arm}', unit='题', position=shard,
              dynamic_ncols=True) as bar:
        for index, row in enumerate(bar):
            rid = row['root_id']
            seed = request['seed_base']+100*index
            source_path = _original_path(arm, rid, seed)
            path = _replay_path(arm, rid, seed)
            if path.exists():
                revised = json.loads(path.read_text())
                if (revised['status'] != 'COMPLETE' or revised['root_id'] != rid
                        or revised['seed'] != seed or revised['arm'] != arm
                        or revised['original_case_sha256'] != file_hash(source_path)
                        or revised['request_sha256'] != file_hash(ROOT/'request.json')):
                    raise ValueError(f'Saved interface replay differs: {arm}/{rid}')
                continue
            if path.parent.exists():
                raise ValueError(f'Unfinished interface replay exists: {path.parent}')
            original = json.loads(source_path.read_text())
            path.parent.mkdir(parents=True)
            snapshot = original['final_program']
            if snapshot is None:
                adapted, scores, cost = (None, original['final_scores'],
                                         original['scoring_cost'])
            else:
                _, _, contract_spec, _, verifier = load_task(row, tests, request['framework'])
                contract = Contract(**contract_spec.model_dump(),
                                    contract_id='C_N0_1', version=1)
                source = (Path(snapshot['path'])/snapshot['artifact']['module_path']).read_text()
                store = ArtifactStore(path.parent/'final_program')
                artifact = store.save_initial('N0', contract, bridge_source(source))
                adapted = {'path': str(path.parent/'final_program'), 'root_id': rid,
                           'artifact': artifact.model_dump(mode='json'),
                           'contract': contract.model_dump(mode='json')}
                scores, cost = score_program(adapted, verifier, rid)
            status = ('ERROR' if any(value['status'] == 'ERROR' for value in scores.values())
                      else 'COMPLETE')
            revised = {'status': status, 'root_id': rid, 'arm': arm, 'seed': seed,
                       'request_sha256': file_hash(ROOT/'request.json'),
                       'original_case_path': str(source_path),
                       'original_case_sha256': file_hash(source_path),
                       'bridge_applied': snapshot is not None,
                       'final_program': adapted, 'final_scores': scores,
                       'scoring_cost': cost, 'run_dir': str(path.parent)}
            write_json(path, revised)
            if status != 'COMPLETE':
                raise RuntimeError(f'Interface replay scoring error: {arm}/{rid}')
    write_json(done, {'status': 'COMPLETE', 'shard': shard, 'arm': arm,
                      'request_sha256': file_hash(ROOT/'request.json'),
                      'roots': [row['root_id'] for row in rows]})


def report():
    from tqdm import tqdm

    check()
    if (ROOT/'result.json').exists():
        raise FileExistsError(f'TACO interface replay result exists: {ROOT/"result.json"}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    difficulties = {f'taco_test_{r["id"]:04d}': r['difficulty']
                    for r in read_jsonl(transfer.DATA/'selected_records.jsonl')}
    for shard, arm in enumerate(ARMS):
        done = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (done['status'] != 'COMPLETE' or done['shard'] != shard
                or done['arm'] != arm or done['roots'] != [r['root_id'] for r in rows]
                or done['request_sha256'] != file_hash(ROOT/'request.json')):
            raise ValueError(f'TACO interface replay shard incomplete: {arm}')
    totals = {arm: {state: Counter() for state in ('original', 'bridged')}
              for arm in ARMS}
    by_difficulty = {d: {arm: {state: Counter() for state in ('original', 'bridged')}
                         for arm in ARMS} for d in transfer.QUOTAS}
    paired = {arm: Counter() for arm in ARMS}
    paired_atomic = {arm: Counter() for arm in ARMS[1:]}
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='TACO接口复评汇总', unit='题',
                                     dynamic_ncols=True)):
        rid = row['root_id']
        seed = request['seed_base']+100*index
        difficulty = difficulties[rid]
        by_root[rid] = {'difficulty': difficulty}
        for arm in ARMS:
            original_path = _original_path(arm, rid, seed)
            original = json.loads(original_path.read_text())
            revised = json.loads(_replay_path(arm, rid, seed).read_text())
            if (revised['status'] != 'COMPLETE' or revised['root_id'] != rid
                    or revised['arm'] != arm or revised['seed'] != seed
                    or revised['original_case_sha256'] != file_hash(original_path)
                    or revised['request_sha256'] != file_hash(ROOT/'request.json')):
                raise ValueError(f'TACO interface replay differs: {arm}/{rid}')
            signals = {'original': _scores(original), 'bridged': _scores(revised)}
            by_root[rid][arm] = signals
            for state, signal in signals.items():
                for counter in (totals[arm][state], by_difficulty[difficulty][arm][state]):
                    counter['roots'] += 1
                    counter.update(signal)
            a, b = signals['original']['strict'], signals['bridged']['strict']
            paired[arm]['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
            if arm in paired_atomic:
                atomic = json.loads((transfer.ROOT/'evaluation'/arm/rid/
                                     f'seed_{seed}'/'case_result.json').read_text())
                first = all(atomic['first_scores'][s]['status'] == 'PASS'
                            for s in ('visible', 'private'))
                paired_atomic[arm]['win' if b and not first else
                                   'loss' if first and not b else 'tie'] += 1
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'roots': len(rows),
              'summary': {arm: {state: dict(c) for state, c in values.items()}
                          for arm, values in totals.items()},
              'by_difficulty': {d: {arm: {state: dict(c) for state, c in values.items()}
                                    for arm, values in arms.items()}
                                for d, arms in by_difficulty.items()},
              'paired_bridged_vs_original': {arm: dict(c) for arm, c in paired.items()},
              'paired_bridged_vs_ATOMIC_first': {arm: dict(c) for arm, c in paired_atomic.items()},
              'by_root': by_root}
    write_json(ROOT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'by_root'},
                     ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'replay', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'replay':
        replay(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
