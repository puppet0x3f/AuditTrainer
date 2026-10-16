"""CPU-only diagnostic replay of saved DIRECT code with a uniform stdin bridge."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from . import apps_direct_rejudge as rejudge, apps_diverse_sft as diverse
from . import apps_taco_sft as taco, apps_taco_transfer as transfer
from .apps_passk_select import _score
from .artifacts import ArtifactStore
from .dataset import file_hash
from .evaluate import load_task, score_program, write_json
from .sandbox import ExecutionLimits, Sandbox
from .schema import Contract


BRIDGE = '''
def solve(text):
    _lines = iter(text.splitlines())
    _output = []
    def _input(_prompt=None):
        try:
            return next(_lines)
        except StopIteration:
            raise EOFError('input exhausted')
    def _print(*values, sep=' ', end='\\n', file=None, flush=False):
        _output.append((' ' if sep is None else sep).join(map(str, values))
                       + ('\\n' if end is None else end))
    _namespace = _original_solve.__globals__
    _missing = object()
    _before_input = _namespace.get('input', _missing)
    _before_print = _namespace.get('print', _missing)
    _namespace['input'] = _input
    _namespace['print'] = _print
    try:
        _answer = _original_solve(text)
    finally:
        for _name, _before in (('input', _before_input), ('print', _before_print)):
            if _before is _missing:
                _namespace.pop(_name, None)
            else:
                _namespace[_name] = _before
    return _answer if isinstance(_answer, str) else ''.join(_output)
'''


def needs_bridge(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'solve']
    if len(functions) != 1:
        return False
    solve = functions[0]
    uses_input = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'input' for node in ast.walk(solve))
    binds_input_locally = any((isinstance(node, ast.FunctionDef) and node.name == 'input')
                              or (isinstance(node, ast.Name) and node.id == 'input'
                                  and isinstance(node.ctx, ast.Store)) for node in ast.walk(solve))
    return uses_input and not binds_input_locally


def bridge_source(source):
    if not needs_bridge(source):
        raise ValueError('Source has no unbound input() to diagnose')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'solve':
            node.name = '_original_solve'
            break

    class RenameRecursiveCalls(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'solve' and isinstance(node.ctx, ast.Load):
                node.id = '_original_solve'
            return node

    tree = RenameRecursiveCalls().visit(tree)
    ast.fix_missing_locations(tree)
    output = ast.unparse(tree) + '\n' + BRIDGE
    compile(output, '<direct_stdin_bridge>', 'exec')
    return output


def _old_signals(parent):
    return {rid: signals['fresh_signals'][rid]['DIRECT']
            for i in range(3)
            for signals in [json.loads((parent/f'shard_{i}.json').read_text())]
            for rid in signals['fresh_signals']}


def replay():
    from tqdm import tqdm

    transfer.check()
    folder = transfer.ROOT/'interface'
    result_path = folder/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['status'] == 'COMPLETE' and result['request_sha256'] == file_hash(transfer.ROOT/'request.json'):
            print(json.dumps({'status': 'ALREADY_COMPLETE', 'summary': result['summary']},
                             ensure_ascii=False), flush=True)
            return
        raise ValueError('Prior direct-interface replay differs from this request')
    rows = rejudge._rows()
    tests = rejudge.fresh._tests()
    framework = json.loads((diverse.ROOT/'request.json').read_text())['framework']
    sandbox = Sandbox(ExecutionLimits(**framework['execution']))
    saved = {'DIVERSE': _old_signals(diverse.ROOT),
             'TACO': _old_signals(taco.ROOT)}
    totals = {arm: {state: {str(k): Counter() for k in (1, 3)}
                    for state in ('original', 'bridged')}
              for arm in saved}
    applied, recovered = Counter(), Counter()
    by_root = {}
    with tqdm(total=len(rows)*3*len(saved), desc='DIRECT标准输入适配复评',
              unit='候选', dynamic_ncols=True) as bar:
        for row in rows:
            rid = row['root_id']
            spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
            sample = rejudge._sample(spec)
            by_root[rid] = {}
            for arm, all_signals in saved.items():
                originals = all_signals[rid]
                updated = []
                for signal in originals:
                    case = json.loads(Path(signal['case_path']).read_text())
                    snapshot = case['final_program']
                    if snapshot is None:
                        updated.append(signal)
                        bar.update(1)
                        continue
                    source = (Path(snapshot['path'])/
                              snapshot['artifact']['module_path']).read_text()
                    if not needs_bridge(source):
                        updated.append(signal)
                        bar.update(1)
                        continue
                    applied[arm] += 1
                    output = folder/arm/rid/f'seed_{case["seed"]}'
                    diagnostic = output/'case_result.json'
                    if diagnostic.exists():
                        revised = json.loads(diagnostic.read_text())
                        if revised['original_case_path'] != signal['case_path']:
                            raise ValueError(f'Diagnostic case changed: {diagnostic}')
                    else:
                        if output.exists():
                            raise ValueError(f'Incomplete diagnostic case: {output}')
                        output.mkdir(parents=True)
                        _, _, contract_spec, _, verifier = load_task(row, tests, framework)
                        contract = Contract(**contract_spec.model_dump(),
                                            contract_id='C_N0_1', version=1)
                        store = ArtifactStore(output/'final_program')
                        artifact = store.save_initial('N0', contract, bridge_source(source))
                        patched = {'path': str(output/'final_program'), 'root_id': rid,
                                   'artifact': artifact.model_dump(mode='json'),
                                   'contract': contract.model_dump(mode='json')}
                        scores, cost = score_program(patched, verifier, rid)
                        revised = {**case, 'final_program': patched,
                                   'final_scores': scores, 'scoring_cost': cost,
                                   'run_dir': str(output),
                                   'original_case_path': signal['case_path'],
                                   'diagnostic_bridge': 'capture input() and print() around saved solve(text)'}
                        write_json(diagnostic, revised)
                    new_signal = _score(diagnostic, rid, sample, sandbox)
                    updated.append(new_signal)
                    if (case['final_scores']['private']['status'] != 'PASS'
                            and revised['final_scores']['private']['status'] == 'PASS'):
                        recovered[arm] += 1
                    bar.update(1)
                by_root[rid][arm] = {'bridge_applied': sum(
                    x['case_path'] != y['case_path'] for x, y in zip(originals, updated))}
                for state, signals in (('original', originals), ('bridged', updated)):
                    counts, selected = diverse._counts(signals)
                    by_root[rid][arm][state] = {'k3_selected_strict': int(selected),
                                                 'counts': counts}
                    for k in (1, 3):
                        totals[arm][state][str(k)]['roots'] += 1
                        totals[arm][state][str(k)].update(counts[str(k)])
    summary = {'arms': {arm: {state: {k: dict(value) for k, value in ks.items()}
                              for state, ks in states.items()}
                        for arm, states in totals.items()},
               'bridge_applied_candidates': dict(applied),
               'private_pass_candidates_recovered': dict(recovered)}
    write_json(result_path, {'status': 'COMPLETE',
                             'request_sha256': file_hash(transfer.ROOT/'request.json'),
                             'candidate_count_per_arm': len(rows)*3,
                             'summary': summary, 'by_root': by_root})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('replay',))
    args = parser.parse_args(argv)
    if args.command == 'replay':
        replay()


if __name__ == '__main__':
    main()
