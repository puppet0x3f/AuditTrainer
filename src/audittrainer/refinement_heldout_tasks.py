"""Predeclared, unseen compositions of known A4 component algorithms.

Construction uses reference/oracle checks only, never model success rates.
"""
from copy import deepcopy
import json
from pathlib import Path

from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .refinement_curriculum import transfer_definition
from .training_tasks import FAMILIES
from .verifier import canonical

# Six fresh inputs per preparation algorithm: two public and four private.
INPUTS = {
    'graph': [
        [['oak', 'pi'], ['oak', 'elm'], ['elm', 'ash'], ['oak', 'pi']],
        [['willow', 'willow'], ['fir', 'oak'], ['fir', 'birch']],
        [['x', 'y'], ['y', 'x'], ['x', 'x']],
        [['MAPLE', 'maple'], ['maple', 'ash'], ['MAPLE', 'ash']],
        [['spruce', 'spruce']],
        [['ant', 'beech'], ['ant', 'cedar'], ['ant', 'dogwood'], ['ant', 'elm'], ['beech', 'elm']],
    ],
    'calendar': [
        ['07:03', '07:01', '07:02', '07:03', '08:10'],
        ['11:11', '11:12', '11:13', '11:14', '12:01', '12:02', '12:03'],
        ['03:04', '03:04'],
        ['00:02', '00:01', '00:00', '23:57', '23:58', '23:59'],
        ['06:00', '06:02', '06:04', '06:06'],
        ['18:04', '18:00', '18:03', '18:01', '18:02', '18:05', '19:11'],
    ],
    'intervals': [
        [[-8, -3], [-5, 0], [7, 10]], [[-29, -28], [28, 30]],
        [[11, 11], [-12, -12]], [[-30, -26], [-26, -23], [21, 25]],
        [[-3, -2], [0, 1], [3, 4], [6, 7]], [[9, 14], [11, 17], [17, 20], [24, 26]],
    ],
    'words': [
        [' Elm ', 'OAK', 'elm', ' a ', 'Beech', 'cedar', ' FIR '],
        ['Harbor', 'hare', 'HARP', 'hay', 'HARE', '  '],
        [' ', '    ', ''], [' I ', 'j', 'I', 'K', 'j'],
        [' north star ', 'NORTH STAR', 'South', 'east', 'WEST'],
        ['willow', 'birch', 'ash', 'pine', 'maple', 'elm', 'fir', 'oak'],
    ],
    'paths': [
        ['/red//oak', './blue/./fir/', 'red/oak/tall', '///', 'green'],
        ['a/b/c/d', 'a/b/e', 'a//b/./c', 'z/y', './q', 'r', 's/t'],
        ['///./', '././', ''], ['//iris', 'iris/', './iris', 'iris/./leaf'],
        ['n/e/s/w/up/down', 'n/e/s', 'n/e', 'n', '/'],
        ['k/l', 'm/n', 'o/p', 'q/r', 's/t', 'u/v', 'w/x', 'y/z'],
    ],
    'polynomial': [
        [[4, 3], [0, -7], [2, 5], [4, -1], [1, 2]],
        [[6, 2], [5, -3], [4, 4], [3, -5], [2, 6], [1, -7], [0, 8]],
        [[2, 9], [2, -9], [0, 0]], [[5, -6], [0, 11]],
        [[6, 0], [3, 8], [1, 4], [3, -8], [1, -1]],
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [2, -1]],
    ],
    'matrix': [
        [[2, 7, -3, 4], [5, -2, 9, 1]],
        [[1, 2, 3, 4, 5, 6, 7], [-1, 3, -3, 2, -5, 1, -7]],
        [[], [], []], [[-8], [3], [-2]],
        [[9, -4, 0], [-9, 4, 0]], [[6, 1, -2, 8, 3], [2, 5, 2, -1, 4], [-3, 7, 0, 1, -2]],
    ],
    'chunks': [
        [['m', 2], ['N', 4], [' ', 1], ['m', 1]], [['r', 5], ['s', 2], ['t', 4]],
        [['?', 0], ['!', 0]], [[' ', 3], ['A', 2], ['a', 2], [' ', 1]],
        [['/', 2], ['.', 1], ['/', 1]], [['z', 10], ['x', 10], ['z', 3], ['y', 1]],
    ],
}

TRANSFERS = [
    ('graph', 'words', 3), ('calendar', 'intervals', 3), ('intervals', 'calendar', 3),
    ('words', 'chunks', 3), ('paths', 'chunks', 3), ('polynomial', 'chunks', 3),
    ('matrix', 'chunks', 3), ('intervals', 'chunks', 3), ('calendar', 'chunks', 3),
    ('words', 'paths', 3), ('chunks', 'words', 1), ('chunks', 'paths', 3),
]


def definition(first, second, p):
    goal = None
    if second == 'chunks':
        goal = (f'Split the prepared list into consecutive sublists of width {p}, retaining a shorter last sublist. '
                'Preserve each element exactly, including nested lists. Return chunks and length of the original prepared list.')
    elif (first, second) == ('graph', 'words'):
        goal = (f'Treat the node names (dictionary keys) as words, preserving case and whitespace. Keep distinct names of length >= {p}, '
                'sorted by length then lexicographically. Return words and count. Degree values do not affect this summary.')
    elif (first, second) == ('words', 'paths'):
        goal = (f'Treat each normalized word as a path whose individual characters, including internal spaces, are its components. '
                f"Take at most its first {p} characters and join them with '/'. Count prefixes and return sorted prefixes and parallel counts.")
    elif (first, second) == ('chunks', 'words'):
        goal = ('Treat every character of the expanded string, including spaces and punctuation, as a one-character word. '
                'Keep distinct characters of length >= 1, sorted lexicographically with case preserved. Return words and count.')
    elif (first, second) == ('chunks', 'paths'):
        goal = ('Treat each character of the expanded string, including spaces and punctuation, as a path with one component. '
                "The prefix of each such path is that same character. Count prefixes and return sorted prefixes and parallel counts.")
    return transfer_definition(first, second, p, split='VALIDATION', input_cases=INPUTS[first], aggregate_goal=goal)


def build(directory):
    directory = Path(directory)
    prior = {r['family_id'] for path in Path('data').rglob('roots.jsonl')
             if directory.resolve() not in path.resolve().parents for r in read_jsonl(path)}
    rows, tests = [], []
    directory.mkdir(parents=True, exist_ok=True)
    checks = 0
    for first, second, p in TRANSFERS:
        d, private, reference = definition(first, second, p)
        task = d['task']
        if task['family_id'] in prior or first == second:
            raise ValueError('Held-out composition was already in a project task manifest')
        if any(items in FAMILIES[first]['cases'] for items in INPUTS[first]):
            raise ValueError('Expected fresh held-out input cases')
        namespace = {}
        exec(reference['prepare']+'\n'+reference['aggregate'], namespace)
        for suite in d['suites']+private['suites']:
            key = suite.get('contract_key', 'root')
            for case in suite['cases']:
                x = deepcopy(case['args'][0])
                if key == 'root':
                    actual = json.dumps(namespace['aggregate'](namespace['prepare'](json.loads(x))), separators=(',', ':'))
                elif key == 'compute':
                    actual = namespace['aggregate'](namespace['prepare'](x))
                else:
                    actual = namespace[key](x)
                if actual != case['expected']:
                    raise ValueError(f'Reference/oracle disagreement: {task["root_id"]} {case["case_id"]}')
                checks += 1
        folder = directory/task['root_id']; folder.mkdir(exist_ok=True)
        for name, value in [('task.json', d), ('private_tests.json', private), ('reference.json', reference)]:
            path = folder/name
            if path.exists() and json.loads(path.read_text()) != value:
                raise ValueError(f'Changed frozen held-out task: {path}')
            write_json(path, value)
        rows.append({**{k: task[k] for k in ('root_id', 'family_id', 'split')},
            'source': 'b3_unseen_component_compositions_v1', 'original_split': 'VALIDATION', 'io_type': 'solve_text',
            'fixture_path': str(folder/'task.json'), 'source_hash': file_hash(folder/'task.json'),
            'reference_path': str(folder/'reference.json'), 'reference_hash': file_hash(folder/'reference.json')})
        tests.append({'root_id': task['root_id'], 'path': str(folder/'private_tests.json'), 'sha256': file_hash(folder/'private_tests.json')})
    for name, values in [('roots.jsonl', rows), ('tests.jsonl', tests)]:
        content = ''.join(canonical(r)+'\n' for r in values); path = directory/name
        if path.exists() and path.read_text() != content:
            raise ValueError('Changed held-out manifest')
        path.write_text(content)
    return {'roots': len(rows), 'reference_oracle_checks': checks, 'prior_project_families': len(prior),
            'novel_compositions': [list(t[:2]) for t in TRANSFERS], 'model_calibration_rollouts': 0}
