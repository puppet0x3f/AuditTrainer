"""A4 authored TRAIN catalog: 50 specifications in ten explicitly shared families.

Oracle functions below are independent implementations of the reference sources.
This is a small supervised pilot, not fifty independent benchmark families.
"""

from collections import Counter, defaultdict
import itertools
import json
import textwrap


FAMILIES = {
    "ledger": {
        "input": "a list of [account string, signed integer cents] entries",
        "prepare": "Sum cents for each account, returning an account-to-balance dict.",
        "aggregate": "Return accounts with balance <= -{p}, sorted by name, and their total positive debt, as keys accounts and debt.",
        "keys": ["accounts", "debt"],
        "a": "return {name: sum(v for key, v in items if key == name) for name in sorted({r[0] for r in items})}",
        "b": "names = sorted(k for k, v in prepared.items() if v <= -{p})\nreturn {{'accounts': names, 'debt': -sum(prepared[k] for k in names)}}",
        "cases": [[], [["a", -9], ["b", 3], ["a", 2], ["c", -2]], [["z", 0]],
                  [["a", -1], ["b", -5], ["a", 1]], [["x", 9], ["x", -19], ["q", -4]], [["a", 2], ["b", 3]]],
    },
    "intervals": {
        "input": "a list of half-open integer intervals [start, end], with -30 <= start <= end <= 30",
        "prepare": "Return the sorted distinct occupied integer slots of the intervals.",
        "aggregate": "Find maximal consecutive runs of slots. Keep runs of length >= {p}; return half-open segments and their covered slot count as keys segments and covered.",
        "keys": ["segments", "covered"],
        "a": "return sorted({t for start, end in items for t in range(start, end)})",
        "b": "segments = []\nfor value in prepared:\n    if segments and segments[-1][1] == value:\n        segments[-1][1] += 1\n    else:\n        segments.append([value, value + 1])\nkept = [s for s in segments if s[1] - s[0] >= {p}]\nreturn {{'segments': kept, 'covered': sum(b-a for a,b in kept)}}",
        "cases": [[], [[0, 3], [2, 8], [10, 12]], [[2, 2]], [[-5, -1], [-1, 1], [4, 5]], [[0, 1], [2, 3], [4, 5]], [[-9, 9], [0, 3]]],
    },
    "graph": {
        "input": "a list of undirected edges [node string, node string]; duplicates and self loops are allowed",
        "prepare": "Return a dict mapping every mentioned node to its number of distinct other neighbors; ignore self loops but retain their nodes.",
        "aggregate": "Return sorted nodes of degree >= {p} and the sum of their degrees as keys nodes and degree_sum.",
        "keys": ["nodes", "degree_sum"],
        "a": "neighbors = {}\nfor a, b in items:\n    neighbors.setdefault(a, set())\n    neighbors.setdefault(b, set())\n    if a != b:\n        neighbors[a].add(b)\n        neighbors[b].add(a)\nreturn {k: len(v) for k, v in neighbors.items()}",
        "b": "chosen = sorted(n for n, degree in prepared.items() if degree >= {p})\nreturn {{'nodes': chosen, 'degree_sum': sum(prepared[n] for n in chosen)}}",
        "cases": [[], [["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"], ["a", "f"], ["b", "c"]], [["x", "x"]],
                  [["a", "b"], ["b", "a"], ["b", "b"]], [["x", "y"], ["z", "w"]], [["k", "l"], ["l", "m"], ["m", "k"]]],
    },
    "inventory": {
        "input": "a list of [SKU string, stock integer, reserved integer] records, with nonnegative stock and reserved",
        "prepare": "Return the total stock minus reserved for each SKU as a dict.",
        "aggregate": "Keep SKUs with available units >= {p}. Sort by decreasing units then SKU; return parallel lists order and units.",
        "keys": ["order", "units"],
        "a": "totals = {}\nfor sku, stock, reserved in items:\n    totals[sku] = totals.get(sku, 0) + stock - reserved\nreturn totals",
        "b": "pairs = sorted(((k,v) for k,v in prepared.items() if v >= {p}), key=lambda row: (-row[1], row[0]))\nreturn {{'order': [r[0] for r in pairs], 'units': [r[1] for r in pairs]}}",
        "cases": [[], [["b", 8, 1], ["a", 7, 0], ["c", 2, 1]], [["a", 1, 9]],
                  [["x", 3, 0], ["x", 5, 2], ["y", 3, 0]], [["a", 0, 0]], [["z", 20, 3], ["q", 10, 4], ["z", 2, 1]]],
    },
    "words": {
        "input": "a list of ASCII strings containing letters and spaces only",
        "prepare": "Strip surrounding whitespace, lowercase, and discard empty strings; retain duplicates and order.",
        "aggregate": "Keep distinct words of length >= {p}, sorted by length then lexicographically. Return words and count.",
        "keys": ["words", "count"],
        "a": "return [word.strip().lower() for word in items if word.strip()]",
        "b": "words = sorted({{w for w in prepared if len(w) >= {p}}}, key=lambda w: (len(w), w))\nreturn {{'words': words, 'count': len(words)}}",
        "cases": [[], [" Cat ", "a", "STONE", "cat", "", "  ", "apple"], ["A", "a"],
                  [" zebra", "pie", "ALPHA ", "b"], ["   ", ""], ["hello", "WORLD", "x y", "hello"]],
    },
    "chunks": {
        "input": "a list of [one ASCII character, repetition count] pairs, with integer counts between 0 and 10",
        "prepare": "Expand repetitions into a single string in input order.",
        "aggregate": "Split the string into consecutive chunks of width {p}, retaining a shorter last chunk. Return chunks and length of the original string.",
        "keys": ["chunks", "length"],
        "a": "return ''.join(symbol * count for symbol, count in items)",
        "b": "return {{'chunks': [prepared[i:i+{p}] for i in range(0, len(prepared), {p})], 'length': len(prepared)}}",
        "cases": [[], [["a", 3], ["b", 4], ["x", 0]], [["z", 1]], [["x", 0], ["y", 0]], [["A", 5], [" ", 2], ["B", 1]], [["q", 10], ["r", 3]]],
    },
    "matrix": {
        "input": "a rectangular list of integer rows; [] and rows of width zero are valid",
        "prepare": "Return the sum of each column, in column order; return [] for no columns.",
        "aggregate": "Select columns whose sums are divisible by {p} (Python modulo). Return zero-based indices and the sum of selected column sums as total.",
        "keys": ["indices", "total"],
        "a": "if not items:\n    return []\nreturn [sum(row[j] for row in items) for j in range(len(items[0]))]",
        "b": "indices = [i for i, value in enumerate(prepared) if value % {p} == 0]\nreturn {{'indices': indices, 'total': sum(prepared[i] for i in indices)}}",
        "cases": [[], [[1, -2, 4], [4, 2, 8]], [[], []], [[0, 3, -5]], [[-2, -3], [-4, -6]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]],
    },
    "calendar": {
        "input": "a list of valid 24-hour HH:MM strings on one day",
        "prepare": "Convert times to integer minutes after midnight, remove duplicates, and sort ascending.",
        "aggregate": "Consider adjacent times only; return breaks with gaps >= {p15} minutes as [start minute,end minute] pairs and max_gap across all adjacent times, or 0 with fewer than two times.",
        "keys": ["breaks", "max_gap"],
        "a": "return sorted({int(t[:2]) * 60 + int(t[3:]) for t in items})",
        "b": "pairs = list(zip(prepared, prepared[1:]))\nreturn {{'breaks': [[a,b] for a,b in pairs if b-a >= {p15}], 'max_gap': max([b-a for a,b in pairs] or [0])}}",
        "cases": [[], ["10:00", "09:15", "09:00", "10:00", "12:30"], ["00:00"], ["23:59", "00:01"], ["01:00", "01:15", "01:30"], ["02:00", "04:00", "03:10"]],
    },
    "polynomial": {
        "input": "a list of [degree, coefficient] integer pairs, with degrees from 0 to 6",
        "prepare": "Combine equal degrees, remove zero coefficients, and return [degree, coefficient] pairs sorted by degree ascending.",
        "aggregate": "Evaluate the normalized polynomial at x={p}; return value and highest nonzero degree, or degree=-1 for the zero polynomial.",
        "keys": ["value", "degree"],
        "a": "totals = {}\nfor degree, coefficient in items:\n    totals[degree] = totals.get(degree, 0) + coefficient\nreturn [[d, totals[d]] for d in sorted(totals) if totals[d] != 0]",
        "b": "return {{'value': sum(c * {p} ** d for d, c in prepared), 'degree': prepared[-1][0] if prepared else -1}}",
        "cases": [[], [[0, 2], [2, 3], [1, -1], [2, -1]], [[3, 4], [3, -4]], [[0, -5]], [[6, 1], [1, 2], [0, 0]], [[2, -2], [5, 3], [2, 1]]],
    },
    "paths": {
        "input": "a list of slash-separated ASCII paths; repeated slashes and '.' components are allowed, '..' is excluded",
        "prepare": "Split each path on '/', discarding empty and '.' components; retain empty paths and order.",
        "aggregate": "Take at most the first {p} components of each normalized path and join with '/'; empty paths yield ''. Count prefixes and return sorted prefixes and parallel counts.",
        "keys": ["prefixes", "counts"],
        "a": "return [[part for part in path.split('/') if part not in ('', '.')] for path in items]",
        "b": "counts = {{}}\nfor parts in prepared:\n    prefix = '/'.join(parts[:{p}])\n    counts[prefix] = counts.get(prefix, 0) + 1\nkeys = sorted(counts)\nreturn {{'prefixes': keys, 'counts': [counts[k] for k in keys]}}",
        "cases": [[], ["/a/b/c/d/e/f", "a/b/z", "/a//b/./c", "", "/"], ["./x"], ["a", "a/", "./a"], ["m/n/o/p/q", "m/n/r/s/t", "z"], [".", "//./", "/a/.//b/"]],
    },
}


def oracle_a(family, items):
    if family in {"ledger", "inventory"}:
        result = defaultdict(int)
        for row in items:
            result[row[0]] += row[1] if family == "ledger" else row[1] - row[2]
        return dict(result)
    if family == "intervals":
        return [t for t in range(-30, 30) if any(a <= t < b for a, b in items)]
    if family == "graph":
        nodes = sorted(set(itertools.chain.from_iterable(items)))
        return {n: sum(any((a == n and b == other) or (b == n and a == other) for a, b in items)
                       for other in nodes if other != n) for n in nodes}
    if family == "words":
        result = []
        for word in items:
            normalized = word.strip().lower()
            if normalized:
                result.append(normalized)
        return result
    if family == "chunks":
        result = []
        for char, count in items:
            for _ in range(count):
                result.append(char)
        return ''.join(result)
    if family == "matrix":
        return list(map(sum, zip(*items)))
    if family == "calendar":
        values = []
        for stamp in items:
            h, m = map(int, stamp.split(':'))
            if h * 60 + m not in values:
                values.append(h * 60 + m)
        return sorted(values)
    if family == "polynomial":
        return [[d, sum(c for degree, c in items if degree == d)] for d in range(7)
                if sum(c for degree, c in items if degree == d)]
    if family == "paths":
        result = []
        for path in items:
            parts = []
            for part in path.split('/'):
                if part and part != '.':
                    parts.append(part)
            result.append(parts)
        return result
    raise ValueError(family)


def oracle_b(family, prepared, p):
    if family == "ledger":
        rows = sorted((name, -balance) for name, balance in prepared.items() if -balance >= p)
        return {'accounts': [r[0] for r in rows], 'debt': sum(r[1] for r in rows)}
    if family == "intervals":
        runs = [list(map(lambda r: r[1], group)) for _, group in
                itertools.groupby(enumerate(prepared), key=lambda r: r[1] - r[0])]
        kept = [r for r in runs if len(r) >= p]
        return {'segments': [[r[0], r[-1]+1] for r in kept], 'covered': sum(map(len, kept))}
    if family == "graph":
        nodes, total = [], 0
        for node in sorted(prepared):
            if prepared[node] >= p:
                nodes.append(node)
                total += prepared[node]
        return {'nodes': nodes, 'degree_sum': total}
    if family == "inventory":
        names = [k for k in sorted(prepared) if prepared[k] >= p]
        names.sort(key=lambda k: prepared[k], reverse=True)
        return {'order': names, 'units': [prepared[k] for k in names]}
    if family == "words":
        values = []
        for word in sorted(prepared):
            if len(word) >= p and word not in values:
                values.append(word)
        values.sort(key=len)
        return {'words': values, 'count': len(values)}
    if family == "chunks":
        chunks = []
        rest = prepared
        while rest:
            chunks.append(rest[:p])
            rest = rest[p:]
        return {'chunks': chunks, 'length': len(prepared)}
    if family == "matrix":
        pairs = [(i, value) for i, value in enumerate(prepared) if value // p * p == value]
        return {'indices': [i for i, _ in pairs], 'total': sum(value for _, value in pairs)}
    if family == "calendar":
        # Composed tasks can supply decreasing values (e.g. matrix column sums).
        # The contract says max adjacent difference, not a clamp at zero.
        breaks = []
        largest = prepared[1] - prepared[0] if len(prepared) >= 2 else 0
        for i in range(1, len(prepared)):
            gap = prepared[i] - prepared[i-1]
            largest = max(gap, largest)
            if gap >= p * 15:
                breaks.append([prepared[i-1], prepared[i]])
        return {'breaks': breaks, 'max_gap': largest}
    if family == "polynomial":
        coeff = dict(prepared)
        degree = max(coeff, default=-1)
        value = 0
        for d in range(degree, -1, -1):
            value = value * p + coeff.get(d, 0)
        return {'value': value, 'degree': degree}
    if family == "paths":
        counts = Counter('/'.join(parts[:p]) for parts in prepared)
        pairs = sorted(counts.items())
        return {'prefixes': [k for k, _ in pairs], 'counts': [v for _, v in pairs]}
    raise ValueError(family)


def function(name, parameter, body):
    return f"def {name}({parameter}):\n" + textwrap.indent(body, '    ') + '\n'


def contract(name, parameter, goal, owns, requires, returns="JSON value"):
    return {'signature': {'name': name, 'parameters': [{'name': parameter, 'type': 'str' if parameter == 'text' else 'JSON value'}],
                          'returns': returns},
            'requires': [requires], 'ensures': [{'id': key.upper(), 'text': value} for key, value in goal.items()],
            'owns': owns, 'uses': [], 'must_not': ['No input mutation or external I/O.'],
            'frame': {'public_symbol': name, 'private_helper_prefix': '_', 'mutate_inputs': False, 'external_io': False}}


def definition(family, p):
    entry = FAMILIES[family]
    goal = entry['aggregate'].format(p=p, p15=p*15)
    root_id = f'train_{family}_{p:02d}'
    root = contract('solve', 'text', {'parse': 'Parse the JSON input.', 'prepare': entry['prepare'],
                    'aggregate': goal, 'format': "Return compact JSON with separators=(',', ':') and no newline."},
                    ['parse', 'prepare', 'aggregate', 'format'], 'Input text encodes ' + entry['input'], 'str')
    a = contract('prepare', 'items', {'prepare': entry['prepare']}, ['prepare'], entry['input'])
    b = contract('aggregate', 'prepared', {'aggregate': goal}, ['aggregate'], 'The exact output of: ' + entry['prepare'])
    combined = contract('compute', 'items', {'prepare': entry['prepare'], 'aggregate': goal},
                        ['prepare', 'aggregate'], entry['input'])
    local = {'prepare': a, 'aggregate': b, 'compute': combined}
    suites, private = [], []
    for visibility, indices in [('VISIBLE', [0, 1]), ('PRIVATE', [2, 3, 4, 5])]:
        for key, spec in [('root', root), *local.items()]:
            cases = []
            for i in indices:
                items = entry['cases'][i]
                prepared = oracle_a(family, items)
                answer = oracle_b(family, prepared, p)
                args = [json.dumps(items)] if key == 'root' else [prepared if key == 'aggregate' else items]
                expected = json.dumps(answer, separators=(',', ':')) if key == 'root' else prepared if key == 'prepare' else answer
                cases.append({'case_id': f'{key}_{i}', 'args': args, 'expected': expected})
            suite = {'suite_id': f'{root_id}_{key}_{visibility.lower()}', 'check_kind': 'ROOT' if key == 'root' else 'LOCAL',
                     'visibility': visibility, 'oracle_origin': 'A4 independent authored oracle v1',
                     'covered_obligations': [o['id'] for o in spec['ensures']], 'cases': cases}
            if key != 'root':
                suite['contract_key'] = key
            (suites if visibility == 'VISIBLE' else private).append(suite)
    task = {'root_id': root_id, 'family_id': f'a4_{family}_v1', 'split': 'TRAIN',
            'spec': f"Input is JSON encoding {entry['input']}. {entry['prepare']} Then: {goal} Return compact JSON without a newline. Implement solve(text: str) -> str.",
            'entrypoint': 'solve', 'visible_suite_ref': f'{root_id}_root_visible', 'private_suite_ref': f'{root_id}_root_private'}
    reference = {'family': family, 'parameter': p, 'keys': entry['keys'],
                 'prepare': function('prepare', 'items', entry['a']),
                 'aggregate': function('aggregate', 'prepared', entry['b'].format(p=p, p15=p*15))}
    return {'task': task, 'contract': root, 'local_contracts': local, 'suites': suites}, {'suites': private}, reference
