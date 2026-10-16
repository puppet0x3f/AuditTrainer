"""Additional APPS diagnostic inputs for the remaining original-test successes."""

from itertools import permutations

from .apps_next import PROBE_INPUTS as EARLIER_INPUTS, oracle as earlier_oracle, valid_output as earlier_valid


EXTRA_INPUTS = {
    'apps_train_0517': ['1 97\n', '4 1000\n', '12 97\n'],
    'apps_train_0572': [
        '4\n1 100 2\n8 3 10\n5 5 1\n40 41 1\n',
        '3\n100 1 50\n3 10 2\n7 4 3\n',
        '2\n20 20 100\n25 5 4\n',
    ],
    'apps_train_0998': [
        '3 3\nRowAdd 1 5\nRowAdd 1 2\nColAdd 3 4\n',
        '2 2\nColAdd 1 3\nColAdd 2 7\n',
        '1 2\nRowAdd 1 3\nColAdd 1 6\n',
    ],
    'apps_train_1011': [
        '4\n3 0\nabc\n3 0\nABC\n4 2\naBcD\n4 1\naBcD\n',
        '3\n1 1\nz\n5 2\nAbCdE\n5 1\nabcDE\n',
        '2\n6 3\nAbCdEf\n6 2\nabcdef\n',
    ],
    'apps_train_1151': [
        '2\n4 2\n0 1\n2 3\n5 1\n0 4\n',
        '1\n6 5\n0 1\n1 2\n2 3\n3 4\n4 5\n',
        '2\n3 1\n0 1\n4 3\n0 1\n1 2\n2 3\n',
    ],
    'apps_train_1533': [
        '4 4 3\n1 2 3\n1 4 9\n4 2 9\n4 3 9\n2 3 1\n',
        '3 2 2\n1 3\n1 2 4\n2 3 5\n',
        '5 4 3\n1 3 4\n1 2 10\n2 3 1\n3 4 1\n4 5 1\n',
    ],
    'apps_train_2597': ['1\n', '8\n', '17\n'],
    'apps_train_2604': ['ABC 2\n', 'ZXY 1\n', 'WXYZ 3\n'],
}


def oracle(root_id, text):
    if root_id in EARLIER_INPUTS:
        return earlier_oracle(root_id, text)
    tokens = iter(text.split())
    if root_id == 'apps_train_0517':
        n, modulus = int(next(tokens)), int(next(tokens))
        primitive = [0] * (n + 1)
        for length in range(1, n + 1):
            primitive[length] = 2**length - sum(primitive[d] for d in range(1, length) if length % d == 0)
        return str(primitive[n] % modulus)
    if root_id == 'apps_train_0572':
        return '\n'.join(str(max(0, abs(int(next(tokens)) - int(next(tokens))) - int(next(tokens))))
                         for _ in range(int(next(tokens))))
    if root_id == 'apps_train_0998':
        n, q = int(next(tokens)), int(next(tokens))
        rows, cols = [0] * n, [0] * n
        for _ in range(q):
            kind, index, value = next(tokens), int(next(tokens))-1, int(next(tokens))
            (rows if kind == 'RowAdd' else cols)[index] += value
        return str(max(rows) + max(cols))
    if root_id == 'apps_train_1011':
        out = []
        for _ in range(int(next(tokens))):
            n, k, word = int(next(tokens)), int(next(tokens)), next(tokens)
            assert len(word) == n
            chef = sum(char.isupper() for char in word) <= k
            brother = sum(char.islower() for char in word) <= k
            out.append('both' if chef and brother else 'chef' if chef else 'brother' if brother else 'none')
        return '\n'.join(out)
    if root_id == 'apps_train_1151':
        out = []
        for _ in range(int(next(tokens))):
            n, m = int(next(tokens)), int(next(tokens))
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            for _ in range(m):
                a, b = int(next(tokens)), int(next(tokens))
                parent[find(a)] = find(b)
            out.append(str(len({find(i) for i in range(n)})))
        return '\n'.join(out)
    if root_id == 'apps_train_1533':
        n, m, k = int(next(tokens)), int(next(tokens)), int(next(tokens))
        special = [int(next(tokens)) for _ in range(k)]
        inf = 10**20
        dist = [[0 if i == j else inf for j in range(n)] for i in range(n)]
        for _ in range(m):
            a, b, weight = int(next(tokens))-1, int(next(tokens))-1, int(next(tokens))
            dist[a][b] = dist[b][a] = weight
        for via in range(n):
            for src in range(n):
                for dst in range(n):
                    dist[src][dst] = min(dist[src][dst], dist[src][via]+dist[via][dst])
        return str(min(dist[a-1][b-1] for i, a in enumerate(special) for b in special[i+1:]))
    if root_id == 'apps_train_2597':
        n = int(next(tokens)); width = len(f'{n:b}')
        return '\n'.join(' '.join(f'{value:>{width}}' for value in (str(i), f'{i:o}', f'{i:X}', f'{i:b}'))
                         for i in range(1, n+1))
    if root_id == 'apps_train_2604':
        letters, k = next(tokens), int(next(tokens))
        return '\n'.join(''.join(value) for value in permutations(sorted(letters), k))
    raise ValueError(f'Unknown APPS probe root: {root_id}')


def valid_output(root_id, text, value, expected):
    if root_id in EARLIER_INPUTS:
        return earlier_valid(root_id, text, value, expected)
    if not isinstance(value, str):
        return False
    if root_id == 'apps_train_2597':
        return value.rstrip('\n') == expected.rstrip('\n')
    return value.strip() == expected.strip()
