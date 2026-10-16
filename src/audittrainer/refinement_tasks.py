"""Finite PR1 task pack, fixed before policy sampling; private cases stay offline."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .training_tasks import contract


SPECS = {
    'fifo': (
        'an object: invoices is a list of [account,id,time,amount] arrays; payments is a list of [account,amount] arrays; void is a list of invoice IDs; now is an integer. Amounts are nonnegative; account and invoice IDs are strings',
        'Discard void invoice IDs. For each remaining ID keep the greatest time, breaking ties by the last input occurrence. Return {"invoices":{account:[[time,id,amount],...]},"payments":{account:summed_amount},"now":original_now}. Sort each invoice list by (time,id); include only accounts with kept invoices in invoices and only accounts appearing in payments in payments. Sum repeated payments.',
        'Within each account allocate its payment to invoices in order, never below zero. An unpaid invoice is overdue when now-time>{p}. Return rows sorted by account as [account,total_unpaid,overdue_unpaid,number_of_unpaid_invoices], including accounts with invoices even if fully paid; ignore payment-only accounts. Also return total unpaid as total.'),
    'timesheets': (
        'a list of [user,start,end,rate] integer half-open work intervals with 0<=start<end and nonnegative rate',
        'For each user partition the union at every start/end boundary. On each covered segment use the maximum rate among active intervals. Return a dict from user to ordered [start,end,rate] disjoint segments; gaps are absent.',
        'Split segments at day boundaries (a day has 10 time units). For each user/day pay only its first {p} covered time units chronologically, retaining the rate of each portion; gaps consume no allowance. Return rows sorted by (user,day) as [user,day,all_covered_units,paid_amount], and total paid_amount as total.'),
    'reconcile': (
        'an object: prices is a list of [sku,time,unit_price,category] arrays; orders is a list of [id,sku,time,quantity] arrays; returns is a list of [order_id,quantity] arrays. Order IDs are unique strings; SKU/category are strings; times, quantities and prices are nonnegative integers',
        'For each order use the latest price for its SKU at or before order time, resolving equal times by last input occurrence; omit orders with no applicable price. Sum all returns for its ID, cap at ordered quantity, and subtract. Return [category,net_quantity,net_quantity*unit_price] for every priced order, retaining zero quantities.',
        'Group rows by category. If category gross is at least {p}, discount is floor(gross*15/100), otherwise zero. Return rows sorted by category as [category,net_quantity,gross,discount,due], with due=gross-discount, and total due as total.'),
    'schedule': (
        'an object: jobs is a list of [id,positive_duration,prerequisite_ids] arrays, where prerequisite_ids is a list of strings; blackout is a list of [start,end] arrays of nonnegative integers with start<end, denoting half-open intervals. Job IDs are unique strings and all prerequisites exist',
        'Perform Kahn topological sorting, always taking the lexicographically smallest currently ready job; duplicate prerequisite IDs count once. If cyclic return {"cycle":true}; otherwise return {"cycle":false,"order":ordered_ids,"jobs":original_jobs,"blackout":original_blackout}.',
        'If cycle return {"cycle":true}. Otherwise visit topological order and schedule on {p} workers initially free at 0. For each worker start at max(worker_free, all prerequisite finish times), repeatedly shift start to the end of any blackout overlapping [start,start+duration) until none overlaps. Choose smallest feasible start then smallest worker index. Record [id,start,finish,worker] sorted by id, plus makespan (0 if empty), in {"rows":...,"makespan":...}.'),
    'reservations': (
        'an object: stock maps SKU strings to nonnegative integers; events is a list of [time,id,kind,sku,quantity] arrays. IDs are strings; kind is reserve, cancel, or ship; time and quantity are nonnegative integers',
        'Sort events by (time,input_position). A reserve for an inactive ID succeeds only if quantity<=available stock; otherwise count a rejection. Accepted reserves reduce stock and store [SKU,quantity]. Cancel/ship closes an active ID, using its stored SKU/quantity (ignoring the event SKU/quantity); cancel returns stock, ship adds shipped units. Inactive cancel/ship count a rejection. Closed IDs may be reused. Return {"available":{sku:stock_left},"active":{id:[sku,quantity]},"shipped":{sku:total_shipped},"rejected":count}. available includes all original SKUs; active only active IDs; shipped only SKUs with an accepted ship (including zero-quantity ships).',
        'Return rows sorted by SKU from original stock as [sku,available,total_active_reserved,shipped] for SKUs with available<{p} or positive reserved/shipped; also return rejected. No event refers to an SKU absent from stock.'),
    'revisions': (
        'a list of [revision_id,parent_id_or_null,position,delete_count,insert_text] with unique IDs; a parent always appears earlier, positions/counts nonnegative',
        'Each revision starts from its parent text, or empty text if parent null. Clamp position to text length, delete up to delete_count characters there, then insert insert_text. Return {"texts":{revision_id:resulting_text},"leaves":sorted_leaf_ids}; texts contains all revisions; leaves contains IDs never used as a parent.',
        'For each leaf text collapse whitespace with split/join, then keep only texts with at least {p} words. Return rows in leaf ID order as [id,normalized_text,word_count], and total word count as total.')}


def prepare(family, items):
    if family == 'fifo':
        last = {}
        for row in items['invoices']:
            if row[1] not in items['void'] and (row[1] not in last or row[2] >= last[row[1]][2]):
                last[row[1]] = row
        invoices, payments = defaultdict(list), defaultdict(int)
        for a, i, t, amount in last.values(): invoices[a].append([t, i, amount])
        for a, amount in items['payments']: payments[a] += amount
        return {'invoices': {a: sorted(v) for a, v in invoices.items()}, 'payments': dict(payments), 'now': items['now']}
    if family == 'timesheets':
        result = {}
        for user in sorted({r[0] for r in items}):
            rows = [r for r in items if r[0] == user]
            cuts = sorted({x for r in rows for x in r[1:3]})
            result[user] = [[a, b, max(r[3] for r in rows if r[1] <= a < r[2])]
                            for a, b in zip(cuts, cuts[1:]) if any(r[1] <= a < r[2] for r in rows)]
        return result
    if family == 'reconcile':
        result = []
        for oid, sku, when, quantity in items['orders']:
            prices = [(r[1], i, r) for i, r in enumerate(items['prices']) if r[0] == sku and r[1] <= when]
            if not prices: continue
            price = max(prices)[2]
            net = max(0, quantity-sum(r[1] for r in items['returns'] if r[0] == oid))
            result.append([price[3], net, net*price[2]])
        return result
    if family == 'schedule':
        pending = {r[0]: set(r[2]) for r in items['jobs']}
        order = []
        while pending:
            ready = sorted(k for k, v in pending.items() if not v)
            if not ready: return {'cycle': True}
            key = ready[0]; order.append(key); del pending[key]
            for v in pending.values(): v.discard(key)
        return {'cycle': False, 'order': order, 'jobs': deepcopy(items['jobs']), 'blackout': deepcopy(items['blackout'])}
    if family == 'reservations':
        available, active, shipped, rejected = dict(items['stock']), {}, defaultdict(int), 0
        for _, row in sorted(enumerate(items['events']), key=lambda x: (x[1][0], x[0])):
            _, rid, kind, sku, q = row
            if kind == 'reserve':
                if rid in active or q > available[sku]: rejected += 1
                else: available[sku] -= q; active[rid] = [sku, q]
            elif rid not in active: rejected += 1
            else:
                old_sku, old_q = active.pop(rid)
                if kind == 'cancel': available[old_sku] += old_q
                else: shipped[old_sku] += old_q
        return {'available': available, 'active': active, 'shipped': dict(shipped), 'rejected': rejected}
    if family == 'revisions':
        texts, parents = {}, set()
        for rid, parent, pos, count, inserted in items:
            text = texts[parent] if parent is not None else ''
            pos = min(pos, len(text)); texts[rid] = text[:pos]+inserted+text[pos+count:]
            if parent is not None: parents.add(parent)
        return {'texts': texts, 'leaves': sorted(set(texts)-parents)}
    raise ValueError(family)


def aggregate(family, value, p):
    rows = []
    if family == 'fifo':
        for account, invoices in sorted(value['invoices'].items()):
            payment, total, overdue, count = value['payments'].get(account, 0), 0, 0, 0
            for when, _, amount in invoices:
                paid = min(payment, amount); payment -= paid; left = amount-paid
                total += left; overdue += left if value['now']-when > p else 0; count += int(left > 0)
            rows.append([account, total, overdue, count])
        return {'rows': rows, 'total': sum(r[1] for r in rows)}
    if family == 'timesheets':
        for user, segments in sorted(value.items()):
            days = defaultdict(lambda: [0, 0])
            for start, end, rate in segments:
                while start < end:
                    day = start//10; stop = min(end, (day+1)*10)
                    covered, paid = days[day]; amount = stop-start
                    days[day] = [covered+amount, paid+min(amount, max(0, p-covered))*rate]; start = stop
            rows.extend([user, day, *stats] for day, stats in sorted(days.items()))
        return {'rows': rows, 'total': sum(r[3] for r in rows)}
    if family == 'reconcile':
        totals = defaultdict(lambda: [0, 0])
        for category, quantity, gross in value: totals[category][0] += quantity; totals[category][1] += gross
        for category, (q, gross) in sorted(totals.items()):
            discount = gross*15//100 if gross >= p else 0
            rows.append([category, q, gross, discount, gross-discount])
        return {'rows': rows, 'total': sum(r[4] for r in rows)}
    if family == 'schedule':
        if value['cycle']: return {'cycle': True}
        jobs = {r[0]: r for r in value['jobs']}; free, finished = [0]*p, {}
        for key in value['order']:
            _, duration, parents = jobs[key]; choices = []
            for worker, when in enumerate(free):
                start = max([when]+[finished[x] for x in parents])
                while overlaps := [b for a, b in value['blackout'] if start < b and start+duration > a]: start = max(overlaps)
                choices.append((start, worker))
            start, worker = min(choices); finished[key] = free[worker] = start+duration
            rows.append([key, start, start+duration, worker])
        return {'rows': sorted(rows), 'makespan': max(free, default=0)}
    if family == 'reservations':
        for sku, available in sorted(value['available'].items()):
            reserved = sum(q for s, q in value['active'].values() if s == sku); shipped = value['shipped'].get(sku, 0)
            if available < p or reserved or shipped: rows.append([sku, available, reserved, shipped])
        return {'rows': rows, 'rejected': value['rejected']}
    if family == 'revisions':
        for rid in value['leaves']:
            words = value['texts'][rid].split()
            if len(words) >= p: rows.append([rid, ' '.join(words), len(words)])
        return {'rows': rows, 'total': sum(r[2] for r in rows)}
    raise ValueError(family)


def cases(family):
    """One simple visible case, empty visible case, then fixed edge/random cases."""
    rng = random.Random(924)
    if family == 'fifo':
        empty = {'invoices': [], 'payments': [], 'void': [], 'now': 20}
        values = [empty, {**empty, 'invoices': [['a','i',1,10]], 'payments': [['a',3]]}]
        for k in range(8):
            values.append({'invoices': [[rng.choice('ab'), f'i{rng.randrange(5)}', rng.randrange(20), rng.randrange(20)] for _ in range(12)],
                           'payments': [[rng.choice('abc'), rng.randrange(25)] for _ in range(5)], 'void': ['i4'] if k%2 else [], 'now': 20})
    elif family == 'timesheets':
        values = [[], [['a',0,3,2]], [['a',0,12,2],['a',3,8,5],['b',9,22,1]]]
        values += [[[rng.choice('ab'), (a:=rng.randrange(25)), a+rng.randrange(1,16), rng.randrange(1,8)] for _ in range(8)] for _ in range(7)]
    elif family == 'reconcile':
        empty = {'prices': [], 'orders': [], 'returns': []}
        values = [empty, {'prices': [['x',0,3,'A']], 'orders': [['o','x',1,4]], 'returns': []}]
        for _ in range(8):
            values.append({'prices': [[rng.choice('xy'), rng.randrange(5), rng.randrange(1,11), rng.choice('AB')] for _ in range(9)],
                           'orders': [[f'o{i}', rng.choice('xyz'), rng.randrange(7), rng.randrange(7)] for i in range(8)],
                           'returns': [[f'o{rng.randrange(9)}', rng.randrange(5)] for _ in range(8)]})
    elif family == 'schedule':
        values = [{'jobs': [], 'blackout': []}, {'jobs': [['a',2,[]],['b',3,['a']]], 'blackout': []},
                  {'jobs': [['a',1,['b']],['b',1,['a']]], 'blackout': []}]
        for _ in range(7):
            values.append({'jobs': [[str(i), rng.randrange(1,6), [str(j) for j in range(i) if rng.random()<.35]] for i in range(7)],
                           'blackout': [[4,7],[6,9],[15,18]]})
    elif family == 'reservations':
        values = [{'stock': {}, 'events': []}, {'stock': {'x':5}, 'events': [[0,'a','reserve','x',2]]}]
        values += [{'stock': {'x':8,'y':6}, 'events': [[rng.randrange(8),rng.choice('abcd'),rng.choice(['reserve','cancel','ship']),rng.choice('xy'),rng.randrange(6)] for _ in range(15)]} for _ in range(8)]
    else:
        values = [[], [['a',None,0,0,'one two']]]
        values += [[[str(i), None if i==0 or rng.random()<.2 else str(rng.randrange(i)), rng.randrange(10),rng.randrange(5),rng.choice([' a ','two words','\nX\tY',''])] for i in range(9)] for _ in range(8)]
    return values


def definition(family, p, split):
    description, first, second = SPECS[family]; second = second.replace('{p}', str(p))
    rid = f'pr1_{split.lower()}_{family}_{p:02d}'
    keys = 'rows,makespan (or only cycle for a cycle)' if family=='schedule' else 'rows,rejected' if family=='reservations' else 'rows,total'
    formatting = "Return compact JSON with separators=(',', ':') and no newline; top-level keys in this order: "+keys+'.'
    goals = {'parse': 'Parse the JSON input.', 'prepare': first, 'aggregate': second,
             'format': formatting}
    root = contract('solve', 'text', goals, list(goals), 'Input text encodes '+description, 'str')
    local = {'prepare': contract('prepare','items',{'prepare':first},['prepare'],description),
             'aggregate': contract('aggregate','prepared',{'aggregate':second},['aggregate'],'The exact output of: '+first),
             'compute': contract('compute','items',{'prepare':first,'aggregate':second},['prepare','aggregate'],description)}
    public, private = [], []
    for visibility, indices in [('VISIBLE',range(2)),('PRIVATE',range(2,10))]:
        for key, spec in [('root',root),*local.items()]:
            checks = []
            for i in indices:
                x = cases(family)[i]; prepared = prepare(family,x); answer = aggregate(family,prepared,p)
                args = [json.dumps(x)] if key=='root' else [prepared if key=='aggregate' else x]
                expected = json.dumps(answer,separators=(',', ':')) if key=='root' else prepared if key=='prepare' else answer
                checks.append({'case_id':f'{key}_{i}','args':args,'expected':expected})
            suite = {'suite_id':f'{rid}_{key}_{visibility.lower()}', 'check_kind':'ROOT' if key=='root' else 'LOCAL',
                     'visibility':visibility,'oracle_origin':'PR1 authored oracle v1; fixed before sampling',
                     'covered_obligations':[o['id'] for o in spec['ensures']],'cases':checks}
            if key!='root': suite['contract_key']=key
            (public if visibility=='VISIBLE' else private).append(suite)
    spec = f'Input is JSON encoding {description}. {first} Then: {second} {formatting} Implement solve(text: str) -> str.'
    spec += '\nOptional reviewed subcontracts with visible local checks (copy a contract exactly to use its checks; these are interfaces, not code or a required plan): '+json.dumps(local,separators=(',', ':'))
    task = {'root_id':rid,'family_id':f'pr1_{family}_v1','split':split,'spec':spec,'entrypoint':'solve',
            'visible_suite_ref':f'{rid}_root_visible','private_suite_ref':f'{rid}_root_private'}
    return {'task':task,'contract':root,'local_contracts':local,'suites':public}, {'suites':private}


def build(directory='data/refinement/pr1'):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    roots, tests = [], []
    choices = [('fifo',[5,10]),('timesheets',[5,8]),('reconcile',[20,40]),('schedule',[2,3]),
               ('reservations',[2,4]),('revisions',[1,2])]
    for family, parameters in choices:
        split = 'VALIDATION' if family in {'reservations','revisions'} else 'TRAIN'
        for p in parameters:
            d, private=definition(family,p,split); task=d['task']; folder=directory/task['root_id']; folder.mkdir()
            write_json(folder/'task.json',d); write_json(folder/'private_tests.json',private)
            roots.append({**{k:task[k] for k in ('root_id','family_id','split')},'source':'pr1_authored_v1',
                          'original_split':split,'io_type':'solve_text','fixture_path':str(folder/'task.json'),
                          'source_hash':file_hash(folder/'task.json')})
            tests.append({'root_id':task['root_id'],'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')})
    old=load_root_manifest('data/manifests/roots.jsonl'); old_tests={r['root_id']:r for r in read_jsonl('data/manifests/tests.jsonl')}
    for rid in ['train_inventory_04','train_ledger_04']: roots.append(old[rid]); tests.append(old_tests[rid])
    for name, data in [('roots.jsonl',roots),('tests.jsonl',tests)]:
        (directory/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    write_json(directory/'selection.json',{'train_roots':10,'validation_roots':4,'generator_sha256':file_hash(__file__),
        'selection':'Fixed before sampling; all roots retained. TRAIN difficulty pilot, family-disjoint VALIDATION used only after updates.',
        'public_contract_catalog':True,'training_code_or_private_examples_exposed':False})


if __name__=='__main__': build()
