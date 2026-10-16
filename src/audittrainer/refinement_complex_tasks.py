"""Four preselected multi-stage internal contracts with independent test oracles."""
from copy import deepcopy
from itertools import product
import json
from .training_tasks import contract
from .verifier import canonical

TASKS={
 'weighted_routes': {
 'input': 'an object with edges (directed [u,v,w], string nodes, integer 0<=w<=9), source (string), and limits (integer list)',
 'prepare': "Compute shortest directed path distances from source; duplicate edges and zero-weight cycles are allowed. Include source at distance 0, omit unreachable nodes. Return native {'distances': [[node,distance],...], 'limits': original limits}, distances sorted by node lexicographically.",
 'aggregate': "For each limit in original order select distance entries with distance<=limit. Return {'queries': [{'count': selected count, 'total': sum of selected distances, 'farthest': node with maximum selected distance, ties lexicographically smallest, or null if none}, ...]}. Source counts if selected.",
 'prepare_code': '''def prepare(items):
    import heapq
    edges = {}
    for u, v, w in items['edges']:
        edges.setdefault(u, []).append((v, w))
    dist = {items['source']: 0}
    queue = [(0, items['source'])]
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != dist[u]:
            continue
        for v, w in edges.get(u, []):
            candidate = cost + w
            if v not in dist or candidate < dist[v]:
                dist[v] = candidate
                heapq.heappush(queue, (candidate, v))
    return {'distances': [[u, dist[u]] for u in sorted(dist)], 'limits': list(items['limits'])}
''',
 'aggregate_code': '''def aggregate(prepared):
    result = []
    for limit in prepared['limits']:
        chosen = [(u, d) for u, d in prepared['distances'] if d <= limit]
        ordered = sorted(chosen, key=lambda x: (-x[1], x[0]))
        result.append({'count': len(chosen), 'total': sum(d for u, d in chosen), 'farthest': ordered[0][0] if ordered else None})
    return {'queries': result}
''',
 'cases': [
 {'edges':[],'source':'s','limits':[-1,0,5]},
 {'edges':[['s','a',5],['s','b',2],['b','a',1],['a','c',1]],'source':'s','limits':[0,2,4]},
 {'edges':[['s','b',0],['b','s',0],['s','a',0],['x','y',1]],'source':'s','limits':[-2,0,3]},
 {'edges':[['s','a',8],['s','a',2],['a','b',2],['s','c',4]],'source':'s','limits':[4,1,2,4]},
 {'edges':[['z','a',1],['a','b',1],['b','z',1],['a','c',1]],'source':'z','limits':[1,2,100]},
 {'edges':[['a','b',1]],'source':'isolated','limits':[]} ]},
 'stock_events': {
 'input': "an object with initial (SKU->nonnegative integer), events ([time,SKU,delta] with integer time and delta), and targets (SKU->nonnegative integer)",
 'prepare': "Process events ordered by (time, original zero-based event index). Each SKU starts at initial.get(SKU,0). Accept an event iff stock+delta>=0, then update stock; otherwise reject it with stock unchanged. Include every SKU in initial, events, or targets. Return {'stocks': [[SKU,final_stock],...], 'rejected': rejected original indices sorted ascending, 'targets': [[SKU,target],...]}, stocks and targets lexicographically sorted. Missing targets mean 0.",
 'aggregate': "Using final stocks, compute need=max(target-stock,0). Keep positive needs, sort by need descending then SKU ascending. Return {'orders': [[SKU,need],...], 'total': sum of needs, 'rejected': unchanged rejected indices}.",
 'prepare_code': '''def prepare(items):
    stocks = dict(items['initial'])
    for sku in items['targets']:
        stocks.setdefault(sku, 0)
    rejected = []
    for index, event in sorted(enumerate(items['events']), key=lambda pair: (pair[1][0], pair[0])):
        time, sku, delta = event
        stocks.setdefault(sku, 0)
        if stocks[sku] + delta < 0:
            rejected.append(index)
        else:
            stocks[sku] += delta
    return {'stocks': [[k, stocks[k]] for k in sorted(stocks)], 'rejected': sorted(rejected), 'targets': [[k, items['targets'][k]] for k in sorted(items['targets'])]}
''',
 'aggregate_code': '''def aggregate(prepared):
    targets = dict(prepared['targets'])
    orders = [[sku, targets.get(sku, 0)-stock] for sku, stock in prepared['stocks'] if targets.get(sku, 0)>stock]
    orders.sort(key=lambda pair: (-pair[1], pair[0]))
    return {'orders': orders, 'total': sum(n for sku, n in orders), 'rejected': prepared['rejected']}
''',
 'cases': [
 {'initial':{},'events':[],'targets':{}},
 {'initial':{'a':2},'events':[[2,'a',-4],[1,'a',3],[2,'a',-2]],'targets':{'a':5,'b':2}},
 {'initial':{},'events':[[0,'x',-1],[0,'x',2],[0,'x',-2]],'targets':{'x':3}},
 {'initial':{'a':4,'b':1},'events':[[3,'a',-5],[1,'b',-1],[2,'a',1]],'targets':{'a':4,'b':4,'c':4}},
 {'initial':{'Z':0},'events':[[5,'q',0],[-1,'q',-2],[0,'q',3],[4,'q',-1]],'targets':{'q':2,'Z':1}},
 {'initial':{'a':2},'events':[[0,'a',-2],[0,'a',-1],[0,'a',1]],'targets':{}} ]},
 'coverage_queries': {
 'input': 'an object with intervals and queries, each a list of half-open integer [start,end] pairs, -12<=start<=end<=12; duplicate intervals allowed',
 'prepare': "Sweep intervals with multiplicity. Return {'segments': [[start,end,count],...], 'queries': original queries}. Segments must be sorted, nonempty, positive-count, disjoint, and maximal: merge adjacent segments with equal count. Ignore empty intervals. Queries retain order.",
 'aggregate': "For each half-open query return covered length where count>0, exposure (integral of count), and peak count on the query, or 0 if empty/no overlap. Return {'queries': [{'covered': integer, 'exposure': integer, 'peak': integer}, ...]} in query order.",
 'prepare_code': '''def prepare(items):
    changes = {}
    for a, b in items['intervals']:
        if a < b:
            changes[a] = changes.get(a, 0)+1
            changes[b] = changes.get(b, 0)-1
    points = sorted(changes)
    segments = []
    count = 0
    for i in range(len(points)-1):
        a, b = points[i], points[i+1]
        count += changes[a]
        if count:
            if segments and segments[-1][1] == a and segments[-1][2] == count:
                segments[-1][1] = b
            else:
                segments.append([a, b, count])
    return {'segments': segments, 'queries': [list(q) for q in items['queries']]}
''',
 'aggregate_code': '''def aggregate(prepared):
    result = []
    for left, right in prepared['queries']:
        covered = exposure = peak = 0
        for a, b, count in prepared['segments']:
            length = max(0, min(b, right)-max(a, left))
            if length:
                covered += length
                exposure += count*length
                peak = max(peak, count)
        result.append({'covered': covered, 'exposure': exposure, 'peak': peak})
    return {'queries': result}
''',
 'cases': [
 {'intervals':[],'queries':[[0,0],[-2,3]]},
 {'intervals':[[0,4],[2,6]],'queries':[[1,5],[4,6],[0,2]]},
 {'intervals':[[-4,0],[0,4],[0,0]],'queries':[[-3,3],[0,0],[-12,-4]]},
 {'intervals':[[1,3],[1,3],[2,5]],'queries':[[0,6],[3,4],[5,7]]},
 {'intervals':[[-10,-8],[-6,-2],[0,1]],'queries':[[-12,12],[-7,-3],[-8,-6]]},
 {'intervals':[[-3,5],[-2,4],[-1,3]],'queries':[[-1,1],[-3,5],[8,10]]} ]},
 'polynomial_chain': {
 'input': 'an object with polynomials (list of polynomials, each a list of [degree,coefficient] integer pairs, 0<=degree<=4, -3<=coefficient<=3), derivative (integer 0..4), points (integer list)',
 'prepare': "Multiply all polynomials exactly over integers, combining duplicate degrees and dropping zero coefficients. An empty polynomial is zero; an empty list of polynomials has product 1. Return {'terms': [[degree,coefficient],...] ascending degree with nonzero coefficients, 'derivative': unchanged derivative, 'points': unchanged points}.",
 'aggregate': "Differentiate terms the specified number of times, evaluate that derivative at every point in order, and return {'values': evaluation list, 'degree': highest nonzero derivative degree or -1 for zero}. Order zero means no differentiation.",
 'prepare_code': '''def prepare(items):
    result = {0: 1}
    for polynomial in items['polynomials']:
        combined = {}
        for degree, coefficient in polynomial:
            combined[degree] = combined.get(degree, 0)+coefficient
        new = {}
        for a, x in result.items():
            for b, y in combined.items():
                new[a+b] = new.get(a+b, 0)+x*y
        result = {d: c for d, c in new.items() if c}
    return {'terms': [[d, result[d]] for d in sorted(result)], 'derivative': items['derivative'], 'points': list(items['points'])}
''',
 'aggregate_code': '''def aggregate(prepared):
    terms = dict(prepared['terms'])
    for _ in range(prepared['derivative']):
        terms = {d-1: d*c for d, c in terms.items() if d}
    degree = max(terms, default=-1)
    values = []
    for x in prepared['points']:
        value = 0
        for d in range(degree, -1, -1):
            value = value*x + terms.get(d, 0)
        values.append(value)
    return {'values': values, 'degree': degree}
''',
 'cases': [
 {'polynomials':[],'derivative':0,'points':[-1,0,2]},
 {'polynomials':[[[1,1],[0,1]],[[1,1],[0,-1]]],'derivative':1,'points':[-2,0,3]},
 {'polynomials':[[[2,2],[2,-2],[0,1]],[[3,1],[0,2]]],'derivative':2,'points':[0,1,-1]},
 {'polynomials':[[],[[0,2]]],'derivative':0,'points':[0,3]},
 {'polynomials':[[[1,1],[0,1]],[[1,1],[0,1]],[[1,1],[0,-1]]],'derivative':3,'points':[-2,0,2]},
 {'polynomials':[[[0,2]],[[0,3]]],'derivative':4,'points':[]} ]}
}


def oracle(name, items):
    """Alternative algorithms for expected prepared and aggregate outputs."""
    if name=='weighted_routes':
        nodes={items['source']}|{n for u,v,w in items['edges'] for n in (u,v)}
        dist={n:float('inf') for n in nodes};dist[items['source']]=0
        for _ in range(len(nodes)):
            previous=dict(dist)
            for u,v,w in items['edges']:dist[v]=min(dist[v],previous[u]+w)
        pairs=[[n,dist[n]] for n in sorted(nodes) if dist[n]!=float('inf')]
        prep={'distances':pairs,'limits':items['limits']};answers=[]
        for limit in items['limits']:
            chosen={n:d for n,d in pairs if d<=limit};best=max(chosen.values(),default=None)
            answers.append({'count':len(chosen),'total':sum(chosen.values()),'farthest':min((n for n,d in chosen.items() if d==best),default=None)})
        return prep,{'queries':answers}
    if name=='stock_events':
        keys=sorted(set(items['initial'])|set(items['targets'])|{e[1] for e in items['events']});stocks=[];rejected=[]
        for sku in keys:
            stock=items['initial'].get(sku,0)
            for time,i,delta in sorted((e[0],i,e[2]) for i,e in enumerate(items['events']) if e[1]==sku):
                next_stock=stock+delta
                if next_stock<0:rejected.append(i)
                else:stock=next_stock
            stocks.append([sku,stock])
        prep={'stocks':stocks,'rejected':sorted(rejected),'targets':sorted([k,v] for k,v in items['targets'].items())}
        needs={sku:max(0,items['targets'].get(sku,0)-n) for sku,n in stocks}
        order=sorted((k for k in needs if needs[k]),key=lambda k:(-needs[k],k))
        return prep,{'orders':[[k,needs[k]] for k in order],'total':sum(needs.values()),'rejected':sorted(rejected)}
    if name=='coverage_queries':
        density={t:sum(a<=t<b for a,b in items['intervals']) for t in range(-12,12)};segments=[]
        for t,c in density.items():
            if c:
                if segments and segments[-1][1]==t and segments[-1][2]==c:segments[-1][1]=t+1
                else:segments.append([t,t+1,c])
        return {'segments':segments,'queries':items['queries']},{'queries':[{'covered':sum(density[t]>0 for t in range(a,b)), 'exposure':sum(density[t] for t in range(a,b)), 'peak':max((density[t] for t in range(a,b)),default=0)} for a,b in items['queries']]}
    assert name=='polynomial_chain'
    coeff={}
    for choices in product(*items['polynomials']):
        degree=sum(d for d,c in choices);c=1
        for d,value in choices:c*=value
        coeff[degree]=coeff.get(degree,0)+c
    coeff={d:c for d,c in coeff.items() if c};k=items['derivative'];derived={}
    for d,c in coeff.items():
        if d>=k:
            factor=1
            for j in range(k):factor*=d-j
            derived[d-k]=c*factor
    prep={'terms':sorted([d,c] for d,c in coeff.items()),'derivative':k,'points':items['points']}
    return prep,{'values':[sum(c*x**d for d,c in derived.items()) for x in items['points']],'degree':max(derived,default=-1)}


def pack(name, *, cases=None, root_id=None, split='VALIDATION', task_spec=None, oracle_fn=None):
    specification = TASKS[name] if task_spec is None else task_spec
    evaluate = oracle if oracle_fn is None else oracle_fn
    t={**specification, 'cases': specification['cases'] if cases is None else cases};rid=root_id or 'validation_internal_complex_'+name
    obligations={'prepare':t['prepare'],'aggregate':t['aggregate']}
    root=contract('solve','text',{'parse':'Parse JSON input.',**obligations,'format':"Return compact JSON with separators=(',', ':') and no newline; preserve the stated output key order."},['parse','prepare','aggregate','format'],'Input JSON encodes '+t['input'],'str')
    local={'prepare':contract('prepare','items',{'prepare':t['prepare']},['prepare'],t['input']),
           'aggregate':contract('aggregate','prepared',{'aggregate':t['aggregate']},['aggregate'],'Exact native output of: '+t['prepare']),
           'compute':contract('compute','items',obligations,['prepare','aggregate'],t['input'])}
    ref={'prepare':t['prepare_code'],'aggregate':t['aggregate_code'],'component_lineage':[name,name],'structure':'recursive'}
    ns={};exec(ref['prepare']+'\n'+ref['aggregate'],ns);suites=[];private=[]
    for visibility,indices in [('VISIBLE',[0,1]),('PRIVATE',[2,3,4,5])]:
        for key,c in [('root',root),*local.items()]:
            cases=[]
            for i in indices:
                items=deepcopy(t['cases'][i]);prepared,answer=evaluate(name,deepcopy(items))
                assert ns['prepare'](deepcopy(items))==prepared
                assert ns['aggregate'](deepcopy(prepared))==answer
                cases.append({'case_id':f'{key}_{i}','args':[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items],
                    'expected':json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer})
            suite={'suite_id':f'{rid}_{key}_{visibility.lower()}','check_kind':'ROOT' if key=='root' else 'LOCAL','visibility':visibility,
                   'oracle_origin':'Independent algorithm cross-check; frozen before model execution','covered_obligations':[o['id'] for o in c['ensures']],'cases':cases}
            if key!='root':suite['contract_key']=key
            (suites if visibility=='VISIBLE' else private).append(suite)
    task={'root_id':rid,'family_id':'internal_complex_'+name+'_v1','split':split,'entrypoint':'solve','visible_suite_ref':rid+'_root_visible','private_suite_ref':rid+'_root_private',
          'spec':f"Input is JSON encoding {t['input']}. {t['prepare']} Then {t['aggregate']} Return compact JSON with no newline and the stated output key order. Implement solve(text: str) -> str. Optional reviewed subcontracts (no implementation): "+canonical(local)}
    return {'task':task,'contract':root,'local_contracts':local,'suites':suites},{'suites':private},ref
