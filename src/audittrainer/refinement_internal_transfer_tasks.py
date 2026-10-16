"""Prespecified new aggregation contracts in four known algorithm families."""
from copy import deepcopy
from .refinement_complex_tasks import TASKS, oracle as original_oracle, pack

VARIANTS = {
 'routes_outside': ('weighted_routes',
  "For each limit in original order select reachable nodes with distance strictly GREATER than limit. Return {'queries': [{'count': selected count, 'total': sum of selected distances, 'nearest': node with minimum selected distance, ties lexicographically smallest, or null if none}, ...]}. Source participates under the same strict comparison; unreachable nodes never participate.",
  '''def aggregate(prepared):
    result = []
    for limit in prepared['limits']:
        chosen = [(u, d) for u, d in prepared['distances'] if d > limit]
        ordered = sorted(chosen, key=lambda x: (x[1], x[0]))
        result.append({'count': len(chosen), 'total': sum(d for u, d in chosen), 'nearest': ordered[0][0] if ordered else None})
    return {'queries': result}
'''),
 'stock_surplus': ('stock_events',
  "Using final stocks, compute surplus=max(stock-target,0), with missing targets equal to 0. Keep positive surpluses, sort by surplus descending then SKU ascending. Return {'surplus': [[SKU,surplus],...], 'total': sum of surpluses, 'rejected': unchanged rejected indices}. Do not report replenishment needs.",
  '''def aggregate(prepared):
    targets = dict(prepared['targets'])
    surplus = [[sku, stock-targets.get(sku, 0)] for sku, stock in prepared['stocks'] if stock > targets.get(sku, 0)]
    surplus.sort(key=lambda pair: (-pair[1], pair[0]))
    return {'surplus': surplus, 'total': sum(n for sku, n in surplus), 'rejected': prepared['rejected']}
'''),
 'coverage_redundancy': ('coverage_queries',
  "For each half-open query return overlap (length where count>=2), excess (integral of max(count-1,0)), and peak count on the query, or 0 if empty/no overlap. Return {'queries': [{'overlap': integer, 'excess': integer, 'peak': integer}, ...]} in query order. Single coverage contributes zero to overlap and excess.",
  '''def aggregate(prepared):
    result = []
    for left, right in prepared['queries']:
        overlap = excess = peak = 0
        for a, b, count in prepared['segments']:
            length = max(0, min(b, right)-max(a, left))
            if length:
                overlap += length if count >= 2 else 0
                excess += length*max(count-1, 0)
                peak = max(peak, count)
        result.append({'overlap': overlap, 'excess': excess, 'peak': peak})
    return {'queries': result}
'''),
 'polynomial_joint': ('polynomial_chain',
  "Evaluate BOTH the original product polynomial and its derivative of the requested order at each point. Return {'values': original product values in point order, 'derivatives': requested derivative values in point order, 'degree': degree of the ORIGINAL product, or -1 for the zero polynomial}. Do not replace values or degree with those of the derivative. Derivative order zero repeats original values.",
  '''def aggregate(prepared):
    terms = dict(prepared['terms'])
    derived = dict(terms)
    for _ in range(prepared['derivative']):
        derived = {d-1: d*c for d, c in derived.items() if d > 0}
    values = [sum(c*x**d for d, c in terms.items()) for x in prepared['points']]
    derivatives = [sum(c*x**d for d, c in derived.items()) for x in prepared['points']]
    return {'values': values, 'derivatives': derivatives, 'degree': max(terms, default=-1)}
''')
}

CASES = {
 'routes_outside': [
  {'edges':[], 'source':'origin','limits':[-1,0,2]},
  {'edges':[['b','c',2],['origin','a',3],['origin','b',1],['a','c',0]],'source':'origin','limits':[0,1,3,9]},
  {'edges':[['q','r',0],['r','q',0],['q','s',4],['x','y',1]],'source':'q','limits':[4,-1,0,3]},
  {'edges':[['m','z',6],['m','z',2],['m','a',2],['a','b',1]],'source':'m','limits':[1,2,3,1]},
  {'edges':[['v','w',1],['u','v',2],['w','u',0]],'source':'u','limits':[2,0,3]},
  {'edges':[['u','v',9]],'source':'absent','limits':[]}],
 'stock_surplus': [
  {'initial':{},'events':[],'targets':{}},
  {'initial':{'pear':5,'plum':2},'events':[[2,'pear',-8],[0,'plum',2],[1,'pear',1]],'targets':{'pear':3,'plum':1,'apple':5}},
  {'initial':{},'events':[[0,'kiwi',-2],[0,'kiwi',5],[0,'kiwi',-1]],'targets':{'kiwi':2}},
  {'initial':{'a':7,'b':6,'c':2},'events':[[3,'a',-9],[1,'b',2],[2,'c',3]],'targets':{'a':4,'b':5,'c':9}},
  {'initial':{'Z':3},'events':[[5,'q',1],[-1,'q',-1],[0,'q',5],[4,'q',-2]],'targets':{'q':2,'Z':1}},
  {'initial':{'a':4},'events':[[0,'a',-4],[0,'a',-2],[0,'a',2]],'targets':{}}],
 'coverage_redundancy': [
  {'intervals':[],'queries':[[-2,2],[0,0]]},
  {'intervals':[[-5,1],[-2,4],[-2,4]],'queries':[[-6,5],[-2,1],[1,4]]},
  {'intervals':[[0,2],[2,4],[0,4]],'queries':[[0,4],[2,2],[4,7]]},
  {'intervals':[[-8,-3],[-6,0],[-6,0],[-3,3]],'queries':[[-12,12],[-3,0],[3,4]]},
  {'intervals':[[1,1],[1,3],[1,3],[1,3]],'queries':[[0,4],[1,2],[3,3]]},
  {'intervals':[[-12,12]],'queries':[]}],
 'polynomial_joint': [
  {'polynomials':[],'derivative':2,'points':[0,1,-2]},
  {'polynomials':[[[2,1],[0,2]],[[1,2],[0,-1]]],'derivative':1,'points':[-2,0,3]},
  {'polynomials':[[[1,3],[1,-3]],[[2,1]]],'derivative':0,'points':[1,0]},
  {'polynomials':[[[3,2],[0,-1]],[[1,1],[0,2]]],'derivative':2,'points':[-1,2,0]},
  {'polynomials':[[[0,5]],[[2,2],[0,1]]],'derivative':5,'points':[2,-2]},
  {'polynomials':[[[1,1],[0,1]],[[1,1],[0,-1]]],'derivative':0,'points':[]}]
}


def oracle(name, items):
    family = VARIANTS[name][0]
    prep, previous = original_oracle(family, deepcopy(items))
    if name == 'routes_outside':
        result=[]
        for limit in items['limits']:
            chosen={node:distance for node,distance in prep['distances'] if distance>limit}
            best=min(chosen.values(),default=None)
            result.append({'count':len(chosen),'total':sum(chosen.values()),'nearest':min((u for u,d in chosen.items() if d==best),default=None)})
        return prep,{'queries':result}
    if name == 'stock_surplus':
        remaining={sku:max(stock-items['targets'].get(sku,0),0) for sku,stock in prep['stocks']}
        order=sorted((sku for sku in remaining if remaining[sku]),key=lambda sku:(-remaining[sku],sku))
        return prep,{'surplus':[[sku,remaining[sku]] for sku in order],'total':sum(remaining.values()),'rejected':prep['rejected']}
    if name == 'coverage_redundancy':
        density={t:sum(a<=t<b for a,b in items['intervals']) for t in range(-12,12)}
        return prep,{'queries':[{'overlap':sum(density[t]>=2 for t in range(a,b)), 'excess':sum(max(0,density[t]-1) for t in range(a,b)), 'peak':max((density[t] for t in range(a,b)),default=0)} for a,b in items['queries']]}
    assert name == 'polynomial_joint'
    values=[]
    for x in items['points']:
        value=1
        for factor in items['polynomials']:value*=sum(c*x**d for d,c in factor)
        values.append(value)
    return prep,{'values':values,'derivatives':previous['values'],'degree':max((d for d,c in prep['terms']),default=-1)}


def task_pack(name):
    family,aggregate,code=VARIANTS[name]
    spec={**TASKS[family], 'aggregate':aggregate,'aggregate_code':code,'cases':CASES[name]}
    return pack(name,task_spec=spec,oracle_fn=oracle,root_id='validation_internal_transfer_'+name)
