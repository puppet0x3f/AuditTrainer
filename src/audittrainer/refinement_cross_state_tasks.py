"""Eight prespecified new aggregation contracts; no model-based selection."""
from copy import deepcopy
from .refinement_complex_tasks import TASKS,pack,oracle as base_oracle
from .refinement_internal_transfer_tasks import CASES as OLD_CASES

VARIANTS={
 'routes_below':('weighted_routes',"For each limit select reachable nodes with distance strictly LESS than limit. Return {'queries':[{'count': count, 'total': sum of distances, 'nodes': selected node names sorted lexicographically},...]} in limit order. Source participates under the same comparison; empty selections have count/total zero and nodes [].",'''def aggregate(prepared):
    queries=[]
    for limit in prepared['limits']:
        chosen=[(u,d) for u,d in prepared['distances'] if d<limit]
        queries.append({'count':len(chosen),'total':sum(d for u,d in chosen),'nodes':sorted(u for u,d in chosen)})
    return {'queries':queries}
'''),
 'routes_exact':('weighted_routes',"For each limit select reachable nodes with distance EQUAL to limit. Return {'queries':[{'count': count, 'total': sum of distances, 'nodes': selected node names sorted lexicographically},...]} in limit order. Source participates; empty selections have count/total zero and nodes [].",'''def aggregate(prepared):
    queries=[]
    for limit in prepared['limits']:
        chosen=[(u,d) for u,d in prepared['distances'] if d==limit]
        queries.append({'count':len(chosen),'total':sum(d for u,d in chosen),'nodes':sorted(u for u,d in chosen)})
    return {'queries':queries}
'''),
 'stock_balance':('stock_events',"For EVERY final SKU return signed balance=stock-target (missing target 0), including zero and negative balances, sorted by SKU ascending. Return {'balances':[[SKU,balance],...],'total':sum of signed balances,'rejected':unchanged rejected indices}. Do not clamp, filter or sort by balance.",'''def aggregate(prepared):
    targets=dict(prepared['targets'])
    balances=[[sku,stock-targets.get(sku,0)] for sku,stock in prepared['stocks']]
    balances.sort(key=lambda x:x[0])
    return {'balances':balances,'total':sum(x[1] for x in balances),'rejected':prepared['rejected']}
'''),
 'stock_status':('stock_events',"Classify EVERY final SKU relative to target (missing target 0): below if stock<target, equal if stock==target, above if stock>target. Return {'below':lexicographically sorted SKU list,'equal':sorted SKU list,'above':sorted SKU list,'rejected':unchanged rejected indices}. Include zero-target SKUs.",'''def aggregate(prepared):
    targets=dict(prepared['targets'])
    return {'below':sorted(s for s,n in prepared['stocks'] if n<targets.get(s,0)), 'equal':sorted(s for s,n in prepared['stocks'] if n==targets.get(s,0)), 'above':sorted(s for s,n in prepared['stocks'] if n>targets.get(s,0)), 'rejected':prepared['rejected']}
'''),
 'coverage_single':('coverage_queries',"For each half-open query return {'single':length covered by EXACTLY ONE interval,'multiple':length covered by TWO OR MORE intervals}. Return {'queries':[...]} in query order; uncovered length contributes to neither; empty queries return zeros.",'''def aggregate(prepared):
    result=[]
    for left,right in prepared['queries']:
        single=multiple=0
        for a,b,count in prepared['segments']:
            length=max(0,min(right,b)-max(left,a))
            if count==1:single+=length
            elif count>=2:multiple+=length
        result.append({'single':single,'multiple':multiple})
    return {'queries':result}
'''),
 'coverage_gaps':('coverage_queries',"For each half-open query return {'uncovered':length covered by ZERO intervals,'covered':length covered by at least one interval}. Return {'queries':[...]} in query order. Count gaps outside and between prepared segments; duplicates do not multiply covered length. Empty queries return zeros.",'''def aggregate(prepared):
    result=[]
    for left,right in prepared['queries']:
        covered=sum(max(0,min(right,b)-max(left,a)) for a,b,count in prepared['segments'])
        result.append({'uncovered':right-left-covered,'covered':covered})
    return {'queries':result}
'''),
 'polynomial_signs':('polynomial_chain',"Evaluate the ORIGINAL product at each point and return its sign (-1 negative, 0 zero, 1 positive), not its value or derivative. Return {'signs':signs in point order,'degree':degree of original product or -1 for zero}. The derivative field does not affect this output.",'''def aggregate(prepared):
    terms=dict(prepared['terms'])
    values=[sum(c*x**d for d,c in terms.items()) for x in prepared['points']]
    return {'signs':[(v>0)-(v<0) for v in values],'degree':max(terms,default=-1)}
'''),
 'polynomial_sum':('polynomial_chain',"For each point compute ORIGINAL product value PLUS its requested-order derivative value. Return {'sums':these sums in point order,'degree':degree of ORIGINAL product or -1 for zero}. Order zero returns twice the original values; do not report only derivative values.",'''def aggregate(prepared):
    terms=dict(prepared['terms']);derived=dict(terms)
    for _ in range(prepared['derivative']):derived={d-1:d*c for d,c in derived.items() if d>0}
    sums=[sum(c*x**d for d,c in terms.items())+sum(c*x**d for d,c in derived.items()) for x in prepared['points']]
    return {'sums':sums,'degree':max(terms,default=-1)}
''')}


def inputs(name):
    family=VARIANTS[name][0]
    old={'weighted_routes':'routes_outside','stock_events':'stock_surplus','coverage_queries':'coverage_redundancy','polynomial_chain':'polynomial_joint'}[family]
    cases=deepcopy(OLD_CASES[old])
    for x in cases:
        if family=='weighted_routes':
            x['edges']=[[name+'_'+u,name+'_'+v,w] for u,v,w in x['edges']];x['source']=name+'_'+x['source']
        elif family=='stock_events':
            x['initial']={name+'_'+k:v for k,v in x['initial'].items()};x['targets']={name+'_'+k:v for k,v in x['targets'].items()};x['targets'][name+'_zero']=0
            x['events']=[[t,name+'_'+k,v] for t,k,v in x['events']]
        elif family=='coverage_queries':
            x['intervals']=[[-b,-a] for a,b in x['intervals']]+[[-11,-11]];x['queries']=[[-b,-a] for a,b in x['queries']]
        else:
            x['polynomials']=[[[d,max(-3,min(c,3))] for d,c in factor] for factor in x['polynomials']]
            x['polynomials'].append([[0,1]]);x['derivative']=min(x['derivative'],4)
    return cases


def oracle(name,items):
    family=VARIANTS[name][0];prep,prior=base_oracle(family,deepcopy(items))
    if family=='weighted_routes':
        queries=[]
        for limit in items['limits']:
            ds={u:d for u,d in prep['distances']};nodes=sorted(u for u in ds if (ds[u]<limit if name=='routes_below' else ds[u]==limit))
            queries.append({'count':len(nodes),'total':sum(ds[u] for u in nodes),'nodes':nodes})
        answer={'queries':queries}
    elif family=='stock_events':
        balances={s:n-items['targets'].get(s,0) for s,n in prep['stocks']}
        if name=='stock_balance':answer={'balances':[[s,balances[s]] for s in sorted(balances)],'total':sum(balances.values()),'rejected':prep['rejected']}
        else:answer={'below':sorted(s for s in balances if balances[s]<0),'equal':sorted(s for s in balances if balances[s]==0),'above':sorted(s for s in balances if balances[s]>0),'rejected':prep['rejected']}
    elif family=='coverage_queries':
        queries=[]
        for a,b in items['queries']:
            counts=[sum(l<=x<r for l,r in items['intervals']) for x in range(a,b)]
            queries.append({'single':sum(c==1 for c in counts),'multiple':sum(c>=2 for c in counts)} if name=='coverage_single' else {'uncovered':sum(c==0 for c in counts),'covered':sum(c>0 for c in counts)})
        answer={'queries':queries}
    else:
        values=[]
        for x in items['points']:
            value=1
            for factor in items['polynomials']:value*=sum(c*x**d for d,c in factor)
            values.append(value)
        degree=max((d for d,c in prep['terms']),default=-1)
        answer={'signs':[(v>0)-(v<0) for v in values],'degree':degree} if name=='polynomial_signs' else {'sums':[v+w for v,w in zip(values,prior['values'])],'degree':degree}
    return prep,answer


def task_pack(name):
    family,aggregate,code=VARIANTS[name]
    return pack(name,task_spec={**TASKS[family],'aggregate':aggregate,'aggregate_code':code},cases=inputs(name),oracle_fn=oracle,root_id='validation_cross_state_'+name)
