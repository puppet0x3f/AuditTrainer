"""Small executor correction for output contracts sharing a prepare algorithm."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('runs/refinement/contract_correction')
DATA = Path('data/refinement/contract_correction')
INITIAL = 'runs/refinement/contract_course/training/final_adapter'
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in {'train', 'evaluate'}:
    from .gpu_resource import acquire
    _reservation = acquire(ROOT/'gpu_resources.json', 'train' if sys.argv[1] == 'train' else 'collect')

from . import refinement_contract_course as previous
from .refinement_complex_tasks import TASKS, pack, oracle as base_oracle
from .refinement_cross_state_tasks import VARIANTS, inputs as cross_inputs, oracle as cross_oracle
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, score_program, write_json
from .refinement_responsibility_course import CourseReference, internal_variant
from .refinement_handoff_sft import NestedPlanner
from .refinement_rollout import run_case, runtime_for
from .refinement_separate_gate import fingerprints
from .refinement_compute_values import local_checks
from .refinement_complex_executor import prepare_checks
from .verifier import canonical

CORRECTIONS = ['stock_balance', 'coverage_gaps', 'polynomial_sum']
FRESH = {
    'stock_absolute': ('stock_events',
        "For EVERY final SKU, compute abs(stock-target), with missing target 0. Return {'differences':[[SKU,absolute difference],...] sorted by SKU,'total':sum of differences,'rejected':unchanged rejected indices}. Keep zero differences.",
        """def aggregate(prepared):
    targets = dict(prepared['targets'])
    differences = [[sku, abs(stock-targets.get(sku,0))] for sku,stock in prepared['stocks']]
    return {'differences': differences, 'total': sum(v for _,v in differences), 'rejected': prepared['rejected']}
"""),
    'coverage_exact_two': ('coverage_queries',
        "For each half-open query, return {'exactly_two':length covered by EXACTLY TWO intervals,'three_plus':length covered by THREE OR MORE intervals}. Return {'queries':[...]} in query order; empty queries return zeros.",
        """def aggregate(prepared):
    queries=[]
    for left,right in prepared['queries']:
        two=more=0
        for a,b,count in prepared['segments']:
            length=max(0,min(right,b)-max(left,a))
            if count==2: two+=length
            elif count>=3: more+=length
        queries.append({'exactly_two':two,'three_plus':more})
    return {'queries':queries}
"""),
    'polynomial_difference': ('polynomial_chain',
        "At each point return ORIGINAL product value MINUS requested-order derivative value. Return {'differences':values in point order,'degree':degree of ORIGINAL product or -1 for zero}. Order zero therefore returns zeros.",
        """def aggregate(prepared):
    terms=dict(prepared['terms']); derived=dict(terms)
    for _ in range(prepared['derivative']):
        derived={d-1:d*c for d,c in derived.items() if d>0}
    values=[]
    for x in prepared['points']:
        original=sum(c*x**d for d,c in terms.items())
        derivative=sum(c*x**d for d,c in derived.items())
        values.append(original-derivative)
    return {'differences':values,'degree':max(terms,default=-1)}
""")}

# Six independent TRAIN inputs per algorithm. None are copied from failed model
# traces or the previous development fixtures. All requested derivative orders
# and different original/derivative values occur in the polynomial examples.
INPUTS = {
    'stock_events': [
        {'initial': {'cor_a': k+1, 'cor_b': 3},
         'events': [[2,'cor_a',-k-3],[0,'cor_a',2],[2,'cor_b',-4],[1,'cor_c',k+1]],
         'targets': {'cor_a': k+2, 'cor_b': 1, 'cor_c': 2, 'cor_zero': 0}}
        for k in range(6)],
    'coverage_queries': [
        {'intervals': [[-10,-k-2],[-4,4],[-4,4],[6,9]],
         'queries': [[-12,12],[-5,k],[-k,7],[0,0]]}
        for k in range(6)],
    'polynomial_chain': [
        {'polynomials': [], 'derivative': 0, 'points': [-2,0,3]},
        {'polynomials': [[[1,1],[0,2]],[[1,1],[0,-1]]], 'derivative': 1, 'points': [-2,0,3]},
        {'polynomials': [[[2,2],[0,-1]],[[1,1],[0,3]]], 'derivative': 2, 'points': [-2,0,2]},
        {'polynomials': [[]], 'derivative': 3, 'points': [-1,0,2]},
        {'polynomials': [[[1,1],[0,1]],[[1,1],[0,1]]], 'derivative': 4, 'points': [-2,1,3]},
        {'polynomials': [[[2,1],[2,-1],[1,2],[0,1]]], 'derivative': 1, 'points': [-2,0,2]}]}


def fresh_oracle(name, items):
    family = FRESH[name][0]
    prepared, prior = base_oracle(family, deepcopy(items))
    if family == 'stock_events':
        values = [[sku, abs(stock-items['targets'].get(sku,0))] for sku,stock in prepared['stocks']]
        answer = {'differences':values,'total':sum(x[1] for x in values),'rejected':prepared['rejected']}
    elif family == 'coverage_queries':
        values=[]
        for a,b in items['queries']:
            counts=[sum(left<=x<right for left,right in items['intervals']) for x in range(a,b)]
            values.append({'exactly_two':sum(c==2 for c in counts),'three_plus':sum(c>=3 for c in counts)})
        answer={'queries':values}
    else:
        original=[]
        for x in items['points']:
            product=1
            for factor in items['polynomials']:
                product*=sum(c*x**d for d,c in factor)
            original.append(product)
        answer={'differences':[a-b for a,b in zip(original,prior['values'])],
                'degree':max((d for d,_ in prepared['terms']),default=-1)}
    return prepared, answer


def save_root(rows, tests, name, d, private, ref, source):
    rid=d['task']['root_id'];folder=DATA/rid;folder.mkdir(parents=True,exist_ok=True)
    for label,obj in [('task',d),('private_tests',private),('reference',ref)]:
        write_json(folder/f'{label}.json',obj)
    rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},
               'source':source,'io_type':'solve_text','fixture_path':str(folder/'task.json'),
               'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),
               'reference_hash':file_hash(folder/'reference.json')}
    tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),
                'sha256':file_hash(folder/'private_tests.json')}


def configure():
    previous.ROOT=ROOT;previous.DATA=DATA;previous.INITIAL=INITIAL
    previous.configure()


def prepare():
    assert not (ROOT/'request.json').exists(), 'request.json already exists; do not overwrite the frozen experiment'
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('runs/refinement/contract_course/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])}
    targets=[];selection=[]
    for name in CORRECTIONS:
        family,aggregate,code=VARIANTS[name]
        cases=INPUTS[family]
        assert all(x not in TASKS[family]['cases'] and x not in cross_inputs(name) and x not in previous.INPUTS[family] for x in cases)
        rid='train_contract_correction_'+name
        spec={**TASKS[family],'aggregate':aggregate,'aggregate_code':code}
        d,private,ref=pack(name,task_spec=spec,oracle_fn=cross_oracle,cases=cases,root_id=rid,split='TRAIN')
        save_root(rows,tests,name,d,private,ref,'independent_train_output_contrast')
        for variant in ['atomic','parent_aggregate']:
            selection.append({'group':'fit','root_id':rid,'variant':variant})
        for variant in ['atomic','parent_aggregate','pipeline']:
            teacher=CourseReference(d,ref,'recursive')
            planner=previous.study.fixed(SimpleNamespace(cost={},identity={}),d,variant)
            policy=SimpleNamespace(act=lambda obs,p=planner,t=teacher:p.act(obs) if obs['phase']=='NEW' else t.act(obs))
            out=ROOT/'teacher'/rid/variant
            runtime,full=runtime_for(rows[rid],tests,old['framework'],policy,out/'runtime');runtime.run()
            scores,_=score_program(capture_program(runtime,out/'program'),full,rid)
            assert all(v['status']=='PASS' for v in scores.values())
            write_json(out/'result.json',scores)
            for event in read_jsonl(runtime.trace.path):
                action=event.get('parsed_action') or {}
                if action.get('action')=='write_code':
                    assert event['host_result']['code']=='SOURCE_SAVED'
                    targets.append({'sample_id':rid+'_'+variant+'_'+event['event_id'],
                                    'root_id':rid,'split':'TRAIN','kind':'target',
                                    'observation':event['observation_before'],'target_action':action})
    for name in CORRECTIONS:
        for variant in ['atomic','parent_aggregate']:
            selection.append({'group':'development','root_id':'validation_cross_state_'+name,'variant':variant})
    for name in FRESH:
        family,aggregate,code=FRESH[name]
        source={'stock_events':'stock_balance','coverage_queries':'coverage_gaps',
                'polynomial_chain':'polynomial_sum'}[family]
        cases=cross_inputs(source)
        spec={**TASKS[family],'aggregate':aggregate,'aggregate_code':code}
        rid='train_contract_correction_fresh_'+name
        d,private,ref=pack(name,task_spec=spec,oracle_fn=fresh_oracle,cases=cases,root_id=rid,split='TRAIN')
        save_root(rows,tests,name,d,private,ref,'prespecified_new_output_contract_development')
        for variant in ['atomic','parent_aggregate']:
            selection.append({'group':'fresh_contract','root_id':rid,'variant':variant})
    selection.extend({**s,'group':'taught_retention'} for s in old['selection'] if s['group']=='development')
    selection.extend(s for s in old['selection'] if s['group']=='retention')
    configure();previous.register_development(rows,selection)
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/contract_course/training.jsonl')]
    assert len(targets)==27 and len(retention)==150 and len(selection)==34
    assert len({x['sample_id'] for x in targets+retention})==177
    for label,values in [('roots',list(rows.values())),('tests',list(tests.values())),
                         ('training',targets+retention)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    adapters={**old['adapters'],'executor':INITIAL,'initial':INITIAL}
    request={'framework':old['framework'],
             'config':{**old['config'],'root_manifest':str(DATA/'roots.jsonl'),
                       'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},
             'selection':selection,'steps':108,'learning_rate':1e-5,'seed':601000713,
             'eval_seed':603000713,'repetitions':2,'initial_executor':INITIAL,
             'adapters':adapters,
             'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
             'training_sha256':file_hash(DATA/'training.jsonl'),
             'target_examples':27,'retention_examples':150,'evaluation_cases':136,
             'design':'108 fixed updates, 2 new targets + 2 retention each; every new target 8 exposures. Three previously failed output contracts taught on independent TRAIN inputs. Previous 150 supervised writes retained. Frozen final checkpoint and fixed authored plans.',
             'scope':'Known prior failures are development, three prespecified new output contracts are exploratory development (not sealed), prior taught and legacy tasks are retention. No natural-gate or sampled-SPLIT claim.',
             'decision':'Advance only if corrected development improves, prior taught and legacy retention do not regress, and new output contracts show credible positive transfer. Report all fresh contract outcomes independently; no checkpoint selection.'}
    write_json(ROOT/'request.json',request)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],
                                          'min_free_mib':{'train':22000,'collect':16000}})


def check():
    r=json.loads((ROOT/'request.json').read_text())
    assert file_hash(DATA/'training.jsonl')==r['training_sha256']
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k]
               for k,p in r['adapters'].items())
    rows=load_root_manifest(r['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    for selection in r['selection']:
        rid=selection['root_id'];row=rows[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(tests[rid]['path'])==tests[rid]['sha256']
    assert r['adapters']['internal_gate']=='runs/refinement/transfer_gate/training/final_adapter'
    assert r['adapters']['executor']==INITIAL
    return r


def evaluate(method,shard):
    from tqdm import tqdm
    check();configure()
    r,policy,executor,rows,tests=previous.study.context(method)
    before=fingerprints(policy);results=[]
    jobs=[(i,s,rep) for i,s in enumerate(r['selection']) if i%3==shard
          for rep in range(r['repetitions'])]
    for i,s,rep in tqdm(jobs,desc=f'{method} 分片{shard}',unit='条',dynamic_ncols=True):
        rid=s['root_id'];d=json.loads(Path(rows[rid]['fixture_path']).read_text())
        if 'variant' in s:planner=previous.study.fixed(policy,d,s['variant'])
        elif 'first' in s:planner=previous.study.fixed(policy,d,internal_variant(s['first']))
        else:planner=NestedPlanner(policy,d)
        directory=ROOT/'evaluation'/method/f'{i:02d}_{rid}'/f'repeat_{rep:02d}'
        case=run_case(rows[rid],tests,r['framework'],r['config'],planner,executor,
                      directory,r['eval_seed']+i*100+rep)
        checks={}
        if s['group']!='retention':
            checks=local_checks(case,rows[rid],tests,r['framework'])
            checks['prepare_local']=prepare_checks(case,rows[rid],tests,r['framework'])
        results.append({'selection_index':i,'group':s['group'],
                        'variant':s.get('variant',s.get('first','nested')),
                        'case':case,'checks':checks})
        write_json(directory/'execution_checks.json',checks)
    assert before==fingerprints(policy)
    write_json(ROOT/'evaluation'/method/f'shard_{shard}.json',
               {'results':results,'frozen':before})


def report():
    r=check();arms={};summary={};errors=[]
    groups=['fit','development','fresh_contract','taught_retention','retention']
    for method in ['BEFORE','AFTER']:
        parts=[json.loads((ROOT/'evaluation'/method/f'shard_{i}.json').read_text())
               for i in range(3)]
        assert all(p['frozen']==parts[0]['frozen'] for p in parts)
        results=[x for p in parts for x in p['results']]
        assert len(results)==68
        arms[method]={'results':results,'frozen':parts[0]['frozen']};summary[method]={}
        for group in groups:
            values=[x for x in results if x['group']==group]
            passed=lambda x:x['case']['final_scores']['private']['status']=='PASS'
            summary[method][group]={'n':len(values),'success':sum(map(passed,values)),
                'by_root':{rid:sum(passed(x) for x in values if x['case']['root_id']==rid)
                           for rid in sorted({x['case']['root_id'] for x in values})},
                'by_variant':{v:sum(passed(x) for x in values if x['variant']==v)
                              for v in sorted({x['variant'] for x in values})},
                'compute_private_pass':sum(any(y['visibility']=='PRIVATE' and y['status']=='PASS'
                                                for y in x['checks'].get('local',[])) for x in values)
                                       if group!='retention' else None,
                'root_called_compute':sum(x['checks'].get('parent_called',False) for x in values)
                                      if group!='retention' else None}
        for x in results:
            if x['case']['errors'] or any(y['status']=='ERROR' for kind in ['local','prepare_local']
                                          for y in x['checks'].get(kind,[])):
                errors.append({'method':method,'root_id':x['case']['root_id'],
                               'seed':x['case']['seed'],'errors':x['case']['errors']})
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k]
               for k in ['default','body'])
    keyed={method:{(x['selection_index'],x['case']['seed']):x for x in arm['results']}
           for method,arm in arms.items()}
    assert keyed['BEFORE'].keys()==keyed['AFTER'].keys()
    paired=Counter();pairs=[]
    for key,before in keyed['BEFORE'].items():
        after=keyed['AFTER'][key]
        def first(x):
            return next(e['observation_before'] for e in read_jsonl(
                Path(x['case']['run_dir'])/'runtime/trace.jsonl')
                if e['observation_before']['phase']=='IMPLEMENT')
        assert first(before)==first(after)
        a=before['case']['final_scores']['private']['status']=='PASS'
        b=after['case']['final_scores']['private']['status']=='PASS'
        paired['wins' if b>a else 'losses' if b<a else 'ties']+=1
        pairs.append({'selection_index':key[0],'seed':key[1],
                      'same_first_implementation_observation':True})
    training=json.loads((ROOT/'training/result.json').read_text())
    counts=Counter(i for h in training['history'] for i in h['indices'][:2])
    assert len(counts)==27 and set(counts.values())=={8}
    retained=all(summary['AFTER'][g]['by_root'][rid]>=n
                 for g in ['taught_retention','retention']
                 for rid,n in summary['BEFORE'][g]['by_root'].items())
    corrected=summary['AFTER']['development']['success']>summary['BEFORE']['development']['success']
    fresh=(summary['AFTER']['fresh_contract']['success']>
           summary['BEFORE']['fresh_contract']['success'] and
           all(summary['AFTER']['fresh_contract']['by_root'][rid]>=n
               for rid,n in summary['BEFORE']['fresh_contract']['by_root'].items()))
    proceed=not errors and retained and corrected and fresh
    write_json(ROOT/'result.json',{'status':'COMPLETE','summary':summary,
        'paired':dict(paired),'errors':errors,'corrected_development':corrected,
        'retention_nonregressing':retained,'fresh_contract_improved':fresh,
        'proceed_to_fresh_values':proceed,'training':training,'evaluation':arms})
    write_json(ROOT/'validation.json',{'pairs':pairs,'target_exposures':dict(counts),
        'original_adapters_unchanged':True,'errors':errors})
    print(json.dumps({'summary':summary,'paired':dict(paired),
                      'proceed_to_fresh_values':proceed},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','train','evaluate','report'])
    parser.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE')
    parser.add_argument('--shard',type=int,choices=[0,1,2],default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare();check()
    elif args.command=='check':
        r=check();print(json.dumps({'status':'READY','steps':r['steps'],
                                     'evaluation_cases':r['evaluation_cases'],
                                     'model_loaded':False}))
    elif args.command=='train':
        check();previous.study.train(ROOT,DATA,INITIAL,progress=True)
    elif args.command=='evaluate':evaluate(args.method,args.shard)
    else:report()
