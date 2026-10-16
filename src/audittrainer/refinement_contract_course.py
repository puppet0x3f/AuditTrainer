"""Paired-contract executor SFT and fixed-plan autonomous controls."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path('runs/refinement/contract_course');DATA=Path('data/refinement/contract_course')
INITIAL='runs/refinement/root_retention/training/final_adapter'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from . import refinement_complex_sft as study
from .refinement_complex_tasks import TASKS,pack,oracle as basic_oracle
from .refinement_cross_state_tasks import VARIANTS as CROSS,oracle as cross_oracle,inputs as cross_inputs
from .refinement_internal_transfer_tasks import VARIANTS as TRANSFER,oracle as transfer_oracle,CASES as TRANSFER_INPUTS
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,capture_program,score_program
from .refinement_responsibility_course import CourseReference,internal_variant
from .refinement_handoff_sft import NestedPlanner
from .refinement_rollout import runtime_for,run_case
from .refinement_separate_gate import fingerprints
from .refinement_compute_values import local_checks
from .refinement_complex_executor import prepare_checks
from .verifier import canonical

NAMES=['stock_status','stock_events','coverage_single','coverage_redundancy','polynomial_signs','polynomial_joint']
UNSUPERVISED=['stock_balance','coverage_gaps','polynomial_sum']
INPUTS={
 'stock_events':[{'initial':{'course_a':k,'course_b':2},'events':[[3,'course_a',-8],[1,'course_a',3],[1,'course_b',-4],[2,'course_c',k]],'targets':{'course_a':4,'course_b':0,'course_c':2,'course_zero':0}} for k in range(6)],
 'coverage_queries':[{'intervals':[[-9,k-1],[-5,5],[-5,5],[7,7]],'queries':[[-10,10],[-6,k+1],[8,8]]} for k in range(6)],
 'polynomial_chain':[{'polynomials':([] if k==0 else [[]] if k==1 else [[[2,1],[2,-1]]] if k==2 else [[[2,2],[0,-1]],[[1,-1],[0,2]]]),'derivative':k%5,'points':[-3,k-2,1]} for k in range(6)]}


def specification(name):
    if name in CROSS:
        f,a,c=CROSS[name];return f,{**TASKS[f],'aggregate':a,'aggregate_code':c},cross_oracle
    if name in TRANSFER:
        f,a,c=TRANSFER[name];return f,{**TASKS[f],'aggregate':a,'aggregate_code':c},transfer_oracle
    return name,TASKS[name],basic_oracle


def configure():
    study.ROOT=ROOT;study.INITIAL=INITIAL


def prepare():
    if (ROOT/'request.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('runs/refinement/root_retention/request.json').read_text());cross=json.loads(Path('runs/refinement/cross_state/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])}
    rows.update(load_root_manifest(cross['config']['root_manifest']));tests.update({x['root_id']:x for x in read_jsonl(cross['config']['test_manifest'])})
    targets=[];selection=[]
    for i,name in enumerate(NAMES):
        family,spec,oracle=specification(name);rid='train_contract_course_'+name
        assert not any(x in TASKS[family]['cases'] for x in INPUTS[family])
        if name in CROSS:assert not any(x in cross_inputs(name) for x in INPUTS[family])
        if name in TRANSFER:assert not any(x in TRANSFER_INPUTS[name] for x in INPUTS[family])
        d,private,ref=pack(name,task_spec=spec,oracle_fn=oracle,cases=INPUTS[family],root_id=rid,split='TRAIN')
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for label,obj in [('task',d),('private_tests',private),('reference',ref)]:write_json(folder/f'{label}.json',obj)
        rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'independent_train_paired_contracts','io_type':'solve_text','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        selection.append({'group':'fit','root_id':rid,'variant':'atomic' if i%2==0 else 'parent_aggregate'})
        for variant in ['atomic','parent_aggregate','pipeline']:
            teacher=CourseReference(d,ref,'recursive');planner=study.fixed(SimpleNamespace(cost={},identity={}),d,variant)
            policy=SimpleNamespace(act=lambda obs,p=planner,t=teacher:p.act(obs) if obs['phase']=='NEW' else t.act(obs))
            out=ROOT/'teacher'/rid/variant;runtime,full=runtime_for(rows[rid],tests,old['framework'],policy,out/'runtime');runtime.run()
            scores,_=score_program(capture_program(runtime,out/'program'),full,rid);assert all(x['status']=='PASS' for x in scores.values());write_json(out/'result.json',scores)
            for e in read_jsonl(runtime.trace.path):
                a=e.get('parsed_action') or {}
                if a.get('action')=='write_code':
                    assert e['host_result']['code']=='SOURCE_SAVED'
                    targets.append({'sample_id':rid+'_'+variant+'_'+e['event_id'],'root_id':rid,'split':'TRAIN','kind':'target','observation':e['observation_before'],'target_action':a})
    for group,names in [('development',['stock_status','coverage_single','polynomial_signs']),('unsupervised_contract',UNSUPERVISED)]:
        for name in names:
            for variant in ['atomic','parent_aggregate']:selection.append({'group':group,'root_id':'validation_cross_state_'+name,'variant':variant})
    selection.extend({**s,'group':'retention'} for s in old['selection'] if s['group'] in ['development','retention'])
    register_development(rows,selection)
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/root_retention/training.jsonl')]
    assert len(targets)==54 and len(retention)==96 and len(selection)==28
    for label,items in [('roots',list(rows.values())),('tests',list(tests.values())),('training',targets+retention)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(x)+'\n' for x in items))
    adapters={**cross['adapters'],'initial':INITIAL}
    r={'framework':old['framework'],'config':{**old['config'],'root_manifest':str(DATA/'roots.jsonl'),'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},
       'selection':selection,'steps':216,'learning_rate':1e-5,'seed':591000713,'eval_seed':593000713,'initial_executor':INITIAL,
       'adapters':adapters,'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
       'training_sha256':file_hash(DATA/'training.jsonl'),'target_examples':54,'retention_examples':96,'evaluation_cases':112,'repetitions':2,
       'design':'Single final checkpoint; 216 updates, 2 target+2 retention per step, each target 8 exposures. Six paired TRAIN contracts with common within-family inputs, full atomic/parent_aggregate/pipeline writes; all 96 old examples retained. Executor copy only; no gate/body training.',
       'scope':'Fit 12, taught-contract development 12, unsupervised-contract known development 12, retention 20 per executor. No sealed-test or natural-policy/value claim. Diagnostic failure traces and test answers excluded from supervision.',
       'decision':'Require taught-contract development improvement and every retention root nonregressing before fresh value collection. Report unsupervised-contract transfer separately; no claim of transfer without improvement there. No adaptive budget/checkpoint selection.'}
    write_json(ROOT/'request.json',r);write_json(ROOT/'gpu_resources.json',{'devices':['0','1'],'min_free_mib':{'train':22000,'collect':16000}})


def register_development(rows,selection):
    # Taught contracts are development, never a nominal heldout family.
    for rid in {s['root_id'] for s in selection if s['group']=='development'}:
        row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text());d['task']['split']='TRAIN'
        folder=DATA/'development'/rid;folder.mkdir(parents=True,exist_ok=True);write_json(folder/'task.json',d)
        rows[rid]={**row,'split':'TRAIN','original_split':'VALIDATION','source':'known_development_contract_now_taught','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json')}


def check():
    r=json.loads((ROOT/'request.json').read_text());assert file_hash(DATA/'training.jsonl')==r['training_sha256']
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in r['adapters'].items())
    assert r['adapters']['internal_gate']=='runs/refinement/transfer_gate/training/final_adapter'
    assert r['adapters']['executor']==INITIAL
    assert study.GATE==r['adapters']['gate'] and study.BODY==r['adapters']['body']
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    for s in r['selection']:
        row=rows[s['root_id']];assert file_hash(row['fixture_path'])==row['source_hash'];assert file_hash(tests[s['root_id']]['path'])==tests[s['root_id']]['sha256']
    return r


def evaluate(method,shard):
    from tqdm import tqdm
    check();configure();r,policy,executor,rows,tests=study.context(method);before=fingerprints(policy);results=[]
    jobs=[(i,s,rep) for i,s in enumerate(r['selection']) if i%2==shard for rep in range(2)]
    for i,s,rep in tqdm(jobs,desc=f'{method} 分片{shard}',unit='条',dynamic_ncols=True):
        rid=s['root_id'];d=json.loads(Path(rows[rid]['fixture_path']).read_text())
        if 'variant' in s:planner=study.fixed(policy,d,s['variant'])
        elif 'first' in s:planner=study.fixed(policy,d,internal_variant(s['first']))
        else:planner=NestedPlanner(policy,d)
        directory=ROOT/'evaluation'/method/f'{i:02d}_{rid}'/f'repeat_{rep:02d}'
        c=run_case(rows[rid],tests,r['framework'],r['config'],planner,executor,directory,r['eval_seed']+i*100+rep)
        assert c['status']=='COMPLETE' or c['errors']==['SCORING_ERROR']
        checks={}
        if s['group']!='retention':
            checks=local_checks(c,rows[rid],tests,r['framework']);checks['prepare_local']=prepare_checks(c,rows[rid],tests,r['framework'])
        results.append({'selection_index':i,'group':s['group'],'variant':s.get('variant',s.get('first','nested')),'case':c,'checks':checks})
        write_json(directory/'execution_checks.json',checks)
    assert before==fingerprints(policy)
    write_json(ROOT/'evaluation'/method/f'shard_{shard}.json',{'results':results,'frozen':before})


def report():
    r=check();arms={};summary={};errors=[]
    for m in ['BEFORE','AFTER']:
        ps=[json.loads((ROOT/'evaluation'/m/f'shard_{i}.json').read_text()) for i in range(2)];assert ps[0]['frozen']==ps[1]['frozen']
        arms[m]={'results':[x for p in ps for x in p['results']],'frozen':ps[0]['frozen']};assert len(arms[m]['results'])==56;summary[m]={}
        for group in ['fit','development','unsupervised_contract','retention']:
            xs=[x for x in arms[m]['results'] if x['group']==group]
            summary[m][group]={'n':len(xs),'success':sum(x['case']['final_scores']['private']['status']=='PASS' for x in xs),
                'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in xs)/len(xs),
                'compute_private_pass':sum(any(e['visibility']=='PRIVATE' and e['status']=='PASS' for e in x['checks'].get('local',[])) for x in xs) if group!='retention' else None,
                'prepare_private_pass':sum(any(e['visibility']=='PRIVATE' and e['status']=='PASS' for e in x['checks'].get('prepare_local',[])) for x in xs) if group!='retention' else None,
                'root_called_compute':sum(x['checks'].get('parent_called',False) for x in xs) if group!='retention' else None,
                'by_root':{rid:sum(x['case']['final_scores']['private']['status']=='PASS' for x in xs if x['case']['root_id']==rid) for rid in sorted({x['case']['root_id'] for x in xs})},
                'by_variant':{v:{'n':sum(x['variant']==v for x in xs),'success':sum(x['variant']==v and x['case']['final_scores']['private']['status']=='PASS' for x in xs)} for v in sorted({x['variant'] for x in xs})}}
        for x in arms[m]['results']:
            if x['case']['errors'] or any(e['status']=='ERROR' for k in ['local','prepare_local'] for e in x['checks'].get(k,[])):errors.append({'method':m,'root_id':x['case']['root_id'],'seed':x['case']['seed'],'errors':x['case']['errors'],'checks':x['checks']})
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    keyed={m:{(x['selection_index'],x['case']['seed']):x for x in a['results']} for m,a in arms.items()};assert keyed['BEFORE'].keys()==keyed['AFTER'].keys()
    paired=Counter();pairs=[]
    for key,x in keyed['BEFORE'].items():
        y=keyed['AFTER'][key]
        def observation(z):return next(e['observation_before'] for e in read_jsonl(Path(z['case']['run_dir'])/'runtime/trace.jsonl') if e['observation_before']['phase']=='IMPLEMENT')
        assert observation(x)==observation(y)
        a=x['case']['final_scores']['private']['status']=='PASS';b=y['case']['final_scores']['private']['status']=='PASS';paired['wins' if b>a else 'losses' if b<a else 'ties']+=1
        pairs.append({'selection_index':key[0],'seed':key[1],'same_first_implementation_observation':True})
    training=json.loads((ROOT/'training/result.json').read_text());counts=Counter(i for h in training['history'] for i in h['indices'][:2]);assert len(counts)==54 and set(counts.values())=={8}
    proceed=not errors and summary['AFTER']['development']['success']>summary['BEFORE']['development']['success'] and all(summary['AFTER']['retention']['by_root'][k]>=v for k,v in summary['BEFORE']['retention']['by_root'].items())
    write_json(ROOT/'result.json',{'status':'COMPLETE','summary':summary,'paired':dict(paired),'errors':errors,'proceed_to_fresh_values':proceed,'training':training,'evaluation':arms})
    write_json(ROOT/'validation.json',{'pairs':pairs,'target_exposures':dict(counts),'original_adapters_unchanged':True,'errors':errors})
    print(json.dumps({'summary':summary,'paired':dict(paired),'proceed_to_fresh_values':proceed},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','train','evaluate','report']);p.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE');p.add_argument('--shard',type=int,choices=[0,1],default=0);a=p.parse_args()
    if a.command=='prepare':prepare();check()
    elif a.command=='check':print(json.dumps({'status':'READY','steps':check()['steps'],'model_loaded':False}))
    elif a.command=='train':check();study.train(ROOT,DATA,INITIAL,progress=True)
    elif a.command=='evaluate':evaluate(a.method,a.shard)
    else:report()
