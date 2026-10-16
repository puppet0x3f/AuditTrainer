"""Fixed-budget complete-algorithm executor SFT across two responsibility structures."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/complex_sft');DATA=Path('data/refinement/complex_sft')
INITIAL='runs/refinement/compute_sft/training/final_adapter'
from .refinement_handoff_sft import GATE,BODY,NestedPlanner
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from .dataset import load_root_manifest,read_jsonl,file_hash
from .evaluate import write_json,capture_program,score_program
from .refinement_generalization import guarded_json
from .refinement_responsibility_course import task_pack,CourseReference,allocation,internal_variant
from .refinement_executor_sft import train
from .refinement_rollout import runtime_for,run_case
from .refinement_policy import load_policies,load_adapter_metadata,atomic_plan
from .refinement_separate_gate import RootGatePolicy,fingerprints
from .refinement_interface_gate import FixedPlanner
from .refinement_compute_values import local_checks
from .schema import PolicyOutput,TokenUsage
from .training_tasks import oracle_a,oracle_b
from .verifier import canonical

from types import SimpleNamespace
from .refinement_complex_tasks import TASKS,pack
from .refinement_complex_executor import prepare_checks

# New TRAIN inputs. The original development cases and edge-order counterexample
# are never used for teacher traces. Algorithm families are intentionally shared.
INPUTS={
 'weighted_routes': [dict(edges=[["q","r",3],["r","t",2],["q","t",8],["t","u",1]],source="q",limits=[k,6,10]) for k in range(6)],
 'stock_events': [dict(initial={"pear":k},events=[[3,"pear",-4],[1,"pear",2],[1,"pear",-3],[2,"plum",5]],targets={"pear":7,"plum":k}) for k in range(6)],
 'coverage_queries': [dict(intervals=[[-8,k],[k,9],[-3,7],[-3,7]],queries=[[-9,10],[k,8],[-4,2]]) for k in range(6)],
 'polynomial_chain': [dict(polynomials=[[[2,2],[0,-1]],[[1,3],[0,2]],[[2,-1],[1,1]]],derivative=k%5,points=[k-3,2,4]) for k in range(6)]
}

def fixed(base,d,variant):
    p=FixedPlanner(base,{'N0':allocation(d['contract'],d,'fused'),'N0_p1_compute':allocation(d['local_contracts']['compute'],d,variant)})
    p.identity={'backend':'authored_fixed_plans','study':'complex_sft','variant':variant}
    return p

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/complex_executor/request.json').read_text())
    rows=load_root_manifest('data/refinement/compute_sft/roots.jsonl');tests={x['root_id']:x for x in read_jsonl('data/refinement/compute_sft/tests.jsonl')}
    rows.update(load_root_manifest(old['config']['root_manifest']));tests.update({x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])})
    # These algorithm families are now taught; preserve original fixtures and
    # explicitly register local development copies in TRAIN, never a heldout split.
    for selected in old['selection']:
        rid=selected['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text())
        d['task']['split']='TRAIN';folder=DATA/'development'/rid;folder.mkdir(parents=True,exist_ok=True)
        guarded_json(folder/'task.json',d)
        rows[rid]={**row,'split':'TRAIN','original_split':'VALIDATION','source':'known_development_same_family_not_supervised','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json')}
    fit=[];targets=[]
    for i,name in enumerate(TASKS):
        rid='train_complex_sft_'+name
        assert not any(x in TASKS[name]['cases'] for x in INPUTS[name])
        d,private,ref=pack(name,cases=INPUTS[name],root_id=rid,split='TRAIN')
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for key,obj in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{key}.json',obj)
        rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'independent_train_inputs_same_algorithm_family','original_split':'TRAIN','io_type':'solve_text','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        fit.append({'root_id':rid,'variant':'parent_aggregate' if i%2==0 else 'pipeline'})
        for variant in ['parent_aggregate','pipeline']:
            teacher=CourseReference(d,ref,'recursive');planner=fixed(SimpleNamespace(cost={},identity={}),d,variant)
            policy=SimpleNamespace(act=lambda obs,p=planner,t=teacher:p.act(obs) if obs['phase']=='NEW' else t.act(obs))
            out=ROOT/'teacher'/rid/variant
            runtime,full=runtime_for(rows[rid],tests,old['framework'],policy,out/'runtime');runtime.run()
            scores,_=score_program(capture_program(runtime,out/'program'),full,rid)
            assert all(x['status']=='PASS' for x in scores.values());write_json(out/'result.json',scores)
            for e in read_jsonl(runtime.trace.path):
                action=e.get('parsed_action')
                if action and action['action']=='write_code':
                    assert e['host_result']['code']=='SOURCE_SAVED'
                    targets.append({'sample_id':rid+'_'+variant+'_'+e['event_id'],'root_id':rid,'split':'TRAIN','kind':'target','observation':e['observation_before'],'target_action':action})
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/compute_sft/training.jsonl')]
    assert len(targets)==28 and len(retention)==46 and all(x['split']=='TRAIN' for x in targets+retention)
    for name,values in [('roots',list(rows.values())),('tests',list(tests.values())),('training',targets+retention)]:
        (DATA/f'{name}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    config={**old['config'],'root_manifest':str(DATA/'roots.jsonl'),'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)}
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'fit':fit,'development':[{'root_id':s['root_id'],'variant':'parent_aggregate'} for s in old['selection']],
        'retention':[{'root_id':'train_course_calendar_matrix_04','first':'calendar'},{'root_id':'train_course_chunks_chunks_04','first':'chunks'},{'root_id':'train_execution_polynomial_matrix_05'},{'root_id':'train_executor_check_paths_matrix_07'}],
        'steps':32,'learning_rate':2e-5,'seed':491000713,'eval_seed':492000713,'initial_executor':INITIAL,'gate':GATE,'body':BODY,
        'target_examples':len(targets),'retention_examples':len(retention),'training_sha256':file_hash(DATA/'training.jsonl'),
        'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)]},
        'design':'32 fixed updates, 2 target +2 retention each. Four same-algorithm TRAIN tasks with distinct inputs; parent_aggregate and pipeline teacher structures; all prepare/aggregate/compute/root writes included. Original development traces/cases and edge-order counterexample excluded from supervision.',
        'evaluation':'48 autonomous paired runs: 4 FIT,4 known same-family development,4 retention roots x2 seeds x2 executors. Fixed plans, original prompts, all failures counted. Not unseen-family generalization or sampled-SPLIT value.',
        'split_note':'Known development copies are registered in TRAIN because the same algorithm and contract families are explicitly supervised; original fixtures/tests unchanged, development traces excluded. Not a heldout benchmark.',
        'decision':'Fixed final checkpoint only; gate/body frozen. No gate training. Require development improvement and retention protection before refreshing values.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})
    print(json.dumps({'targets':len(targets),'retention':len(retention),'evaluations':48}))


def context(method):
    r=json.loads((ROOT/'request.json').read_text());adapter=INITIAL if method=='BEFORE' else str(ROOT/'training/final_adapter')
    base,executor=load_policies(r['framework'],GATE,adapter);base.model.load_adapter(BODY,adapter_name='body',is_trainable=False,local_files_only=True)
    base.model.set_adapter('default');base.model.requires_grad_(False);policy=RootGatePolicy(base.llm,BODY)
    _,metadata=load_adapter_metadata(Path(BODY),r['framework']['model'],base.tokenizer);policy.body.identity.update(metadata)
    executor.llm.identity['role']='executor_compute_'+method
    return r,policy,executor,load_root_manifest(r['config']['root_manifest']),{x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}


def evaluate(method,shard):
    r,policy,executor,rows,tests=context(method);before=fingerprints(policy);results=[]
    group=['fit','development','retention'][shard]
    for i,s in enumerate(r[group]):
        rid=s['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text())
        if group!='retention':planner=fixed(policy,d,s['variant'])
        elif 'first' in s:planner=fixed(policy,d,internal_variant(s['first']))
        else:planner=NestedPlanner(policy,d)
        for rep in range(2):
            c=run_case(row,tests,r['framework'],r['config'],planner,executor,ROOT/'evaluation'/method/group/rid/f'repeat_{rep:02d}',r['eval_seed']+shard*1000+i*100+rep)
            assert c['status']=='COMPLETE' and not c['errors']
            checks=local_checks(c,row,tests,r['framework']) if group!='retention' else {}
            if group!='retention':checks['prepare_local']=prepare_checks(c,row,tests,r['framework'])
            results.append({'group':group,'case':c,'checks':checks})
            print(json.dumps({'method':method,'group':group,'root':rid,'rep':rep,'success':int(c['final_scores']['private']['status']=='PASS')}),flush=True)
    assert fingerprints(policy)==before
    write_json(ROOT/'evaluation'/method/f'shard_{shard}.json',{'status':'COMPLETE','results':results,'frozen':before})


def report():
    r=json.loads((ROOT/'request.json').read_text());arms={};summary={}
    for method in ['BEFORE','AFTER']:
        ps=[json.loads((ROOT/'evaluation'/method/f'shard_{i}.json').read_text()) for i in range(3)]
        assert all(p['frozen']==ps[0]['frozen'] for p in ps)
        arms[method]={'results':[x for p in ps for x in p['results']],'frozen':ps[0]['frozen']};summary[method]={}
        for group in ['fit','development','retention']:
            cs=[x for x in arms[method]['results'] if x['group']==group]
            summary[method][group]={'n':len(cs),'root_success':sum(int(x['case']['final_scores']['private']['status']=='PASS') for x in cs),
                'compute_local_private_pass':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks'].get('local',[])) for x in cs) if group!='retention' else None,
                'root_called_compute':sum(x['checks'].get('parent_called',False) for x in cs) if group!='retention' else None,
                'prepare_local_private_pass':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks'].get('prepare_local',[])) for x in cs) if group!='retention' else None,
                'by_root':{rid:sum(int(x['case']['final_scores']['private']['status']=='PASS') for x in cs if x['case']['root_id']==rid) for rid in sorted({x['case']['root_id'] for x in cs})}}
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)])
    pairs={m:{(x['group'],x['case']['root_id'],x['case']['seed']):int(x['case']['final_scores']['private']['status']=='PASS') for x in v['results']} for m,v in arms.items()}
    assert pairs['BEFORE'].keys()==pairs['AFTER'].keys() and len(pairs['BEFORE'])==24
    paired={'wins':sum(pairs['AFTER'][k]>v for k,v in pairs['BEFORE'].items()),'losses':sum(pairs['AFTER'][k]<v for k,v in pairs['BEFORE'].items()),'ties':sum(pairs['AFTER'][k]==v for k,v in pairs['BEFORE'].items())}
    proceed=all(summary['AFTER'][g]['root_success']>summary['BEFORE'][g]['root_success'] for g in ['fit','development']) and all(summary['AFTER']['retention']['by_root'][k]>=v for k,v in summary['BEFORE']['retention']['by_root'].items())
    write_json(ROOT/'result.json',{'status':'COMPLETE','summary':summary,'paired':paired,'proceed_to_fresh_values':proceed,'evaluation':arms,'training':json.loads((ROOT/'training/result.json').read_text())})
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','report']);p.add_argument('--method',choices=['BEFORE','AFTER']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':train(ROOT,DATA,INITIAL)
    elif a.command=='evaluate':evaluate(a.method,a.shard)
    else:report()
