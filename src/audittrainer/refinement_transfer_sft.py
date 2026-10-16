"""Bounded executor SFT for new-contract fulfillment, with paired fixed-plan checks."""
import argparse
from copy import deepcopy
from collections import Counter
from pathlib import Path
import json
import sys
from types import SimpleNamespace
ROOT=Path('runs/refinement/transfer_sft');DATA=Path('data/refinement/transfer_sft')
INITIAL='runs/refinement/complex_fit/training/final_adapter'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from . import refinement_complex_sft as study
from .refinement_internal_transfer_tasks import VARIANTS,CASES,oracle
from .refinement_complex_tasks import TASKS,pack
from .dataset import file_hash,read_jsonl,load_root_manifest
from .evaluate import write_json,capture_program,score_program
from .refinement_generalization import guarded_json
from .refinement_responsibility_course import CourseReference,internal_variant
from .refinement_handoff_sft import NestedPlanner
from .refinement_rollout import runtime_for,run_case
from .refinement_separate_gate import fingerprints
from .refinement_compute_values import local_checks
from .refinement_complex_executor import prepare_checks
from .verifier import canonical

INPUTS={
 'coverage_redundancy':[{'intervals':[[-10,k-4],[k-4,10],[-7,k],[-7,k]],'queries':[[-11,11],[k-4,9],[-8,-1]]} for k in range(6)],
 'polynomial_joint':[{'polynomials':[[[3,1],[0,k-2]],[[1,-2],[0,3]]] if k<4 else [] if k==4 else [[]], 'derivative':k%5,'points':[k-2,-3,1]} for k in range(6)]}


def configure():
    study.ROOT=ROOT;study.INITIAL=INITIAL


def prepare():
    if (ROOT/'request.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('runs/refinement/complex_fit/request.json').read_text());transfer=json.loads(Path('runs/refinement/internal_transfer/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])}
    new_rows=load_root_manifest(transfer['config']['root_manifest']);new_tests={x['root_id']:x for x in read_jsonl(transfer['config']['test_manifest'])}
    fit=[];development=[];targets=[]
    for i,name in enumerate(INPUTS):
        rid='validation_internal_transfer_'+name;row=new_rows[rid];d=json.loads(Path(row['fixture_path']).read_text());d['task']['split']='TRAIN'
        folder=DATA/'development'/rid;folder.mkdir(parents=True,exist_ok=True);guarded_json(folder/'task.json',d)
        rows[rid]={**row,'split':'TRAIN','source':'known_development_new_contract_now_supervised','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json')};tests[rid]=new_tests[rid]
        development.append({'root_id':rid,'variant':'parent_aggregate','group':'development'})
        rid='train_transfer_sft_'+name;family,aggregate,code=VARIANTS[name]
        assert not any(x in CASES[name] for x in INPUTS[name])
        specification={**TASKS[family],'aggregate':aggregate,'aggregate_code':code}
        d,private,ref=pack(name,cases=INPUTS[name],root_id=rid,split='TRAIN',task_spec=specification,oracle_fn=oracle)
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for label,obj in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{label}.json',obj)
        rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'independent_train_inputs_new_contract','original_split':'TRAIN','io_type':'solve_text','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        fit.append({'root_id':rid,'variant':'parent_aggregate' if i==0 else 'pipeline','group':'fit'})
        for variant in ['parent_aggregate','pipeline']:
            teacher=CourseReference(d,ref,'recursive');planner=study.fixed(SimpleNamespace(cost={},identity={}),d,variant)
            policy=SimpleNamespace(act=lambda obs,p=planner,t=teacher:p.act(obs) if obs['phase']=='NEW' else t.act(obs))
            out=ROOT/'teacher'/rid/variant;runtime,full=runtime_for(rows[rid],tests,old['framework'],policy,out/'runtime');runtime.run()
            scores,_=score_program(capture_program(runtime,out/'program'),full,rid);assert all(x['status']=='PASS' for x in scores.values());write_json(out/'result.json',scores)
            for e in read_jsonl(runtime.trace.path):
                a=e.get('parsed_action')
                if a and a['action']=='write_code':
                    assert e['host_result']['code']=='SOURCE_SAVED'
                    targets.append({'sample_id':rid+'_'+variant+'_'+e['event_id'],'root_id':rid,'split':'TRAIN','kind':'target','observation':e['observation_before'],'target_action':a})
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/complex_sft/training.jsonl')]
    assert len(targets)==14 and len(retention)==74
    keep=[{**s,'group':'retention'} for s in old['development']+old['retention']]
    for label,items in [('roots',list(rows.values())),('tests',list(tests.values())),('training',targets+retention)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(x)+'\n' for x in items))
    adapters={**transfer['adapters'],'initial':INITIAL}
    config={**old['config'],'root_manifest':str(DATA/'roots.jsonl'),'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)}
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'selection':fit+development+keep,'fit':fit,'development':development,'retention':keep,
        'steps':56,'learning_rate':2e-5,'seed':541000713,'eval_seed':542000713,'initial_executor':INITIAL,'gate':adapters['gate'],'body':adapters['body'],'internal_gate':adapters['internal_gate'],
        'adapters':adapters,'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},'training_sha256':file_hash(DATA/'training.jsonl'),
        'target_examples':14,'retention_examples':74,'evaluation_cases':48,
        'design':'Single fixed 56-step checkpoint from complex_fit; 2 target +2 retention per update, each target exactly 8 exposures. Independent TRAIN inputs, parent_aggregate and pipeline full write_code targets. All previous 74 executor examples retained. No gate/body training.',
        'scope':'Known development contracts are now taught on independent TRAIN inputs; development traces and evaluation cases excluded from supervision. Not unseen-contract generalization. Fixed planning decisions; model autonomously implements/consumes/verifies/composes.',
        'decision':'Proceed to fresh A/S only if FIT and development improve over paired baseline and every retention root is nonregressing. All failures retained; no adaptive checkpoint selection.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})
    print(json.dumps({'targets':14,'retention':74,'steps':56,'evaluations':48}))


def evaluate(method,shard):
    configure();r,policy,executor,rows,tests=study.context(method);before=fingerprints(policy);results=[]
    for i,s in enumerate(r['selection']):
        if i%3!=shard:continue
        rid=s['root_id'];d=json.loads(Path(rows[rid]['fixture_path']).read_text())
        if 'variant' in s:planner=study.fixed(policy,d,s['variant'])
        elif 'first' in s:planner=study.fixed(policy,d,internal_variant(s['first']))
        else:planner=NestedPlanner(policy,d)
        for rep in range(2):
            directory=ROOT/'evaluation'/method/rid/f'repeat_{rep:02d}'
            c=run_case(rows[rid],tests,r['framework'],r['config'],planner,executor,directory,r['eval_seed']+i*100+rep)
            assert c['status']=='COMPLETE' or c['errors']==['SCORING_ERROR']
            checks={}
            if s['group']!='retention':
                cp=directory/'execution_checks.json'
                if cp.exists():checks=json.loads(cp.read_text())
                else:
                    checks=local_checks(c,rows[rid],tests,r['framework']);checks['prepare_local']=prepare_checks(c,rows[rid],tests,r['framework']);write_json(cp,checks)
            results.append({'group':s['group'],'case':c,'checks':checks})
            print(json.dumps({'method':method,'group':s['group'],'root':rid,'rep':rep,'root_score':c['final_scores']['private']['status']}),flush=True)
    assert before==fingerprints(policy)
    write_json(ROOT/'evaluation'/method/f'shard_{shard}.json',{'status':'COMPLETE','results':results,'frozen':before})


def report():
    configure();study.report()
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text())
    counts=Counter(i for h in result['training']['history'] for i in h['indices'][:2]);assert len(counts)==14 and set(counts.values())=={8}
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in r['adapters'].items())
    assert file_hash(DATA/'training.jsonl')==r['training_sha256']
    arms={m:{(x['case']['root_id'],x['case']['seed']):x for x in result['evaluation'][m]['results']} for m in ['BEFORE','AFTER']};pairs=[];errors=[]
    for key,x in arms['BEFORE'].items():
        y=arms['AFTER'][key]
        def observation(c):return next(e['observation_before'] for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl') if e['observation_before']['phase']=='IMPLEMENT')
        assert observation(x['case'])==observation(y['case'])
        pairs.append({'root_id':key[0],'seed':key[1],'first_implementation_observation_identical':True})
        for z in [x,y]:
            if z['case']['errors']:errors.append({'root_id':key[0],'seed':key[1],'errors':z['case']['errors']})
    if errors:result['proceed_to_fresh_values']=False
    write_json(ROOT/'result.json',result)
    write_json(ROOT/'validation.json',{'paired':pairs,'target_exposures':dict(counts),'training_data_unchanged':True,'original_adapters_unchanged':True,'scoring_errors':errors})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','report']);p.add_argument('--method',choices=['BEFORE','AFTER'],default='BEFORE');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':study.train(ROOT,DATA,INITIAL)
    elif a.command=='evaluate':evaluate(a.method,a.shard)
    else:report()
