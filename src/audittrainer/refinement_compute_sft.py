"""Bounded executor SFT for composite internal contract fulfillment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/compute_sft');DATA=Path('data/refinement/compute_sft')
INITIAL='runs/refinement/handoff_sft/training/final_adapter'
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

INPUTS={
 'calendar': [[],['00:06','00:12','00:07','00:06'],['01:00','00:59','23:54'],['12:01'],['00:00','12:00','06:00'],['03:06','02:05','04:18','03:06']],
 'intervals': [[],[[0,7],[5,13]], [[-12,-5],[-6,1]],[[8,8]], [[-3,4],[12,20]],[[6,9],[0,2],[7,14]]],
 'chunks': [[],[['a',3],['b',5]], [['Z',13]], [['x',0],['y',7]], [['p',6],['q',6]], [['m',2],['n',3],['o',4]]],
 'words': [[],[' CEDARS ','maple','cedars','birches'], ['ORANGES','oranges','peaches',' '],[' elm ',' OAK '],['sequoia','redwood','sequoia','maples'],[' willows ','chestnut','cypress','aspens']]
}

class ComputePlanner:
    def __init__(self,base,d,first):self.cost=base.cost;self.identity={**base.identity,'intervention':'fixed_recursive_responsibilities'};self.d=d;self.first=first
    def act(self,obs):
        variant='fused' if obs['node_id']=='N0' else internal_variant(self.first) if obs['incoming_contract']['signature']['name']=='compute' else 'atomic'
        return PolicyOutput(raw_text=canonical(allocation(obs['incoming_contract'],self.d,variant)),token_usage=TokenUsage(),model_identity=self.identity)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/handoff_sft/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])}
    course=load_root_manifest('data/refinement/body_transfer/roots.jsonl');course_tests={x['root_id']:x for x in read_jsonl('data/refinement/body_transfer/tests.jsonl')}
    fit=[];dev=[]
    for first,second in [('calendar','matrix'),('intervals','matrix'),('chunks','chunks'),('words','words')]:
        d,private,ref=task_pack(first,second,6,'recursive');oldid=d['task']['root_id'];rid=f'train_compute_sft_{first}_{second}_06'
        d=json.loads(json.dumps(d).replace(oldid,rid));private=json.loads(json.dumps(private).replace(oldid,rid))
        ns={};exec(ref['prepare']+'\n'+ref['aggregate'],ns)
        for suite in d['suites']+private['suites']:
            key=suite.get('contract_key','root');suite['cases']=[]
            for i in ([0,1] if suite['visibility']=='VISIBLE' else [2,3,4,5]):
                items=deepcopy(INPUTS[first][i]);prepared=oracle_a(first,items);answer=oracle_b(second,prepared,6)
                assert ns['prepare'](deepcopy(items))==prepared and ns['aggregate'](deepcopy(prepared))==answer
                suite['cases'].append({'case_id':f'{key}_{i}','args':[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items],
                    'expected':json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer})
        folder=DATA/rid;folder.mkdir()
        for name,obj in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{name}.json',obj)
        rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'compute_sft_train_variant','original_split':'TRAIN','io_type':'solve_text',
            'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        fit.append({'root_id':rid,'first':first})
        oldrid=f'train_course_{first}_{second}_04';rows[oldrid]=course[oldrid];tests[oldrid]=course_tests[oldrid];dev.append({'root_id':oldrid,'first':first})
    targets=[]
    for s in fit:
        rid=s['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text());ref=json.loads(Path(row['reference_path']).read_text())
        runtime,full=runtime_for(row,tests,old['framework'],CourseReference(d,ref,'recursive'),ROOT/'teacher'/rid/'runtime');result=runtime.run()
        scores,_=score_program(capture_program(runtime,ROOT/'teacher'/rid/'program'),full,rid)
        assert all(x['status']=='PASS' for x in scores.values())
        write_json(ROOT/'teacher'/rid/'result.json',{'scores':scores,'runtime':result})
        for e in read_jsonl(runtime.trace.path):
            a=e.get('parsed_action');obs=e['observation_before']
            if a and a['action']=='write_code' and obs['incoming_contract']['signature']['name'] in {'aggregate','compute','solve'}:
                assert e['host_result']['status']=='APPLIED'
                targets.append({'sample_id':rid+'_'+e['event_id'],'root_id':rid,'split':'TRAIN','kind':'target','observation':obs,'target_action':a})
    assert len(targets)==10
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/handoff_sft/training.jsonl')]
    for name,values in [('roots',list(rows.values())),('tests',list(tests.values())),('training',targets+retention)]:
        (DATA/f'{name}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    config=deepcopy(old['config']);config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),output_dir=str(ROOT))
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'fit':fit,'development':dev,
        'retention':['train_execution_polynomial_matrix_05','train_executor_check_paths_matrix_07','train_polynomial_03','train_paths_03'],
        'steps':32,'learning_rate':2e-5,'seed':451000713,'eval_seed':452000713,'initial_executor':INITIAL,'gate':GATE,'body':BODY,
        'target_examples':len(targets),'retention_examples':len(retention),'training_sha256':file_hash(DATA/'training.jsonl'),
        'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)]},
        'design':'Fixed final checkpoint, 32 steps; 2 target +2 retention per update. Four independent TRAIN mod/width/length6 variants with new inputs,10 aggregate/compute/root writes,36 conversion/handoff/original retention actions.',
        'evaluation':'48 paired autonomous runs: 4 FIT +4 known-development compute roots +4 retention roots,2 seeds each,BEFORE/AFTER. Plans fixed; executor autonomously consumes,implements,and completes root. No manual code/action changes. No claim of heldout family generalization.',
        'decision':'Report final fixed-budget model, no checkpoint selection. Do not train gate or treat fixed plans as current sampled-SPLIT value. Fresh values deferred until executor assessment.'})
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
        rid=s if group=='retention' else s['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text())
        if group!='retention':planner=ComputePlanner(policy,d,s['first'])
        elif rid in r['retention'][:2]:planner=NestedPlanner(policy,d)
        else:planner=FixedPlanner(policy,{'N0':atomic_plan(d['contract'])})
        for rep in range(2):
            c=run_case(row,tests,r['framework'],r['config'],planner,executor,ROOT/'evaluation'/method/group/rid/f'repeat_{rep:02d}',r['eval_seed']+shard*1000+i*100+rep)
            assert c['status']=='COMPLETE'
            checks=local_checks(c,row,tests,r['framework']) if group!='retention' else {}
            results.append({'group':group,'case':c,'checks':checks})
            print(json.dumps({'method':method,'group':group,'root':rid,'rep':rep,'success':c['reward']['y']}),flush=True)
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
            summary[method][group]={'n':len(cs),'root_success':sum(x['case']['reward']['y'] for x in cs),
                'compute_local_private_pass':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks'].get('local',[])) for x in cs) if group!='retention' else None,
                'root_called_compute':sum(x['checks'].get('parent_called',False) for x in cs) if group!='retention' else None,
                'by_root':{rid:sum(x['case']['reward']['y'] for x in cs if x['case']['root_id']==rid) for rid in sorted({x['case']['root_id'] for x in cs})}}
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)])
    pairs={m:{(x['group'],x['case']['root_id'],x['case']['seed']):x['case']['reward']['y'] for x in v['results']} for m,v in arms.items()}
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
