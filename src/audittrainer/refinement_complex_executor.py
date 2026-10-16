"""Sixteen fixed-plan old/new executor controls on known complex development tasks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path('runs/refinement/complex_executor');DATA=Path('data/refinement/complex_executor')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,load_task,capture_program,score_program
from .refinement_generalization import guarded_json
from .refinement_compute_sft import context,INITIAL,GATE,BODY
from .refinement_compute_values import local_checks
from .refinement_interface_gate import FixedPlanner
from .refinement_responsibility_course import allocation,CourseReference
from .refinement_rollout import run_case,runtime_for
from .refinement_separate_gate import fingerprints
from .schema import Contract
from .artifacts import ArtifactStore
from .verifier import canonical


def plans(d):
    return {'N0':allocation(d['contract'],d,'fused'),'N0_p1_compute':allocation(d['local_contracts']['compute'],d,'parent_aggregate')}


def fixed(base,d):
    policy=FixedPlanner(base,plans(d));policy.identity={'backend':'authored_fixed_plans','study':'complex_executor_control'}
    return policy


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/complex_values/request.json').read_text());rows=load_root_manifest(old['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])};rid='validation_internal_complex_weighted_routes'
    extra=Path('runs/refinement/complex_values/edge_order_check/private_tests.json')
    tests[rid]={**tests[rid],'path':str(extra),'sha256':file_hash(extra)}
    (DATA/'tests.jsonl').write_text(''.join(canonical(x)+'\n' for x in tests.values()))
    configs={**old['config'],'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)}
    for s in old['selection']:
        row=rows[s['root_id']];d=json.loads(Path(row['fixture_path']).read_text());ref=json.loads(Path(row['reference_path']).read_text())
        teacher=CourseReference(d,ref,'recursive');p=fixed(SimpleNamespace(cost={},identity={}),d)
        teacher_policy=SimpleNamespace(act=lambda obs,p=p,t=teacher:p.act(obs) if obs['phase']=='NEW' else t.act(obs))
        runtime,full=runtime_for(row,tests,old['framework'],teacher_policy,ROOT/'teacher'/s['root_id']/'runtime');runtime.run()
        scores,_=score_program(capture_program(runtime,ROOT/'teacher'/s['root_id']/'program'),full,s['root_id'])
        assert all(x['status']=='PASS' for x in scores.values())
        write_json(ROOT/'teacher'/s['root_id']/'result.json',{'scores':scores,'plans':plans(d)})
    adapters={'BEFORE':INITIAL,'AFTER':'runs/refinement/compute_sft/training/final_adapter','gate':GATE,'body':BODY}
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':configs,'selection':old['selection'],'adapters':adapters,
        'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},'seed':481000713,'cases':16,'repetitions':2,'updates':0,
        'design':'Four known development tasks, two paired seeds, BEFORE handoff_sft vs AFTER compute_sft. Identical authored fixed root fused/compute parent_aggregate/atomic prepare plans. All implementation and consumption autonomous, no action rewrites or prompt changes.',
        'scoring':'Original suites plus existing private graph edge-order counterexample for both arms, fixed before new execution. No training reward or parameter updates.',
        'scope':'Bounded diagnosis of latest compute SFT influence, not original F01 comparison, not current sampled-SPLIT value or heldout generalization. Zero/zero cannot prove executor equivalence or rule out other regressions.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'cases':16,'teacher_root_pass':4}))


def prepare_checks(case,row,tests,framework):
    directory=Path(case['run_dir'])/'runtime';events=read_jsonl(directory/'trace.jsonl');node='N0_p1_compute_p1_prepare'
    writes=[e for e in events if e['node_id']==node and e['host_result']['code']=='SOURCE_SAVED']
    if not writes:return []
    _,_,_,_,full=load_task(row,tests,framework);contract=Contract.model_validate(writes[-1]['observation_before']['incoming_contract'])
    store=ArtifactStore(directory);artifact=store.load(node,store.latest_version(node));result=[]
    for suite in full.suites.values():
        if suite.check_kind=='LOCAL' and suite.applies(row['root_id'],contract):
            e=full.verify(directory,row['root_id'],contract,artifact,suite.suite_id,f'POST_PREPARE_{suite.visibility}',allow_private=True)
            result.append({'visibility':suite.visibility,'status':e.status,'cases':e.details['executed_cases']})
    return result


def run(method,shard):
    r=json.loads((ROOT/'request.json').read_text());_,policy,executor,_,_=context(method);before=fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])};results=[]
    for i,s in enumerate(r['selection']):
        if i%2!=shard:continue
        row=rows[s['root_id']];d=json.loads(Path(row['fixture_path']).read_text());planner=fixed(policy,d)
        for rep in range(2):
            c=run_case(row,tests,r['framework'],r['config'],planner,executor,ROOT/method/s['root_id']/f'repeat_{rep:02d}',r['seed']+i*100+rep)
            assert c['status']=='COMPLETE' and not c['errors'] and c['reward'] is None
            checks=local_checks(c,row,tests,r['framework']);checks['prepare_local']=prepare_checks(c,row,tests,r['framework'])
            results.append({'case':c,'checks':checks,'rep':rep});write_json(Path(c['run_dir'])/'execution_checks.json',checks)
            print(json.dumps({'method':method,'root':s['root_id'],'rep':rep,'success':c['final_scores']['private']['status']}),flush=True)
    assert fingerprints(policy)==before
    write_json(ROOT/method/f'shard_{shard}.json',{'status':'COMPLETE','frozen':before,'results':results})


def report():
    r=json.loads((ROOT/'request.json').read_text());arms={};summary=[]
    for method in ['BEFORE','AFTER']:
        ps=[json.loads((ROOT/method/f'shard_{i}.json').read_text()) for i in range(2)]
        assert ps[0]['frozen']==ps[1]['frozen'];arms[method]={'frozen':ps[0]['frozen'],'results':[x for p in ps for x in p['results']]}
    for s in r['selection']:
        item={'root_id':s['root_id']}
        for method in arms:
            cs=[x for x in arms[method]['results'] if x['case']['root_id']==s['root_id']];assert len(cs)==2
            rejects=Counter()
            for x in cs:rejects.update(x['case']['protocol_rejections'])
            item[method]={'n':2,'root_success':sum(x['case']['final_scores']['private']['status']=='PASS' for x in cs),
                'compute_private':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks']['local']) for x in cs),
                'prepare_private':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks']['prepare_local']) for x in cs),
                'root_called_compute':sum(x['checks']['parent_called'] for x in cs),'rejections':dict(rejects)}
        summary.append(item)
    before={(x['case']['root_id'],x['case']['seed']):x for x in arms['BEFORE']['results']};paired={'wins':0,'losses':0,'ties':0}
    for x in arms['AFTER']['results']:
        c=x['case'];y=before[(c['root_id'],c['seed'])]['case']
        a=c['final_scores']['private']['status']=='PASS';b=y['final_scores']['private']['status']=='PASS';paired['wins' if a>b else 'losses' if a<b else 'ties']+=1
        getplans=lambda case:[(e['node_id'],e['parsed_action']) for e in read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl') if e['host_result']['code']=='PLAN_REGISTERED']
        assert getplans(c)==getplans(y)
    assert sum(paired.values())==8
    assert all(arms['BEFORE']['frozen'][k]==arms['AFTER']['frozen'][k] for k in ['default','body'])
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':16,'updates':0,'summary':summary,'paired':paired,'evaluation':arms})
    print(json.dumps({'summary':summary,'paired':paired}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--method',choices=['BEFORE','AFTER']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.method,a.shard)
    else:report()
