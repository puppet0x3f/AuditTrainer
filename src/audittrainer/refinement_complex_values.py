"""Frozen preselected complex internal task diagnostic, four states x A/S x4."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/complex_values');DATA=Path('data/refinement/complex_values')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from . import refinement_compute_values as probe
from .refinement_compute_sft import context
from .refinement_complex_tasks import TASKS,pack
from .refinement_responsibility_course import CourseReference,allocation
from .refinement_interface_gate import FixedPlanner
from .refinement_rollout import runtime_for
from .evaluate import write_json,capture_program,score_program
from .dataset import file_hash,read_jsonl
from .refinement_generalization import guarded_json
from .verifier import canonical


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/compute_refresh/request.json').read_text());rows=[];tests={};selection=[]
    # The list is fixed in code before any model results. All outcomes remain included.
    for name in TASKS:
        d,private,ref=pack(name);t=d['task'];rid=t['root_id'];folder=DATA/rid;folder.mkdir(exist_ok=True)
        for key,value in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{key}.json',value)
        row={**{k:t[k] for k in ['root_id','family_id','split']},'source':'complex_internal_development_v1','original_split':'VALIDATION','io_type':'solve_text',
             'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        rows.append(row);tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        teacher,full=runtime_for(row,tests,old['framework'],CourseReference(d,ref,'recursive'),ROOT/'teacher'/rid/'runtime')
        outcome=teacher.run();scores,_=score_program(capture_program(teacher,ROOT/'teacher'/rid/'program'),full,rid)
        assert all(v['status']=='PASS' for v in scores.values())
        write_json(ROOT/'teacher'/rid/'result.json',{'scores':scores,'runtime':outcome})
        base=ROOT/rid;base.mkdir(exist_ok=True)
        planner=FixedPlanner(type('Base',(),{'cost':{},'identity':{}})(),{'N0':allocation(d['contract'],d,'fused')})
        runtime,_=runtime_for(row,tests,old['framework'],planner,base/'decision')
        assert runtime.step().host_result.code=='PLAN_REGISTERED' and runtime.step().host_result.code=='CHILD_ENTERED'
        obs=runtime.build_observation();assert obs['node_id']==probe.NODE and obs['phase']=='NEW'
        event=next(e for e in read_jsonl(runtime.trace.path) if e['event_kind']=='POLICY')
        write_json(base/'state.json',{'root_event':event,'observation':obs})
        selection.append({'root_id':rid,'state_path':str(base/'state.json'),'split':'VALIDATION','structure':name})
    for key,values in [('roots',rows),('tests',list(tests.values()))]:
        (DATA/f'{key}.jsonl').write_text(''.join(canonical(x)+'\n' for x in values))
    r={**old,'config':{**old['config'],'root_manifest':str(DATA/'roots.jsonl'),'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},'selection':selection,
       'plan_seed':471000713,'execution_seed':472000713,
       'scope':'Four newly authored development tasks with multi-stage algorithmic dependencies, selected by structure before model outcomes; no model updates. Not a sealed benchmark or proof these tasks are empirically harder.',
       'selection_rule':'All four fixed: shortest routes with threshold summaries, chronological stock rejection/reordering, multiplicity coverage queries, polynomial products/derivatives. No filtering by measured atomic or split success.',
       'oracle_validation':'Independent algorithm vs reference for every fixture; authored recursive teacher passes visible/private ROOT suites. Teacher plans never used as sampled SPLIT rewards.',
       'decision':'Success-count difference <=1 of4 is uncertain; larger differences exploratory only. Keep all failed plans/retries. Diagnose reward and costs before any gate training.'}
    guarded_json(ROOT/'request.json',r);guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'states':4,'cases':32,'teacher_root_pass':4}))

def check_edge_order():
    """Post-hoc contract counterexample, equally applied to all eight saved programs."""
    import shutil
    from .refinement_complex_tasks import oracle
    from .dataset import load_root_manifest
    from .evaluate import load_task
    r=json.loads((ROOT/'request.json').read_text());rid='validation_internal_complex_weighted_routes'
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    private=json.loads(Path(tests[rid]['path']).read_text());original_root=next(s for s in private['suites'] if s['check_kind']=='ROOT')
    suite={**original_root,'suite_id':rid+'_root_edge_order_private','cases':[]}
    for i,edges in enumerate([[['s','a',1],['a','b',1]],[['a','b',1],['s','a',1]]]):
        items={'edges':edges,'source':'s','limits':[2]};_,answer=oracle('weighted_routes',items)
        assert answer=={'queries':[{'count':3,'total':3,'farthest':'b'}]}
        suite['cases'].append({'case_id':f'edge_order_{i}','args':[json.dumps(items)],'expected':json.dumps(answer,separators=(',',':'))})
    private['suites'].append(suite);out=ROOT/'edge_order_check';out.mkdir(exist_ok=False)
    write_json(out/'private_tests.json',private)
    tests[rid]={**tests[rid],'path':str(out/'private_tests.json'),'sha256':file_hash(out/'private_tests.json')}
    _,_,_,_,full=load_task(rows[rid],tests,r['framework']);group=json.loads((ROOT/rid/'result.json').read_text());results=[]
    for x in group['cases']:
        c=x['case'];snapshot=c['final_program'];folder=out/x['arm']/f"repeat_{x['rep']:02d}";folder.mkdir(parents=True)
        if snapshot:
            shutil.copytree(Path(snapshot['path'])/'artifacts',folder/'artifacts')
            copied={**snapshot,'path':str(folder)}
        else:copied=None
        scores,_=score_program(copied,full,rid)
        results.append({'arm':x['arm'],'rep':x['rep'],'original_private':c['final_scores']['private']['status'],'checked_scores':scores,'source':c['run_dir']})
    write_json(out/'result.json',{'status':'COMPLETE','model_calls':0,'updates':0,
        'scope':'Post-hoc same-graph edge-order counterexample after inspecting one-pass relaxation in original successful SPLIT. All eight existing programs checked equally; original fixtures, programs, and scores untouched. Not an independent model resampling or prespecified holdout.',
        'results':results})
    print(json.dumps({a:sum(x['checked_scores']['private']['status']=='PASS' for x in results if x['arm']==a) for a in ['ATOMIC','SPLIT']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report','edge_order']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':probe.run(a.index,root=ROOT,policy_context=context)
    elif a.command=='edge_order':check_edge_order()
    else:probe.report(root=ROOT)
