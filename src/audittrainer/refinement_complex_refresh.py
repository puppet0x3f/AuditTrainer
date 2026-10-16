"""Fresh internal A/S values after complete-algorithm executor fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/complex_refresh')
FIT=Path('runs/refinement/complex_fit')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from . import refinement_compute_values as probe
from . import refinement_complex_sft as executor_study
from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import write_json
from .refinement_interface_gate import FixedPlanner
from .refinement_responsibility_course import allocation
from .refinement_rollout import runtime_for


def context(method):
    executor_study.ROOT=FIT
    return executor_study.context(method)


def prepare():
    if (ROOT/'request.json').exists():return
    assert json.loads((FIT/'result.json').read_text())['proceed_to_fresh_values']
    ROOT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((FIT/'request.json').read_text())
    rows=load_root_manifest(prior['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(prior['config']['test_manifest'])};selection=[]
    for selected in prior['development']:
        rid=selected['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text());base=ROOT/rid;base.mkdir()
        planner=FixedPlanner(type('Base',(),{'cost':{},'identity':{}})(),{'N0':allocation(d['contract'],d,'fused')})
        runtime,_=runtime_for(row,tests,prior['framework'],planner,base/'decision')
        assert runtime.step().host_result.code=='PLAN_REGISTERED' and runtime.step().host_result.code=='CHILD_ENTERED'
        obs=runtime.build_observation();assert obs['node_id']==probe.NODE and obs['phase']=='NEW'
        assert {o['id'] for o in obs['incoming_contract']['ensures']}=={'PREPARE','AGGREGATE'}
        event=next(e for e in read_jsonl(runtime.trace.path) if e['event_kind']=='POLICY')
        write_json(base/'state.json',{'root_event':event,'observation':obs})
        selection.append({'root_id':rid,'state_path':str(base/'state.json'),'split':row['split'],'structure':rid.removeprefix('validation_internal_complex_')})
    adapters={'gate':prior['gate'],'body':prior['body'],'executor':str(FIT/'training/final_adapter')}
    config={**prior['config'],'output_dir':str(ROOT)}
    write_json(ROOT/'request.json',{'framework':prior['framework'],'config':config,'selection':selection,'adapters':adapters,
        'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},'repetitions':4,'cases':32,'updates':0,
        'plan_seed':501000713,'execution_seed':502000713,
        'scope':'Four known same-algorithm/contract development states after executor SFT; registered as TRAIN development. Not unseen-family generalization.',
        'design':'Fixed root delegates compute. Force only the first compute decision ATOMIC or fresh MULTI_BODY SPLIT. Same four continuation seeds per arm; all subsequent choices, refusals, retries and failures retained. Frozen complex_fit executor, root gate and body.',
        'scoring':'Final private root success including preexisting weighted-route edge-order counterexample. Local compute, root calls and total billed tokens/calls/nodes secondary. Original suites/prompt/protocol unchanged.',
        'classification':'Reviewed parent_aggregate/parent_prepare/pipeline are responsibility allocations; same-contract delegation separate; all outcomes count including unconfirmed/invalid plans.',
        'decision':'No updates. Success difference <=1 of4 uncertain; larger differences exploratory, not sufficient alone for general labels. Evaluate mean values and costs before choosing next gate work.'})
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'states':4,'cases':32,'executor':adapters['executor']}))


def report():
    probe.report(root=ROOT)
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text());rejects={a:Counter() for a in ['ATOMIC','SPLIT']};pairs=[];score_errors=[]
    for g in result['groups']:
        state=json.loads((ROOT/g['root_id']/'state.json').read_text())['observation']
        for rep in range(4):
            cases={x['arm']:x for x in g['cases'] if x['rep']==rep}
            assert set(cases)=={'ATOMIC','SPLIT'}
            assert cases['ATOMIC']['case']['seed']==cases['SPLIT']['case']['seed']
            for arm,x in cases.items():
                c=x['case'];assert not c['errors'] or c['errors']==['SCORING_ERROR'];rejects[arm].update(c['protocol_rejections'])
                if c['final_scores']['private']['status']=='ERROR':score_errors.append({'root_id':g['root_id'],'arm':arm,'rep':rep,'reason':c['final_scores']['private']['reason']})
                events=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
                first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==probe.NODE)
                assert first['observation_before']==state
            pairs.append({'root_id':g['root_id'],'rep':rep,'seed':cases['ATOMIC']['case']['seed'],'observation_identical':True})
    summary=result['summary']
    totals={a:{'n':16,'success':sum(s[a]['success'] for s in summary),'local_private_pass':sum(s[a]['local_private_pass'] for s in summary),
        'parent_called':sum(s[a]['parent_called'] for s in summary),'mean_tokens':sum(s[a]['mean_tokens'] for s in summary)/4,'rejections':dict(rejects[a])} for a in ['ATOMIC','SPLIT']}
    result.update(totals=totals,private_scoring_errors=score_errors,paired={'split_wins':sum(s['split_wins'] for s in summary),'split_losses':sum(s['split_losses'] for s in summary),'ties':sum(4-s['split_wins']-s['split_losses'] for s in summary)})
    write_json(ROOT/'result.json',result)
    write_json(ROOT/'validation.json',{'paired_decision_observations':pairs,'rollout_errors':0,'private_scoring_errors':score_errors,'scoring_replay':str(ROOT/'scoring_check/result.json'),'all_frozen':True,'updates':0,'tests_log':str(ROOT/'tests.log')})
    print(json.dumps({'totals':totals,'paired':result['paired']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':probe.run(a.index,root=ROOT,policy_context=context,allow_scoring_errors=True)
    else:report()
