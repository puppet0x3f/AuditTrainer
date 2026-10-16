"""Fresh internal and root A/S values after executor handoff SFT; no updates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/post_handoff_values')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from .dataset import read_jsonl,file_hash
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_handoff_sft import context,INITIAL,GATE,BODY
from .refinement_internal_probe import FixedInternal,NODE
from .refinement_rollout import run_case,runtime_for
from .refinement_separate_gate import fingerprints
from .refinement_execution_repair import local_checks


def same_contract_delegation(candidate,contract):
    action=candidate.get('parsed_action')
    if not action:return False
    children=action['args']['children']
    clean=lambda c:{k:v for k,v in c.items() if k not in {'contract_id','version'}}
    return len(children)==1 and clean(children[0]['contract'])==clean(contract)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/internal_probe/request.json').read_text())
    current=json.loads(Path('runs/refinement/handoff_sft/request.json').read_text())
    executor='runs/refinement/handoff_sft/training/final_adapter'
    adapters={'gate':GATE,'body':BODY,'executor':executor}
    guarded_json(ROOT/'request.json',{'framework':current['framework'],'config':current['config'],'selection':old['selection'],
        'adapters':adapters,'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
        'repetitions':4,'internal_cases':16,'root_cases':16,'plan_seed':431000713,'execution_seed':432000713,
        'objective':'Frozen current policies. Two exact old internal NEW states: ATOMIC versus four fresh conditional MULTI_BODY SPLITs each. Then same two roots: root ATOMIC versus four new root SPLITs each; internal policy remains natural/frozen.',
        'metrics':'Final root private success, local prepare contract and actual use, logical tokens/calls/nodes, initial refusals/retries/failures. Describe same-contract delegation separately from successful registration.',
        'scope':'Two failure-selected development roots only; no new task generalization, no gate calibration or global value refresh. Old negative labels superseded for these states under the new executor; single continuation per sampled plan.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})


def run(level,index):
    from transformers import set_seed
    r=json.loads((ROOT/'request.json').read_text());_,policy,executor,rows,tests=context('AFTER');before=fingerprints(policy)
    s=r['selection'][index];rid=s['root_id'];base=ROOT/level/rid;base.mkdir(parents=True,exist_ok=True)
    if level=='internal':state=json.loads(Path(s['state_path']).read_text());obs=state['observation']
    else:
        runtime,_=runtime_for(rows[rid],tests,r['framework'],None,base/'decision');obs=runtime.build_observation()
    gate=policy.score_modes(obs);write_json(base/'gate.json',gate);cases=[]
    offset=(0 if level=='internal' else 10000)+index*100
    for rep in range(r['repetitions']):
        set_seed(r['plan_seed']+offset+rep);candidate=policy.sample_split(gate);write_json(base/f'candidate_{rep:02d}.json',candidate)
        for arm in ['ATOMIC','SPLIT']:
            chosen=policy.atomic(gate) if arm=='ATOMIC' else candidate
            if level=='internal':planner=FixedInternal(policy,state,gate,chosen);initial=None
            else:planner=policy;initial=(gate,chosen)
            c=run_case(rows[rid],tests,r['framework'],r['config'],planner,executor,base/arm/f'repeat_{rep:02d}',r['execution_seed']+offset+rep,initial=initial)
            assert c['status']=='COMPLETE'
            if level=='internal':assert planner.root_sent and planner.internal_sent
            events=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl');node=NODE if level=='internal' else 'N0'
            first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==node)
            assert first['model_input']['refinement']['selected_mode']==arm
            checks=local_checks(c,rows[rid],tests,r['framework'])
            write_json(Path(c['run_dir'])/'execution_checks.json',checks)
            cases.append({'arm':arm,'rep':rep,'case':c,'checks':checks,'first_registered':first['host_result']['code']=='PLAN_REGISTERED',
                'same_contract_delegation':arm=='SPLIT' and same_contract_delegation(candidate,obs['incoming_contract'])})
    assert fingerprints(policy)==before
    write_json(base/'result.json',{'status':'COMPLETE','level':level,'root_id':rid,'frozen':before,'cases':cases})


def report():
    r=json.loads((ROOT/'request.json').read_text());groups=[];summary=[]
    for level in ['internal','root']:
        for s in r['selection']:
            g=json.loads((ROOT/level/s['root_id']/'result.json').read_text());groups.append(g);item={'level':level,'root_id':s['root_id']}
            for arm in ['ATOMIC','SPLIT']:
                cs=[x for x in g['cases'] if x['arm']==arm];assert len(cs)==4
                item[arm]={'n':4,'success':sum(x['case']['reward']['y'] for x in cs),
                    'local_private_pass':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks']['local']) for x in cs),
                    'first_registered':sum(x['first_registered'] for x in cs),
                    'same_contract_delegation':sum(x['same_contract_delegation'] for x in cs),
                    'mean_policy_calls':sum(x['case']['usage']['policy_calls'] for x in cs)/4,
                    'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in cs)/4,
                    'mean_nodes':sum(x['case']['node_count'] for x in cs)/4,
                    'mean_wall_seconds':sum(x['case']['wall_seconds'] for x in cs)/4}
            pairs={a:{x['rep']:x['case']['reward']['y'] for x in g['cases'] if x['arm']==a} for a in ['ATOMIC','SPLIT']}
            item['split_wins']=sum(pairs['SPLIT'][i]>pairs['ATOMIC'][i] for i in range(4));item['split_losses']=sum(pairs['SPLIT'][i]<pairs['ATOMIC'][i] for i in range(4))
            item['delta_success']=(item['SPLIT']['success']-item['ATOMIC']['success'])/4;summary.append(item)
    assert all(g['frozen']==groups[0]['frozen'] for g in groups)
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':32,'updates':0,'summary':summary,'groups':groups});print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--level',choices=['internal','root']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.level,a.index)
    else:report()
