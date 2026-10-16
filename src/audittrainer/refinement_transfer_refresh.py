"""Fresh A/S values for the repaired executor on four known internal contracts."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/transfer_refresh')
SOURCE=Path('runs/refinement/internal_transfer')
FIT=Path('runs/refinement/root_retention')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from .refinement_internal_transfer import BranchInternal,gate_study,reviewed_plans,plan_key,same_contract_delegation
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json
from .refinement_rollout import run_case


def prepare():
    if (ROOT/'request.json').exists():return
    assert json.loads((FIT/'result.json').read_text())['proceed_to_fresh_values']
    ROOT.mkdir(parents=True,exist_ok=True)
    r=deepcopy(json.loads((SOURCE/'request.json').read_text()))
    r['adapters']['executor']=str(FIT/'training/final_adapter')
    r['hashes']={k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in r['adapters'].items()}
    for s in r['selection']:
        folder=ROOT/s['root_id'];folder.mkdir(exist_ok=True)
        state=json.loads(Path(s['state_path']).read_text());write_json(folder/'state.json',state);s['state_path']=str(folder/'state.json')
    r['config']['output_dir']=str(ROOT)
    r.update(execution_seed=562000713,cases=32,arms=['ATOMIC','SPLIT'],updates=0,
        scope='Four known development contracts. Coverage and polynomial now explicitly supervised on separate TRAIN inputs. Not unseen-contract generalization.',
        design='Frozen root_retention executor plus unchanged root/internal gates and BODY. Fixed registered root; only first compute action forced A or freshly generated S. Shared seed and mode draw per pair; all later decisions and retries unchanged. All failures count.',
        decision='Absolute difference <=1 of4 uncertain; >=2 of4 exploratory direction. No gate training in this round; refresh mean value with current execution policy.',
        source_request=str(SOURCE/'request.json'))
    write_json(ROOT/'request.json',r)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'states':4,'cases':32,'executor':r['adapters']['executor']}))


def context():
    r=json.loads((ROOT/'request.json').read_text());a=r['adapters']
    base,executor=gate_study.load_policies(r['framework'],a['internal_gate'],a['executor'])
    for name,path in [('body',a['body']),('root_gate',a['gate'])]:
        base.model.load_adapter(path,adapter_name=name,is_trainable=False,local_files_only=True)
    base.model.set_adapter('default');base.model.requires_grad_(False)
    policy=gate_study.InternalGatePolicy(base.llm);policy.identity['frozen_adapter_paths']=json.dumps(a,sort_keys=True)
    return r,policy,executor,None,None


def run(index):
    r=json.loads((ROOT/'request.json').read_text());_,policy,executor,_,_=context();before=gate_study.fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    selected=r['selection'][index];rid=selected['root_id'];state=json.loads(Path(selected['state_path']).read_text());base=ROOT/rid
    definition=json.loads(Path(rows[rid]['fixture_path']).read_text());reviewed=reviewed_plans(state['observation'],definition)
    gate=policy.score_modes(state['observation']);prior=policy.body.score_modes(state['observation'])
    write_json(base/'gate.json',gate);write_json(base/'body_gate.json',prior);cases=[]
    for rep in range(4):
        for arm in ['ATOMIC','SPLIT']:
            fixed=BranchInternal(policy,state,arm);directory=base/arm/f'repeat_{rep:02d}';cached=(directory/'case_result.json').exists()
            c=run_case(rows[rid],tests,r['framework'],r['config'],fixed,executor,directory,r['execution_seed']+index*100+rep)
            assert c['status']=='COMPLETE' or c['errors']==['SCORING_ERROR']
            if not cached:assert fixed.root_sent and fixed.internal_sent
            events=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
            first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==gate_study.NODE)
            assert first['observation_before']==state['observation']
            mode=first['model_input']['refinement']['selected_mode'];assert arm=='NATURAL' or mode==arm
            variant=reviewed.get(plan_key(first['parsed_action'],definition),'UNCONFIRMED') if mode=='SPLIT' else 'ATOMIC'
            cases.append({'arm':arm,'rep':rep,'mode':mode,'case':c,'first_registered':first['host_result']['code']=='PLAN_REGISTERED','variant':variant,
                          'same_contract_delegation':mode=='SPLIT' and same_contract_delegation({'parsed_action':first['parsed_action']},state['observation']['incoming_contract'])})
            print(json.dumps({'root':rid,'arm':arm,'rep':rep,'mode':mode,'root_score':c['final_scores']['private']['status']}),flush=True)
    assert gate_study.fingerprints(policy)==before
    write_json(base/'result.json',{'status':'COMPLETE','root_id':rid,'frozen':before,'p_split':gate['mode_probabilities']['SPLIT'],'body_prior_p_split':prior['mode_probabilities']['SPLIT'],'cases':cases})


def report():
    r=json.loads((ROOT/'request.json').read_text());groups=[json.loads((ROOT/s['root_id']/'result.json').read_text()) for s in r['selection']]
    summary=[];pairs=[];errors=[];rejects={a:Counter() for a in r['arms']}
    for g in groups:
        item={k:g[k] for k in ['root_id','p_split','body_prior_p_split']};paired={}
        expected=json.loads((ROOT/g['root_id']/'state.json').read_text())
        for arm in r['arms']:
            cs=[x for x in g['cases'] if x['arm']==arm];assert len(cs)==4
            paired[arm]={x['rep']:x for x in cs}
            item[arm]={'n':4,'success':sum(x['case']['final_scores']['private']['status']=='PASS' for x in cs),
                'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in cs)/4,
                'variants':dict(Counter(x['variant'] for x in cs)),
                'registered':sum(x['first_registered'] for x in cs),'same_contract_delegations':sum(x['same_contract_delegation'] for x in cs)}
            for x in cs:
                c=x['case'];rejects[arm].update(c['protocol_rejections'])
                if c['errors']:errors.append({'root_id':g['root_id'],'arm':arm,'rep':x['rep'],'errors':c['errors']})
                es=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
                first=next(e for e in es if e['event_kind']=='POLICY' and e['node_id']==gate_study.NODE)
                root=next(e for e in es if e['event_kind']=='POLICY' and e['node_id']=='N0')
                assert first['observation_before']==expected['observation']
                assert root['parsed_action']==expected['root_event']['parsed_action']
        wins=losses=0
        for rep in range(4):
            a=paired['ATOMIC'][rep]['case'];s=paired['SPLIT'][rep]['case'];assert a['seed']==s['seed']
            ay=a['final_scores']['private']['status']=='PASS';sy=s['final_scores']['private']['status']=='PASS'
            wins+=sy>ay;losses+=sy<ay
            pairs.append({'root_id':g['root_id'],'rep':rep,'seed':a['seed'],'same_observation_root_plan':True})
        delta=item['SPLIT']['success']-item['ATOMIC']['success']
        item.update(delta_success=delta/4,split_wins=wins,split_losses=losses,direction='UNCERTAIN' if abs(delta)<=1 else 'SPLIT' if delta>0 else 'ATOMIC')
        summary.append(item)
    assert all(g['frozen']==groups[0]['frozen'] for g in groups)
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    totals={a:{'n':16,'success':sum(s[a]['success'] for s in summary),'mean_tokens':sum(s[a]['mean_tokens'] for s in summary)/4,'rejections':dict(rejects[a])} for a in r['arms']}
    paired={'wins':sum(s['split_wins'] for s in summary),'losses':sum(s['split_losses'] for s in summary)};paired['ties']=16-paired['wins']-paired['losses']
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':32,'updates':0,'summary':summary,'totals':totals,'paired':paired,'groups':groups,'errors':errors})
    write_json(ROOT/'validation.json',{'pairs':pairs,'all_frozen':True,'errors':errors,'updates':0})
    print(json.dumps({'summary':summary,'totals':totals,'paired':paired,'errors':errors}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.index)
    else:report()
