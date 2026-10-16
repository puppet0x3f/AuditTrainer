"""Frozen value probes at four genuinely composite internal TRAIN states."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('runs/refinement/compute_values');NODE='N0_p1_compute'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,load_task
from .refinement_generalization import guarded_json
from .refinement_handoff_sft import context,GATE,BODY
from .refinement_internal_probe import FixedInternal
from .refinement_interface_gate import FixedPlanner
from .refinement_responsibility_course import allocation
from .refinement_body_transfer import reviewed_plans
from .refinement_curriculum_rl import plan_key
from .refinement_post_handoff_values import same_contract_delegation
from .refinement_rollout import runtime_for,run_case
from .refinement_separate_gate import fingerprints
from .schema import Contract
from .artifacts import ArtifactStore


def prepare(root=ROOT, prior_path='runs/refinement/handoff_sft/request.json', executor_adapter='runs/refinement/handoff_sft/training/final_adapter', plan_seed=441000713, execution_seed=442000713):
    ROOT=root
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    prior=json.loads(Path(prior_path).read_text())
    rows=load_root_manifest('data/refinement/body_transfer/roots.jsonl');tests={x['root_id']:x for x in read_jsonl('data/refinement/body_transfer/tests.jsonl')}
    states=[x for x in read_jsonl('data/refinement/body_transfer/states.jsonl') if x['split']=='TRAIN' and x['node_id']==NODE]
    assert len(states)==4;selection=[]
    for state in states:
        rid=state['root_id'];row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text());plan=allocation(d['contract'],d,'fused')
        base=ROOT/rid;base.mkdir(exist_ok=True)
        # Only register the fixed root and enter compute; no future code or scores.
        runtime,_=runtime_for(row,tests,prior['framework'],FixedPlanner(type('Base',(),{'cost':{},'identity':{}})(),{'N0':plan}),base/'decision')
        assert runtime.step().host_result.code=='PLAN_REGISTERED'
        assert runtime.step().host_result.code=='CHILD_ENTERED'
        obs=runtime.build_observation();assert obs['node_id']==NODE and obs['phase']=='NEW'
        assert {'PREPARE','AGGREGATE'}=={o['id'] for o in obs['incoming_contract']['ensures']}
        # Curriculum state equality also prevents changing the decision budget.
        assert obs==json.loads(state['messages'][1]['content'])['observation']
        e=next(e for e in read_jsonl(runtime.trace.path) if e['event_kind']=='POLICY')
        write_json(base/'state.json',{'root_event':e,'observation':obs})
        selection.append({'root_id':rid,'state_path':str(base/'state.json'),'split':'TRAIN','structure':'compute_PREPARE_AGGREGATE'})
    config={**prior['config'],'root_manifest':'data/refinement/body_transfer/roots.jsonl','test_manifest':'data/refinement/body_transfer/tests.jsonl'}
    adapters={'gate':GATE,'body':BODY,'executor':executor_adapter}
    guarded_json(ROOT/'request.json',{'framework':prior['framework'],'config':config,'selection':selection,'adapters':adapters,
        'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},'repetitions':4,'cases':32,'plan_seed':plan_seed,'execution_seed':execution_seed,
        'scope':'All four existing TRAIN internal compute curriculum states, selected before new outcomes, previously exposed in body SFT; capability/value diagnosis, not generalization.',
        'design':'Fixed root delegates compute. Force only its first ATOMIC or newly sampled MULTI_BODY SPLIT; all refusals, retries and later choices natural and retained. Root private success primary; local compute, actual root use and costs secondary.',
        'classification':'Reviewed parent_aggregate/parent_prepare/pipeline count as distinct responsibility allocation; fused/same-full-contract child are delegation; unmatched is unconfirmed, not automatically invalid. No filtering by classification or outcome.',
        'decision':'No gate or executor training. Report empirical mean differences; a single success difference among four is uncertain, not a strong label.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})


def local_checks(case,row,tests,framework):
    runtime=Path(case['run_dir'])/'runtime';events=read_jsonl(runtime/'trace.jsonl')
    writes=[e for e in events if e['node_id']==NODE and e['host_result']['code']=='SOURCE_SAVED']
    local=[]
    if writes:
        _,_,_,_,full=load_task(row,tests,framework);contract=Contract.model_validate(writes[-1]['observation_before']['incoming_contract'])
        store=ArtifactStore(runtime);artifact=store.load(NODE,store.latest_version(NODE))
        for suite in full.suites.values():
            if suite.check_kind=='LOCAL' and suite.applies(row['root_id'],contract):
                ev=full.verify(runtime,row['root_id'],contract,artifact,suite.suite_id,f'POST_COMPUTE_{suite.visibility}',allow_private=True)
                local.append({'visibility':suite.visibility,'status':ev.status,'cases':ev.details['executed_cases']})
    used=[]
    for score in case['final_scores']['private']['suites']:
        ev=json.loads(Path(score['evidence_path']).read_text());used.extend(ev['details'].get('dependency_usage',[]))
    return {'local':local,'parent_called':any(d['child_id']==NODE and d.get('called') for d in used),
        'consumed':any(e['node_id']=='N0' and e['host_result']['code']=='CHILD_CONSUMED' and e['host_result']['details']['consumption']['child_node']==NODE for e in events)}


def run(index, root=ROOT, policy_context=context, allow_scoring_errors=False):
    ROOT=root
    from transformers import set_seed
    r=json.loads((ROOT/'request.json').read_text());_,policy,executor,_,_=policy_context('AFTER');before=fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    s=r['selection'][index];rid=s['root_id'];row=rows[rid];base=ROOT/rid;state=json.loads(Path(s['state_path']).read_text());obs=state['observation']
    definition=json.loads(Path(row['fixture_path']).read_text());reviewed=reviewed_plans(obs,definition)
    gate_path=base/'gate.json'
    if gate_path.exists():gate=json.loads(gate_path.read_text())
    else:gate=policy.score_modes(obs);write_json(gate_path,gate)
    cases=[]
    for rep in range(4):
        candidate_path=base/f'candidate_{rep:02d}.json'
        if candidate_path.exists():candidate=json.loads(candidate_path.read_text())
        else:
            set_seed(r['plan_seed']+index*100+rep);candidate=policy.sample_split(gate);write_json(candidate_path,candidate)
        variant=reviewed.get(plan_key(candidate.get('parsed_action'),definition),'UNCONFIRMED')
        for arm in ['ATOMIC','SPLIT']:
            chosen=policy.atomic(gate) if arm=='ATOMIC' else candidate
            fixed=FixedInternal(policy,state,gate,chosen,node=NODE)
            directory=base/arm/f'repeat_{rep:02d}';cached=(directory/'case_result.json').exists()
            c=run_case(row,tests,r['framework'],r['config'],fixed,executor,directory,r['execution_seed']+index*100+rep)
            assert c['status']=='COMPLETE' or (allow_scoring_errors and c['errors']==['SCORING_ERROR'])
            if not cached:assert fixed.root_sent and fixed.internal_sent
            checks_path=Path(c['run_dir'])/'execution_checks.json'
            if checks_path.exists():checks=json.loads(checks_path.read_text())
            else:checks=local_checks(c,row,tests,r['framework']);write_json(checks_path,checks)
            e=next(e for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl') if e['event_kind']=='POLICY' and e['node_id']==NODE)
            assert e['model_input']['refinement']['selected_mode']==arm and e['observation_before']==obs
            cases.append({'arm':arm,'rep':rep,'case':c,'checks':checks,'first_registered':e['host_result']['code']=='PLAN_REGISTERED',
                'variant':variant if arm=='SPLIT' else 'ATOMIC','same_contract_delegation':arm=='SPLIT' and same_contract_delegation(candidate,obs['incoming_contract'])})
    assert fingerprints(policy)==before
    write_json(base/'result.json',{'status':'COMPLETE','root_id':rid,'frozen':before,'p_split':gate['mode_probabilities']['SPLIT'],'cases':cases})


def report(root=ROOT):
    ROOT=root
    from collections import Counter
    r=json.loads((ROOT/'request.json').read_text());groups=[json.loads((ROOT/s['root_id']/'result.json').read_text()) for s in r['selection']];summary=[]
    for g in groups:
        item={'root_id':g['root_id'],'p_split':g['p_split']}
        for arm in ['ATOMIC','SPLIT']:
            cs=[x for x in g['cases'] if x['arm']==arm];assert len(cs)==4
            item[arm]={'n':4,'success':sum(int(x['case']['final_scores']['private']['status']=='PASS') for x in cs),'local_private_pass':sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['checks']['local']) for x in cs),
                'parent_called':sum(x['checks']['parent_called'] for x in cs),'first_registered':sum(x['first_registered'] for x in cs),'variants':dict(Counter(x['variant'] for x in cs)),
                'same_contract_delegation':sum(x['same_contract_delegation'] for x in cs),'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in cs)/4,
                'mean_calls':sum(x['case']['usage']['policy_calls'] for x in cs)/4,'mean_nodes':sum(x['case']['node_count'] for x in cs)/4}
        pair={a:{x['rep']:int(x['case']['final_scores']['private']['status']=='PASS') for x in g['cases'] if x['arm']==a} for a in ['ATOMIC','SPLIT']}
        item['split_wins']=sum(pair['SPLIT'][i]>pair['ATOMIC'][i] for i in range(4));item['split_losses']=sum(pair['SPLIT'][i]<pair['ATOMIC'][i] for i in range(4))
        item['delta_success']=(item['SPLIT']['success']-item['ATOMIC']['success'])/4;summary.append(item)
    assert all(g['frozen']==groups[0]['frozen'] for g in groups)
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':32,'updates':0,'summary':summary,'groups':groups});print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.index)
    else:report()
