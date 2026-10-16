"""Executor-only nested binding SFT with autonomous and retention controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path('runs/refinement/handoff_sft');DATA=Path('data/refinement/handoff_sft')
INITIAL='runs/refinement/executor_sft/training/final_adapter'
GATE='runs/refinement/gate_refresh/training/final_adapter'
BODY='runs/refinement/body_transfer/MULTI_BODY/final_adapter'
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,capture_program,score_program
from .refinement_generalization import guarded_json
from .refinement_rollout import runtime_for,run_case
from .refinement_executor_sft import NEW_INPUTS,train as executor_train
from .refinement_responsibility_course import CourseReference,allocation
from .refinement_policy import atomic_plan,load_policies,load_adapter_metadata
from .refinement_separate_gate import RootGatePolicy,fingerprints
from .refinement_interface_gate import FixedPlanner
from .refinement_execution_repair import local_checks
from .refinement_handoff import Replay
from .training_tasks import oracle_a,oracle_b
from .verifier import canonical


def nested_plan(obs,definition):
    contract=obs['incoming_contract'];node=obs['node_id']
    if node=='N0':return allocation(contract,definition,'pipeline')
    action=atomic_plan(contract)
    if node.count('_p1_')==1:
        c={k:v for k,v in contract.items() if k not in {'contract_id','version'}}
        key=contract['signature']['name']
        action['args'].update(mode='SPLIT',children=[{'local_key':key,'contract':c}],
            obligation_map={o['id']:['child:'+key] for o in contract['ensures']},retained_responsibilities=[],dataflow=[])
    return action


class NestedTeacher(CourseReference):
    def decide(self,obs):
        if obs['phase']=='NEW':return 'plan',nested_plan(obs,self.d)['args']
        return super().decide(obs)
    def source(self,obs):
        symbol=obs['incoming_contract']['signature']['name'];deps=obs['approved_dependencies']
        if symbol in {'prepare','aggregate'} and deps:
            assert len(deps)==1;d=deps[0];arg=obs['incoming_contract']['signature']['parameters'][0]['name']
            return f"from nodes.{d['child_id']} import {symbol} as {d['alias']}\n\ndef {symbol}({arg}):\n    return {d['alias']}({arg})\n"
        return super().source(obs)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads(Path('runs/refinement/executor_sft/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest']);tests={t['root_id']:t for t in read_jsonl(old['config']['test_manifest'])};fit=[]
    for first in ['polynomial','paths']:
        source=rows[f'train_execution_{first}_matrix_05'];rid=f'train_handoff_{first}_matrix_11'
        d=json.loads(Path(source['fixture_path']).read_text());private=json.loads(Path(tests[source['root_id']]['path']).read_text());ref=json.loads(Path(source['reference_path']).read_text())
        d=json.loads(json.dumps(d).replace(source['root_id'],rid).replace('divisible by 5','divisible by 11'))
        private=json.loads(json.dumps(private).replace(source['root_id'],rid))
        ref['aggregate']=ref['aggregate'].replace('% 5','% 11');ref['parameter']=11
        inputs=deepcopy(NEW_INPUTS[first])
        if first=='polynomial': inputs=[[[degree,coefficient*11+1] for degree,coefficient in row] for row in inputs]
        else:inputs=[[path+'/newpart' for path in row] for row in inputs]
        ns={};exec(ref['prepare']+'\n'+ref['aggregate'],ns)
        for suite in d['suites']+private['suites']:
            key=suite.get('contract_key','root');cases=[]
            for i in ([0,1] if suite['visibility']=='VISIBLE' else [2,3,4,5]):
                items=inputs[i];raw=oracle_a(first,items);prepared=[x[1] for x in raw] if first=='polynomial' else [len(x) for x in raw]
                answer=oracle_b('matrix',prepared,11)
                assert ns['prepare'](deepcopy(items))==prepared and ns['aggregate'](deepcopy(prepared))==answer
                cases.append({'case_id':f'{key}_{i}','args':[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items],
                    'expected':json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer})
            suite['cases']=cases
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for name,obj in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{name}.json',obj)
        rows[rid]={**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'handoff_sft_independent_train_variant','original_split':'TRAIN','io_type':'solve_text',
            'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')};fit.append(rid)
    targets=[]
    for rid in fit:
        row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text());ref=json.loads(Path(row['reference_path']).read_text())
        runtime,full=runtime_for(row,tests,old['framework'],NestedTeacher(d,ref,'pipeline'),ROOT/'teacher'/rid/'runtime');result=runtime.run()
        scores,_=score_program(capture_program(runtime,ROOT/'teacher'/rid/'program'),full,rid);assert scores['private']['status']=='PASS'
        write_json(ROOT/'teacher'/rid/'result.json',{'scores':scores,'runtime':result})
        for e in read_jsonl(runtime.trace.path):
            a=e.get('parsed_action');obs=e['observation_before']
            if a and (a['action']=='consume' or a['action']=='write_code' and obs['node_id'].count('_p1_')==1):
                assert e['host_result']['status']=='APPLIED'
                targets.append({'sample_id':rid+'_'+e['event_id'],'root_id':rid,'split':'TRAIN','kind':'target','observation':obs,'target_action':a})
    assert len(targets)==12
    retention=[{**x,'kind':'retention'} for x in read_jsonl('data/refinement/executor_sft/training.jsonl')]
    for name,values in [('roots',list(rows.values())),('tests',list(tests.values())),('training',targets+retention)]:
        (DATA/f'{name}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    config=deepcopy(old['config']);config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),output_dir=str(ROOT))
    replay=json.loads(Path('runs/refinement/handoff/request.json').read_text())['selection']
    eval_roots=fit+['train_execution_polynomial_matrix_05','train_executor_check_paths_matrix_07']
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'fit_roots':fit,'recursive_roots':eval_roots,'replay':replay,
        'retention_roots':['train_execution_polynomial_matrix_05','train_executor_check_paths_matrix_07','train_polynomial_03','train_paths_03'],
        'initial_executor':INITIAL,'gate':GATE,'body':BODY,'steps':32,'learning_rate':2e-5,'seed':421000713,'eval_seed':422000713,
        'target_examples':12,'retention_examples':24,'training_sha256':file_hash(DATA/'training.jsonl'),
        'source_hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)]},
        'objective':'32 steps, two complete consume/wrapper targets plus two retention actions per update, token-mean CE per example. Original prompts; no clarified summary or action rewriting.',
        'evaluation':'Before/after: 8 frozen replay prefixes (qualified or missing artifact), 4 recursive roots x2 seeds, 4 atomic-prepare/original-family roots x2 seeds: 24/model,48 total. Reuse five exact BEFORE original-prompt cases; run 43 new continuations.',
        'scope':'New TRAIN mod11/input variants for teacher only. Previous failure states are development evaluation, not heldout family generalization. Fixed final checkpoint, no evaluation selection.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})


class NestedPlanner:
    def __init__(self,base,definition):self.base=base;self.d=definition;self.cost=base.cost;self.identity={**base.identity,'intervention':'fixed_nested_plan'}
    def act(self,obs):
        from .schema import PolicyOutput,TokenUsage
        return PolicyOutput(raw_text=canonical(nested_plan(obs,self.d)),token_usage=TokenUsage(),model_identity=self.identity)


def context(method):
    r=json.loads((ROOT/'request.json').read_text());adapter=INITIAL if method=='BEFORE' else str(ROOT/'training/final_adapter')
    base,executor=load_policies(r['framework'],GATE,adapter);base.model.load_adapter(BODY,adapter_name='body',is_trainable=False,local_files_only=True)
    base.model.set_adapter('default');base.model.requires_grad_(False);policy=RootGatePolicy(base.llm,BODY)
    _,metadata=load_adapter_metadata(Path(BODY),r['framework']['model'],base.tokenizer);policy.body.identity.update(metadata)
    executor.llm.identity['role']='executor_handoff_'+method
    return r,policy,executor,load_root_manifest(r['config']['root_manifest']),{x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}


def evaluate(method,shard):
    r,policy,executor,rows,tests=context(method);before=fingerprints(policy);results=[]
    if shard==0:
        for i,s in enumerate(r['replay']):
            if method=='BEFORE' and s['qualified']:
                p=Path('runs/refinement/handoff/ORIGINAL')/s['root_id']/f"repeat_{s['rep']:02d}"
                c=json.loads((p/'case_result.json').read_text());reused=True
            else:
                replay=Replay(policy,executor,s,'ORIGINAL');planner=SimpleNamespace(act=replay.act,cost=policy.cost,identity=policy.identity)
                execution=SimpleNamespace(act=replay.act,cost=executor.cost,llm=executor.llm)
                c=run_case(rows[s['root_id']],tests,r['framework'],r['config'],planner,execution,ROOT/'evaluation'/method/'replay'/s['root_id']/f"repeat_{s['rep']:02d}",411000713+i)
                assert replay.cursor>replay.boundary and not replay.interventions;reused=False
            checks=json.loads((p/'execution_checks.json').read_text()) if reused else local_checks(c,rows[s['root_id']],tests,r['framework'])
            results.append({'group':'replay_qualified' if s['qualified'] else 'replay_missing','case':c,'checks':checks,'reused':reused})
    else:
        for group,roots in [('recursive',r['recursive_roots']),('retention',r['retention_roots'])]:
            for i,rid in enumerate(roots):
                row=rows[rid];d=json.loads(Path(row['fixture_path']).read_text())
                if group=='recursive':planner=NestedPlanner(policy,d)
                else:
                    plan=atomic_plan(d['contract']) if rid in ['train_polynomial_03','train_paths_03'] else allocation(d['contract'],d,'parent_aggregate')
                    planner=FixedPlanner(policy,{'N0':plan})
                for rep in range(2):
                    c=run_case(row,tests,r['framework'],r['config'],planner,executor,ROOT/'evaluation'/method/group/rid/f'repeat_{rep:02d}',r['eval_seed']+(1000 if group=='retention' else 0)+i*100+rep)
                    checks=local_checks(c,row,tests,r['framework'])
                    results.append({'group':group,'case':c,'checks':checks,'fit':rid in r['fit_roots'],'reused':False})
    assert all(x['case']['status']=='COMPLETE' for x in results) and fingerprints(policy)==before
    write_json(ROOT/'evaluation'/method/f'shard_{shard}.json',{'status':'COMPLETE','results':results,'frozen':before})


def report():
    r=json.loads((ROOT/'request.json').read_text());allresults={};summary={}
    for method in ['BEFORE','AFTER']:
        pieces=[json.loads((ROOT/'evaluation'/method/f'shard_{i}.json').read_text()) for i in [0,1]]
        assert pieces[0]['frozen']==pieces[1]['frozen'];items=[x for p in pieces for x in p['results']];assert len(items)==24
        allresults[method]={'results':items,'frozen':pieces[0]['frozen']};summary[method]={}
        for group in ['replay_qualified','replay_missing','recursive','retention']:
            cs=[x for x in items if x['group']==group];consumed=0
            for x in cs:
                es=read_jsonl(Path(x['case']['run_dir'])/'runtime/trace.jsonl')
                consumed+=any(e['node_id']=='N0' and e['host_result']['code']=='CHILD_CONSUMED' and e['host_result']['details']['consumption']['child_node']=='N0_p1_prepare' for e in es)
            summary[method][group]={'n':len(cs),'root_success':sum(x['case']['reward']['y'] for x in cs),'direct_prepare_consumed':consumed}
    assert all(allresults['BEFORE']['frozen'][k]==allresults['AFTER']['frozen'][k] for k in ['default','body'])
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in [('initial',INITIAL),('gate',GATE),('body',BODY)])
    out={'status':'COMPLETE','summary':summary,'evaluation':allresults,'training':json.loads((ROOT/'training/result.json').read_text())};write_json(ROOT/'result.json',out);print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','report']);p.add_argument('--method',choices=['BEFORE','AFTER']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':executor_train(ROOT,DATA,INITIAL)
    elif a.command=='evaluate':evaluate(a.method,a.shard)
    else:report()
