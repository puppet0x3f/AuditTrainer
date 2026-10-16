"""Minimal artifact counterfactuals and 16 fixed-plan TRAIN continuations."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import yaml

ROOT=Path('runs/refinement/execution_repair')
DATA=Path('data/refinement/execution_repair')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')

from .artifacts import ArtifactStore
from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import load_task,score_program,write_json
from .refinement_fresh_gate import pack
from .refinement_generalization import guarded_json
from .refinement_interface_gate import FixedPlanner
from .refinement_parent_composition import summary_prompt
from .refinement_policy import atomic_plan
from .refinement_responsibility_course import allocation
from .refinement_rollout import run_case
from .refinement_separate_gate import load,fingerprints
from .schema import Contract,Artifact,Dependency
from .training_tasks import oracle_a,oracle_b
from .verifier import canonical

INPUTS={
 'polynomial': [[[3,2],[0,5],[3,-1],[1,10]], [[6,-5],[2,7],[0,0],[2,3]],
                [], [[5,4],[5,-4],[0,15]], [[4,-10],[1,6],[2,0]], [[6,1],[0,-5],[3,5],[1,10],[3,5]]],
 'paths': [['/cedar//leaf/','./oak','///'], ['a/b/c/d/e','a/./b','././'],
           [], ['a/b/c/d/e/f/g/h/i/j','x'], ['///./','./ash//leaf/./bud'], ['one/two/three/four/five','one','./','six/seven/eight/nine/ten']]
}
NOTES={
 'polynomial':'Current function prepare: the final return value must be a native list[int]. After combining equal degrees and removing zero coefficients, you must still extract ONLY the coefficient from each normalized pair, in ascending degree order. Do not return degree-coefficient pairs or JSON text.',
 'paths':'Current function prepare: the final return value must be a native list[int]. After normalizing each path into components, you must still convert each path to its number of components. Preserve input order; an empty path contributes 0. Do not return component lists or JSON text.'
}


def causal():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('runs/refinement/fresh_gate/request.json').read_text())
    rows=load_root_manifest(old['config']['root_manifest']);tests={r['root_id']:r for r in read_jsonl(old['config']['test_manifest'])};results=[]
    for first in ['polynomial','paths']:
        rid=f'validation_fresh_{first}_matrix_03';source=Path('runs/refinement/fresh_gate')/rid/'SPLIT/repeat_00'
        snapshot=json.loads((source/'final_program/snapshot.json').read_text())
        events=read_jsonl(source/'runtime/trace.jsonl')
        child_contract=Contract.model_validate(next(e['observation_before']['incoming_contract'] for e in events if e['node_id']=='N0_p1_prepare' and e['observation_before']['phase']=='IMPLEMENT'))
        parent_contract=Contract.model_validate(snapshot['contract'])
        code=(source/'final_program/artifacts/N0_p1_prepare/v1.py').read_text()
        parent=(source/'final_program/artifacts/N0/v1.py').read_text()
        old_text='return [[d, totals[d]] for d in sorted(totals) if totals[d] != 0]' if first=='polynomial' else "return [[part for part in path.split('/') if part not in ('', '.')] for path in items]"
        new_text='return [totals[d] for d in sorted(totals) if totals[d] != 0]' if first=='polynomial' else "return [len([part for part in path.split('/') if part not in ('', '.')]) for path in items]"
        assert code.count(old_text)==1
        _,_,_,_,full=load_task(rows[rid],tests,old['framework'])
        conditions={}
        for condition,child_code in [('ORIGINAL',code),('PATCHED',code.replace(old_text,new_text))]:
            out=ROOT/'causal'/first/condition;out.mkdir(parents=True,exist_ok=False)
            store=ArtifactStore(out);child=store.save_initial('N0_p1_prepare',child_contract,child_code)
            deps=[Dependency(child_id=child.node_id,artifact_version=child.version,artifact_hash=child.source_hash,alias='_prepare_child')]
            root=store.save_initial('N0',parent_contract,parent,deps)
            assert (out/root.module_path).read_text()==parent
            store.bundle(root).materialize(out/'program')
            local=[]
            for suite in full.suites.values():
                if suite.check_kind=='LOCAL' and suite.applies(rid,child_contract):
                    ev=full.verify(out,rid,child_contract,child,suite.suite_id,f'LOCAL_{suite.visibility}',allow_private=True)
                    local.append({'visibility':suite.visibility,'status':ev.status,'cases':ev.details['executed_cases']})
            scores,_=score_program({'path':str(out),'contract':parent_contract.model_dump(mode='json'),'artifact':root.model_dump(mode='json')},full,rid)
            conditions[condition]={'local':local,'root_scores':scores,'child_sha256':child.source_hash,'parent_sha256':root.source_hash}
        assert conditions['ORIGINAL']['parent_sha256']==conditions['PATCHED']['parent_sha256']
        results.append({'root_id':rid,'source':str(source),'edit':{'before':old_text,'after':new_text},'conditions':conditions})
    result={'status':'COMPLETE','scope':'Human one-line code intervention; not model performance. New dependency hashes rebound, all other source unchanged.','model_calls':0,'results':results}
    guarded_json(ROOT/'causal_result.json',result)
    print(json.dumps(result,ensure_ascii=False))


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    causal_=json.loads((ROOT/'causal_result.json').read_text())
    assert all(x['conditions']['PATCHED']['root_scores']['private']['status']=='PASS' and all(v['status']=='PASS' for v in x['conditions']['PATCHED']['local']) for x in causal_['results'])
    rows=[];tests=[];selections=[]
    for first,structure in [('polynomial','coefficient_summary'),('paths','path_length_summary')]:
        d,private,ref,_=pack(first,'matrix',structure)
        oldid=d['task']['root_id'];rid=f'train_execution_{first}_matrix_05'
        # New TRAIN variants: same mechanism/family, disjoint inputs and modulus 5.
        d=json.loads(json.dumps(d).replace(oldid,rid).replace('divisible by 3','divisible by 5'));d['task']['split']='TRAIN'
        ref['aggregate']=ref['aggregate'].replace('% 3','% 5');ref['parameter']=5
        refns={};exec(ref['prepare']+'\n'+ref['aggregate'],refns)
        for suites in [d['suites'],private['suites']]:
            for suite in suites:
                suite['suite_id']=suite['suite_id'].replace(oldid,rid);key=suite.get('contract_key','root')
                indices=[0,1] if suite['visibility']=='VISIBLE' else [2,3,4,5];new_cases=[]
                for i in indices:
                    items=deepcopy(INPUTS[first][i]);raw=oracle_a(first,items)
                    prepared=[pair[1] for pair in raw] if first=='polynomial' else [len(parts) for parts in raw]
                    answer=oracle_b('matrix',prepared,5)
                    assert refns['prepare'](deepcopy(items))==prepared and refns['aggregate'](deepcopy(prepared))==answer
                    args=[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items]
                    expected=json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer
                    new_cases.append({'case_id':f'{key}_{i}','args':args,'expected':expected})
                suite['cases']=new_cases
        # Keep the public optional contracts synchronized after replacing the modulus.
        folder=DATA/rid;folder.mkdir(exist_ok=True)
        for name,value in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{name}.json',value)
        rows.append({**{k:d['task'][k] for k in ['root_id','family_id','split']},'source':'execution_repair_train_v1','original_split':'TRAIN','io_type':'solve_text',
          'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')})
        tests.append({'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')})
        plan=allocation(d['contract'],d,'parent_aggregate');guarded_json(folder/'fixed_plan.json',plan)
        selections.append({'root_id':rid,'first':first,'plan_path':str(folder/'fixed_plan.json'),'clarification':NOTES[first]})
    for name,values in [('roots',rows),('tests',tests)]: (DATA/f'{name}.jsonl').write_text(''.join(canonical(x)+'\n' for x in values))
    old=json.loads(Path('runs/refinement/separate_gate/request.json').read_text());config=deepcopy(old['config'])
    config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),root_limit=2,output_dir=str(ROOT))
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'selection':selections,'seed':350000713,'planned_cases':16,
      'scope':'Independent TRAIN states with new inputs and modulus 5; same composition families as diagnostic cases, not held-out-family generalization.',
      'plan':'Fixed complete parent_aggregate root; prepare child fixed ATOMIC. Gate/body/F01 frozen; only current prepare IMPLEMENT prompt differs.',
      'adoption_rule':'Clarified total root successes strictly increase and neither task regresses; inspect local and real parent consumption too.',
      'budget':'2 states x 2 conditions x 4 paired seeds; no parameter updates; no subsequent SFT within this experiment.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'prepared':2,'planned_cases':16}))


def prompt_for(first,condition):
    def builder(tokenizer,observation,**kwargs):
        obs=deepcopy(observation)
        if condition=='CLARIFIED' and obs['phase']=='IMPLEMENT' and obs['node_id']=='N0_p1_prepare':
            assert obs['incoming_contract']['signature']['name']=='prepare'
            obs['root_spec']+='\nCurrent function final-output clarification:\n'+NOTES[first]
        return summary_prompt(tokenizer,obs,**kwargs)
    return builder


def local_checks(case,row,tests,framework):
    _,_,_,_,full=load_task(row,tests,framework);runtime=Path(case['run_dir'])/'runtime';events=read_jsonl(runtime/'trace.jsonl')
    child='N0_p1_prepare';writes=[e for e in events if e['node_id']==child and e['host_result']['code']=='SOURCE_SAVED']
    if not writes:return {'local':[],'consumed':False,'parent_called':False}
    contract=Contract.model_validate(writes[-1]['observation_before']['incoming_contract'])
    store=ArtifactStore(runtime);artifact=store.load(child,store.latest_version(child));local=[]
    for suite in full.suites.values():
        if suite.check_kind=='LOCAL' and suite.applies(row['root_id'],contract):
            ev=full.verify(runtime,row['root_id'],contract,artifact,suite.suite_id,f'POST_LOCAL_{suite.visibility}',allow_private=True)
            local.append({'visibility':suite.visibility,'status':ev.status,'cases':ev.details['executed_cases']})
    consumed=any(e['node_id']=='N0' and e['host_result']['code']=='CHILD_CONSUMED' for e in events)
    dependency=[]
    for s in case['final_scores']['private']['suites']:
        ev=json.loads(Path(s['evidence_path']).read_text());dependency.extend(ev['details'].get('dependency_usage',[]))
    return {'local':local,'consumed':consumed,'parent_called':any(d.get('called') for d in dependency),
            'dependency_usage':dependency,'child_sha256':artifact.source_hash}


def run(condition):
    study=yaml.safe_load(Path('configs/refinement_separate_gate.yaml').read_text());request=json.loads((ROOT/'request.json').read_text())
    _,_,policy,executor=load(study,'runs/refinement/separate_gate/CALIBRATED/final_adapter');before=fingerprints(policy)
    assert before==json.loads(Path('runs/refinement/separate_gate/CALIBRATED/result.json').read_text())['frozen_after']
    rows=load_root_manifest(request['config']['root_manifest']);tests={t['root_id']:t for t in read_jsonl(request['config']['test_manifest'])};cases=[]
    for i,selected in enumerate(request['selection']):
        row=rows[selected['root_id']];root_plan=json.loads(Path(selected['plan_path']).read_text())
        fixed=FixedPlanner(policy,{'N0':root_plan});executor.llm.prompt_builder=prompt_for(selected['first'],condition)
        for rep in range(4):
            case=run_case(row,tests,request['framework'],request['config'],fixed,executor,ROOT/'model'/condition/row['root_id']/f'repeat_{rep:02d}',request['seed']+i*100+rep,reset_seed_after_root_plan=True)
            if case['status']!='COMPLETE':raise ValueError('Infrastructure failure')
            events=read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
            for e in events:
                if e['host_result']['code']=='PLAN_REGISTERED':
                    assert e['parsed_action']==(root_plan if e['node_id']=='N0' else atomic_plan(e['observation_before']['incoming_contract']))
            checks=local_checks(case,row,tests,request['framework']);write_json(Path(case['run_dir'])/'execution_checks.json',checks)
            cases.append({**case,'execution_checks':checks})
    assert fingerprints(policy)==before
    write_json(ROOT/'model'/condition/'result.json',{'status':'COMPLETE','condition':condition,'cases':cases,'frozen':before})


def report():
    arms={a:json.loads((ROOT/'model'/a/'result.json').read_text()) for a in ['ORIGINAL','CLARIFIED']}
    assert arms['ORIGINAL']['frozen']==arms['CLARIFIED']['frozen']
    summaries=[]
    for selected in json.loads((ROOT/'request.json').read_text())['selection']:
        summary={'root_id':selected['root_id']};outcomes={}
        for a,r in arms.items():
            cs=sorted([c for c in r['cases'] if c['root_id']==selected['root_id']],key=lambda c:c['seed']);assert len(cs)==4
            outcomes[a]={c['seed']:c['final_scores']['private']['status']=='PASS' for c in cs}
            summary[a]={'root_success':sum(outcomes[a].values()),'local_visible':sum(any(e['visibility']=='VISIBLE' and e['status']=='PASS' for e in c['execution_checks']['local']) for c in cs),
              'local_private':sum(any(e['visibility']=='PRIVATE' and e['status']=='PASS' for e in c['execution_checks']['local']) for c in cs),
              'consumed':sum(c['execution_checks']['consumed'] for c in cs),'parent_called':sum(c['execution_checks']['parent_called'] for c in cs),
              'outcomes':[c['final_scores']['private']['status'] for c in cs],'rejections':[c['protocol_rejections'] for c in cs]}
        assert outcomes['ORIGINAL'].keys()==outcomes['CLARIFIED'].keys()
        summary['paired']={'wins':sum(outcomes['CLARIFIED'][k]>v for k,v in outcomes['ORIGINAL'].items()),'losses':sum(outcomes['CLARIFIED'][k]<v for k,v in outcomes['ORIGINAL'].items())}
        summaries.append(summary)
    adopt=all(s['CLARIFIED']['root_success']>=s['ORIGINAL']['root_success'] for s in summaries) and sum(s['CLARIFIED']['root_success']-s['ORIGINAL']['root_success'] for s in summaries)>0
    result={'status':'COMPLETE','model_cases':16,'optimizer_updates':0,'causal':json.loads((ROOT/'causal_result.json').read_text()),'by_root':summaries,'clarification_passed':adopt,'frozen':arms['ORIGINAL']['frozen']}
    write_json(ROOT/'result.json',result);print(json.dumps({'clarification_passed':adopt,'by_root':summaries},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['causal','prepare','run','report']);p.add_argument('--condition',choices=['ORIGINAL','CLARIFIED'],default='ORIGINAL');args=p.parse_args()
    if args.command=='causal':causal()
    elif args.command=='prepare':prepare()
    elif args.command=='run':run(args.condition)
    else:report()
