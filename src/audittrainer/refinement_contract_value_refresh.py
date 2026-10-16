"""Frozen A/new-SPLIT values for three previously used output-contract states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path('runs/refinement/contract_value_refresh')
DATA=Path('data/refinement/contract_value_refresh')
SOURCE=Path('runs/refinement/contract_correction')
EXECUTOR=str(SOURCE/'training/final_adapter')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')

from . import refinement_transfer_refresh as loader
from .refinement_cross_state import BranchInternal
from .refinement_internal_gate import NODE,fingerprints
from .refinement_post_handoff_values import same_contract_delegation
from .refinement_responsibility_course import allocation
from .refinement_interface_gate import FixedPlanner
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,capture_program,score_program
from .refinement_rollout import runtime_for,run_case
from .refinement_responsibility_course import CourseReference
from .verifier import canonical

NAMES=['stock_absolute','coverage_exact_two','polynomial_difference']


def prepare():
    assert not (ROOT/'request.json').exists(), 'request.json already exists; frozen design will not be overwritten'
    old=json.loads((SOURCE/'request.json').read_text())
    completed=json.loads((SOURCE/'result.json').read_text())
    assert completed['status']=='COMPLETE' and completed['errors']==[]
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    source_rows=load_root_manifest(old['config']['root_manifest'])
    source_tests={x['root_id']:x for x in read_jsonl(old['config']['test_manifest'])}
    rows=[];tests=[];selection=[]
    for name in NAMES:
        rid='train_contract_correction_fresh_'+name
        row=source_rows[rid];test=source_tests[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(test['path'])==test['sha256']
        d=json.loads(Path(row['fixture_path']).read_text())
        ref=json.loads(Path(row['reference_path']).read_text())
        rows.append(row);tests.append(test)
        base=ROOT/rid;base.mkdir(parents=True,exist_ok=True)
        teacher,full=runtime_for(row,source_tests,old['framework'],
                                 CourseReference(d,ref,'recursive'),base/'teacher/runtime')
        teacher.run()
        scores,_=score_program(capture_program(teacher,base/'teacher/program'),full,rid)
        assert all(x['status']=='PASS' for x in scores.values())
        write_json(base/'teacher/result.json',scores)
        fixed=FixedPlanner(SimpleNamespace(cost={},identity={}),
                           {'N0':allocation(d['contract'],d,'fused')})
        runtime,_=runtime_for(row,source_tests,old['framework'],fixed,base/'decision')
        assert runtime.step().host_result.code=='PLAN_REGISTERED'
        assert runtime.step().host_result.code=='CHILD_ENTERED'
        observation=runtime.build_observation()
        assert observation['node_id']==NODE and observation['phase']=='NEW'
        event=next(e for e in read_jsonl(runtime.trace.path)
                   if e['event_kind']=='POLICY' and e['node_id']=='N0')
        write_json(base/'state.json',{'root_event':event,'observation':observation})
        selection.append({'root_id':rid,'state_path':str(base/'state.json'),
                          'contract':name,'family_id':row['family_id']})
    for label,values in [('roots',rows),('tests',tests)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    adapters={**old['adapters'],'executor':EXECUTOR,'initial':EXECUTOR}
    request={'framework':old['framework'],
             'config':{**old['config'],'root_manifest':str(DATA/'roots.jsonl'),
                       'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},
             'adapters':adapters,
             'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
             'selection':selection,'execution_seed':613000713,'repetitions':4,
             'cases':24,'updates':0,'arms':['ATOMIC','SPLIT'],
             'source_request':str(SOURCE/'request.json'),
             'scope':'Three already-used new-output development contracts. Fixed registered root; only first internal compute choice is intervened. Not sealed generalization or natural gate performance.',
             'design':'Freeze contract_correction executor, root/internal gates, MULTI_BODY and host. For each state compare forced ATOMIC with freshly sampled current-BODY SPLIT using four new paired seeds. Keep rejects, retries, failures and final root scores.',
             'decision':'Use current-policy mean A/S root success and cost. Absolute success difference <=1/4 remains uncertain; >=2/4 is exploratory direction. No parameter update or gate label in this run.'}
    write_json(ROOT/'request.json',request)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1'],
                                          'min_free_mib':{'collect':16000}})


def check():
    r=json.loads((ROOT/'request.json').read_text())
    assert r['updates']==0 and r['selection'] and r['cases']==(
        len(r['selection'])*len(r['arms'])*r['repetitions'])
    assert r['adapters']['executor']==EXECUTOR
    assert r['adapters']['internal_gate']=='runs/refinement/transfer_gate/training/final_adapter'
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k]
               for k,p in r['adapters'].items())
    rows=load_root_manifest(r['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    for selected in r['selection']:
        rid=selected['root_id'];row=rows[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(tests[rid]['path'])==tests[rid]['sha256']
        teacher=json.loads((ROOT/rid/'teacher/result.json').read_text())
        assert all(x['status']=='PASS' for x in teacher.values())
        state=json.loads(Path(selected['state_path']).read_text())
        assert state['observation']['node_id']==NODE
    return r


def context():
    loader.ROOT=ROOT
    return loader.context()


def run(index):
    from tqdm import tqdm
    r=check();_,policy,executor,_,_=context()
    frozen=fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest'])
    tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    selected=r['selection'][index];rid=selected['root_id']
    state=json.loads(Path(selected['state_path']).read_text())
    base=ROOT/rid
    gate=policy.score_modes(state['observation'])
    prior=policy.body.score_modes(state['observation'])
    write_json(base/'gate.json',gate)
    write_json(base/'body_gate.json',prior)
    cases=[]
    jobs=[(rep,arm) for rep in range(r['repetitions']) for arm in r['arms']]
    for rep,arm in tqdm(jobs,desc=f'{selected["contract"]} A/S',unit='条',dynamic_ncols=True):
        branch=BranchInternal(policy,state,arm)
        directory=base/arm/f'repeat_{rep:02d}'
        cached=(directory/'case_result.json').exists()
        case=run_case(rows[rid],tests,r['framework'],r['config'],branch,executor,
                      directory,r['execution_seed']+index*100+rep)
        assert case['status']=='COMPLETE' or case['errors']==['SCORING_ERROR']
        if not cached:assert branch.root_sent and branch.internal_sent
        events=read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
        first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==NODE)
        assert first['observation_before']==state['observation']
        mode=first['model_input']['refinement']['selected_mode']
        assert mode==arm
        action=first['parsed_action']
        cases.append({'arm':arm,'rep':rep,'mode':mode,'case':case,
                      'first_registered':first['host_result']['code']=='PLAN_REGISTERED',
                      'same_contract_delegation':arm=='SPLIT' and
                          same_contract_delegation({'parsed_action':action},
                                                   state['observation']['incoming_contract'])})
    assert fingerprints(policy)==frozen
    write_json(base/'result.json',{'status':'COMPLETE','root_id':rid,
        'frozen':frozen,'p_split':gate['mode_probabilities']['SPLIT'],
        'body_prior_p_split':prior['mode_probabilities']['SPLIT'],'cases':cases})


def report():
    r=check();groups=[json.loads((ROOT/s['root_id']/'result.json').read_text())
                      for s in r['selection']]
    assert all(g['frozen']==groups[0]['frozen'] for g in groups)
    summary=[];pairs=[];errors=[];rejects={arm:Counter() for arm in r['arms']}
    for g in groups:
        item={k:g[k] for k in ['root_id','p_split','body_prior_p_split']}
        state=json.loads((ROOT/g['root_id']/'state.json').read_text())
        per_arm={}
        for arm in r['arms']:
            cases=[x for x in g['cases'] if x['arm']==arm]
            assert len(cases)==r['repetitions']
            per_arm[arm]={x['rep']:x for x in cases}
            item[arm]={'n':len(cases),
                       'success':sum(x['case']['final_scores']['private']['status']=='PASS' for x in cases),
                       'mean_tokens':sum(x['case']['usage']['input_tokens']+
                                         x['case']['usage']['output_tokens'] for x in cases)/len(cases),
                       'registered':sum(x['first_registered'] for x in cases),
                       'same_contract_delegations':sum(x['same_contract_delegation'] for x in cases)}
            for x in cases:
                case=x['case'];rejects[arm].update(case['protocol_rejections'])
                if case['errors']:
                    errors.append({'root_id':g['root_id'],'arm':arm,'rep':x['rep'],
                                   'errors':case['errors']})
                events=read_jsonl(Path(case['run_dir'])/'runtime/trace.jsonl')
                first=next(e for e in events if e.get('event_kind')=='POLICY' and e['node_id']==NODE)
                root=next(e for e in events if e.get('event_kind')=='POLICY' and e['node_id']=='N0')
                assert first['observation_before']==state['observation']
                assert root['parsed_action']==state['root_event']['parsed_action']
        wins=losses=0
        for rep in range(r['repetitions']):
            a=per_arm['ATOMIC'][rep]['case'];s=per_arm['SPLIT'][rep]['case']
            assert a['seed']==s['seed']
            ay=a['final_scores']['private']['status']=='PASS'
            sy=s['final_scores']['private']['status']=='PASS'
            wins+=sy>ay;losses+=sy<ay
            pairs.append({'root_id':g['root_id'],'rep':rep,'seed':a['seed'],
                          'same_observation_root_plan':True})
        delta=item['SPLIT']['success']-item['ATOMIC']['success']
        item.update(delta_success=delta/r['repetitions'],split_wins=wins,
                    split_losses=losses,
                    direction='UNCERTAIN' if abs(delta)<=1 else
                    'SPLIT' if delta>0 else 'ATOMIC')
        summary.append(item)
    n_pairs=len(r['selection'])*r['repetitions']
    totals={arm:{'n':n_pairs,'success':sum(s[arm]['success'] for s in summary),
                 'mean_tokens':sum(s[arm]['mean_tokens'] for s in summary)/len(summary),
                 'rejections':dict(rejects[arm])} for arm in r['arms']}
    paired={'wins':sum(s['split_wins'] for s in summary),
            'losses':sum(s['split_losses'] for s in summary)}
    paired['ties']=n_pairs-paired['wins']-paired['losses']
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':r['cases'],'updates':0,
        'summary':summary,'totals':totals,'paired':paired,'groups':groups,'errors':errors})
    write_json(ROOT/'validation.json',{'pairs':pairs,'all_frozen':True,
        'errors':errors,'updates':0})
    print(json.dumps({'summary':summary,'totals':totals,'paired':paired,
                      'errors':errors},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','run','report'])
    parser.add_argument('--index',type=int,choices=range(3),default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare();check()
    elif args.command=='check':
        print(json.dumps({'status':'READY','cases':check()['cases'],
                          'model_loaded':False}))
    elif args.command=='run':run(args.index)
    else:report()
