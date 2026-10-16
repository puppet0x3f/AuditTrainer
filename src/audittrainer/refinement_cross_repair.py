"""User-started offline 3x4 responsibility isolation; no model loading."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import difflib
import json
from pathlib import Path

from .artifacts import ArtifactStore
from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import load_task,score_program,write_json
from .schema import Contract,Dependency
from .refinement_transfer_repair import replace_function
from .refinement_compute_repair import WRAPPER,NODE

ROOT=Path('runs/refinement/cross_repair')
SOURCE=Path('runs/refinement/cross_state')
PREPARE=NODE+'_p1_prepare'
SELECTION=[('stock_status',0),('coverage_single',0),('polynomial_signs',1)]
CONDITIONS={'ORIGINAL':[], 'PREPARE_ONLY':[PREPARE], 'AGGREGATE_ONLY':[NODE], 'BOTH':[PREPARE,NODE]}


def prepare():
    if (ROOT/'request.json').exists():return
    previous=json.loads((SOURCE/'request.json').read_text());result=json.loads((SOURCE/'result.json').read_text())
    assert result['status']=='COMPLETE' and not result['errors']
    rows=load_root_manifest(previous['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(previous['config']['test_manifest'])}
    ROOT.mkdir(parents=True,exist_ok=True);selections=[];input_hashes={str(SOURCE/'result.json'):file_hash(SOURCE/'result.json')}
    for variant,rep in SELECTION:
        rid='validation_cross_state_'+variant
        group=next(g for g in result['groups'] if g['root_id']==rid)
        c=next(x['case'] for x in group['cases'] if x['arm']=='SPLIT' and x['rep']==rep)
        assert c['final_scores']['private']['status']!='PASS'
        runtime=Path(c['run_dir'])/'runtime';events=read_jsonl(runtime/'trace.jsonl')
        contracts={e['node_id']:Contract.model_validate(e['observation_before']['incoming_contract']).model_dump(mode='json') for e in events if e.get('host_result',{}).get('code')=='SOURCE_SAVED'}
        contracts['N0']=Contract.model_validate(next(e['observation_before']['incoming_contract'] for e in events if e['node_id']=='N0')).model_dump(mode='json')
        assert set(contracts)=={'N0',NODE,PREPARE}
        store=ArtifactStore(runtime);artifacts={n:store.load(n,store.latest_version(n)) for n in contracts if store.latest_version(n)}
        sources={n:(runtime/a.module_path).read_text() for n,a in artifacts.items()};root_present='N0' in sources
        if not root_present:sources['N0']=WRAPPER
        ref=json.loads(Path(rows[rid]['reference_path']).read_text())
        patches={PREPARE:replace_function(sources[PREPARE],'prepare',ref['prepare']),NODE:replace_function(sources[NODE],'_aggregate',ref['aggregate'].replace('def aggregate(','def _aggregate(',1))}
        dependencies={n:[d.model_dump(mode='json') for d in a.dependency_bindings] for n,a in artifacts.items()}
        if not root_present:dependencies['N0']=[Dependency(child_id=NODE,artifact_version=artifacts[NODE].version,artifact_hash=artifacts[NODE].source_hash,alias='_compute_child').model_dump(mode='json')]
        scenario={'root_id':rid,'variant':variant,'rep':rep,'source_runtime':str(runtime),'original_root_present':root_present,
                  'original_root_status':c['final_scores']['private']['status'],'contracts':contracts,'sources':sources,'patches':patches,'dependencies':dependencies,
                  'row':rows[rid],'tests':tests[rid]}
        folder=ROOT/rid;folder.mkdir(exist_ok=True)
        for n,code in patches.items():
            ast.parse(code)
            (folder/f'{n}.diff').write_text(''.join(difflib.unified_diff(sources[n].splitlines(True),code.splitlines(True),fromfile=n+'/ORIGINAL',tofile=n+'/REFERENCE_FUNCTION')))
        path=folder/'scenario.json';write_json(path,scenario)
        selections.append({'root_id':rid,'variant':variant,'rep':rep,'scenario_path':str(path),'original_root_present':root_present})
        for f in [runtime/'trace.jsonl',Path(c['run_dir'])/'case_result.json',Path(rows[rid]['fixture_path']),Path(rows[rid]['reference_path']),Path(tests[rid]['path']),*[runtime/a.module_path for a in artifacts.values()]]:
            input_hashes[str(f)]=file_hash(f)
    write_json(ROOT/'request.json',{'source':str(SOURCE),'selection':selections,'conditions':CONDITIONS,'program_conditions':12,'framework':previous['framework'],
        'input_hashes':input_hashes,'scenario_hashes':{s['scenario_path']:file_hash(s['scenario_path']) for s in selections},
        'model_calls':0,'updates':0,'cpu_workers':3,'adapters':previous['adapters'],'hashes':previous['hashes'],
        'design':'Prespecified representative SPLIT failures, same parent_aggregate topology. Replace only prepare or _aggregate function; identical missing-root wrapper across conditions. Preserve all other source/contracts/tests. Rebind transitive hashes with identical aliases. Full reference functions, not minimal line patches.',
        'scope':'Offline human functional causal diagnosis, not autonomous generation/consumption recovery, model scores, success-rate estimate, or heldout evaluation. Two existing roots, one reconstruction. No adaptive repair if BOTH still fails.'})


def check():
    r=json.loads((ROOT/'request.json').read_text())
    assert r['program_conditions']==12 and r['conditions']==CONDITIONS
    assert all(file_hash(p)==h for p,h in {**r['input_hashes'],**r['scenario_hashes']}.items())
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    return r


def condition_sources(scenario,condition):
    sources=dict(scenario['sources'])
    for n in CONDITIONS[condition]:sources[n]=scenario['patches'][n]
    return sources


def execute(job):
    # Worker loads only CPU verifier and saved code, never a model or tokenizer.
    scenario_path,condition=job
    r=json.loads((ROOT/'request.json').read_text());s=json.loads(Path(scenario_path).read_text());rid=s['root_id']
    out=ROOT/'evaluations'/rid/condition;out.mkdir(parents=True,exist_ok=False)
    store=ArtifactStore(out);artifacts={};sources=condition_sources(s,condition)
    nodes=sorted(sources,key=lambda n:(-n.count('_'),n));contracts={n:Contract.model_validate(c) for n,c in s['contracts'].items()}
    for n in nodes:
        deps=[Dependency(child_id=d['child_id'],artifact_version=artifacts[d['child_id']].version,artifact_hash=artifacts[d['child_id']].source_hash,alias=d['alias']) for d in s['dependencies'][n]]
        artifacts[n]=store.save_initial(n,contracts[n],sources[n],deps)
    _,_,_,_,verifier=load_task(s['row'],{rid:s['tests']},r['framework'])
    local={}
    for n in nodes:
        if n=='N0':continue
        local[n]=[]
        for suite in verifier.suites.values():
            if suite.check_kind=='LOCAL' and suite.applies(rid,contracts[n]):
                ev=verifier.verify(out,rid,contracts[n],artifacts[n],suite.suite_id,f'LOCAL_{n}_{suite.visibility}',allow_private=True)
                local[n].append({'visibility':suite.visibility,'status':ev.status,'cases':ev.details.get('executed_cases',0)})
    scores,_=score_program({'path':str(out),'contract':contracts['N0'].model_dump(mode='json'),'artifact':artifacts['N0'].model_dump(mode='json')},verifier,rid)
    usage=[]
    for suite in scores['private']['suites']:usage.extend(json.loads(Path(suite['evidence_path']).read_text())['details'].get('dependency_usage',[]))
    record={'root_id':rid,'condition':condition,'original_root_present':s['original_root_present'],'local':local,'root_scores':scores,
        'root_called_compute':any(u['child_id']==NODE and u.get('called') for u in usage),
        'replaced_functions_in_nodes':CONDITIONS[condition],
        'actually_changed_nodes':[n for n in nodes if sources[n]!=s['sources'][n]],'source_hashes':{n:a.source_hash for n,a in artifacts.items()}}
    assert set(record['actually_changed_nodes'])<=set(CONDITIONS[condition])
    write_json(out/'result.json',record)
    return record


def report():
    r=check();summary=[];errors=[]
    for selected in r['selection']:
        rid=selected['root_id'];s=json.loads(Path(selected['scenario_path']).read_text());conditions={}
        for condition in CONDITIONS:
            c=json.loads((ROOT/'evaluations'/rid/condition/'result.json').read_text())
            for n,source in condition_sources(s,condition).items():
                assert file_hash(ROOT/'evaluations'/rid/condition/'artifacts'/n/'v1.py')==c['source_hashes'][n]
            local=lambda node:any(e['visibility']=='PRIVATE' and e['status']=='PASS' for e in c['local'][node])
            conditions[condition]={'prepare_private_pass':local(PREPARE),'compute_private_pass':local(NODE),'root_status':c['root_scores']['private']['status'],'root_called_compute':c['root_called_compute'],'actually_changed_nodes':c['actually_changed_nodes']}
            if any(v['status']=='ERROR' for v in c['root_scores'].values()) or any(e['status']=='ERROR' for es in c['local'].values() for e in es):errors.append({'root_id':rid,'condition':condition})
        summary.append({**selected,'conditions':conditions})
    result={'status':'SCORING_ERRORS' if errors else 'COMPLETE','program_conditions':12,'model_calls':0,'updates':0,'summary':summary,'errors':errors,'original_results_preserved':True,'adapters_preserved':True}
    write_json(ROOT/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def run(workers):
    from tqdm import tqdm
    r=check()
    # Refuse to overwrite an earlier partial or completed experiment.
    (ROOT/'evaluations').mkdir(exist_ok=False)
    jobs=[(s['scenario_path'],c) for s in r['selection'] for c in CONDITIONS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        with tqdm(total=len(jobs),desc='离线职责诊断',unit='条件',dynamic_ncols=True) as progress:
            for f in as_completed(futures):
                c=f.result();progress.set_postfix_str(c['root_id'].removeprefix('validation_cross_state_')+' '+c['condition']+' '+c['root_scores']['private']['status']);progress.update(1)
    report()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','run','report']);p.add_argument('--workers',type=int,choices=[1,2,3],default=3);a=p.parse_args()
    if a.command=='prepare':prepare();check()
    elif a.command=='check':print(json.dumps({'status':'READY','conditions':check()['program_conditions'],'model_loaded':False,'evaluation_started':(ROOT/'evaluations').exists()}))
    elif a.command=='run':run(a.workers)
    else:report()
