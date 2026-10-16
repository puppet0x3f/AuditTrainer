"""Offline staged contract repairs on all eight failed transfer SPLIT artifacts."""
import ast
import difflib
import json
from pathlib import Path
from .artifacts import ArtifactStore
from .dataset import load_root_manifest, read_jsonl, file_hash
from .evaluate import load_task, score_program, write_json
from .schema import Contract, Dependency
from .refinement_compute_repair import WRAPPER, NODE
from .refinement_internal_transfer_tasks import VARIANTS

ROOT=Path('runs/refinement/transfer_repair')
PRIOR=Path('runs/refinement/internal_transfer')
PREPARE=NODE+'_p1_prepare'


def replace_function(source, name, replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])


def repair_prepare(source, rep):
    # Retain the original per-case structure while correcting product identity,
    # empty factors, convolution, and coefficient normalization.
    source=source.replace('    result = {}\n','    result = {0: 1}\n',1)
    if rep in (0,2,3):
        source=source.replace('        if not polynomial:\n            continue',
            "        if not polynomial:\n            return {'terms': [], 'derivative': items['derivative'], 'points': list(items['points'])}",1)
    if rep==0:
        source=source.replace('        for degree in sorted(terms):\n            result.setdefault(degree, 0)\n            result[degree] += terms[degree]',
            '        new = {}\n        for a, ca in result.items():\n            for b, cb in terms.items():\n                new[a+b] = new.get(a+b, 0)+ca*cb\n        result = new')
    elif rep==1:
        source=source.replace('        if not result:\n            result = dict(polynomial)\n        else:\n            new = {}\n            for a, ca in result.items():\n                for b, cb in polynomial:\n                    new[a+b] = new.get(a+b, 0)+ca*cb\n            result = new',
            '        new = {}\n        for a, ca in result.items():\n            for b, cb in polynomial:\n                new[a+b] = new.get(a+b, 0)+ca*cb\n        result = new')
        source=source.replace('[[a, result[a]] for a in sorted(result)]','[[a, result[a]] for a in sorted(result) if result[a]]')
    elif rep==2:
        source=replace_function(source,'_multiply','''def _multiply(a, b):
    result = {}
    for i, ca in a.items():
        for j, cb in b.items():
            result[i+j] = result.get(i+j, 0)+ca*cb
    return result
''')
    else:
        assert rep==3
        source=source.replace('        for degree, coefficient in result.items():\n            terms[degree] = terms.get(degree, 0)+coefficient\n        result = {k:v for k,v in terms.items() if v}',
            '        new = {}\n        for a, ca in result.items():\n            for b, cb in terms.items():\n                new[a+b] = new.get(a+b, 0)+ca*cb\n        result = {k:v for k,v in new.items() if v}')
    return source


def repair_aggregate(source, variant, rep):
    code=VARIANTS[variant][2].replace('def aggregate(', 'def _aggregate(',1)
    fixed=replace_function(source,'_aggregate',code)
    if variant=='polynomial_joint' and rep==2:
        bad="    result['terms'] = [[d, terms[d]] for d in sorted(terms) if terms[d]]\n"
        assert fixed.count(bad)==1
        fixed=fixed.replace(bad,'')
    return fixed


def run():
    ROOT.mkdir(parents=True,exist_ok=False)
    request=json.loads((PRIOR/'request.json').read_text());prior=json.loads((PRIOR/'result.json').read_text())
    rows=load_root_manifest(request['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(request['config']['test_manifest'])}
    result_hash=file_hash(PRIOR/'result.json');records=[]
    write_json(ROOT/'request.json',{'source':str(PRIOR),'source_result_sha256':result_hash,'selection':'All four forced SPLIT failures for each of coverage_redundancy and polynomial_joint; no outcome filtering within these two tasks.',
        'conditions':{'coverage_redundancy':['ORIGINAL','AGGREGATE_ONLY'],'polynomial_joint':['ORIGINAL','PREPARE_ONLY','AGGREGATE_ONLY','BOTH']},
        'scope':'Offline human causal intervention; unchanged plans, contracts, retained code and root source. Missing root code gets identical explicit wrapper in every condition. Bind dependent artifact versions/hashes consistently; not autonomous consume/protocol replay or model score.',
        'model_calls':0,'updates':0,'adapters':request['adapters'],'hashes':request['hashes']})
    for group in prior['groups']:
        variant=group['root_id'].removeprefix('validation_internal_transfer_')
        if variant not in ('coverage_redundancy','polynomial_joint'):continue
        rid=group['root_id']
        for x in group['cases']:
            if x['arm']!='SPLIT':continue
            c=x['case'];rep=x['rep'];assert c['final_scores']['private']['status']!='PASS'
            original=Path(c['run_dir'])/'runtime';trace_hash=file_hash(original/'trace.jsonl');events=read_jsonl(original/'trace.jsonl')
            contracts={e['node_id']:Contract.model_validate(e['observation_before']['incoming_contract']) for e in events if e['host_result']['code']=='SOURCE_SAVED'}
            contracts['N0']=Contract.model_validate(next(e['observation_before']['incoming_contract'] for e in events if e['node_id']=='N0'))
            store=ArtifactStore(original);nodes=sorted(contracts,key=lambda n:(-n.count('_'),n));old={n:store.load(n,store.latest_version(n)) for n in nodes if store.latest_version(n)}
            sources={n:(original/a.module_path).read_text() for n,a in old.items()};root_present='N0' in sources
            if not root_present:sources['N0']=WRAPPER
            patches={NODE:repair_aggregate(sources[NODE],variant,rep)}
            if variant=='polynomial_joint':patches[PREPARE]=repair_prepare(sources[PREPARE],rep)
            base=ROOT/rid/f'repeat_{rep:02d}';base.mkdir(parents=True)
            for n,code in patches.items():
                assert code!=sources[n];ast.parse(code)
                (base/f'{n}.diff').write_text(''.join(difflib.unified_diff(sources[n].splitlines(True),code.splitlines(True),fromfile=n+'/ORIGINAL',tofile=n+'/PATCHED')))
            _,_,_,_,verifier=load_task(rows[rid],tests,request['framework'])
            condition_targets={'ORIGINAL':[],'AGGREGATE_ONLY':[NODE]}
            if variant=='polynomial_joint':condition_targets={'ORIGINAL':[],'PREPARE_ONLY':[PREPARE],'AGGREGATE_ONLY':[NODE],'BOTH':[PREPARE,NODE]}
            conditions={}
            for name,targets in condition_targets.items():
                out=base/name;out.mkdir();new_store=ArtifactStore(out);artifacts={}
                for n in nodes:
                    old_deps=old[n].dependency_bindings if n in old else [Dependency(child_id=NODE,artifact_version=old[NODE].version,artifact_hash=old[NODE].source_hash,alias='_compute_child')]
                    deps=[Dependency(child_id=d.child_id,artifact_version=artifacts[d.child_id].version,artifact_hash=artifacts[d.child_id].source_hash,alias=d.alias) for d in old_deps]
                    artifacts[n]=new_store.save_initial(n,contracts[n],patches[n] if n in targets else sources[n],deps)
                local={}
                for n in nodes:
                    if n=='N0':continue
                    local[n]=[]
                    for suite in verifier.suites.values():
                        if suite.check_kind=='LOCAL' and suite.applies(rid,contracts[n]):
                            ev=verifier.verify(out,rid,contracts[n],artifacts[n],suite.suite_id,f'LOCAL_{n}_{suite.visibility}',allow_private=True)
                            local[n].append({'visibility':suite.visibility,'status':ev.status,'cases':ev.details['executed_cases']})
                scores,_=score_program({'path':str(out),'contract':contracts['N0'].model_dump(mode='json'),'artifact':artifacts['N0'].model_dump(mode='json')},verifier,rid)
                usage=[]
                for suite in scores['private']['suites']:usage.extend(json.loads(Path(suite['evidence_path']).read_text())['details'].get('dependency_usage',[]))
                conditions[name]={'local':local,'root_scores':scores,'root_called_compute':any(u['child_id']==NODE and u.get('called') for u in usage),'source_hashes':{n:a.source_hash for n,a in artifacts.items()},'changed_nodes':targets}
                changed={n for n in nodes if conditions['ORIGINAL']['source_hashes'][n]!=conditions[name]['source_hashes'][n]}
                assert changed==set(targets)
                assert all(v['status']!='ERROR' for v in scores.values())
            assert file_hash(original/'trace.jsonl')==trace_hash
            assert all(file_hash(original/a.module_path)==a.source_hash for a in old.values())
            record={'root_id':rid,'variant':variant,'rep':rep,'source':str(original),'original_root_present':root_present,'conditions':conditions}
            write_json(base/'result.json',record);records.append(record)
            print(json.dumps({'variant':variant,'rep':rep,'original_root_present':root_present,'root':{k:v['root_scores']['private']['status'] for k,v in conditions.items()}}),flush=True)
    assert len(records)==8 and file_hash(PRIOR/'result.json')==result_hash
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==request['hashes'][k] for k,p in request['adapters'].items())
    summary=[]
    for variant in ['coverage_redundancy','polynomial_joint']:
        selected=[x for x in records if x['variant']==variant];item={'variant':variant,'n':4,'existing_roots':sum(x['original_root_present'] for x in selected),'conditions':{}}
        for name in selected[0]['conditions']:
            item['conditions'][name]={'root_pass':sum(x['conditions'][name]['root_scores']['private']['status']=='PASS' for x in selected),'existing_root_pass':sum(x['original_root_present'] and x['conditions'][name]['root_scores']['private']['status']=='PASS' for x in selected),
              'prepare_private_pass':sum(any(t['visibility']=='PRIVATE' and t['status']=='PASS' for t in x['conditions'][name]['local'][PREPARE]) for x in selected),
              'compute_private_pass':sum(any(t['visibility']=='PRIVATE' and t['status']=='PASS' for t in x['conditions'][name]['local'][NODE]) for x in selected),
              'root_called_compute':sum(x['conditions'][name]['root_called_compute'] for x in selected)}
        summary.append(item)
    write_json(ROOT/'result.json',{'status':'COMPLETE','model_calls':0,'updates':0,'conditions':24,'original_results_preserved':True,'adapters_preserved':True,'summary':summary,'results':records})
    print(json.dumps(summary))

if __name__=='__main__':run()
