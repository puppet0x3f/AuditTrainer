"""Frozen-policy, preselected unseen-composition diagnostic (8 x 3 x 4)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import yaml

ROOT = Path('runs/refinement/fresh_gate')
DATA = Path('data/refinement/fresh_gate')
if __name__ == '__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')

from .dataset import file_hash, read_jsonl, load_root_manifest
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_heldout_tasks import INPUTS
from .refinement_separate_gate import load, fingerprints
from .refinement_rollout import runtime_for, run_case
from .training_tasks import FAMILIES, contract, definition, function, oracle_a, oracle_b
from .verifier import canonical

# Chosen by interface structure and absence from all prior manifests, before model execution.
TASKS=[('ledger','words','labels_filter'),('inventory','words','labels_filter'),
       ('ledger','paths','label_prefix'),('inventory','paths','label_prefix'),
       ('polynomial','matrix','coefficient_summary'),('polynomial','calendar','coefficient_summary'),
       ('paths','matrix','path_length_summary'),('paths','calendar','path_length_summary')]
EXTRA_INPUTS={
 'ledger': [[['elm',-8],['oak',3],['elm',2]], [['alpha',5],['alpine',-4],['beech',0]],
            [], [[' A ',1],['a',-1]], [['cedar',0],['birch',9],['cedar',-9]], [['oak',0],['ox',1],['oat',-2],['oak',4]]],
 'inventory': [[['elm',8,1],['oak',2,6],['elm',0,2]], [['alpha',1,1],['alpine',4,8],['beech',0,0]],
               [], [[' A ',0,1],['a',1,0]], [['cedar',2,1],['birch',0,9],['cedar',0,1]], [['oak',0,0],['ox',1,0],['oat',0,2],['oak',4,0]]]
}


def pack(first,second,structure):
    p=3; a=FAMILIES[first]; b=FAMILIES[second]
    if structure in {'labels_filter','label_prefix'}:
        bridge='Then return the dictionary keys as a lexicographically sorted list of strings, preserving case and whitespace; discard the numeric values.'
        expr='sorted(base)'; convert=lambda base: sorted(base)
    elif structure=='coefficient_summary':
        bridge='Then return only the coefficient from each normalized pair, in ascending degree order, as a list of integers.'
        expr='[pair[1] for pair in base]'; convert=lambda base:[pair[1] for pair in base]
    else:
        bridge='Then return the number of components in each normalized path, retaining input order and empty paths as zero, as a list of integers.'
        expr='[len(parts) for parts in base]'; convert=lambda base:[len(parts) for parts in base]
    prep=a['prepare']+' '+bridge
    goal=b['aggregate'].format(p=p,p15=p*15)
    if second=='paths':
        goal="Treat each prepared label as a sequence of individual characters, including whitespace. Take at most the first 3 characters and join with '/'. Count these prefixes and return sorted prefixes and parallel counts."
    elif second=='matrix':
        goal='Select positions whose integer values are divisible by 3 (Python modulo). Return their zero-based indices and the sum of selected values as total.'
    elif second=='calendar':
        goal='Consider adjacent integer values in the supplied order without sorting. Return breaks with differences (next minus previous) >= 45 as pairs and max_gap across all adjacent differences, which may be negative, or 0 with fewer than two values.'
    root_id=f'validation_fresh_{first}_{second}_03'
    root=contract('solve','text',{'parse':'Parse the JSON input.','prepare':prep,'aggregate':goal,
          'format':"Return compact JSON with separators=(',', ':') and no newline."},
          ['parse','prepare','aggregate','format'],'Input text encodes '+a['input'],'str')
    local={'prepare':contract('prepare','items',{'prepare':prep},['prepare'],a['input']),
           'aggregate':contract('aggregate','prepared',{'aggregate':goal},['aggregate'],'The exact output of: '+prep),
           'compute':contract('compute','items',{'prepare':prep,'aggregate':goal},['prepare','aggregate'],a['input'])}
    reference={'prepare':definition(first,p)[2]['prepare'].replace('def prepare(', 'def _base(')+'\n'+function('prepare','items','base = _base(items)\nreturn '+expr),
               'aggregate':definition(second,p)[2]['aggregate'],'component_lineage':[first,second],'structure':structure,'parameter':p}
    ns={};exec(reference['prepare']+'\n'+reference['aggregate'],ns)
    inputs=EXTRA_INPUTS[first] if first in EXTRA_INPUTS else INPUTS[first]
    suites=[];private=[];checks=0
    for visibility,indices in [('VISIBLE',[0,1]),('PRIVATE',[2,3,4,5])]:
        for key,c in [('root',root),*local.items()]:
            cases=[]
            for i in indices:
                items=deepcopy(inputs[i]);prepared=convert(oracle_a(first,items));answer=oracle_b(second,prepared,p)
                assert ns['prepare'](deepcopy(items))==prepared
                assert ns['aggregate'](deepcopy(prepared))==answer
                args=[json.dumps(items)] if key=='root' else [prepared if key=='aggregate' else items]
                expected=json.dumps(answer,separators=(',',':')) if key=='root' else prepared if key=='prepare' else answer
                cases.append({'case_id':f'{key}_{i}','args':args,'expected':expected});checks+=1
            suite={'suite_id':f'{root_id}_{key}_{visibility.lower()}','check_kind':'ROOT' if key=='root' else 'LOCAL',
                   'visibility':visibility,'oracle_origin':'Independent A4 oracle composition with explicit interface projection',
                   'covered_obligations':[o['id'] for o in c['ensures']],'cases':cases}
            if key!='root':suite['contract_key']=key
            (suites if visibility=='VISIBLE' else private).append(suite)
    task={'root_id':root_id,'family_id':f'pr1c_transfer_{first}_{second}_v1','split':'VALIDATION','entrypoint':'solve',
          'visible_suite_ref':root_id+'_root_visible','private_suite_ref':root_id+'_root_private',
          'spec':f"Input is JSON encoding {a['input']}. {prep} Then: {goal} Return compact JSON without a newline. Implement solve(text: str) -> str. Optional reviewed subcontracts (no implementation): "+canonical(local)}
    return {'task':task,'contract':root,'local_contracts':local,'suites':suites},{'suites':private},reference,checks


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    prior=set();paths=[]
    for path in Path('data').rglob('*.jsonl'):
        if DATA in path.parents:continue
        # Includes copied manifests and exported training/evaluation states.
        for row in read_jsonl(path):
            if 'family_id' in row:prior.add(row['family_id'])
        paths.append(str(path))
    rows=[];tests=[];selection=[];checks=0
    for first,second,structure in TASKS:
        d,private,reference,n=pack(first,second,structure);checks+=n;t=d['task']
        assert t['family_id'] not in prior,t['family_id']
        folder=DATA/t['root_id'];folder.mkdir(exist_ok=True)
        for name,value in [('task',d),('private_tests',private),('reference',reference)]:guarded_json(folder/f'{name}.json',value)
        rows.append({**{k:t[k] for k in ['root_id','family_id','split']},'source':'fresh_gate_development_diagnostic_v1',
          'original_split':'VALIDATION','io_type':'solve_text','fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),
          'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')})
        tests.append({'root_id':t['root_id'],'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')})
        selection.append({'root_id':t['root_id'],'components':[first,second],'structure':structure})
    for name,values in [('roots',rows),('tests',tests)]: (DATA/f'{name}.jsonl').write_text(''.join(canonical(x)+'\n' for x in values))
    old=json.loads(Path('runs/refinement/separate_gate/request.json').read_text())
    config=deepcopy(old['config']);config.update(root_manifest=str(DATA/'roots.jsonl'),test_manifest=str(DATA/'tests.jsonl'),root_limit=8,output_dir=str(ROOT))
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':config,'selection':selection,
      'gate_adapter':'runs/refinement/separate_gate/CALIBRATED/final_adapter','body_adapter':old['study']['body_adapter'],
      'seed':330000713,'plan_seed':331000713,'arms':['ATOMIC','SPLIT','NATURAL'],'repetitions':4,'planned_cases':96,
      'selection_rule':'Eight unseen component pairs; two per explicit interface-composition structure. No model executions used for selection.',
      'scope':'Fresh development diagnostic, not a final sealed confirmation; known component algorithms with new pairings and explicit projections.',
      'uncertainty_rule':'Absolute success-count gap <=1/4 is inconclusive; >=2/4 is exploratory directional evidence only, not a strong training label.',
      'forcing':'Only initial root choice forced; all original retries and downstream policies retained. Four independently sampled SPLIT bodies, one continuation each.',
      'seed_pairing':'Same initial execution seed across arms; SPLIT plan sampling has a separate seed. No claim of identical downstream random streams.',
      'reference_checks':checks,'prior_family_count':len(prior),'prior_data_files':paths})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'prepared':len(rows),'oracle_checks':checks,'selection':selection}))


def run(shard):
    from transformers import set_seed
    study=yaml.safe_load(Path('configs/refinement_separate_gate.yaml').read_text())
    request=json.loads((ROOT/'request.json').read_text())
    _,_,policy,executor=load(study,request['gate_adapter']);before=fingerprints(policy)
    expected=json.loads(Path('runs/refinement/separate_gate/CALIBRATED/result.json').read_text())['frozen_after']
    assert before==expected
    rows=list(load_root_manifest(request['config']['root_manifest']).values());tests={r['root_id']:r for r in read_jsonl(request['config']['test_manifest'])}
    records=[]
    for i,row in enumerate(rows):
        if i%2!=shard:continue
        base=ROOT/row['root_id'];base.mkdir(exist_ok=True)
        if (base/'gate.json').exists():gate=json.loads((base/'gate.json').read_text())
        else:
            runtime,_=runtime_for(row,tests,request['framework'],None,base/'decision')
            gate=policy.score_modes(runtime.build_observation());write_json(base/'gate.json',gate)
        for rep in range(4):
            cp=base/f'candidate_{rep:02d}.json'
            if cp.exists():candidate=json.loads(cp.read_text())
            else:
                set_seed(request['plan_seed']+i*100+rep);candidate=policy.sample_split(gate);write_json(cp,candidate)
            for arm in request['arms']:
                initial=None if arm=='NATURAL' else (gate,policy.atomic(gate) if arm=='ATOMIC' else candidate)
                c=run_case(row,tests,request['framework'],request['config'],policy,executor,base/arm/f'repeat_{rep:02d}',request['seed']+i*100+rep,initial=initial)
                if c['status']!='COMPLETE':raise ValueError('Infrastructure failure; preserve and inspect before proceeding')
                assert c['reward'] is None
                records.append({'arm':arm,**c})
    assert fingerprints(policy)==before
    write_json(ROOT/f'shard_{shard}.json',{'status':'COMPLETE','cases':records,'frozen':before})


def report():
    from .refinement_body_transfer import reviewed_plans
    from .refinement_curriculum_rl import plan_key
    pieces=[json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(2)]
    assert pieces[0]['frozen']==pieces[1]['frozen']
    cases=[c for piece in pieces for c in piece['cases']];assert len(cases)==96
    request=json.loads((ROOT/'request.json').read_text());summaries=[]
    for selected in request['selection']:
        rid=selected['root_id'];gate=json.loads((ROOT/rid/'gate.json').read_text())
        entry={**selected,'p_split':gate['mode_probabilities']['SPLIT']}
        definition_=json.loads((DATA/rid/'task.json').read_text())
        reviewed=reviewed_plans(gate['observation'],definition_)
        for arm in request['arms']:
            cs=sorted([c for c in cases if c['root_id']==rid and c['arm']==arm],key=lambda c:c['seed']);assert len(cs)==4
            choices=[]
            variants=[]
            for c in cs:
                event=next(e for e in read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl') if e['event_kind']=='POLICY' and e['node_id']=='N0')
                choices.append((event.get('parsed_action') or {}).get('args',{}).get('mode'))
                if arm=='SPLIT':
                    action=event.get('parsed_action')
                    variants.append(reviewed.get(plan_key(action,definition_)) if action and event['host_result']['code']=='PLAN_REGISTERED' else None)
            entry[arm]={'success':sum(c['final_scores']['private']['status']=='PASS' for c in cs),'runs':4,
                        'initial_modes':choices,'initial_registered':sum(c['root_first_plan_registered'] for c in cs),
                        'outcomes':[c['final_scores']['private']['status'] for c in cs],
                        'rejections':[c['protocol_rejections'] for c in cs]}
            if arm=='SPLIT':entry[arm]['reviewed_variants']=variants
        gap=entry['SPLIT']['success']-entry['ATOMIC']['success'];entry['delta']=gap/4
        entry['direction']='UNCERTAIN' if abs(gap)<=1 else 'SPLIT' if gap>0 else 'ATOMIC'
        summaries.append(entry)
    result={'status':'COMPLETE','cases':96,'optimizer_updates':0,'frozen':pieces[0]['frozen'],'by_root':summaries,
            'totals':{a:sum(r[a]['success'] for r in summaries) for a in request['arms']}}
    write_json(ROOT/'result.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','report']);parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run':run(args.shard)
    else:report()
