"""Frozen-summary validation on new, interface-stratified parent states."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_parent_validation.yaml'
    _gpu_reservation = acquire(Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json', 'collect')

from . import prompts
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_parent_composition import (
    composition_summary, summary_prompt, framework_for, selected_state, evaluate, paired_summary,
)


def presentation_identity():
    return {'summary_functions_sha256': hashlib.sha256(
        (inspect.getsource(composition_summary)+inspect.getsource(summary_prompt)).encode()).hexdigest(),
        'base_prompts_sha256': file_hash(prompts.__file__)}


def write_fixed(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError(f'Changed frozen file: {path}')
    else:
        write_json(path,value)


def prepare(settings):
    output = Path(settings['output_dir']); output.mkdir(parents=True, exist_ok=True)
    previous = Path(settings['previous_study'])
    old = json.loads((previous/'request.json').read_text())
    prior_roots = {s['root_id'] for s in old['states']}
    states = [dict(selected_state(path,i,allowed_splits=('TRAIN','VALIDATION')),interface_type=kind)
              for i,(path,kind) in enumerate(zip(settings['source_cases'],settings['interface_types'],strict=True))]
    if len(states)!=16 or Counter(s['interface_type'] for s in states)!={'dictionary':6,'list':6,'string':4}:
        raise ValueError('Expected 16 preselected interface-stratified states')
    if prior_roots & {s['root_id'] for s in states}:
        raise ValueError('Prior design task reused')
    signatures = [json.dumps([s['root_id'],s['observation']['current_plan'],s['child_hashes']],sort_keys=True) for s in states]
    if len(set(signatures)) != len(states): raise ValueError('Duplicate plan/child state')
    rows = {}; tests = {}
    for source in settings['source_manifests']:
        rows.update(load_root_manifest(source['roots']))
        tests.update({r['root_id']:r for r in read_jsonl(source['tests'])})
    selected_roots = sorted({s['root_id'] for s in states})
    for key,values in [('root_manifest',[rows[r] for r in selected_roots]),('test_manifest',[tests[r] for r in selected_roots])]:
        path = Path(settings[key]); path.parent.mkdir(parents=True,exist_ok=True)
        content = ''.join(json.dumps(r,sort_keys=True)+'\n' for r in values)
        if path.exists() and path.read_text()!=content: raise ValueError('Changed task manifest')
        path.write_text(content)
    write_fixed(output/'request.json',{'settings':settings,'states':states,'framework':framework_for(settings),
        'presentation':presentation_identity(),
        'adapters':{k:file_hash(Path(settings[k])/'adapter_model.safetensors') for k in ('initial_adapter','executor_adapter')},
        'previous_result_sha256':file_hash(previous/'result.json'),
        'previous_report_sha256':file_hash('reports/B3_parent_composition.md'),
        'planned_continuations':128,'optimizer_updates':0,
        'decision':settings['decision'],
        'interpretation':'Use overall paired net wins as primary. Require positive net gains in at least two states from two task roots for broader support. Individual losses are allowed and all interface strata, recurrent key/value mistakes and token costs are reported. No retrospective change to the prior decision.'})
    write_fixed(output/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    return states


def verify_frozen(settings):
    request = json.loads((Path(settings['output_dir'])/'request.json').read_text())
    if (request['settings']!=settings or request['presentation']!=presentation_identity()
            or request['previous_result_sha256']!=file_hash(Path(settings['previous_study'])/'result.json')
            or request['previous_report_sha256']!=file_hash('reports/B3_parent_composition.md')):
        raise ValueError('Frozen summary, request or prior conclusion changed')
    return request


def broader_support(paired, by_state, by_root, decision):
    return (paired['wins']-paired['losses'] >= decision['minimum_net_wins']
        and sum(p['wins']>p['losses'] for p in by_state.values()) >= decision['minimum_positive_states']
        and sum(p['wins']>p['losses'] for p in by_root.values()) >= decision['minimum_positive_roots'])


def report(settings, shards):
    output = Path(settings['output_dir']); request = verify_frozen(settings)
    records = []
    for i,s in enumerate(request['states']):
        for j in range(settings['repetitions']):
            pair = []
            for arm in ('control','summary'):
                c = json.loads((output/'evaluation'/s['id']/f'repeat_{j:02d}'/arm/'case_result.json').read_text())
                if (c['status']!='COMPLETE' or c['state_id']!=s['id'] or c['root_id']!=s['root_id'] or c['arm']!=arm
                        or c['seed']!=settings['evaluation_seed']+i*1000+j
                        or not c['frozen_children_and_plans'] or c['role_calls'].get('planner',0)):
                    raise ValueError('Incomplete or changed paired replay')
                pair.append({**c,'interface_type':s['interface_type']})
            if pair[0]['starting_usage']!=pair[1]['starting_usage']: raise ValueError('Unmatched budgets')
            records.extend(pair)
    checks = [json.loads((output/f'shard_{i}.json').read_text()) for i in range(shards)]
    if (len(records)!=128 or sum(c['cases'] for c in checks)!=128 or
            any(c['frozen_before']!=c['frozen_after'] or c['optimizer_updates'] for c in checks)
            or len({json.dumps(c['frozen_before'],sort_keys=True) for c in checks})!=1):
        raise ValueError('Incomplete or inconsistent model freeze')
    paired = paired_summary(records)
    grouped = {field:{key:paired_summary([r for r in records if r[field]==key]) for key in sorted({r[field] for r in records})}
               for field in ('state_id','root_id','interface_type')}
    costs = {arm:{k:sum(r['continuation_usage'][k] for r in records if r['arm']==arm)
                  for k in ('input_tokens','output_tokens','policy_calls','verify_calls')} for arm in ('control','summary')}
    result = {'status':'COMPLETE','new_parent_continuations':128,'optimizer_updates':0,
        'paired':paired,'by_state':grouped['state_id'],'by_root':grouped['root_id'],
        'by_interface':grouped['interface_type'],'costs':costs,
        'supports_broader_summary':broader_support(paired,grouped['state_id'],grouped['root_id'],settings['decision']),
        'presentation':request['presentation'],'frozen_model_checks':checks,'records':records}
    write_json(output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','frozen_model_checks'}}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','report'))
    parser.add_argument('--config',default='configs/refinement_parent_validation.yaml')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=3)
    args=parser.parse_args();settings=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':
        print(json.dumps([{k:v for k,v in s.items() if k!='observation'} for s in prepare(settings)]))
    elif args.command=='evaluate':
        verify_frozen(settings);evaluate(settings,args.shard,args.shards);verify_frozen(settings)
    else:report(settings,args.shards)


if __name__=='__main__':main()
