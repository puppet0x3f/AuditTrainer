"""Fixed public plans, independently seeded suffix execution, no model updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='evaluate':
    from .gpu_resource import acquire
    _path=sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_fixed_plan.yaml'
    _gpu_reservation=acquire(Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json','collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_coverage import paired_rows
from .refinement_curriculum import plans
from .refinement_curriculum_rl import plan_key
from .refinement_generalization import guarded_json
from .refinement_parent_validation import presentation_identity
from .refinement_policy import adapter_fingerprint, load_policies
from .refinement_pr1 import load_config
from .refinement_rollout import runtime_for, run_case, case_summary
from .schema import PolicyOutput, TokenUsage
from .verifier import canonical


class FixedRootPlanner:
    def __init__(self, base, action):
        self.base=base; self.action=action; self.cost=base.cost
        self.identity={'backend':'authored_fixed_root_plan','leaf_policy':'unchanged_common_SFT'}

    def act(self, observation):
        if observation['node_id']!='N0':
            return self.base.act(observation)  # Same depth-limited atomic leaf policy.
        return PolicyOutput(raw_text=canonical(self.action),token_usage=TokenUsage(),model_identity=self.identity)


def settings_for(study):
    config,framework=load_config(study['source_config'])
    if not framework.get('parent_composition_summary') or framework['runtime']['max_depth']!=1:
        raise ValueError('Expected frozen summary and depth one')
    return config,framework


def prepare(study):
    config,framework=settings_for(study); output=Path(study['output_dir']); output.mkdir(parents=True,exist_ok=True)
    rows=load_root_manifest(config['root_manifest']); tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    actions={rid:{v:plans(json.loads(Path(rows[rid]['fixture_path']).read_text()))[v] for v in study['variants']}
             for rid in study['root_ids']}
    if any(rows[r]['split']!='TRAIN' for r in actions): raise ValueError('Only selected TRAIN roots')
    prior=json.loads((Path(study['source_study'])/'request.json').read_text())
    if presentation_identity()!=prior['presentation']: raise ValueError('Frozen summary changed')
    request={'study':study,'execution_config':config,'framework':framework,'plans':actions,
             'presentation':presentation_identity(),
             'source_result_sha256':file_hash(Path(study['source_study'])/'result.json'),
             'adapters':{k:file_hash(Path(config[k])/'adapter_model.safetensors') for k in ('initial_adapter','executor_adapter')},
             'planned_continuations':72,'optimizer_updates':0,
             'seeding':'Reset all execution RNGs after the fixed root plan registers. Same suffix seed for all three plans on each root. Leaf planning and execution retain the existing policy, budget and retries.'}
    guarded_json(output/'request.json',request)
    guarded_json(output/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    if not (output/'plan_checks.json').exists():
        checks=[]
        for rid,variants in actions.items():
            for variant,action in variants.items():
                policy=SimpleNamespace(act=lambda obs,a=action:PolicyOutput(raw_text=canonical(a),token_usage=TokenUsage(),
                                                                          model_identity={'backend':'authored_plan_check'}))
                runtime,_=runtime_for(rows[rid],tests,framework,policy,output/'plan_checks'/rid/variant)
                event=runtime.step()
                if event.host_result.code!='PLAN_REGISTERED': raise ValueError('Rejected fixed plan')
                suites={cid:runtime.verifier.available_suites(rid,c.node.incoming_contract)
                        for cid,c in runtime.contexts.items() if cid!='N0'}
                if any(v==['NO_APPLICABLE_SUITE'] for v in suites.values()): raise ValueError('Uncovered fixed child')
                checks.append({'root_id':rid,'variant':variant,'registered':True,'local_suites':suites})
        write_json(output/'plan_checks.json',checks)
    return request


def evaluate(study,shard,shards):
    request=prepare(study); config,framework=settings_for(study); output=Path(study['output_dir'])
    rows=load_root_manifest(config['root_manifest']); tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    planner,executor=load_policies(framework,config['initial_adapter'],config['executor_adapter'])
    before={n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}; cases=[]
    for i,rid in enumerate(sorted(study['root_ids'])):
        for variant in study['variants']:
            fixed=FixedRootPlanner(planner,request['plans'][rid][variant])
            for rep in range(study['repetitions']):
                if rep%shards!=shard: continue
                case=run_case(rows[rid],tests,framework,config,fixed,executor,
                    output/'evaluation'/rid/variant/f'repeat_{rep:02d}',study['execution_seed']+i*1000+rep,
                    reset_seed_after_root_plan=True)
                if case['status']!='COMPLETE' or not case['root_first_plan_registered']:
                    raise ValueError('Incomplete fixed-plan execution')
                cases.append(case)
    after={n:adapter_fingerprint(planner.model,n) for n in before}
    if before!=after: raise ValueError('Evaluation changed weights')
    write_json(output/f'shard_{shard}.json',{'status':'COMPLETE','shard':shard,'shards':shards,
        'before':before,'after':after,'summary':case_summary(cases),'optimizer_updates':0})


def repeated_advantage(atomic,split,root_id,block_size):
    a=sorted(atomic,key=lambda r:r['seed']); b=sorted(split,key=lambda r:r['seed'])
    blocks=[paired_rows(a[i:i+block_size],b[i:i+block_size],{root_id}) for i in (0,block_size)]
    return {'all':paired_rows(a,b,{root_id}),'blocks':blocks,
            'repeated_advantage':all(p['wins']>p['losses'] for p in blocks)}


def existing_selections(study,definitions):
    source=json.loads((Path(study['source_study'])/'result.json').read_text()); records=[]
    for seed,group in source['evaluations'].items():
        for model,evaluation in group.items():
            for rec in evaluation['records']:
                if rec['root_id'] not in definitions: continue
                d=definitions[rec['root_id']]; variants={plan_key(a,d):v for v,a in plans(d).items()}
                events=[e for e in read_jsonl(Path(rec['run_dir'])/'runtime/trace.jsonl')
                        if e['node_id']=='N0' and e['event_kind']=='POLICY' and e['observation_before']['phase']=='NEW']
                first=events[0]; accepted=next((e for e in events if e['host_result']['code']=='PLAN_REGISTERED'),None)
                def label(event):
                    if event is None:return 'NO_ACCEPTED_PLAN'
                    if event['host_result']['code']!='PLAN_REGISTERED':return 'REJECTED'
                    return variants.get(plan_key(event['parsed_action'],d),'OTHER_REGISTERED')
                records.append({**rec,'model':model,'evaluation_seed':seed,
                    'first_choice':label(first),'accepted_choice':label(accepted),
                    'first_mode':(first.get('model_input') or {}).get('refinement',{}).get('selected_mode',
                        (first.get('parsed_action') or {}).get('args',{}).get('mode','UNPARSED'))})
    return {'source_trajectories':sum(len(e['records']) for g in source['evaluations'].values() for e in g.values()),
            'selected_trajectories':len(records),'records':records,
            'by_model_root':{model:{rid:{'runs':len(rs),
                'first_choice':dict(Counter(r['first_choice'] for r in rs)),
                'accepted_choice':dict(Counter(r['accepted_choice'] for r in rs)),
                'first_mode':dict(Counter(r['first_mode'] for r in rs))}
                for rid in definitions for rs in [[r for r in records if r['model']==model and r['root_id']==rid]]}
                for model in next(iter(source['evaluations'].values()))},
            'matching':'Exact public contracts, obligation owners, retained responsibilities and declared dataflow; child aliases/order normalized. Near matches are OTHER_REGISTERED, not assumed to inherit a fixed-plan value.'}


def report(study,shards):
    request=prepare(study); config,_=settings_for(study); output=Path(study['output_dir']); results={}; paired={}
    rows=load_root_manifest(config['root_manifest']); definitions={r:json.loads(Path(rows[r]['fixture_path']).read_text()) for r in study['root_ids']}
    for shard in range(shards):
        worker=json.loads((output/f'shard_{shard}.json').read_text())
        if worker['status']!='COMPLETE' or worker['before']!=worker['after'] or worker['optimizer_updates']:
            raise ValueError('Incomplete or changed worker')
    for i,rid in enumerate(sorted(study['root_ids'])):
        variants={}; records={}
        for v in study['variants']:
            cases=[]
            for rep in range(study['repetitions']):
                c=json.loads((output/'evaluation'/rid/v/f'repeat_{rep:02d}'/'case_result.json').read_text())
                expected=study['execution_seed']+i*1000+rep
                if c['status']!='COMPLETE' or c['seed']!=expected or c.get('continuation_seed_reset_after_root_plan')!=expected:
                    raise ValueError('Missing or wrongly seeded fixed-plan case')
                first=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')[0]
                if first['parsed_action']!=request['plans'][rid][v] or first['host_result']['code']!='PLAN_REGISTERED':
                    raise ValueError('Changed fixed plan')
                cases.append(c)
            variants[v]={'summary':case_summary(cases),'cases':cases}
            records[v]=[{'root_id':rid,'seed':c['seed'],'success':int(c['final_scores']['private']['status']=='PASS')} for c in cases]
        results[rid]=variants
        paired[rid]={v:repeated_advantage(records['atomic'],records[v],rid,study['seed_block_size']) for v in study['variants'] if v!='atomic'}
    result={'status':'COMPLETE','new_continuations':sum(v['summary']['runs'] for g in results.values() for v in g.values()),
        'optimizer_updates':0,'fixed_plans':request['plans'],'results':results,'paired_vs_atomic':paired,
        'existing_selections':existing_selections(study,definitions),'decision':study['decision']}
    if result['new_continuations']!=72:raise ValueError('Fixed budget incomplete')
    write_json(output/'result.json',result); print(json.dumps(paired),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_fixed_plan.yaml')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=3)
    args=parser.parse_args();study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':prepare(study)
    elif args.command=='evaluate':evaluate(study,args.shard,args.shards)
    else:report(study,args.shards)


if __name__=='__main__':main()
