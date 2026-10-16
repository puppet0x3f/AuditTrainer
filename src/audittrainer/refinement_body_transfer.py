"""Replicate pure-body SFT across tasks; separately measure held-out transfer."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in {'train', 'evaluate'}:
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_body_transfer.yaml'
    _study = yaml.safe_load(Path(_path).read_text())
    _reservation = acquire(Path(_study['output_dir'])/'gpu_resources.json', 'train' if sys.argv[1]=='train' else 'collect')

import torch
import torch.nn.functional as F
from transformers import set_seed

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, score_program, write_json
from .refinement_curriculum_rl import plan_key
from .refinement_generalization import guarded_json
from .refinement_policy import active_adapter, adapter_fingerprint, load_policies
from .refinement_pr1 import causal_logits, enable_planner
from .refinement_responsibility_course import allocation, CourseReference
from .refinement_rollout import runtime_for, run_case, case_summary
from .schema import PolicyOutput, TokenUsage
from .train_refinement import encode_demo
from .train_sft import changed_tensors, prefix_forward, trainable_snapshots
from .verifier import canonical

INTERNAL = 'N0_p1_compute'
VARIANTS = ('parent_aggregate', 'parent_prepare', 'pipeline', 'fused')


def reviewed_plans(observation, definition):
    """Finite reviewed allocations, not a complete semantic equivalence test."""
    result = {}
    for variant in VARIANTS:
        action = allocation(observation['incoming_contract'], definition, variant)
        args = action['args']
        # A combined RESULT also needs the parent when some computation stays there.
        if 'RESULT' in args['obligation_map'] and set(args['retained_responsibilities']) & {'prepare', 'aggregate'}:
            args['obligation_map']['RESULT'].append('self')
        result[plan_key(action, definition)] = variant
    return result


def at_state(state, row, tests, framework, definition, directory, policy):
    """Recreate NEW from the fixed root, with no future artifacts or budget reset."""
    def act(obs):
        if obs['node_id'] == 'N0' and state['node_id'] != 'N0':
            return PolicyOutput(raw_text=canonical(allocation(obs['incoming_contract'], definition, 'fused')),
                                token_usage=TokenUsage(), model_identity={'backend': 'fixed_reference_root'})
        return policy.act(obs)
    runtime, _ = runtime_for(row, tests, framework, SimpleNamespace(act=act), directory)
    if state['node_id'] != 'N0':
        assert state['node_id'] == INTERNAL
        assert runtime.step().host_result.code == 'PLAN_REGISTERED'
        assert runtime.step().host_result.code == 'CHILD_ENTERED'
    expected = json.loads(state['messages'][1]['content'])['observation']
    if runtime.build_observation() != expected:
        raise ValueError('Recreated planning state differs from the exported observation')
    return runtime


def prepare(study):
    output, data = Path(study['output_dir']), Path(study['data_dir'])
    output.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    course = Path(study['course_dir'])
    original = json.loads((course/'request.json').read_text())
    if json.loads((course/'reference_result.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('Executable course references must be complete')
    config, framework = deepcopy(original['config']), deepcopy(original['framework'])
    framework['allow_validation'] = True
    states = [r for r in read_jsonl(course/'demonstrations.jsonl') if r['target_action']['args']['mode'] == 'SPLIT']
    if len(states) != 24 or any(r['split'] != 'TRAIN' for r in states): raise ValueError('Expected 24 TRAIN SPLIT states')
    train_ids = {r['root_id'] for r in states}; train_families = {r['family_id'] for r in states}
    rows = [r for r in read_jsonl(config['root_manifest']) if r['root_id'] in train_ids]
    tests = [r for r in read_jsonl(config['test_manifest']) if r['root_id'] in train_ids]
    validation = read_jsonl(study['validation_roots'])
    private_rows = {r['root_id']: r for r in read_jsonl(study['validation_tests'])}
    reference_results = []
    for index, old in enumerate(validation):
        if old['split'] != 'VALIDATION' or old['family_id'] in train_families:
            raise ValueError('Validation compositions cross training families')
        definition = json.loads(Path(old['fixture_path']).read_text())
        private = json.loads(Path(private_rows[old['root_id']]['path']).read_text())
        reference = json.loads(Path(old['reference_path']).read_text())
        structure = ('parent_aggregate', 'parent_prepare', 'pipeline', 'joint_obligation')[index % 4]
        if structure == 'joint_obligation':
            ensures = definition['contract']['ensures']; text = {e['id']: e['text'] for e in ensures}
            definition['contract']['ensures'] = [ensures[0],
                {'id': 'RESULT', 'text': text['PREPARE']+' Then: '+text['AGGREGATE']}, ensures[-1]]
            for suite in definition['suites']+private['suites']:
                if suite['check_kind'] == 'ROOT': suite['covered_obligations'] = ['PARSE', 'RESULT', 'FORMAT']
        folder = data/old['root_id']; folder.mkdir(exist_ok=True)
        for name, value in [('task', definition), ('private_tests', private), ('reference', reference)]:
            guarded_json(folder/f'{name}.json', value)
        row = {**old, 'source': 'body_transfer_validation_v1', 'fixture_path': str(folder/'task.json'),
               'source_hash': file_hash(folder/'task.json'), 'reference_path': str(folder/'reference.json'),
               'reference_hash': file_hash(folder/'reference.json')}
        test = {'root_id': row['root_id'], 'path': str(folder/'private_tests.json'), 'sha256': file_hash(folder/'private_tests.json')}
        rows.append(row); tests.append(test)
        for scope, variant in [('root', structure), ('internal', 'recursive')]:
            base = output/'references'/row['root_id']/scope; result_path = base/'result.json'
            if result_path.exists():
                saved = json.loads(result_path.read_text())
            else:
                teacher = CourseReference(definition, reference, variant)
                runtime, full = runtime_for(row, {row['root_id']: test}, framework, teacher, base/'runtime')
                result = runtime.run(); scores, _ = score_program(capture_program(runtime, base/'final_program'), full, row['root_id'])
                local = []
                for node_id, ctx in runtime.contexts.items():
                    if node_id == 'N0' or ctx.artifact is None: continue
                    for suite in full.suites.values():
                        if suite.visibility == 'PRIVATE' and suite.applies(row['root_id'], ctx.node.incoming_contract):
                            ev = full.verify(runtime.run_dir, row['root_id'], ctx.node.incoming_contract, ctx.artifact,
                                suite.suite_id, f'TRANSFER_PRIVATE_{len(local)}', allow_private=True)
                            local.append({'node_id': node_id, 'status': ev.status})
                saved = {'status': 'COMPLETE' if result['functional_status']=='PASS' and scores['private']['status']=='PASS'
                         and len(local)==result['node_count']-1 and all(c['status']=='PASS' for c in local) else 'FAILED',
                         'root_id': row['root_id'], 'scope': scope, 'scores': scores, 'local': local, 'nodes': result['node_count']}
                write_json(result_path, saved)
            if saved['status'] != 'COMPLETE': raise ValueError(f'Reference failed: {base}')
            reference_results.append(saved)
            node_id = 'N0' if scope=='root' else INTERNAL
            event = next(e for e in read_jsonl(base/'runtime/trace.jsonl')
                         if e['node_id']==node_id and e['host_result']['code']=='PLAN_REGISTERED')
            states.append({'sample_id': row['root_id']+'_'+node_id, 'root_id': row['root_id'],
                'family_id': row['family_id'], 'split': 'VALIDATION', 'node_id': node_id,
                'structure': structure if scope=='root' else 'recursive', 'target_action': event['parsed_action'],
                'messages': [{}, {'content': canonical({'observation': event['observation_before']})}],
                'source_trace': str(base/'runtime')})
    if len(states) != 48: raise ValueError('Expected 24 train and 24 held-out states')
    for name, values in [('roots', rows), ('tests', tests), ('states', states)]:
        path = data/f'{name}.jsonl'; content = ''.join(canonical(r)+'\n' for r in values)
        if path.exists() and path.read_text() != content: raise ValueError('Changed frozen transfer data')
        path.write_text(content)
    config.update(root_manifest=str(data/'roots.jsonl'), test_manifest=str(data/'tests.jsonl'))
    request = {'study': study, 'config': config, 'framework': framework, 'train_states': 24,
        'train_roots': len(train_ids), 'train_families': len(train_families), 'validation_roots': len(validation),
        'states_sha256': file_hash(data/'states.jsonl'), 'reference_results': reference_results,
        'adapters': {'B0': config['initial_adapter'], 'SINGLE_BODY': study['single_adapter'],
                     'MULTI_BODY': str(output/'MULTI_BODY/final_adapter')},
        'objective': 'Mean body CE only; no mode supervision, KL, retention, boundary weighting, or RL.',
        'exposures': study['optimizer_steps']*study['gradient_accumulation_steps'],
        'conditional_samples_per_model': len(states)*study['samples_per_state'],
        'natural_runs_per_model': len(validation)*study['natural_repetitions'],
        'validation_scope': 'Untrained known-component compositions from an existing development set; 3 roots regroup upper obligations as RESULT. Not a fresh confirmation set.',
        'primary_endpoints': ['TRAIN exact target under conditional SPLIT, by root/internal and structure',
            'VALIDATION reviewed complete allocations under conditional SPLIT, by root/internal'],
        'secondary_endpoints': ['host registration', 'target preference on held-out states (not correctness)',
            'gate probability drift (not calibrated decision quality)', 'natural recursive root execution on VALIDATION'],
        'reviewed_scope': 'Exact authored child contracts and consistent ownership/retention/dataflow templates; unmatched means unreviewed, not necessarily semantically invalid.'}
    guarded_json(output/'request.json', request)
    guarded_json(output/'gpu_resources.json', {'devices': ['0','1','2'], 'min_free_mib': {'collect':16000, 'train':22000}})
    write_json(output/'reference_result.json', {'status':'COMPLETE', 'runs':len(reference_results), 'results':reference_results})
    print(json.dumps({k:request[k] for k in ['train_states','train_roots','train_families','validation_roots','conditional_samples_per_model']}), flush=True)
    return request


def training_order(states, exposures, seed):
    if any(s['split'] != 'TRAIN' or s['target_action']['args']['mode'] != 'SPLIT' for s in states):
        raise ValueError('Only TRAIN SPLIT targets may be optimized')
    if exposures % len(states): raise ValueError('Use equal exposure per state')
    rng = random.Random(seed); result = []
    for _ in range(exposures//len(states)):
        epoch = list(range(len(states))); rng.shuffle(epoch); result.extend(epoch)
    return result


def train(study):
    root = Path(study['output_dir']); request = json.loads((root/'request.json').read_text())
    states = [s for s in read_jsonl(Path(study['data_dir'])/'states.jsonl') if s['split']=='TRAIN']
    order = training_order(states, request['exposures'], study['seed'])
    output = root/'MULTI_BODY'; output.mkdir(exist_ok=False)
    guarded_json(output/'request.json', {'study':study, 'objective':request['objective'],
        'initial_adapter':request['adapters']['B0'], 'training_state_ids':[s['sample_id'] for s in states],
        'schedule':[states[i]['sample_id'] for i in order], 'states_sha256':request['states_sha256']})
    set_seed(study['seed']); planner, _ = load_policies(request['framework'], request['adapters']['B0'], request['config']['executor_adapter'])
    model, tok = planner.model, planner.tokenizer
    examples = [encode_demo(s, tok, request['framework']) for s in states]
    for state, example in zip(states, examples):
        _, encoded, _ = planner.prepare(json.loads(state['messages'][1]['content'])['observation'])
        if example['prefix'] != encoded['prefix']: raise ValueError('Training/sampling prefix differs')
    (output/'examples.jsonl').write_text(''.join(canonical(e)+'\n' for e in examples))
    frozen = adapter_fingerprint(model, 'executor')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    model.peft_config['default'].lora_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.config.use_cache = False
    enable_planner(model); before, trainable = trainable_snapshots(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=study['learning_rate'], weight_decay=0.)
    history=[]; accumulation=study['gradient_accumulation_steps']; started=time.monotonic()
    for step in range(study['optimizer_steps']):
        optimizer.zero_grad(set_to_none=True); enable_planner(model); total=0.
        for index in order[step*accumulation:(step+1)*accumulation]:
            e=examples[index]
            with planner.llm._autocast():
                logits=causal_logits(model,e['body_prefix'],e['body_target'])
                loss=F.cross_entropy(logits.float(),torch.tensor(e['body_target'],device=model.device))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite body loss')
            (loss/accumulation).backward(); total+=float(loss.detach())/accumulation
            del logits,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or norm<=0: raise FloatingPointError('Invalid body gradient')
        optimizer.step(); record={'step':step+1,'body_ce':total,'gradient_norm':float(norm)}
        history.append(record); print(json.dumps(record),flush=True)
    final=output/'final_adapter';model.save_pretrained(final,selected_adapters=['default']);tok.save_pretrained(final)
    after=adapter_fingerprint(model,'executor');changes=changed_tensors(model,before)
    original=prefix_forward(model,examples[0]['prefix'])
    model.load_adapter(str(final),adapter_name='reload_check',is_trainable=False,local_files_only=True)
    with active_adapter(model,'reload_check'): restored=prefix_forward(model,examples[0]['prefix'])
    delta=float((original-restored).abs().max())
    if after!=frozen or delta>1e-5:raise ValueError('Freeze or reload check failed')
    result={'status':'COMPLETE','optimizer_steps':len(history),'exposures':len(order),'exposures_per_state':len(order)//len(states),
        'train_states':len(states),'rl_updates':0,'objective':request['objective'],'history':history,
        'trainable_parameters':trainable,'parameter_changes':changes,'executor_before':frozen,'executor_after':after,
        'reload_max_abs_difference':delta,'adapter':str(final),'adapter_sha256':file_hash(final/'adapter_model.safetensors'),
        'adapter_config_sha256':file_hash(final/'adapter_config.json'),'train_seconds':time.monotonic()-started}
    write_json(output/'training_report.json',result)


def evaluate(study, method):
    root=Path(study['output_dir']);request=json.loads((root/'request.json').read_text())
    output=root/'evaluation'/method;output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():return
    adapter=request['adapters'][method]
    guarded_json(output/'request.json',{'method':method,'adapter':adapter,'adapter_sha256':file_hash(Path(adapter)/'adapter_model.safetensors'),
        'states_sha256':request['states_sha256'],'probe_seed':study['probe_seed'],'samples_per_state':study['samples_per_state']})
    planner,executor=load_policies(request['framework'],adapter,request['config']['executor_adapter'])
    before={n:adapter_fingerprint(planner.model,n) for n in ('default','executor')}
    states=read_jsonl(Path(study['data_dir'])/'states.jsonl');rows=load_root_manifest(request['config']['root_manifest'])
    tests={r['root_id']:r for r in read_jsonl(request['config']['test_manifest'])}; records=[]; state_scores=[]
    for index,state in enumerate(states):
        folder=output/'conditional'/state['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        obs=json.loads(state['messages'][1]['content'])['observation'];definition=json.loads(Path(rows[state['root_id']]['fixture_path']).read_text())
        gate=planner.score_modes(obs);write_json(folder/'gate.json',gate)
        target=plan_key(state['target_action'],definition);variants=reviewed_plans(obs,definition)
        example=encode_demo(state,planner.tokenizer,request['framework'])
        with active_adapter(planner.model,'default'),torch.no_grad(),planner.llm._autocast():
            ce=float(F.cross_entropy(causal_logits(planner.model,example['body_prefix'],example['body_target']).float(),
                torch.tensor(example['body_target'],device=planner.model.device)))
        state_scores.append({'sample_id':state['sample_id'],'root_id':state['root_id'],'split':state['split'],
            'node_id':state['node_id'],'structure':state['structure'],'p_split':gate['mode_probabilities']['SPLIT'],'target_body_ce':ce})
        for rep in range(study['samples_per_state']):
            base=folder/f'sample_{rep:02d}';path=base/'result.json'
            if path.exists():records.append(json.loads(path.read_text()));continue
            base.mkdir(exist_ok=False);seed=study['probe_seed']+100*index+rep;set_seed(seed)
            candidate=planner.sample_split(gate);write_json(base/'candidate.json',candidate)
            policy=SimpleNamespace(act=lambda o,c=candidate:planner.as_output(gate,c))
            runtime=at_state(state,rows[state['root_id']],tests,request['framework'],definition,base/'registration',policy)
            event=runtime.step();registered=event.host_result.code=='PLAN_REGISTERED'
            key=plan_key(candidate['parsed_action'],definition) if registered else None
            record={**{k:state[k] for k in ('sample_id','root_id','family_id','split','node_id','structure')},
                'rep':rep,'seed':seed,'registered':registered,'host_code':event.host_result.code,
                'target_match':registered and key is not None and key==target,
                'reviewed':registered and key is not None and key in variants,'variant':variants.get(key),
                'matching_child_contracts':registered and key is not None}
            write_json(path,record);records.append(record)
        current=[r for r in records if r['sample_id']==state['sample_id']]
        print(json.dumps({'method':method,'state':index+1,'total_states':len(states),'split':state['split'],'node':state['node_id'],
            'registered':sum(r['registered'] for r in current),'target':sum(r['target_match'] for r in current),
            'reviewed':sum(r['reviewed'] for r in current)}),flush=True)
    write_json(output/'conditional_result.json',{'status':'COMPLETE','records':records,'state_scores':state_scores})
    cases=[]
    for index,row in enumerate(r for r in rows.values() if r['split']=='VALIDATION'):
        for rep in range(study['natural_repetitions']):
            cases.append(run_case(row,tests,request['framework'],request['config'],planner,executor,
                output/'natural'/row['root_id']/f'repeat_{rep:02d}',study['natural_seed']+index*100+rep))
    if any(c['status']!='COMPLETE' or c['reward'] is not None for c in cases):raise ValueError('Incomplete or training-scored validation run')
    after={n:adapter_fingerprint(planner.model,n) for n in before}
    if after!=before:raise ValueError('Evaluation changed weights')
    result={'status':'COMPLETE','method':method,'records':records,'state_scores':state_scores,'natural':case_summary(cases),
        'natural_by_root':{c['root_id']:c['final_scores']['private']['status']=='PASS' for c in cases},
        'frozen_before':before,'frozen_after':after,'actual_model_cost':dict(planner.cost+executor.cost),'optimizer_updates':0}
    write_json(output/'result.json',result)


def summarize(records):
    return {'samples':len(records), **{key:sum(r[key] for r in records) for key in ('registered','target_match','reviewed')},
            'by_variant':dict(Counter(r['variant'] or 'UNREVIEWED' for r in records))}


def report(study):
    output=Path(study['output_dir']);models={m:json.loads((output/'evaluation'/m/'result.json').read_text()) for m in study['methods']}
    summaries={}
    for method,result in models.items():
        if result['status']!='COMPLETE' or len(result['records'])!=192 or result['natural']['runs']!=12:raise ValueError('Incomplete finite evaluation')
        groups={}
        for split in ('TRAIN','VALIDATION'):
            for level in ('root','internal'):
                rows=[r for r in result['records'] if r['split']==split and (r['node_id']=='N0')==(level=='root')]
                scores=[r for r in result['state_scores'] if r['split']==split and (r['node_id']=='N0')==(level=='root')]
                groups[split+'/'+level]={**summarize(rows),'states':len(scores),'roots':len({r['root_id'] for r in rows}),
                    'mean_p_split':sum(s['p_split'] for s in scores)/len(scores),
                    'mean_target_body_ce':sum(s['target_body_ce'] for s in scores)/len(scores)}
        summaries[method]={'groups':groups,'train_structures':{structure:summarize([r for r in result['records']
            if r['split']=='TRAIN' and r['structure']==structure]) for structure in sorted({r['structure'] for r in result['records'] if r['split']=='TRAIN'})},
            'natural':result['natural'],'natural_by_root':result['natural_by_root']}
    base={(r['sample_id'],r['seed']):r for r in models['B0']['records']};paired={}
    for method,result in models.items():
        pairs={(r['sample_id'],r['seed']):r for r in result['records']}
        if pairs.keys()!=base.keys():raise ValueError('Unpaired conditional seeds')
        paired[method]={}
        for split,key in [('TRAIN','target_match'),('VALIDATION','reviewed')]:
            roots=sorted({r['root_id'] for r in pairs.values() if r['split']==split})
            deltas={root:sum(int(pairs[k][key])-int(base[k][key]) for k in pairs if pairs[k]['root_id']==root) /
                    sum(pairs[k]['root_id']==root for k in pairs) for root in roots}
            paired[method][split]={'metric':key,'root_rate_differences':deltas,'improved_roots':sum(v>0 for v in deltas.values()),
                'worse_roots':sum(v<0 for v in deltas.values()),'tied_roots':sum(v==0 for v in deltas.values()),
                'mean_root_rate_difference':sum(deltas.values())/len(deltas)}
    result={'status':'COMPLETE','summaries':summaries,'paired_vs_B0':paired,'conditional_samples':576,'natural_runs':36,
        'training':json.loads((output/'MULTI_BODY/training_report.json').read_text()),
        'scope':'Pure-body recipe replication and development-set composition transfer; one training seed. Conditional plan quality and natural execution are separate endpoints.'}
    write_json(output/'result.json',result);print(json.dumps({'status':'COMPLETE','summaries':summaries,'paired':paired}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','evaluate','report'])
    parser.add_argument('--config',default='configs/refinement_body_transfer.yaml')
    parser.add_argument('--method',choices=['B0','SINGLE_BODY','MULTI_BODY'])
    args=parser.parse_args();study=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':prepare(study)
    elif args.command=='train':train(study)
    elif args.command=='evaluate':evaluate(study,args.method)
    else:report(study)
