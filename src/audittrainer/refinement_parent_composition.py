"""Frozen parent-state replay: current input versus an existing-contract summary."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_parent_composition.yaml'
    _gpu_reservation = acquire(Path(yaml.safe_load(Path(_path).read_text())['output_dir'])/'gpu_resources.json', 'collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, load_task, score_program, write_json
from .prompts import compact_json, prepare_prompt, public_observation
from .runtime import Budget, Limits, NodeExecution, RecursiveRuntime
from .schema import Artifact, Consumption, HostResult, NodeState, Plan


def composition_summary(observation):
    """Project only existing public fields; never infer or correct a contract."""
    o = public_observation(observation)
    p = o['current_plan']
    dependencies = {d['child_id']: d for d in o['approved_dependencies']}
    return {
        'parent_retained_responsibilities': p['retained_responsibilities'],
        'parent_obligations': [v for v in o['incoming_contract']['ensures']
                               if 'self' in p['obligation_map'].get(v['id'], [])],
        'child_inputs_and_outputs': [
            {'node_id': c['node_id'], 'binding': dependencies.get(c['node_id']),
             'signature': c['contract']['signature'], 'input_requirements': c['contract']['requires'],
             'output_guarantees': c['contract']['ensures'], 'declared_uses': c['contract']['uses']}
            for c in o['child_interfaces']],
        'declared_dataflow': p['dataflow'],
        'obligation_owners': p['obligation_map'],
    }


def summary_prompt(tokenizer, observation, **kwargs):
    visible = public_observation(observation)
    if visible['phase'] == 'IMPLEMENT' and visible['mode'] == 'COMPOSE':
        visible['root_spec'] += ('\nParent composition summary (restatement of the current contract and plan):\n'
                                 + compact_json(composition_summary(visible)))
    return prepare_prompt(tokenizer, visible, **kwargs)


def framework_for(settings):
    framework = yaml.safe_load(Path(settings['framework_config']).read_text())
    framework['protocol_version'] = settings['protocol_version']
    framework['runtime']['max_depth'] = 1
    return framework


def selected_state(path, index, allowed_splits=('TRAIN',)):
    directory = Path(path); case = json.loads((directory/'case_result.json').read_text())
    events = read_jsonl(directory/'runtime/trace.jsonl')
    event = next(e for e in events if e['node_id'] == 'N0'
                 and e['observation_before']['phase'] == 'IMPLEMENT'
                 and e['observation_before']['mode'] == 'COMPOSE')
    observation = event['observation_before']
    snapshot = directory/'runtime'/observation['state_id']
    state = json.loads(snapshot.read_text()); parent = state['nodes']['N0']
    if (case['status'] != 'COMPLETE' or case['split'] not in allowed_splits or state['stop_reason']
            or parent['node']['artifact_ref'] or state['active_node_id'] != 'N0'):
        raise ValueError('Expected an untouched parent implementation state in an allowed split')
    for child in parent['children']:
        evidence = state['nodes'][child]['node']['last_evidence']
        consumed = parent['consumptions'].get(child, {})
        if (not evidence or evidence['status'] != 'PASS' or evidence['details']['executed_cases'] < 1
                or consumed.get('decision') != 'CONSUME'
                or consumed['artifact_hash'] != evidence['binding']['source_hash']):
            raise ValueError('Every frozen child must have passed a bound local check and be consumed')
    return {'id': f'S{index:02d}', 'source_case': path, 'root_id': case['root_id'],
            'source_success': case['final_scores']['private']['status'] == 'PASS',
            'state_ref': observation['state_id'], 'snapshot_sha256': file_hash(snapshot),
            'observation': observation,
            'child_hashes': {c: parent['consumptions'][c]['artifact_hash'] for c in parent['children']}}


def restore_parent(selected, row, tests, framework, policy, directory):
    """Restore this depth-one, no-repair snapshot; copy no future parent source."""
    source = Path(selected['source_case'])/'runtime'
    snapshot = source/selected['state_ref']
    if file_hash(snapshot) != selected['snapshot_sha256']:
        raise ValueError('Changed source snapshot')
    state = json.loads(snapshot.read_text())
    _, task, contract, online, full = load_task(row, tests, framework)
    if task.model_dump(mode='json') != state['root_task']:
        raise ValueError('Different task at replay')
    runtime = RecursiveRuntime(task, contract, policy, directory, limits=Limits(**state['limits']),
        protocol_version=framework['protocol_version'], verifier=online,
        implementation_stage=selected['observation']['implementation_stage'], require_obligation_coverage=True)
    contexts = {}
    for node_id, saved in state['nodes'].items():
        node = NodeState.model_validate_json(json.dumps(saved['node']))
        plan = None; artifact = None
        for relative in (node.plan_ref, node.artifact_ref):
            if relative:
                target = runtime.run_dir/relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source/relative, target)
        if node.plan_ref:
            plan = Plan.model_validate_json((source/node.plan_ref).read_text())
        if node.artifact_ref:
            metadata = Path(node.artifact_ref).with_suffix('.json')
            shutil.copyfile(source/metadata, runtime.run_dir/metadata)
            artifact = Artifact.model_validate_json((source/metadata).read_text())
        contexts[node_id] = NodeExecution(node=node, plan=plan, artifact=artifact,
            children=saved['children'], returned_child=saved['returned_child'],
            consumptions={k: Consumption.model_validate(v) for k,v in saved['consumptions'].items()},
            repair_requests=saved['repair_requests'], repair_request=saved['repair_request'],
            termination_reason=saved['termination_reason'],
            last_action_result=HostResult.model_validate(saved['last_action_result']) if saved['last_action_result'] else None,
            consecutive_errors=saved['consecutive_protocol_errors'])
    runtime.contexts = contexts
    runtime.active_node_id = state['active_node_id']; runtime.stack = state['stack']
    runtime.budget = Budget.model_validate(state['budget'])
    runtime.trace.count = int(Path(selected['state_ref']).stem[1:])
    runtime.state_ref = runtime.trace.snapshot(runtime._state())
    if runtime.build_observation() != selected['observation']:
        raise ValueError('Replay changed the original parent observation')
    for child in runtime.context.children:
        if runtime._node_result(runtime.contexts[child]).functional_status != 'PASS':
            raise ValueError('Child evidence is no longer current')
    return runtime, full


def prepare(settings):
    output = Path(settings['output_dir']); output.mkdir(parents=True, exist_ok=True)
    states = [selected_state(path, i) for i,path in enumerate(settings['source_cases'])]
    from .refinement_generalization import guarded_json
    guarded_json(output/'request.json', {'settings': settings, 'states': states,
        'framework': framework_for(settings),
        'adapters': {key: file_hash(Path(settings[key])/'adapter_model.safetensors')
                     for key in ('initial_adapter', 'executor_adapter')},
        'planned_continuations': len(states)*settings['repetitions']*2, 'optimizer_updates': 0,
        'decision': 'Adopt for a fresh all-planner comparison only if summary has more final passes overall and at least as many on source-success states. Retain every paired loss and inspect responsibility omissions/input misuse; this selected-state pilot is not a generalization claim.'})
    guarded_json(output/'gpu_resources.json', {'devices': ['0','1','2'], 'min_free_mib': {'collect':16000}})
    return states


def run_parent(selected, row, tests, framework, planner, executor, arm, directory, seed):
    from transformers import set_seed
    from .refinement_policy import RefinementRouter
    directory = Path(directory)
    if (directory/'case_result.json').exists():
        return json.loads((directory/'case_result.json').read_text())
    directory.mkdir(parents=True, exist_ok=False)
    router = RefinementRouter(planner, executor)
    runtime, full = restore_parent(selected, row, tests, framework, router, directory/'runtime')
    frozen_children = deepcopy(runtime._state()['nodes']); frozen_children.pop('N0')
    frozen_plans = {k: c.plan.model_dump() for k,c in runtime.contexts.items()}
    before = runtime.budget.model_dump(); started = time.monotonic()
    executor.llm.prompt_builder = summary_prompt if arm == 'summary' else prepare_prompt
    # Reset after restoration and immediately before the parent call, for BOTH arms.
    set_seed(seed)
    while runtime.stop_reason is None:
        if runtime.active_node_id != 'N0' or runtime.node.phase.value in {'NEW', 'RESOLVE_CHILD', 'WAIT_CHILD'}:
            raise ValueError('Continuation left the frozen parent implementation boundary')
        runtime.step()
    result = runtime.run()
    children = runtime._state()['nodes']; children.pop('N0')
    if children != frozen_children or frozen_plans != {k:c.plan.model_dump() for k,c in runtime.contexts.items()}:
        raise ValueError('Frozen plans or children changed')
    for child,sha in selected['child_hashes'].items():
        if file_hash(runtime.run_dir/runtime.contexts[child].artifact.module_path) != sha:
            raise ValueError('Frozen child source changed')
    final = capture_program(runtime, directory/'final_program')
    scores, scoring = score_program(final, full, row['root_id'])
    events = read_jsonl(runtime.trace.path)
    errors = [e['host_result']['code'] for e in events if e['host_result']['status'] == 'ERROR']
    if any(v['status'] == 'ERROR' for v in scores.values()): errors.append('SCORING_ERROR')
    record = {'state_id': selected['id'], 'root_id': row['root_id'], 'arm': arm, 'seed': seed,
        'source_success': selected['source_success'], 'status': 'ERROR' if errors else 'COMPLETE', 'errors': errors,
        'success': scores['private']['status'] == 'PASS', 'final_scores': scores, 'final_program': final,
        'frozen_children_and_plans': True, 'role_calls': dict(router.calls),
        'protocol_rejections': dict(Counter(e['host_result']['code'] for e in events if e['host_result']['status'] == 'REJECTED')),
        'stop_reason': result['stop_reason'], 'starting_usage': before,
        'continuation_usage': {k:result['usage'][k]-v for k,v in before.items()},
        'scoring_cost': scoring, 'wall_seconds': time.monotonic()-started, 'run_dir': str(directory)}
    write_json(directory/'case_result.json', record)
    print(json.dumps({k:record[k] for k in ('state_id','arm','seed','success','status','wall_seconds')}), flush=True)
    return record


def evaluate(settings, shard, shards):
    from .refinement_policy import adapter_fingerprint, load_policies
    output = Path(settings['output_dir']); request = json.loads((output/'request.json').read_text())
    if request['settings'] != settings or request['framework'] != framework_for(settings):
        raise ValueError('Changed frozen request')
    for key,sha in request['adapters'].items():
        if file_hash(Path(settings[key])/'adapter_model.safetensors') != sha: raise ValueError('Changed adapter')
    framework = request['framework']; rows = load_root_manifest(settings['root_manifest'])
    tests = {r['root_id']:r for r in read_jsonl(settings['test_manifest'])}
    planner, executor = load_policies(framework, settings['initial_adapter'], settings['executor_adapter'])
    before = {name:adapter_fingerprint(planner.model, name) for name in ('default','executor')}
    records = []
    for i,selected in enumerate(request['states']):
        if i % shards != shard: continue
        for repetition in range(settings['repetitions']):
            seed = settings['evaluation_seed'] + i*1000 + repetition
            # Alternate arm order; each call independently resets the same RNG seed.
            for arm in (('control','summary') if repetition % 2 == 0 else ('summary','control')):
                records.append(run_parent(selected, rows[selected['root_id']], tests, framework, planner, executor,
                    arm, output/'evaluation'/selected['id']/f'repeat_{repetition:02d}'/arm, seed))
    after = {name:adapter_fingerprint(planner.model, name) for name in ('default','executor')}
    if before != after: raise ValueError('Evaluation changed model weights')
    write_json(output/f'shard_{shard}.json', {'status':'COMPLETE', 'shard':shard, 'shards':shards,
        'frozen_before':before, 'frozen_after':after, 'optimizer_updates':0, 'cases':len(records)})


def paired_summary(records):
    pairs = {}
    for r in records:
        key = (r['state_id'],r['seed'])
        if r['arm'] in pairs.setdefault(key, {}): raise ValueError('Duplicate paired slot')
        pairs[key][r['arm']] = r
    if any(set(p) != {'control','summary'} for p in pairs.values()): raise ValueError('Incomplete pair')
    return {'pairs':len(pairs),
        'control_passes':sum(p['control']['success'] for p in pairs.values()),
        'summary_passes':sum(p['summary']['success'] for p in pairs.values()),
        'wins':sum(p['summary']['success'] and not p['control']['success'] for p in pairs.values()),
        'losses':sum(p['control']['success'] and not p['summary']['success'] for p in pairs.values()),
        'ties':sum(p['control']['success'] == p['summary']['success'] for p in pairs.values())}


def report(settings, shards):
    output = Path(settings['output_dir']); request = json.loads((output/'request.json').read_text())
    records = []
    for i,s in enumerate(request['states']):
        for j in range(settings['repetitions']):
            for arm in ('control','summary'):
                r = json.loads((output/'evaluation'/s['id']/f'repeat_{j:02d}'/arm/'case_result.json').read_text())
                if (r['status'] != 'COMPLETE' or r['seed'] != settings['evaluation_seed']+i*1000+j
                        or not r['frozen_children_and_plans'] or r['role_calls'].get('planner',0)):
                    raise ValueError('Incomplete or changed frozen replay')
                records.append(r)
    checks = [json.loads((output/f'shard_{i}.json').read_text()) for i in range(shards)]
    if (sum(c['cases'] for c in checks) != len(records) or
            any(c['frozen_before'] != c['frozen_after'] or c['optimizer_updates'] for c in checks)):
        raise ValueError('Incomplete frozen model checks')
    paired = paired_summary(records)
    strata = {name:paired_summary([r for r in records if r['source_success'] == flag])
              for name,flag in [('source_success',True),('source_failure',False)]}
    result = {'status':'COMPLETE', 'new_parent_continuations':len(records), 'optimizer_updates':0,
        'paired':paired, 'by_source_outcome':strata,
        'by_state':{s['id']:paired_summary([r for r in records if r['state_id'] == s['id']]) for s in request['states']},
        'supports_summary': paired['summary_passes'] > paired['control_passes'] and
            strata['source_success']['summary_passes'] >= strata['source_success']['control_passes'],
        'records':records, 'frozen_model_checks':checks}
    write_json(output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','frozen_model_checks'}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','evaluate','report'))
    parser.add_argument('--config', default='configs/refinement_parent_composition.yaml')
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=3)
    args = parser.parse_args(); settings = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'prepare':
        print(json.dumps([{k:v for k,v in s.items() if k != 'observation'} for s in prepare(settings)]))
    elif args.command == 'evaluate': evaluate(settings,args.shard,args.shards)
    else: report(settings,args.shards)


if __name__ == '__main__':
    main()
