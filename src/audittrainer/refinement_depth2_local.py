"""Fixed-root depth-two diagnostic, with one open internal planning decision."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_depth2_local.yaml'
    _study = yaml.safe_load(Path(_path).read_text())
    _reservation = acquire(Path(_study['output_dir'])/'gpu_resources.json', 'collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_parent_validation import presentation_identity
from .refinement_policy import adapter_fingerprint, atomic_plan, load_policies
from .refinement_responsibility_course import allocation, internal_variant
from .refinement_rollout import run_case, case_summary
from .schema import PolicyOutput, TokenUsage
from .verifier import canonical

INTERNAL = 'N0_p1_compute'


class LocalPlanner:
    def __init__(self, base, definition, first_component, arm):
        self.base, self.definition, self.first_component, self.arm = base, definition, first_component, arm
        self.cost = base.cost
        self.identity = {'backend': 'fixed_root_open_internal', 'arm': arm, 'learned_planner': canonical(base.identity)}

    def act(self, observation):
        node = observation['node_id']
        if node == INTERNAL and self.arm == 'NATURAL': return self.base.act(observation)
        if node == 'N0': action = allocation(observation['incoming_contract'], self.definition, 'fused')
        elif node == INTERNAL and self.arm == 'REFERENCE_SPLIT':
            action = allocation(observation['incoming_contract'], self.definition, internal_variant(self.first_component))
        else: action = atomic_plan(observation['incoming_contract'])
        return PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(), model_identity=self.identity)


def prepare(study):
    course = yaml.safe_load(Path(study['course_config']).read_text())
    request = json.loads((Path(course['output_dir'])/'request.json').read_text())
    reference = json.loads((Path(course['output_dir'])/'reference_result.json').read_text())
    if reference['status'] != 'COMPLETE': raise ValueError('Reference execution is incomplete')
    selections = [s for s in request['selection'] if s['structure'] == 'recursive']
    if len(selections) != 4: raise ValueError('Expected four depth-two diagnostic roots')
    output = Path(study['output_dir']); output.mkdir(parents=True, exist_ok=True)
    guarded_json(output/'request.json', {'study': study, 'course': course, 'config': request['config'],
        'framework': request['framework'], 'selection': selections, 'planned_runs': 48,
        'presentation': presentation_identity(), 'internal_node': INTERNAL,
        'scope': 'Known TRAIN roots and fixed authored root plan; one internal choice; no RL or functional repair',
        'seeding': 'Common seed reset after fixed root and after accepted internal plan; no budget reset'})
    guarded_json(output/'gpu_resources.json', {'devices': ['0', '1', '2'], 'min_free_mib': {'collect': 16000}})
    return json.loads((output/'request.json').read_text())


def chain_record(case):
    runtime = Path(case['run_dir'])/'runtime'
    result = json.loads((runtime/'result.json').read_text())
    events = read_jsonl(runtime/'trace.jsonl')
    attempts = [e for e in events if e['node_id'] == INTERNAL and e['event_kind'] == 'POLICY'
                and e['observation_before']['phase'] == 'NEW']
    accepted = [e for e in attempts if e['host_result']['code'] == 'PLAN_REGISTERED']
    compositions = [e for e in events if e['event_kind'] == 'POLICY'
                    and e['observation_before']['phase'] == 'IMPLEMENT' and e['observation_before']['mode'] == 'COMPOSE']
    summaries_present = ['Parent composition summary' in json.dumps(e.get('model_input')) for e in compositions]
    if not all(summaries_present): raise ValueError('Frozen composition summary missing at a parent depth')
    return {'seed': case['seed'], 'root_id': case['root_id'], 'success': case['final_scores']['private']['status'] == 'PASS',
        'internal_first_registered': bool(attempts and attempts[0]['host_result']['code'] == 'PLAN_REGISTERED'),
        'internal_mode': accepted[0]['parsed_action']['args']['mode'] if accepted else None,
        'internal_attempts': len(attempts), 'node_count': case['node_count'],
        'grandchildren': sum(n.startswith(INTERNAL+'_') for n in result['node_results']),
        'node_functional_status': {n: r['functional_status'] for n, r in result['node_results'].items()},
        'node_responsibility_status': {n: r['responsibility_status'] for n, r in result['node_results'].items()},
        'consumptions': {n: r.get('responsibility_details', {}).get('consumptions', [])
                         for n, r in result['node_results'].items()},
        'dependency_usage': {n: (r.get('evidence') or {}).get('details', {}).get('dependency_usage', [])
                             for n, r in result['node_results'].items()},
        'parent_compositions': len(compositions), 'summary_present': summaries_present,
        'seed_resets': case.get('continuation_seed_resets_after_plan', []),
        'run_dir': case['run_dir'], 'actual_model_cost': case['actual_model_cost']}


def evaluate(study, arm):
    request = prepare(study); config, framework = deepcopy(request['config']), deepcopy(request['framework'])
    output = Path(study['output_dir'])/'evaluation'/arm; output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists(): return
    # Fixed arms never invoke the learned planner. They can execute with F01
    # while the course trains; only NATURAL uses the new planning adapter.
    adapter = study['planner_adapter'] if arm == 'NATURAL' else config['initial_adapter']
    planner, executor = load_policies(framework, adapter, config['executor_adapter'])
    before = {n: adapter_fingerprint(planner.model, n) for n in ('default', 'executor')}
    rows = load_root_manifest(config['root_manifest']); tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    cases, chains = [], []
    for index, selected in enumerate(request['selection']):
        row = rows[selected['root_id']]; definition = json.loads(Path(row['fixture_path']).read_text())
        wrapped = LocalPlanner(planner, definition, selected['components'][0], arm)
        for rep in range(study['repetitions']):
            seed = study['seed']+1000*index+rep
            case = run_case(row, tests, framework, config, wrapped, executor,
                output/row['root_id']/f'repeat_{rep:02d}', seed, reset_seed_after_root_plan=True,
                reset_seed_after_plan_node=INTERNAL)
            if case['status'] != 'COMPLETE': raise ValueError(f"Infrastructure failure: {case['run_dir']}")
            chain = chain_record(case); cases.append(case); chains.append(chain)
            print(json.dumps({'arm': arm, 'root_id': row['root_id'], 'seed': seed,
                              'success': chain['success'], 'internal_mode': chain['internal_mode'], 'nodes': case['node_count']}), flush=True)
    after = {n: adapter_fingerprint(planner.model, n) for n in before}
    if before != after: raise ValueError('Diagnostic changed planner or executor')
    result = {'status': 'COMPLETE', 'arm': arm, 'adapter': adapter, 'learned_planner_used': arm == 'NATURAL',
        'adapter_sha256': file_hash(Path(adapter)/'adapter_model.safetensors'),
        'summary': case_summary(cases), 'chains': chains, 'frozen_before': before, 'frozen_after': after,
        'optimizer_steps': 0, 'presentation': presentation_identity()}
    write_json(output/'result.json', result)


def report(study):
    output = Path(study['output_dir'])
    from .refinement_curriculum_rl import plan_key
    request = json.loads((output/'request.json').read_text())
    rows = load_root_manifest(request['config']['root_manifest'])
    evaluations = {a: json.loads((output/'evaluation'/a/'result.json').read_text()) for a in study['arms']}
    if len({json.dumps(e['frozen_before']['executor'], sort_keys=True) for e in evaluations.values()}) != 1:
        raise ValueError('The three arms must share the exact frozen executor')
    if sum(e['summary']['runs'] for e in evaluations.values()) != 48: raise ValueError('Incomplete 48-case budget')
    by_root = {}
    for arm, evaluation in evaluations.items():
        for r in evaluation['chains']:
            by_root.setdefault(r['root_id'], {}).setdefault(arm, []).append(r)
    paired = {}
    for root, groups in by_root.items():
        definition = json.loads(Path(rows[root]['fixture_path']).read_text())
        baseline = {r['seed']: r['success'] for r in groups['ATOMIC']}
        paired[root] = {}
        for arm, records in groups.items():
            if set(baseline) != {r['seed'] for r in records}: raise ValueError('Unpaired seeds')
            pairs = [(baseline[r['seed']], r['success']) for r in records]
            allocations = Counter()
            for record in records:
                event = next((e for e in read_jsonl(Path(record['run_dir'])/'runtime/trace.jsonl')
                    if e['node_id'] == INTERNAL and e['host_result']['code'] == 'PLAN_REGISTERED'), None)
                variant = 'NO_REGISTERED_PLAN'
                if event:
                    variant = 'ATOMIC' if event['parsed_action']['args']['mode'] == 'ATOMIC' else 'UNMATCHED'
                    if variant == 'UNMATCHED':
                        key = plan_key(event['parsed_action'], definition)
                        for candidate in ('parent_aggregate', 'parent_prepare', 'pipeline', 'fused'):
                            action = allocation(event['observation_before']['incoming_contract'], definition, candidate)
                            if key is not None and key == plan_key(action, definition):
                                variant = candidate; break
                allocations[variant] += 1
            paired[root][arm] = {'runs': len(records), 'successes': sum(r['success'] for r in records),
                'internal_modes': dict(Counter(r['internal_mode'] for r in records)),
                'registered_allocations': dict(allocations),
                'wins': sum(b > a for a, b in pairs), 'losses': sum(b < a for a, b in pairs), 'ties': sum(b == a for a, b in pairs)}
    result = {'status': 'COMPLETE', 'runs': 48, 'optimizer_steps': 0, 'evaluations': evaluations, 'by_root': paired,
        'scope': 'Controlled internal-node diagnostic on known TRAIN contracts; not natural recursive generalization'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'by_root': paired}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate', 'report'])
    parser.add_argument('--config', default='configs/refinement_depth2_local.yaml')
    parser.add_argument('--arm', choices=['ATOMIC', 'REFERENCE_SPLIT', 'NATURAL'])
    args = parser.parse_args(); study = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'prepare': prepare(study)
    elif args.command == 'evaluate': evaluate(study, args.arm)
    else: report(study)
