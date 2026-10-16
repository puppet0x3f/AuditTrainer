"""Finite responsibility-structure curriculum using the existing plan/runtime."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, load_task, score_program, write_json
from .refinement_curriculum import transfer_definition
from .refinement_generalization import guarded_json
from .refinement_policy import atomic_plan
from .refinement_pr1 import load_config
from .refinement_rollout import runtime_for
from .refinement_scoring_patch import CASES
from .training_data import ReferencePolicy
from .training_tasks import function, oracle_a, oracle_b
from .verifier import canonical


def allocation(contract, definition, variant):
    action = atomic_plan(contract)
    if variant == 'atomic': return action
    keys = {'parent_aggregate': ['prepare'], 'parent_prepare': ['aggregate'],
            'pipeline': ['prepare', 'aggregate'], 'joint_obligation': ['prepare', 'aggregate'],
            'fused': ['compute']}[variant]
    args = action['args']; args['mode'] = 'SPLIT'
    args['children'] = [{'local_key': key, 'contract': deepcopy(definition['local_contracts'][key])} for key in keys]
    delegated = {'prepare': 'compute', 'aggregate': 'compute'} if keys == ['compute'] else {key: key for key in keys}
    args['retained_responsibilities'] = [r for r in contract['owns'] if r not in delegated]
    args['obligation_map'] = {}
    for obligation in contract['ensures']:
        key = obligation['id']
        args['obligation_map'][key] = ([f'child:{name}' for name in keys] if key == 'RESULT'
            else ['child:'+delegated[key.lower()]] if key.lower() in delegated else ['self'])
    if keys == ['prepare', 'aggregate']:
        args['dataflow'] = [{'from_child': 'prepare', 'from_output': 'return', 'to_child': 'aggregate', 'to_parameter': 'prepared'}]
    return action


def internal_variant(first):
    return 'parent_aggregate' if first in {'calendar', 'chunks'} else 'pipeline'


class CourseReference(ReferencePolicy):
    """Authored executable teacher, explicitly distinct from frozen F01."""
    def __init__(self, definition, reference, structure):
        super().__init__(definition, reference, plan='atomic')
        self.structure = structure
        self.identity = {'backend': 'authored_course_reference', 'structure': structure}

    def decide(self, observation):
        phase = observation['phase']; contract = observation['incoming_contract']
        symbol = contract['signature']['name']
        if phase == 'NEW':
            variant = 'atomic'
            if observation['node_id'] == 'N0':
                variant = 'fused' if self.structure == 'recursive' else self.structure
            elif self.structure == 'recursive' and symbol == 'compute':
                variant = internal_variant(self.ref['component_lineage'][0])
            return 'plan', allocation(contract, self.d, variant)['args']
        if phase == 'IMPLEMENT':
            return 'write_code', {'source': self.source(observation), 'contract_version': contract['version'],
                                  'declared_dependencies': observation['approved_dependencies']}
        return super().decide(observation)

    def source(self, observation):
        symbol = observation['incoming_contract']['signature']['name']
        if symbol in {'prepare', 'aggregate'}: return self.ref[symbol]
        dependencies = observation['approved_dependencies']
        names = {c['node_id']: c['contract']['signature']['name'] for c in observation['child_interfaces']}
        bound = {names[d['child_id']]: d['alias'] for d in dependencies}
        code = 'import json\n' if symbol == 'solve' else ''
        for d in dependencies:
            code += f"from nodes.{d['child_id']} import {names[d['child_id']]} as {d['alias']}\n"
        body = 'items = json.loads(text)\n' if symbol == 'solve' else ''
        if 'compute' in bound:
            body += f"result = {bound['compute']}(items)\n"
        else:
            for key in ('prepare', 'aggregate'):
                if key not in bound:
                    code += '\n'+self.ref[key].replace(f'def {key}(', f'def _{key}(')
            body += f"prepared = {bound.get('prepare', '_prepare')}(items)\n"
            body += f"result = {bound.get('aggregate', '_aggregate')}(prepared)\n"
        body += "return json.dumps(result, separators=(',', ':'))" if symbol == 'solve' else 'return result'
        return code+'\n'+function(symbol, 'text' if symbol == 'solve' else 'items', body)


def task_pack(first, second, parameter, structure):
    definition, private, reference = transfer_definition(first, second, parameter)
    old_id = definition['task']['root_id']; root_id = old_id.replace('train_transfer_', 'train_course_')
    definition['task']['root_id'] = root_id
    for key in ('visible_suite_ref', 'private_suite_ref'):
        definition['task'][key] = definition['task'][key].replace(old_id, root_id)
    for suite in definition['suites']+private['suites']:
        suite['suite_id'] = suite['suite_id'].replace(old_id, root_id)
    if structure == 'joint_obligation':
        incoming = definition['contract']
        texts = {o['id']: o['text'] for o in incoming['ensures']}
        incoming['ensures'] = [incoming['ensures'][0],
            {'id': 'RESULT', 'text': texts['PREPARE']+' Then: '+texts['AGGREGATE']}, incoming['ensures'][-1]]
        for suite in definition['suites']+private['suites']:
            if suite['check_kind'] == 'ROOT': suite['covered_obligations'] = ['PARSE', 'RESULT', 'FORMAT']
    if (first, second) == ('calendar', 'matrix'):
        for suite in private['suites']:
            key = suite.get('contract_key', 'root')
            for i, (items, _, _) in enumerate(CASES):
                prepared = oracle_a(first, items); answer = oracle_b(second, prepared, parameter)
                suite['cases'].append({'case_id': f'{key}_score_v2_{i}',
                    'args': [json.dumps(items)] if key == 'root' else [prepared if key == 'aggregate' else items],
                    'expected': json.dumps(answer, separators=(',', ':')) if key == 'root' else prepared if key == 'prepare' else answer})
    reference['structure'] = structure
    return definition, private, reference


def prepare(study):
    output, data = Path(study['output_dir']), Path(study['data_dir'])
    output.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    config, framework = load_config(study['source_config'])
    framework['runtime']['max_depth'] = 2
    heldout = load_root_manifest(study['validation_roots'])
    heldout_families = {r['family_id'] for r in heldout.values()}
    rows, tests, selections = [], [], []
    for structure, entries in study['tasks'].items():
        for first, second, parameter in entries:
            definition, private, reference = task_pack(first, second, parameter, structure)
            task = definition['task']
            if task['family_id'] in heldout_families: raise ValueError('Course crosses a held-out composition family')
            folder = data/task['root_id']; folder.mkdir(exist_ok=True)
            for name, value in [('task', definition), ('private_tests', private), ('reference', reference)]:
                guarded_json(folder/f'{name}.json', value)
            rows.append({**{k: task[k] for k in ('root_id', 'family_id', 'split')}, 'source': 'responsibility_course_v1',
                'original_split': 'TRAIN', 'io_type': 'solve_text', 'fixture_path': str(folder/'task.json'),
                'source_hash': file_hash(folder/'task.json'), 'reference_path': str(folder/'reference.json'),
                'reference_hash': file_hash(folder/'reference.json')})
            tests.append({'root_id': task['root_id'], 'path': str(folder/'private_tests.json'), 'sha256': file_hash(folder/'private_tests.json')})
            selections.append({'root_id': task['root_id'], 'structure': structure, 'components': [first, second],
                               'parameter': parameter, 'role': 'refinement_space' if structure == 'recursive' and first in {'calendar', 'intervals'} else 'stop_control' if structure == 'recursive' else 'course'})
    if len(rows) != 24 or len({r['root_id'] for r in rows}) != 24: raise ValueError('Expected 24 unique roots')
    for name, values in [('roots', rows), ('tests', tests), ('validation_roots', list(heldout.values())),
                         ('validation_tests', read_jsonl(study['validation_tests']))]:
        path = data/f'{name}.jsonl'; text = ''.join(canonical(r)+'\n' for r in values)
        if path.exists() and path.read_text() != text: raise ValueError('Changed course manifest')
        path.write_text(text)
    config.update(root_manifest=str(data/'roots.jsonl'), test_manifest=str(data/'tests.jsonl'), root_limit=24,
                  initial_adapter=study['initial_adapter'], output_dir=str(output), seed=study['seed'])
    config['runtime_overrides']['max_depth'] = 2
    guarded_json(output/'request.json', {'study': study, 'config': config, 'framework': framework,
        'selection': selections, 'train_roots': len(rows), 'train_composition_families': len({r['family_id'] for r in rows}),
        'validation_roots': len(heldout), 'validation_scope': 'Existing untrained development compositions; not a fresh confirmation set',
        'supervision_scope': 'Authored executable structural demonstrations; not optimal-mode labels or on-policy F01 rewards'})
    guarded_json(output/'gpu_resources.json', {'devices': ['0', '1', '2'], 'min_free_mib': {'collect': 16000, 'train': 22000}})
    return config, framework, selections


def references(study):
    config, framework, selections = prepare(study)
    rows = load_root_manifest(config['root_manifest']); tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    output = Path(study['output_dir']); summaries = []; examples = []
    for selected in selections:
        row = rows[selected['root_id']]; folder = output/'references'/row['root_id']
        result_path = folder/'result.json'
        if result_path.exists():
            summary = json.loads(result_path.read_text())
        else:
            definition = json.loads(Path(row['fixture_path']).read_text())
            reference = json.loads(Path(row['reference_path']).read_text())
            policy = CourseReference(definition, reference, selected['structure'])
            runtime, full = runtime_for(row, tests, framework, policy, folder/'runtime')
            result = runtime.run()
            snapshot = capture_program(runtime, folder/'final_program')
            root_scores, cost = score_program(snapshot, full, row['root_id'])
            local = []
            for node_id, ctx in runtime.contexts.items():
                if node_id == 'N0' or ctx.artifact is None: continue
                for suite in full.suites.values():
                    if suite.visibility != 'PRIVATE' or not suite.applies(row['root_id'], ctx.node.incoming_contract): continue
                    evidence = full.verify(runtime.run_dir, row['root_id'], ctx.node.incoming_contract, ctx.artifact,
                        suite.suite_id, f'COURSE_PRIVATE_{len(local)}', allow_private=True)
                    local.append({'node_id': node_id, 'status': evidence.status, 'executed_cases': evidence.details['executed_cases']})
            summary = {'root_id': row['root_id'], 'structure': selected['structure'], 'root_scores': root_scores,
                'local_checks': local, 'node_count': result['node_count'], 'usage': result['usage'], 'root_scoring_cost': cost,
                'status': 'COMPLETE' if result['functional_status'] == 'PASS' and root_scores['private']['status'] == 'PASS'
                    and all(r['status'] == 'PASS' for r in local) and len(local) == result['node_count']-1 else 'FAILED',
                'teacher': 'authored_reference_not_F01', 'runtime_dir': str(runtime.run_dir)}
            write_json(result_path, summary)
        if summary['status'] != 'COMPLETE': raise ValueError(f"Reference execution failed: {row['root_id']}")
        summaries.append(summary)
        for event in read_jsonl(Path(summary['runtime_dir'])/'trace.jsonl'):
            obs = event['observation_before']
            if event['host_result']['code'] != 'PLAN_REGISTERED' or set(obs['allowed_plan_modes']) != {'ATOMIC', 'SPLIT'}: continue
            examples.append({'sample_id': row['root_id']+'_'+event['node_id'], 'root_id': row['root_id'],
                'family_id': row['family_id'], 'split': 'TRAIN', 'node_id': event['node_id'], 'structure': selected['structure'],
                'target_action': event['parsed_action'], 'messages': [{}, {'content': canonical({'observation': obs})}],
                'source_trace': summary['runtime_dir'], 'evidence_scope': 'Authored allocation, exact reference consumption/composition, root and local tests; not universal semantic proof'})
        print(json.dumps({'reference_root': row['root_id'], 'status': summary['status'], 'nodes': summary['node_count']}), flush=True)
    (output/'demonstrations.jsonl').write_text(''.join(canonical(e)+'\n' for e in examples))
    result = {'status': 'COMPLETE', 'roots': len(summaries), 'nodes': sum(s['node_count'] for s in summaries),
        'learnable_plan_states': len(examples), 'nonroot_plan_states': sum(e['node_id'] != 'N0' for e in examples),
        'by_mode': dict(Counter(e['target_action']['args']['mode'] for e in examples)),
        'by_structure': dict(Counter(s['structure'] for s in summaries)), 'references': summaries,
        'optimizer_updates': 0, 'model_calls': 0, 'host_forced_atomic_excluded': True}
    write_json(output/'reference_result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'references'])
    parser.add_argument('--config', default='configs/refinement_responsibility_course.yaml')
    args = parser.parse_args(); study = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'prepare': prepare(study)
    else: references(study)
