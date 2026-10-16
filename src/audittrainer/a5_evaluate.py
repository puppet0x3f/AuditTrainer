"""Local-plan, fixed-contract and free recursive A5 development evaluations."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import yaml

from .cli import fixture_verifier
from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, evaluate, score_program, write_json
from .mechanism_tasks import CATALOG, PRIVATE_INPUTS
from .policy import LLMPolicy
from .runtime import Limits, ProtocolError, RecursiveRuntime
from .schema import ContractSpec, PolicyOutput, RootTask, TokenUsage
from .verifier import canonical


def fixed(action, args):
    return PolicyOutput(raw_text=canonical({'action': action, 'args': args}), token_usage=TokenUsage(),
                        model_identity={'backend': 'a5_fixed_protocol', 'name': 'no model decision'})


def atomic(contract):
    return {'base_plan_version': 0, 'mode': 'ATOMIC', 'children': [], 'dataflow': [],
            'obligation_map': {o['id']: ['self'] for o in contract['ensures']},
            'retained_responsibilities': contract['owns'], 'ticket_id': None}


def parent_plan(d, condition):
    keys = ['a', 'b'] if condition == 'parent_both_children' else ['a']
    return {'base_plan_version': 0, 'mode': 'SPLIT',
            'children': [{'local_key': k, 'contract': d['local_contracts'][k]} for k in keys],
            'obligation_map': {'PARSE': ['self'], 'FORMAT': ['self'], 'TRANSFORM': ['child:a'],
                               'SUMMARIZE': ['child:b'] if 'b' in keys else ['self']},
            'retained_responsibilities': [o['text'] for o in d['contract']['ensures']
                                         if o['id'] in ({'PARSE', 'FORMAT'} if 'b' in keys else {'PARSE', 'FORMAT', 'SUMMARIZE'})],
            'dataflow': [{'from_child': 'a', 'from_output': 'return', 'to_child': 'b', 'to_parameter': 'items'}] if 'b' in keys else [],
            'ticket_id': None}


def component_task(d, condition):
    """Same DEV root, explicitly evaluated as one component, never a new split."""
    d = deepcopy(d)
    key = 'a' if condition == 'transform' else 'b'
    spec = d['local_contracts'][key]
    d['contract'] = spec
    d['task']['entrypoint'] = spec['signature']['name']
    visible = next(s for s in d['suites'] if s.get('contract_key') == key)
    root_id = d['task']['root_id']
    entry = CATALOG[int(root_id[4:6]) - 1]
    cases = []
    for index, items in enumerate(PRIVATE_INPUTS):
        transformed = eval(entry[2], {'items': items})  # Closed authored catalog only.
        argument = items if key == 'a' else transformed
        expected = transformed if key == 'a' else eval(entry[4], {'items': transformed})
        cases.append({'case_id': f'{condition}_private_{index}', 'args': [argument], 'expected': expected})
    d['suites'] = [{**{k: v for k, v in visible.items() if k != 'contract_key'}, 'check_kind': 'ROOT'},
                   {'suite_id': root_id + '_' + condition + '_private', 'check_kind': 'ROOT',
                    'visibility': 'PRIVATE', 'oracle_origin': 'A5 component scoring from frozen A2 catalog',
                    'covered_obligations': ['RESULT'], 'cases': cases}]
    d['local_contracts'] = {}
    return d


class IsolationRuntime(RecursiveRuntime):
    """Frozen contracts and child artifacts: no replanning/repair action is enabled."""
    def build_observation(self):
        obs = super().build_observation()
        obs['allowed_actions'] = [a for a in obs['allowed_actions'] if a != 'request_child_repair']
        return obs

    def _dispatch(self, action):
        if action.action in {'request_child_repair', 'localize', 'repair'}:
            raise ProtocolError('ISOLATION_BOUNDARY', 'Contracts and child artifacts are frozen for this evaluation')
        return super()._dispatch(action)


class IsolationPolicy:
    def __init__(self, student, executor, d, reference, condition):
        self.student, self.executor, self.d, self.reference, self.condition = student, executor, d, reference, condition
        self.student_calls = self.executor_calls = self.fixed_calls = 0

    def act(self, obs):
        root, phase = obs['node_id'] == 'N0', obs['phase']
        if self.condition == 'local_plan':
            if root and phase == 'NEW':
                self.student_calls += 1
                return self.student.act(obs)
            if phase in {'IMPLEMENT', 'RESOLVE_CHILD'}:
                self.executor_calls += 1
                return self.executor.act(obs)
        elif root and phase in {'IMPLEMENT', 'RESOLVE_CHILD'}:
            self.student_calls += 1
            return self.student.act(obs)
        self.fixed_calls += 1
        if phase == 'NEW':
            args = parent_plan(self.d, self.condition) if root and self.condition.startswith('parent_') else atomic(obs['incoming_contract'])
            return fixed('plan', args)
        if phase == 'IMPLEMENT':
            symbol = obs['incoming_contract']['signature']['name']
            key = 'a' if symbol == 'transform' else 'b'
            return fixed('write_code', {'source': self.reference[key], 'contract_version': obs['incoming_contract']['version'],
                                        'declared_dependencies': obs['approved_dependencies']})
        if phase == 'READY_VERIFY':
            return fixed('verify', {'target_ref': obs['current_artifact']['module_path'], 'suite_id': obs['available_suites'][0]})
        if phase == 'REVIEW':
            return fixed('finish', {'artifact_version': obs['current_artifact']['version']})
        return fixed('abstain', {'reason_code': 'INSUFFICIENT_EVIDENCE', 'evidence_ids': []})


class BaseExecutor:
    """Identical frozen theta0 executor in every arm, sharing memory with PEFT."""
    def __init__(self, student, framework):
        self.student, self.seed, self.calls = student, framework['inference']['seed'], 0
        identity = {k: v for k, v in student.identity.items() if not any(
            word in k for word in ('adapter', 'training', 'saved_tokenizer'))}
        identity.update(adapter='none', executor='A5 public theta0; adapter disabled')
        self.policy = LLMPolicy(student.model, student.tokenizer, identity=identity,
            context_tokens=student.context_tokens, generation_tokens=student.generation_tokens, decoding=student.decoding)

    def act(self, obs):
        import torch
        from transformers import set_seed
        with torch.random.fork_rng(devices=[0]), self.student.model.disable_adapter():
            set_seed(self.seed + self.calls)
            self.calls += 1
            return self.policy.act(obs)


def composition_review(runtime, condition):
    """A narrow sound source rule; other functional implementations stay UNKNOWN."""
    ctx = runtime.contexts['N0']
    if ctx.artifact is None:
        return {'status': 'UNKNOWN', 'reason': 'NO_PROGRAM'}
    if condition != 'parent_both_children':
        return {'status': 'UNKNOWN', 'reason': 'SEMANTIC_SOURCE_REVIEW_NOT_AUTOMATED'}
    src = (runtime.run_dir / ctx.artifact.module_path).read_text()
    aliases = {dep.alias: dep.child_id for dep in ctx.artifact.dependency_bindings}
    try:
        tree = ast.parse(src)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve')
        # Imports can be shadowed by a private helper with the same name. Only
        # accept this closed composition form when each alias binds the selected
        # child and no extra definition or assignment can replace that binding.
        imported = set()
        for node in tree.body:
            if node is function:
                continue
            if isinstance(node, ast.Import) and all(v.name == 'json' and v.asname in {None, 'json'} for v in node.names):
                continue
            if isinstance(node, ast.ImportFrom):
                for v in node.names:
                    alias = v.asname or v.name
                    child_id = aliases.get(alias)
                    if (child_id is None or node.module != 'nodes.' + child_id or node.level != 0
                            or v.name != runtime.contexts[child_id].artifact.public_symbol or alias in imported):
                        raise ValueError('Unreviewed import binding')
                    imported.add(alias)
                continue
            raise ValueError('Extra module definition or execution')
        if imported != set(aliases) or not isinstance(function.body[-1], ast.Return):
            raise ValueError('Incomplete composition form')
        env = {function.args.args[0].arg: 'text'}

        def expr(node):
            if isinstance(node, ast.Name):
                return env[node.id]
            if isinstance(node, ast.Call):
                if node.keywords and not (isinstance(node.func, ast.Attribute) and node.func.attr == 'dumps'
                    and len(node.keywords) == 1 and node.keywords[0].arg == 'separators'
                    and ast.literal_eval(node.keywords[0].value) == (',', ':')):
                    raise ValueError('Unreviewed keyword')
                if len(node.args) != 1:
                    raise ValueError('Unreviewed call')
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'json':
                    name = {'loads': 'parse', 'dumps': 'format'}[node.func.attr]
                elif isinstance(node.func, ast.Name) and node.func.id in aliases:
                    name = 'a' if aliases[node.func.id].endswith('_a') else 'b'
                else:
                    raise ValueError('Unreviewed operation')
                return (name, expr(node.args[0]))
            raise ValueError('Unreviewed expression')

        output = None
        for index, stmt in enumerate(function.body):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if stmt.targets[0].id in {'json', *aliases}:
                    raise ValueError('Shadowed import')
                env[stmt.targets[0].id] = expr(stmt.value)
            elif isinstance(stmt, ast.Return) and index == len(function.body) - 1:
                output = expr(stmt.value)
            else:
                raise ValueError('Unreviewed statement')
        if output == ('format', ('b', ('a', ('parse', 'text')))) and len(aliases) == 2:
            return {'status': 'SUPPORTED', 'reason': 'Straight-line parse -> selected transform -> selected summarize -> format; no other algorithm'}
    except (ValueError, KeyError, StopIteration, SyntaxError, IndexError):
        pass
    return {'status': 'UNKNOWN', 'reason': 'OUTSIDE_REVIEWED_STRAIGHT_LINE_COMPOSITION'}


def run_case(d, reference, student, executor, condition, directory, framework):
    if condition in {'transform', 'summarize'}:
        d = component_task(d, condition)
    online_d = {**d, 'suites': [s for s in d['suites'] if s['visibility'] == 'VISIBLE']}
    online, full = fixture_verifier(online_d, framework), fixture_verifier(d, framework)
    policy = IsolationPolicy(student, executor, d, reference, condition)
    limits = Limits(**{k: framework['runtime'][k] for k in Limits.model_fields})
    runtime = IsolationRuntime(RootTask.model_validate(d['task']), ContractSpec.model_validate(d['contract']),
        policy, directory / 'runtime', limits=limits, protocol_version=framework['protocol_version'], implementation_stage='A5', verifier=online)
    write_json(directory / 'request.json', {'condition': condition, 'evaluation_scope': 'local_plan' if condition == 'local_plan' else 'fixed_contract',
        'root_id': d['task']['root_id'], 'split': d['task']['split'], 'incoming_contract': d['contract'],
        'source_selection': 'first and terminal before private scoring', 'model_identity': student.identity})
    first = None
    started = time.monotonic()
    while runtime.stop_reason is None:
        event = runtime.step()
        if event.node_id == 'N0' and event.host_result.code == 'SOURCE_SAVED' and first is None:
            first = capture_program(runtime, directory / 'first_program')
    result = runtime.run()
    final = capture_program(runtime, directory / 'final_program')
    scoring = {'verify_calls': 0, 'executed_cases': 0, 'execution_seconds': 0.0}
    first_scores, _ = score_program(first, full, d['task']['root_id'], scoring)
    final_scores, _ = score_program(final, full, d['task']['root_id'], scoring)
    readonly = True
    if condition.startswith('parent_'):
        for node_id, ctx in runtime.contexts.items():
            if node_id == 'N0' or ctx.artifact is None:
                continue
            symbol = ctx.node.incoming_contract.signature.name
            readonly &= (runtime.run_dir / ctx.artifact.module_path).read_text() == reference['a' if symbol == 'transform' else 'b']
            readonly &= ctx.artifact.version == 1
    events = read_jsonl(runtime.run_dir / 'trace.jsonl')
    accepted_plan = next((e['parsed_action']['args'] for e in events if e['node_id'] == 'N0' and e['host_result']['code'] == 'PLAN_REGISTERED'), None)
    record = {'root_id': d['task']['root_id'], 'condition': condition, 'split': d['task']['split'],
        'stop_reason': result['stop_reason'], 'first_scores': first_scores, 'final_scores': final_scores,
        'student_calls': policy.student_calls, 'executor_calls': policy.executor_calls, 'fixed_actions': policy.fixed_calls,
        'online_cost': result['usage'], 'scoring_cost': scoring, 'node_count': result['node_count'],
        'accepted_plan_mode': accepted_plan['mode'] if accepted_plan else None,
        'host_rejections': dict(Counter(e['host_result']['code'] for e in events if e['host_result']['status'] == 'REJECTED')),
        'responsibility_review': composition_review(runtime, condition), 'child_artifacts_frozen': readonly,
        'first_program': first, 'final_program': final, 'wall_seconds': time.monotonic() - started}
    write_json(directory / 'result.json', record)
    return record


def run_group(group, adapter, config):
    from transformers import set_seed
    framework = yaml.safe_load(Path(config['framework_config']).read_text())
    framework['model']['load_in_4bit'] = True
    framework['inference']['seed'] = config['seed']
    roots = load_root_manifest(config['root_manifest'])
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    directory = Path(config['output_dir']) / group
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', {'config': config, 'group': group, 'adapter': str(adapter),
        'framework': framework, 'evaluation_implementation_sha256': file_hash(__file__),
        'scope': 'development only; A2 roots remain DEV_MECHANISM, never trained or relabelled as unseen'})
    student = LLMPolicy.from_pretrained(framework, adapter_path=adapter)
    records = []
    probe_done = False
    for root_id in config['isolation_roots']:
        row = roots[root_id]
        assert row['split'] == 'DEV_MECHANISM'
        assert file_hash(row['fixture_path']) == row['source_hash']
        assert file_hash(row['reference_path']) == row['reference_hash']
        assert file_hash(tests[root_id]['path']) == tests[root_id]['sha256']
        d = json.loads(Path(row['fixture_path']).read_text())
        d['suites'] += json.loads(Path(tests[root_id]['path']).read_text())['suites']
        ref = json.loads(Path(row['reference_path']).read_text())
        if not probe_done:
            # This probe is model-effect evidence, separate from task performance.
            probe_runtime = IsolationRuntime(RootTask.model_validate(d['task']), ContractSpec.model_validate(d['contract']),
                None, directory / 'probe_state', verifier=fixture_verifier({**d, 'suites': [s for s in d['suites'] if s['visibility'] == 'VISIBLE']}, framework))
            probe, _ = student.check_adapter_effect(probe_runtime.build_observation())
            write_json(directory / 'adapter_probe.json', probe)
            assert probe['status'] == 'PASS'
            probe_done = True
        for condition in config['isolation_conditions']:
            set_seed(config['seed'])
            executor = BaseExecutor(student, framework)
            record = run_case(d, ref, student, executor, condition, directory / condition / root_id, framework)
            records.append(record)
            with (directory / 'isolation.jsonl').open('a') as stream:
                stream.write(canonical(record) + '\n')
            print(group, condition, root_id, record['stop_reason'], record['final_scores']['private']['status'], flush=True)
    free = yaml.safe_load(Path('configs/evaluation_main.yaml').read_text())
    free.update(stage='A5', scope='DEV_ENGINEERING', split=config['free_split'], limit=config['free_limit'], seed=config['seed'], output_dir=str(directory / 'free_recursive'))
    # The factory reuses the loaded model, so it does not perform the usual
    # from_pretrained seed reset. Start the free batch independently of isolation.
    set_seed(config['seed'])
    free_result = evaluate(free, policy_name='sft', adapter=adapter, load_in_4bit=True,
                           policy_factory=lambda *args, **kwargs: student)
    summary = {'group': group, 'status': 'COMPLETE', 'conditions': {}, 'free_result': free_result}
    for condition in config['isolation_conditions']:
        subset = [r for r in records if r['condition'] == condition]
        summary['conditions'][condition] = {'cases': len(subset),
            'private_status': dict(Counter(r['final_scores']['private']['status'] for r in subset)),
            'first_private_pass': sum(r['first_scores']['private']['status'] == 'PASS' for r in subset),
            'accepted_plan_modes': dict(Counter(str(r['accepted_plan_mode']) for r in subset)),
            'responsibility_status': dict(Counter(r['responsibility_review']['status'] for r in subset)),
            'student_calls': sum(r['student_calls'] for r in subset), 'executor_calls': sum(r['executor_calls'] for r in subset),
            'frozen_children': all(r['child_artifacts_frozen'] for r in subset)}
    write_json(directory / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['F00', 'F10', 'F01', 'F11'], required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/a5_evaluation.yaml'))
    args = parser.parse_args()
    run_group(args.group, args.adapter, yaml.safe_load(args.config.read_text()))


if __name__ == '__main__':
    main()
