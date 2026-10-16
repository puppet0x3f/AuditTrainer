"""Thin A4 data commands over the existing runtime, verifier and SFT exporter."""

import argparse
import ast
from collections import Counter
from copy import deepcopy
import io
import inspect
import json
from pathlib import Path
import time
import tokenize

import yaml

from .cli import fixture_verifier
from .dataset import EventReview, RunData, Rejected, classify, file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .policy import LLMPolicy
from .runtime import Limits, RepairRuntime
from .schema import ContractSpec, PolicyOutput, RootTask, TokenUsage, parse_action
from .trace import renderer_identity
from .training_tasks import FAMILIES, definition, function
from .verifier import canonical, digest


PLANS = ('atomic', 'pipeline', 'fused')
TRANSFORMS = ('omit_output', 'move_algorithm', 'bypass_child', 'stale_artifact', 'wrong_argument', 'expand_owns')


def append_json(path, record):
    with Path(path).open('a') as stream:
        stream.write(canonical(record) + '\n')


def import_roots(directory, manifest, test_manifest, records=None):
    """Normalize reviewed task packs; the default is the new authored TRAIN pilot."""
    existing = load_root_manifest(manifest)
    if records is None:
        records = [{'definition': d, 'private': private, 'reference': ref,
                    'source': 'a4_authored_train_v1', 'original_split': 'TRAIN'}
                   for family in FAMILIES for p in range(1, 6)
                   for d, private, ref in [definition(family, p)]]
    roots, families = set(existing), {r['family_id']: r['split'] for r in existing.values()}
    for record in records:
        task = RootTask.model_validate(record['definition']['task'])
        ContractSpec.model_validate(record['definition']['contract'])
        if task.split != 'TRAIN' or record['original_split'] != 'TRAIN':
            raise ValueError('A4 import requires new TRAIN roots, never relabel development or sealed tasks')
        if task.root_id in roots or families.get(task.family_id, 'TRAIN') != 'TRAIN':
            raise ValueError('Duplicate root or family crossing a split')
        roots.add(task.root_id)
        families[task.family_id] = 'TRAIN'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows, tests = [], []
    for record in records:
        d, private = record['definition'], record['private']
        task = d['task']
        target = directory / task['root_id']
        target.mkdir()
        write_json(target / 'task.json', d)
        write_json(target / 'private_tests.json', private)
        row = {**{k: task[k] for k in ('root_id', 'family_id', 'split')},
               'source': record['source'], 'source_task_id': task['root_id'], 'original_split': record['original_split'],
               'io_type': 'solve_text', 'fixture_path': str(target / 'task.json'),
               'source_hash': file_hash(target / 'task.json')}
        if record.get('reference'):
            write_json(target / 'reference.json', record['reference'])
            row.update(reference_path=str(target / 'reference.json'), reference_hash=file_hash(target / 'reference.json'))
        rows.append(row)
        tests.append({'root_id': task['root_id'], 'path': str(target / 'private_tests.json'),
                      'sha256': file_hash(target / 'private_tests.json')})
    for path, values in [(manifest, rows), (test_manifest, tests)]:
        for row in values:
            append_json(path, row)
    result = {'stage': 'A4', 'split': 'TRAIN', 'roots': rows, 'root_count': len(rows),
              'families': dict(Counter(r['family_id'] for r in rows)),
              'source': 'new authored task specifications; five parameter settings per family',
              'generator_sha256': file_hash(Path(__file__).with_name('training_tasks.py'))}
    write_json(directory / 'selection.json', result)
    return result


def selected(config, limit=None):
    ids = {r['root_id'] for r in json.loads(Path(config['selection']).read_text())['roots']}
    rows = [r for r in load_root_manifest(config['root_manifest']).values() if r['root_id'] in ids]
    if len(rows) != len(ids) or any(r['split'] != 'TRAIN' for r in rows):
        raise ValueError('The selected pilot must contain registered TRAIN roots only')
    return rows[:limit] if limit else rows


def load_task(row, config):
    if file_hash(row['fixture_path']) != row['source_hash']:
        raise ValueError('Task changed after root assignment')
    d = json.loads(Path(row['fixture_path']).read_text())
    tests = next(t for t in read_jsonl(config['test_manifest']) if t['root_id'] == row['root_id'])
    if file_hash(tests['path']) != tests['sha256']:
        raise ValueError('Private oracle changed after assignment')
    private = json.loads(Path(tests['path']).read_text())
    ref = None
    if row.get('reference_path'):
        if file_hash(row['reference_path']) != row['reference_hash']:
            raise ValueError('Reviewed reference changed')
        ref = json.loads(Path(row['reference_path']).read_text())
    return d, private, ref


def plan_args(contract, d, plan, is_root):
    args = {'base_plan_version': 0, 'mode': 'ATOMIC', 'children': [],
            'obligation_map': {o['id']: ['self'] for o in contract['ensures']},
            'retained_responsibilities': contract['owns'], 'dataflow': [], 'ticket_id': None}
    if is_root and plan != 'atomic':
        keys = ('prepare', 'aggregate') if plan == 'pipeline' else ('compute',)
        args.update(mode='SPLIT', children=[{'local_key': k, 'contract': d['local_contracts'][k]} for k in keys],
                    retained_responsibilities=['parse', 'format'],
                    obligation_map={'PARSE': ['self'], 'FORMAT': ['self'],
                                    'PREPARE': ['child:prepare' if plan == 'pipeline' else 'child:compute'],
                                    'AGGREGATE': ['child:aggregate' if plan == 'pipeline' else 'child:compute']})
        if plan == 'pipeline':
            args['dataflow'] = [{'from_child': 'prepare', 'from_output': 'return',
                                'to_child': 'aggregate', 'to_parameter': 'prepared'}]
    return args


def restyle(source, style):
    if style == 'renamed':
        names = {'totals': 'balances', 'names': 'selected_names', 'pairs': 'entries', 'result': 'answer', 'chosen': 'selected'}
        tokens = [t._replace(string=names.get(t.string, t.string)) if t.type == tokenize.NAME else t
                  for t in tokenize.generate_tokens(io.StringIO(source).readline)]
        return tokenize.untokenize(tokens)
    return source


def sources(ref, plan, style='plain'):
    a, b = restyle(ref['prepare'], style), restyle(ref['aggregate'], style)
    if style == 'alternative' and ref['family'] == 'matrix':
        a = function('prepare', 'items', 'return list(map(sum, zip(*items)))')
    if style == 'alternative' and ref['family'] == 'chunks':
        a = function('prepare', 'items', "output = ''\nfor char, count in items:\n    for _ in range(count):\n        output += char\nreturn output")
    helpers = a.replace('def prepare(', 'def _prepare(') + '\n' + b.replace('def aggregate(', 'def _aggregate(')
    compute = helpers + '\n' + function('compute', 'items', 'return _aggregate(_prepare(items))')
    if plan == 'atomic':
        root = 'import json\n\n' + helpers + '\n' + function('solve', 'text',
            "items = json.loads(text)\nresult = _aggregate(_prepare(items))\nreturn json.dumps(result, separators=(',', ':'))")
    else:
        keys = ('prepare', 'aggregate') if plan == 'pipeline' else ('compute',)
        imports = ''.join(f'from nodes.N0_p1_{key} import {key} as _{key}_child\n' for key in keys)
        body = "items = json.loads(text)\n" + ('prepared = _prepare_child(items)\nresult = _aggregate_child(prepared)\n'
                  if plan == 'pipeline' else 'result = _compute_child(items)\n')
        root = 'import json\n' + imports + '\n' + function('solve', 'text', body + "return json.dumps(result, separators=(',', ':'))")
    if style == 'helper':
        root = root.replace("return json.dumps(result, separators=(',', ':'))", 'return _format(result)')
        root += '\n' + function('_format', 'value', "return json.dumps(value, separators=(',', ':'))")
    return {'solve': root, 'prepare': a, 'aggregate': b, 'compute': compute}


def transformed_source(source, ref, transform):
    if transform == 'omit_output':
        return source.replace("    return json.dumps(result", f"    result.pop({ref['keys'][1]!r})\n    return json.dumps(result")
    if transform in {'move_algorithm', 'bypass_child'}:
        moved = ref['prepare'].replace('def prepare(', 'def _moved_prepare(')
        call = '_prepare_child(items)\n    ' if transform == 'bypass_child' else ''
        return source.replace('prepared = _prepare_child(items)', call + 'prepared = _moved_prepare(items)') + '\n' + moved
    if transform == 'wrong_argument':
        return source.replace('_aggregate_child(prepared)', '_aggregate_child({})')
    return source


class ReferencePolicy:
    """Fixed E_ref v1, an authored demonstrator; it never pretends to be an LLM."""
    def __init__(self, d, ref, plan='pipeline', style='plain', transform=None):
        self.d, self.ref, self.plan, self.style, self.transform = d, ref, plan, style, transform
        self.code = sources(ref, plan, style)
        self.old_artifact = None
        self.identity = {'backend': 'reviewed_reference', 'name': 'a4_E_ref_v1', 'plan_executor': 'fixed_authored',
                         'implementation_sha256': digest({f.__name__: inspect.getsource(f) for f in
                            (ReferencePolicy, plan_args, restyle, sources, transformed_source)})}

    def act(self, obs):
        name, args = self.decide(obs)
        raw = canonical({'action': name, 'args': args})
        return PolicyOutput(raw_text=raw, parsed_action=parse_action(raw), token_usage=TokenUsage(), model_identity=self.identity)

    def decide(self, obs):
        phase, root = obs['phase'], obs['node_id'] == 'N0'
        previous = obs.get('last_action_result') or {}
        if previous.get('status') in {'REJECTED', 'ERROR'}:
            return 'abstain', {'reason_code': 'INSUFFICIENT_EVIDENCE', 'evidence_ids': [], 'suggested_change': None}
        if phase == 'NEW':
            args = plan_args(obs['incoming_contract'], self.d, self.plan, root)
            if root and self.transform == 'expand_owns':
                args['children'][0]['contract'] = deepcopy(args['children'][0]['contract'])
                args['children'][0]['contract']['owns'].append('aggregate')
            return 'plan', args
        if phase == 'IMPLEMENT':
            name = obs['incoming_contract']['signature']['name']
            source = self.code[name]
            if name == 'prepare' and self.transform in {'stale_artifact', 'repaired_baseline'}:
                source = function('prepare', 'items', 'return None')
            if root and self.transform:
                source = transformed_source(source, self.ref, self.transform)
            return 'write_code', {'source': source, 'contract_version': obs['incoming_contract']['version'],
                                  'declared_dependencies': obs['approved_dependencies']}
        if phase == 'RESOLVE_CHILD':
            child = obs['returned_child']['result']
            artifact = child['artifact']
            args = {'child_id': artifact['node_id'], 'artifact_version': artifact['version'],
                    'artifact_hash': artifact['source_hash'], 'alias': '_' + artifact['public_symbol'] + '_child',
                    'evidence_ids': [child['evidence']['evidence_id']], 'accept_unverified': child['functional_status'] != 'PASS'}
            if self.transform == 'stale_artifact' and artifact['public_symbol'] == 'prepare':
                args.update(artifact_version=self.old_artifact['version'], artifact_hash=self.old_artifact['source_hash'])
            return 'consume', args
        if phase == 'READY_VERIFY':
            if obs['incoming_contract']['signature']['name'] == 'prepare' and self.transform in {'stale_artifact', 'repaired_baseline'}:
                self.old_artifact = deepcopy(obs['current_artifact'])
            return 'verify', {'target_ref': obs['current_artifact']['module_path'], 'suite_id': obs['available_suites'][0]}
        if phase == 'REVIEW':
            if (obs['incoming_contract']['signature']['name'] == 'prepare'
                    and self.transform in {'stale_artifact', 'repaired_baseline'}
                    and obs['current_artifact']['version'] == 1):
                evidence = next(e for e in obs['visible_evidence'] if e['target_ref'] == obs['current_artifact']['module_path'])
                if evidence['status'] == 'FAIL':
                    return 'localize', {'target_type': 'NODE', 'target_id': obs['node_id'],
                                        'evidence_ids': [evidence['evidence_id']], 'boundary_kind': 'OWNER_MODULE',
                                        'rationale': 'The current prepare module returns None instead of its required output.'}
            return 'finish', {'artifact_version': obs['current_artifact']['version']}
        if phase == 'REPAIR_AUTHORIZED' and self.transform in {'stale_artifact', 'repaired_baseline'}:
            ticket = obs['repair_ticket']
            return 'repair', {'ticket_id': ticket['ticket_id'], 'replacements': [{
                'file': ticket['editable_files'][0], 'base_hash': obs['current_artifact']['source_hash'],
                'source': self.code['prepare']}], 'dependency_updates': []}
        return 'abstain', {'reason_code': 'INSUFFICIENT_EVIDENCE', 'evidence_ids': [], 'suggested_change': None}


def execute(d, private, policy, target, framework, request):
    runtime = RepairRuntime(RootTask.model_validate(d['task']), ContractSpec.model_validate(d['contract']),
                            policy, target, limits=Limits(**{k: framework['runtime'][k] for k in Limits.model_fields}),
                            protocol_version=framework['protocol_version'], verifier=fixture_verifier(d, framework),
                            repair_config=framework['repair'], implementation_stage='A4')
    write_json(target / 'request.json', request)
    write_json(target / 'initial_observation.json', runtime.build_observation())
    result = runtime.run()
    # TRAIN private checks are offline labels. They never enter runtime states.
    full = fixture_verifier({**d, 'suites': d['suites'] + private['suites']}, framework)
    cost = {'verify_calls': 0, 'executed_cases': 0, 'execution_seconds': 0.0}
    offline = []
    for node_id, ctx in runtime.contexts.items():
        if ctx.artifact is None:
            continue
        for suite in full.suites.values():
            if suite.visibility != 'PRIVATE' or not suite.applies(d['task']['root_id'], ctx.node.incoming_contract):
                continue
            cost['verify_calls'] += 1
            e = full.verify(target, d['task']['root_id'], ctx.node.incoming_contract, ctx.artifact,
                            suite.suite_id, f'OFFLINE_{cost["verify_calls"]}', allow_private=True)
            offline.append({'node_id': node_id, 'evidence_id': e.evidence_id, 'status': e.status})
            cost['executed_cases'] += e.details['executed_cases']
            cost['execution_seconds'] += e.details['elapsed_seconds']
    summary = {'root_id': d['task']['root_id'], 'family_id': d['task']['family_id'], 'split': 'TRAIN',
               'trajectory_source': request['trajectory_source'], 'functional_status': result['functional_status'],
               'stop_reason': result['stop_reason'], 'usage': result['usage'], 'offline_cost': cost,
               'offline_evidence': offline, 'run_dir': str(target), 'request': request}
    write_json(target / 'collection_result.json', summary)
    return summary


def run_collection(config, *, natural=False, limit=None, device='cuda:0'):
    rows = selected(config)
    if natural and config.get('collection_parameter') is not None:
        rows = [r for r in rows if json.loads(Path(r['reference_path']).read_text())['parameter'] == config['collection_parameter']]
    if limit:
        rows = rows[:limit]
    framework = yaml.safe_load(Path(config['framework_config']).read_text())
    kind = 'collection' if natural else 'demos'
    output = Path(config['trace_dir']) / kind
    output.mkdir(parents=True, exist_ok=False)
    request = {'stage': 'A4', 'kind': kind, 'config': config, 'framework': framework, 'roots': rows,
               'renderer': renderer_identity(), 'executor': 'base_model' if natural else 'a4_E_ref_v1'}
    write_json(output / 'request.json', request)
    model = LLMPolicy.from_pretrained(framework, device=device) if natural else None
    results = []
    for row in rows:
        d, private, ref = load_task(row, config)
        variants = list(range(config['trajectories_per_root'])) if natural else list(PLANS)
        for variant in variants:
            style = ('plain', 'renamed', 'helper', 'alternative', 'plain')[(ref['parameter']-1) % 5] if ref else 'plain'
            policy = model if natural else ReferencePolicy(d, ref, variant, style)
            target = output / row['root_id'] / str(variant)
            job = {'trajectory_source': 'natural_model_rollout' if natural else 'reviewed_reference',
                   'plan': None if natural else variant, 'style': style if not natural else None,
                   'reference_hash': row.get('reference_hash'), 'model_identity': policy.identity,
                   'protocol_version': framework['protocol_version']}
            result = execute(d, private, policy, target, framework, job)
            append_json(output / 'roots.jsonl', result)
            results.append(result)
            print(f'{kind} {len(results)} root={row["root_id"]} variant={variant} {result["stop_reason"]}/{result["functional_status"]}', flush=True)
    summary = {'status': 'COMPLETE', 'kind': kind, 'roots': len(rows), 'trajectories': len(results),
               'generation_calls': model.generation_calls if model else 0,
               'functional_status': dict(Counter(r['functional_status'] for r in results)),
               'stop_reasons': dict(Counter(r['stop_reason'] for r in results))}
    write_json(output / 'result.json', summary)
    return summary


def make_contrasts(config, limit=None):
    rows = [r for r in selected(config) if json.loads(Path(r['reference_path']).read_text())['parameter'] == config['contrast_parameter']]
    if limit:
        rows = rows[:limit]
    framework = yaml.safe_load(Path(config['framework_config']).read_text())
    output = Path(config['trace_dir']) / 'contrasts'
    output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for row in rows:
        d, private, ref = load_task(row, config)
        bases = {}
        for name, transform in [('baseline', None), ('repaired_baseline', 'repaired_baseline')]:
            request = {'trajectory_source': 'reviewed_reference', 'plan': 'pipeline', 'style': 'plain',
                       'transform': transform, 'reference_hash': row['reference_hash']}
            target = output / row['root_id'] / name
            bases[name] = execute(d, private, ReferencePolicy(d, ref, transform=transform), target, framework, request)
        for transform in TRANSFORMS:
            baseline = bases['repaired_baseline' if transform == 'stale_artifact' else 'baseline']
            passed = (baseline['functional_status'] == 'PASS' and baseline['offline_evidence']
                      and all(e['status'] == 'PASS' for e in baseline['offline_evidence']))
            record = {'root_id': row['root_id'], 'family_id': row['family_id'], 'split': 'TRAIN',
                      'transform': transform, 'baseline': baseline['run_dir'], 'same_budget': True,
                      'same_root_contract': True, 'same_child_contracts': transform != 'expand_owns'}
            if passed:
                request = {'trajectory_source': 'controlled_transform', 'plan': 'pipeline', 'style': 'plain',
                           'transform': transform, 'baseline': baseline['run_dir'], 'reference_hash': row['reference_hash']}
                target = output / row['root_id'] / transform
                result = execute(d, private, ReferencePolicy(d, ref, transform=transform), target, framework, request)
                record.update(candidate=result['run_dir'], status='EXECUTED')
            else:
                record.update(candidate=None, status='BASELINE_NOT_SUPPORTED')
            append_json(output / 'pairs.jsonl', record)
            pairs.append(record)
            print(f'contrast {len(pairs)} {row["root_id"]} {transform}: {record["status"]}', flush=True)
    summary = {'status': 'COMPLETE', 'families': len(rows), 'pairs': len(pairs),
               'status_counts': dict(Counter(p['status'] for p in pairs))}
    write_json(output / 'result.json', summary)
    return summary


def responsibility_check(event, d, ref, plan, style):
    """Trusted component/contract rules; verdict never comes from the transform name.

    Unrecognized model code stays UNKNOWN. Test PASS alone is not a source review.
    The checked reference bodies are independently exercised by the host oracles.
    """
    action = event.parsed_action
    if action is None:
        return 'UNKNOWN', 'Unparseable action has no semantic responsibility label.'
    action = action.model_dump(mode='json')
    obs, args, name = event.observation_before, action['args'], action['action']
    spec = {k: v for k, v in obs['incoming_contract'].items() if k not in {'contract_id', 'version'}}
    symbol = spec['signature']['name']
    expected = d['contract'] if event.node_id == 'N0' else d['local_contracts'].get(symbol)
    if name == 'plan':
        if event.node_id == 'N0':
            ownership_fault = plan_args(spec, d, 'pipeline', True)
            ownership_fault['children'][0]['contract'] = deepcopy(ownership_fault['children'][0]['contract'])
            ownership_fault['children'][0]['contract']['owns'].append('aggregate')
            if args == ownership_fault:
                return 'VIOLATION', 'The unchanged obligation map assigns aggregate to the other child, but prepare silently claims it too.'
        if any(not args['obligation_map'].get(o['id']) for o in spec['ensures']):
            return 'VIOLATION', 'The plan omits an incoming obligation.'
        variants = PLANS if event.node_id == 'N0' else ('atomic',)
        if expected == spec and any(args == plan_args(spec, d, variant, event.node_id == 'N0') for variant in variants):
            return 'SUPPORTED', 'All incoming obligations have explicit owners; reviewed component contracts compose through the declared parameter.'
        return 'UNKNOWN', 'This plan has not been independently established by the closed component rules.'
    if name == 'consume':
        returned = obs.get('returned_child')
        artifact = returned['result']['artifact'] if returned else None
        if artifact and (args['artifact_version'], args['artifact_hash']) != (artifact['version'], artifact['source_hash']):
            return 'VIOLATION', 'Consumption does not select the exact currently returned artifact version and hash.'
        if (artifact and returned['result']['functional_status'] == 'PASS'
                and returned['source'] in {sources(ref, p, s).get(artifact['public_symbol'])
                                          for p in PLANS for s in ('plain', 'renamed', 'helper', 'alternative')}
                and returned['contract']['owns'] == d['local_contracts'].get(artifact['public_symbol'], {}).get('owns')):
            return 'SUPPORTED', 'Exact returned component is independently reviewed and visibly verified; subsequent use is reviewed separately.'
        return 'UNKNOWN', 'Child semantics or its decision-time evidence is insufficient for responsibility support.'
    if name != 'write_code':
        return 'UNKNOWN', 'Control/repair actions are handled by the existing protocol-only/core-SFT checks.'
    source = args['source']
    if event.node_id == 'N0' and obs.get('current_plan', {}).get('mode') == 'SPLIT':
        base = sources(ref, 'pipeline', 'plain')['solve']
        known_fault = any(source == transformed_source(base, ref, t)
                          for t in ('move_algorithm', 'bypass_child', 'omit_output', 'wrong_argument'))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 'UNKNOWN', 'Invalid Python source has no responsibility support.'
        if known_fault and any(isinstance(n, ast.FunctionDef) and n.name == '_moved_prepare' for n in ast.walk(tree)):
            moved = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_moved_prepare')
            original = ast.parse(ref['prepare']).body[0]
            if ast.dump(ast.Module(body=moved.body, type_ignores=[])) == ast.dump(ast.Module(body=original.body, type_ignores=[])):
                return 'VIOLATION', 'Parent reimplements the prepare algorithm allocated to its child (possibly after discarding the child return).'
        for node in ast.walk(tree) if known_fault else ():
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'pop':
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'result':
                    return 'VIOLATION', 'Parent removes a required field from the child result before formatting.'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_aggregate_child':
                if node.args and isinstance(node.args[0], ast.Dict):
                    return 'VIOLATION', 'The aggregate handoff receives an unrelated empty mapping instead of the prepare result.'
        if any(child['contract'] != d['local_contracts'].get(child['contract']['signature']['name'])
               for child in obs['current_plan']['children']):
            return 'UNKNOWN', 'The parent dependencies do not have the independently reviewed component contracts.'
    if expected == spec and plan and source == sources(ref, plan, style)[symbol]:
        return 'SUPPORTED', 'Source matches a reviewed ownership-preserving component/composition; independent execution evidence is checked separately.'
    return 'UNKNOWN', 'No independent semantic source proof for this implementation; retain functional evidence separately.'


def validate_labels(config):
    roots = load_root_manifest(config['root_manifest'])
    output = Path(config['report_dir'])
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / 'evidence_audit.jsonl'
    if audit_path.exists():
        raise FileExistsError('Label audit already exists; keep prior audits and choose a new report_dir')
    audits, plan_records = [], []
    cached = {}
    for trace in sorted(Path(config['trace_dir']).rglob('trace.jsonl')):
        run = RunData(trace, roots)
        row = roots[run.task.root_id]
        if row['split'] != 'TRAIN':
            raise ValueError('A4 label review must not consume non-TRAIN trajectories')
        if run.task.root_id not in cached:
            cached[run.task.root_id] = load_task(row, config)
        d, _, ref = cached[run.task.root_id]
        request = json.loads((trace.parent / 'request.json').read_text())
        plan, style = request.get('plan'), request.get('style') or 'plain'
        reviews, local_audits = [], []
        for event in run.events:
            if event.event_kind != 'POLICY':
                continue
            verdict, reason = responsibility_check(event, d, ref, plan, style) if ref else ('UNKNOWN', 'No reviewed reference supplied.')
            if event.parsed_action is not None and verdict != 'UNKNOWN':
                review = EventReview(event_id=event.event_id, root_id=event.root_id, node_id=event.node_id,
                    action_hash=digest(event.parsed_action.model_dump(mode='json')),
                    contract_hash=digest(event.observation_before['incoming_contract']), verdict=verdict,
                    label_origin='controlled_transform' if verdict == 'VIOLATION' else 'reviewed_reference',
                    rationale=reason, meta={'review_method': 'independent_contract_and_component_rules_v1',
                                           'reference_sha256': row.get('reference_hash'), 'human_annotation': False})
                reviews.append(review)
                run.reviews[event.event_id] = review
            record = {'root_id': run.task.root_id, 'family_id': run.task.family_id, 'split': 'TRAIN',
                      'run_dir': str(trace.parent), 'run_id': run.run_id, 'event_id': event.event_id,
                      'node_id': event.node_id, 'action': event.parsed_action.action if event.parsed_action else None,
                      'mode': event.observation_before['mode'], 'verdict': verdict, 'rationale': reason,
                      'trajectory_source': request['trajectory_source'], 'label_method': 'trusted_rules',
                      'human_annotation': False, 'evidence_refs': []}
            try:
                if event.parsed_action is None or event.host_result.status != 'APPLIED':
                    raise Rejected('HOST_REJECTED_OR_INVALID_ACTION')
                label = classify(run, event, event.parsed_action.model_dump(mode='json'), False)
                record.update(core_candidate=True, scope=label['supervision_scope'], evidence_refs=label['evidence_refs'])
            except Rejected as error:
                record.update(core_candidate=False, exclusion_reason=error.reason)
            append_json(audit_path, record)
            audits.append(record)
            local_audits.append(record)
        path = trace.parent / 'supervision.jsonl'
        with path.open('x') as stream:
            for review in reviews:
                stream.write(review.model_dump_json() + '\n')
        if request['trajectory_source'] == 'reviewed_reference' and trace.parent.parent.parent.name == 'demos':
            root_plan = next((r for r in local_audits if r['node_id'] == 'N0' and r['action'] == 'plan'), None)
            initial = json.loads((trace.parent / 'initial_observation.json').read_text())
            plan_records.append({'root_id': run.task.root_id, 'plan': plan, 'initial_state_sha256': digest(initial),
                                 'executor': request['model_identity'], 'functional_status': run.result['functional_status'],
                                 'plan_review': root_plan['verdict'] if root_plan else 'UNKNOWN',
                                 'core_plan_candidate': root_plan['core_candidate'] if root_plan else False,
                                 'usage': run.result['usage'], 'plan_preference': None,
                                 'failure_class': ('PLAN_DEFECT' if root_plan and root_plan['verdict'] == 'VIOLATION' else
                                                  'NONE' if run.result['functional_status'] == 'PASS' else
                                                  'IMPLEMENTATION_FAILURE' if run.result['functional_status'] == 'FAIL' else 'UNKNOWN')})
    for row in plan_records:
        append_json(output / 'plan_candidates.jsonl', row)
    negatives = []
    pair_path = Path(config['trace_dir']) / 'contrasts/pairs.jsonl'
    for pair in read_jsonl(pair_path) if pair_path.exists() else []:
        relevant = [a for a in audits if a['run_dir'] == pair.get('candidate') and a['verdict'] == 'VIOLATION']
        status, matched = 'UNKNOWN', False
        evidence = []
        if pair.get('candidate'):
            run = RunData(Path(pair['candidate']) / 'trace.jsonl', roots)
            evidence = [e.model_dump(mode='json') for e in run.evidence.values() if e.check_kind == 'ROOT']
            matched = bool(evidence) and all(e['status'] == 'PASS' for e in evidence)
            if any(e['status'] == 'FAIL' for e in evidence):
                status = 'FUNCTIONAL_FAILURE'
            elif relevant and matched and pair['same_child_contracts']:
                status = 'FUNCTION_MATCHED_RESPONSIBILITY_VIOLATION'
            elif relevant:
                status = 'PROTOCOL_OR_CONTRACT_VIOLATION'
        record = {**pair, 'label': status, 'root_function_matched': matched,
                  'responsibility_evidence': relevant, 'root_evidence': evidence,
                  'label_strength': 'SUPPORTED' if relevant and status != 'UNKNOWN' else 'UNKNOWN',
                  'preference_ready': False,
                  'pairing_scope': 'same root, reference executor, interfaces and budget; separate real executions'}
        append_json(output / 'contrasts.jsonl', record)
        negatives.append(record)
    # Reviewable stratification, explicitly programmatic; no fictitious human sign-off.
    seen = set()
    for row in audits:
        stratum = 'CONTROLLED_NEGATIVE' if row['verdict'] == 'VIOLATION' else row['mode'] if row['core_candidate'] else 'UNKNOWN_OR_EXCLUDED'
        if row['core_candidate'] and row['action'] == 'consume':
            stratum = 'CONSUME'
        elif row['core_candidate'] and row['mode'] == 'COMPOSE' and row['action'] == 'write_code':
            stratum = 'PARENT_WRITE'
        key = (row['family_id'], stratum)
        if key not in seen:
            seen.add(key)
            append_json(output / 'stratified_review.jsonl', {**row, 'stratum': stratum, 'human_review_status': 'NOT_PERFORMED'})
    summary = {'status': 'COMPLETE', 'audited_actions': len(audits),
               'verdicts': dict(Counter(a['verdict'] for a in audits)),
               'core_candidates_before_tokenization': sum(a['core_candidate'] for a in audits),
               'exclusion_reasons': dict(Counter(a['exclusion_reason'] for a in audits if not a['core_candidate'])),
               'contrasts': dict(Counter(n['label'] for n in negatives)),
               'plan_candidates': len(plan_records), 'human_annotations': 0, 'review_strata': len(seen),
               'review_basis': 'A3 permits trusted construction rules plus independent execution; no human labels claimed.'}
    write_json(output / 'label_report.json', summary)
    return summary


def make_schedule(config):
    from .train_sft import build_schedule, load_config, load_identity_and_tokenizer, load_training_data
    training = load_config(Path(config['sft_config']))
    identity, tokenizer = load_identity_and_tokenizer(training)
    rows, encoded, manifest = load_training_data(training, identity, tokenizer)
    schedule, steps = build_schedule(rows, training['data'], training['training'])
    output = Path(config['report_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'sampling_schedule.jsonl').open('x') as stream:
        for slot in schedule:
            stream.write(canonical(slot) + '\n')
    report = {'status': 'PASS', 'scope': 'TRAIN', 'data_sha256': manifest['output_sha256'],
              'protocol_version': manifest['protocol_version'], 'data_check_protocol_version': manifest['data_check_protocol_version'],
              'samples': len(rows), 'encoded_samples': len(encoded), 'roots': len({r['root_id'] for r in rows}),
              'families': len({r['family_id'] for r in rows}), 'by_bucket': manifest['by_bucket'],
              'by_scope': manifest['by_scope'], 'slots': len(schedule), 'optimizer_steps': steps,
              'unique_sampled_examples': len({s['sample_id'] for s in schedule}),
              'sampled_roots': len({s['root_id'] for s in schedule}),
              'sampled_buckets': dict(Counter(s['bucket'] for s in schedule)),
              'target_tokens': sum(sum(label != -100 for label in encoded[s['index']]['labels']) for s in schedule),
              'training_config': training, 'parameter_updates': 0}
    write_json(output / 'schedule_report.json', report)
    return {k: v for k, v in report.items() if k != 'training_config'}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='audittrainer data')
    parser.add_argument('command', choices=['import-roots', 'build-demos', 'collect', 'make-contrasts', 'validate-labels', 'make-schedule'])
    parser.add_argument('--config', type=Path, default=Path('configs/data_pilot.yaml'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command == 'import-roots':
        result = import_roots(config['task_dir'], config['root_manifest'], config['test_manifest'],
                              read_jsonl(args.input) if args.input else None)
    elif args.command in {'build-demos', 'collect'}:
        result = run_collection(config, natural=args.command == 'collect', limit=args.limit, device=args.device)
    elif args.command == 'make-contrasts':
        result = make_contrasts(config, limit=args.limit)
    elif args.command == 'validate-labels':
        result = validate_labels(config)
    else:
        result = make_schedule(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
