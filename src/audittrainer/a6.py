"""One A6 data-coverage revision, reusing the frozen SFT and A5 evaluator."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .dataset import EventReview, RunData, export_sft, file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .training_data import (ReferencePolicy, execute, load_task, plan_args,
                            responsibility_check, selected, sources)
from .training_tasks import function
from .verifier import canonical, digest


def read(path):
    return json.loads(Path(path).read_text())


def reassignment_plan(contract, definition):
    plan = plan_args(contract, definition, 'pipeline', True)
    plan['children'] = plan['children'][:1]
    plan['obligation_map']['AGGREGATE'] = ['self']
    plan['retained_responsibilities'] = ['parse', 'aggregate', 'format']
    plan['dataflow'] = []
    return plan


def reassignment_sources(reference, style):
    code = sources(reference, 'pipeline', style)
    aggregate = code['aggregate'].replace('def aggregate(', 'def _aggregate(')
    body = ("items = json.loads(text)\nprepared = _prepare_child(items)\n"
            "result = _aggregate(prepared)\nreturn json.dumps(result, separators=(',', ':'))")
    root = ('import json\nfrom nodes.N0_p1_prepare import prepare as _prepare_child\n\n'
            + aggregate + '\n' + function('solve', 'text', body))
    if style == 'helper':
        root = root.replace("return json.dumps(result, separators=(',', ':'))", 'return _format(result)')
        root += '\n' + function('_format', 'value', "return json.dumps(value, separators=(',', ':'))")
    return {**code, 'solve': root}


class ReassignmentPolicy(ReferencePolicy):
    def __init__(self, definition, reference, style='plain'):
        super().__init__(definition, reference, plan='pipeline', style=style)
        self.code = reassignment_sources(reference, style)
        self.identity = {**self.identity, 'name': 'a6_authored_reassignment',
                         'revision_implementation_sha256': file_hash(__file__)}

    def decide(self, obs):
        if obs['node_id'] == 'N0' and obs['phase'] == 'NEW':
            return 'plan', reassignment_plan(obs['incoming_contract'], self.d)
        return super().decide(obs)


def review_reassignment(event, definition, reference, style):
    """Closed reference rule, bound to the changed plan; tests alone do not label it."""
    if event.parsed_action is None:
        return 'UNKNOWN', 'No parsed action.'
    action = event.parsed_action.model_dump(mode='json')
    obs = event.observation_before
    spec = {k: v for k, v in obs['incoming_contract'].items() if k not in {'contract_id', 'version'}}
    if event.node_id == 'N0' and action['action'] in {'plan', 'write_code'}:
        expected_plan = reassignment_plan(spec, definition)
        if spec != definition['contract']:
            return 'UNKNOWN', 'Incoming obligations differ from the registered TRAIN contract.'
        if action['action'] == 'plan':
            supported = action['args'] == expected_plan
        else:
            actual_plan = {k: v for k, v in (obs.get('current_plan') or {}).items()
                           if k not in {'node_id', 'version'}}
            deps = action['args']['declared_dependencies']
            supported = (actual_plan == expected_plan and len(deps) == 1
                and deps == obs['approved_dependencies']
                and deps[0]['child_id'] == 'N0_p1_prepare'
                and deps[0]['alias'] == '_prepare_child'
                and deps[0]['artifact_version'] == 1
                and deps[0]['artifact_hash'] == digest_source(reassignment_sources(reference, style)['prepare'])
                and action['args']['source'] == reassignment_sources(reference, style)['solve'])
        return ('SUPPORTED', 'Reviewed prepare is delegated; parent implements the independently checked aggregate and exact root formatting.') if supported else (
            'UNKNOWN', 'Source, plan, or dependency differs from the reviewed responsibility reassignment.')
    return responsibility_check(event, definition, reference, 'pipeline', style)


def digest_source(source):
    import hashlib
    return hashlib.sha256(source.encode()).hexdigest()


def collect(config):
    data_config = yaml.safe_load(Path(config['data_config']).read_text())
    framework = yaml.safe_load(Path(data_config['framework_config']).read_text())
    roots = load_root_manifest(data_config['root_manifest'])
    rows = selected(data_config)
    assert {r['root_id'] for r in rows} == {r['root_id'] for r in read_jsonl(config['baseline_data'])}
    directory = Path(config['trace_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    output = Path(config['report_dir'])
    output.mkdir(parents=True, exist_ok=True)
    contrasts = read_jsonl('runs/a4/contrasts.jsonl')
    write_json(output / 'request.json', {
        'stage': 'A6', 'config': config, 'baseline_result_sha256': file_hash(config['baseline_result']),
        'root_manifest_sha256': file_hash(data_config['root_manifest']),
        'framework_sha256': file_hash(data_config['framework_config']),
        'training_config_sha256': file_hash(config['training_config']),
        'evaluation_config_sha256': file_hash(config['evaluation_config']),
        'dpo_decision': {'enabled': False, 'prepared_preference_pairs': sum(p['preference_ready'] for p in contrasts),
            'reason': 'Test the missing parent responsibility coverage with one SFT data intervention; no prepared same-observation preferences for this new condition.'},
        'initialization': 'fresh public theta0; matched to A5 F11, no extra continuation steps',
        'retained': 'all A5 data, checkpoints, evaluation, prompts, runtime and oracles',
        'implementation_sha256': file_hash(__file__)})
    results = []
    for row in rows:
        definition, private, reference = load_task(row, data_config)
        style = ('plain', 'renamed', 'helper', 'alternative', 'plain')[(reference['parameter'] - 1) % 5]
        policy = ReassignmentPolicy(definition, reference, style)
        target = directory / row['root_id']
        request = {'stage': 'A6', 'trajectory_source': 'reviewed_reference',
                   'plan': 'child_prepares_parent_aggregates', 'style': style,
                   'reference_hash': row['reference_hash'], 'model_identity': policy.identity}
        result = execute(definition, private, policy, target, framework, request)
        if result['functional_status'] != 'PASS' or not all(e['status'] == 'PASS' for e in result['offline_evidence']):
            raise ValueError(f"Reassignment execution failed: {row['root_id']}")
        run = RunData(target / 'trace.jsonl', roots)
        reviews = []
        for event in run.events:
            if event.event_kind != 'POLICY':
                continue
            verdict, reason = review_reassignment(event, definition, reference, style)
            if verdict == 'SUPPORTED':
                reviews.append(EventReview(event_id=event.event_id, root_id=event.root_id, node_id=event.node_id,
                    action_hash=digest(event.parsed_action.model_dump(mode='json')),
                    contract_hash=digest(event.observation_before['incoming_contract']), verdict=verdict,
                    label_origin='reviewed_reference', rationale=reason,
                    meta={'review_method': 'a6_bound_reassignment_reference', 'reference_sha256': row['reference_hash'],
                          'human_annotation': False}))
        (target / 'supervision.jsonl').write_text(''.join(r.model_dump_json() + '\n' for r in reviews))
        results.append(result)
        print(f"reassignment {len(results)}/{len(rows)} {row['root_id']}: PASS", flush=True)
    exported = export_sft(directory, config['new_data'], root_manifest=data_config['root_manifest'],
                          config=framework, max_length=8192, allowed_splits=('TRAIN',))
    write_json(output / 'collection.json', {'status': 'COMPLETE', 'results': results, 'export': exported})
    return exported


def revise_slots(base, additions, slots):
    parents = {r['root_id']: r for r in additions
               if r['target_action']['action'] == 'write_code' and r['meta']['node_id'] == 'N0'}
    if set(parents) != {r['root_id'] for r in base}:
        raise ValueError('Every baseline TRAIN root needs a reviewed parent example.')
    pool = list(base) + list(parents.values())
    index = {r['sample_id']: i for i, r in enumerate(pool)}
    if len(index) != len(pool):
        raise ValueError('Revised pool must contain distinct real events.')
    seen, schedule = Counter(), []
    for slot in slots:
        original = base[slot['index']]
        if original['sample_id'] != slot['sample_id'] or original['root_id'] != slot['root_id']:
            raise ValueError('Baseline schedule binding mismatch.')
        replace = False
        if original['bucket'] == 'D_COMPOSE' and original['target_action']['action'] == 'write_code':
            replace = seen[original['root_id']] % 2 == 0
            seen[original['root_id']] += 1
        selected_row = parents[original['root_id']] if replace else original
        schedule.append({**slot, 'index': index[selected_row['sample_id']], 'sample_id': selected_row['sample_id'],
                         'source_sample_id': original['sample_id'], 'a6_replaced': replace,
                         'replaced': replace, 'actual_bucket': selected_row['bucket']})
    return pool, schedule


def exposure(pool, schedule):
    selected_rows = [pool[s['index']] for s in schedule]
    return {'pool_examples': len(pool), 'slots': len(schedule), 'roots': len({r['root_id'] for r in selected_rows}),
            'unique_sampled_examples': len({r['sample_id'] for r in selected_rows}),
            'prefix_tokens': sum(r['meta']['prefix_tokens'] for r in selected_rows),
            'target_tokens': sum(r['meta']['target_tokens'] for r in selected_rows),
            'actions': dict(Counter(r['target_action']['action'] for r in selected_rows)),
            'buckets': dict(Counter(r['bucket'] for r in selected_rows))}


def prepare(config):
    base, additions = read_jsonl(config['baseline_data']), read_jsonl(config['new_data'])
    original = read_jsonl(config['baseline_schedule'])
    pool, schedule = revise_slots(base, additions, original)
    path, schedule_path = Path(config['revised_data']), Path(config['revised_schedule'])
    if any(p.exists() for p in [path, schedule_path, path.with_suffix('.manifest.json')]):
        raise FileExistsError('Keep previous A6 datasets; choose new paths.')
    path.write_text(''.join(canonical(r) + '\n' for r in pool))
    schedule_path.write_text(''.join(canonical(r) + '\n' for r in schedule))
    manifest = deepcopy(read(Path(config['baseline_data']).with_suffix('.manifest.json')))
    manifest['a5_source_factorial'] = manifest.pop('factorial')
    manifest.update(dataset_version='a6-parent-reassignment-v1', samples=len(pool), output_sha256=file_hash(path),
                    by_mode=dict(Counter(r['mode'] for r in pool)), by_bucket=dict(Counter(r['bucket'] for r in pool)),
                    by_scope=dict(Counter(r['supervision_scope'] for r in pool)),
                    raw_sources=manifest['raw_sources'] + read(Path(config['new_data']).with_suffix('.manifest.json'))['raw_sources'],
                    a6_revision=config['revision'])
    write_json(path.with_suffix('.manifest.json'), manifest)
    write_json(Path(config['report_dir']) / 'data_comparison.json', {
        'status': 'COMPLETE', 'baseline': exposure(base, original), 'revised': exposure(pool, schedule),
        'replaced_slots': sum(s['a6_replaced'] for s in schedule),
        'replaced_roots': sorted({s['root_id'] for s in schedule if s['a6_replaced']}),
        'only_replaced': 'complete observation/action examples for parent COMPOSE write_code',
        'unchanged': 'root/slot/optimizer order; every plan, component, consume and control sample',
        'baseline_data_sha256': file_hash(config['baseline_data']), 'revised_data_sha256': file_hash(path),
        'revised_schedule_sha256': file_hash(schedule_path)})


def evaluate_revision(config):
    from .a5_evaluate import run_group
    evaluation = yaml.safe_load(Path(config['evaluation_config']).read_text())
    evaluation['output_dir'] = config['evaluation_dir']
    return run_group(config['evaluation_group'], Path(config['training_dir']) / 'final_adapter', evaluation)


def report(config):
    from .train_sft import load_config
    output = Path(config['report_dir'])
    train_dir = Path(config['training_dir'])
    evaluation_dir = Path(config['evaluation_dir']) / config['evaluation_group']
    training, data = read(train_dir / 'training_report.json'), read(output / 'data_comparison.json')
    baseline_train = read(Path(config['baseline_training']) / 'training_report.json')
    baseline_config = yaml.safe_load((Path(config['baseline_training']) / 'resolved_config.yaml').read_text())
    revised_config = load_config(config['training_config'])
    for key in ['model', 'lora']:
        assert baseline_config[key] == revised_config[key]
    assert {k: v for k, v in baseline_config['training'].items() if k != 'output_dir'} == {
        k: v for k, v in revised_config['training'].items() if k != 'output_dir'}
    assert training['status'] == 'COMPLETE' and training['optimizer_steps'] == baseline_train['optimizer_steps'] == 140
    assert training['reload']['max_abs_logit_difference'] <= 1e-5
    assert read(train_dir / 'before_update.json') == read(Path(config['baseline_training']) / 'before_update.json')
    assert sum(r.get('target_tokens', 0) for r in training['history']) == data['revised']['target_tokens']
    assert file_hash(Path(training['adapter']) / 'adapter_model.safetensors') == training['adapter_sha256']
    assert read(evaluation_dir / 'adapter_probe.json')['status'] == 'PASS'
    evaluation = read(evaluation_dir / 'summary.json')
    assert evaluation['status'] == 'COMPLETE'
    request = read(evaluation_dir / 'request.json')
    old_request = read(Path(config['baseline_evaluation']) / 'request.json')
    assert request['framework'] == old_request['framework']
    assert request['evaluation_implementation_sha256'] == old_request['evaluation_implementation_sha256']
    assert {k: v for k, v in request['config'].items() if k != 'output_dir'} == {
        k: v for k, v in old_request['config'].items() if k != 'output_dir'}
    rows = read_jsonl(evaluation_dir / 'isolation.jsonl')
    baseline = read_jsonl(Path(config['baseline_evaluation']) / 'isolation.jsonl')
    old = {(r['root_id'], r['condition']): r for r in baseline}
    assert len(rows) == len(old) == 40 and {(r['root_id'], r['condition']) for r in rows} == set(old)
    comparison = {}
    for condition in request['config']['isolation_conditions']:
        subset = [r for r in rows if r['condition'] == condition]
        old_pass, new_pass, improved, regressed = [], [], [], []
        for row in subset:
            previous = old[(row['root_id'], condition)]
            assert row['child_artifacts_frozen']
            assert row['final_scores']['private']['status'] != 'ERROR'
            identity = read(evaluation_dir / condition / row['root_id'] / 'request.json')['model_identity']
            assert identity['adapter_weights_sha256'] == training['adapter_sha256']
            a, b = previous['final_scores']['private']['status'] == 'PASS', row['final_scores']['private']['status'] == 'PASS'
            if a: old_pass.append(row['root_id'])
            if b: new_pass.append(row['root_id'])
            if b and not a: improved.append(row['root_id'])
            if a and not b: regressed.append(row['root_id'])
        comparison[condition] = {'cases': len(subset), 'baseline_passes': len(old_pass), 'revised_passes': len(new_pass),
                                 'improved_roots': improved, 'regressed_roots': regressed}
    free = evaluation['free_result']['aggregate']
    assert free['roots'] == 5 and not free['root_status'].get('ERROR', 0) and not free['final_private_status'].get('ERROR', 0)
    result = {'stage': 'A6', 'status': 'COMPLETE', 'scope': config['scope'], 'major_revisions': 1,
        'changed_factor': config['revision']['factor'], 'data': data, 'training': training,
        'comparison': comparison, 'evaluation': evaluation,
        'baseline_free': read(Path(config['baseline_evaluation']) / 'summary.json')['free_result']['aggregate'],
        'primary_metric': comparison['parent_transform_child'], 'dpo': read(output / 'request.json')['dpo_decision'],
        'initial_tracked_tensors_identical': True, 'evaluation_protocol_identical': True,
        'next_stage': 'A7', 'stopping': 'No second revision, regardless of capability results.',
        'implementation_sha256': file_hash(__file__)}
    write_json(output / 'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'comparison': comparison, 'free': free['BudgetedSuccess']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'prepare', 'evaluate', 'report'])
    parser.add_argument('--config', type=Path, default=Path('configs/a6.yaml'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    {'collect': collect, 'prepare': prepare, 'evaluate': evaluate_revision, 'report': report}[args.command](config)


if __name__ == '__main__':
    main()
