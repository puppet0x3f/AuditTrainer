"""Finite PR1C RL comparison on fresh seeds, with natural and plan-only probes."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import set_seed

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_curriculum import plans
from .refinement_policy import active_adapter, adapter_fingerprint, load_policies, sequence_log_probs
from .refinement_pr1 import load_config, load_training_groups
from .refinement_rollout import run_case, runtime_for, case_summary, costs
from .verifier import canonical

METHODS = ('B0', 'B3', 'B4')


def plan_key(action, definition):
    """Identify a reviewed allocation up to child aliases and ordering."""
    if not action:
        return None
    args = action['args']; names = {}
    for child in args['children']:
        matches = [name for name, contract in definition['local_contracts'].items() if contract == child['contract']]
        if not matches:
            return None
        names[child['local_key']] = matches[0]
    def owner(value):
        return 'self' if value == 'self' else 'child:'+names.get(value.removeprefix('child:'), '?')
    return canonical({'children': sorted(names.values()), 'retained': sorted(args['retained_responsibilities']),
        'obligations': {k: sorted(owner(v) for v in values) for k, values in args['obligation_map'].items()},
        'dataflow': sorted((names.get(e['from_child'], '?'), e['from_output'],
                           names.get(e['to_child'], '?'), e['to_parameter']) for e in args['dataflow'])})


def natural_seed(config, root_index, repetition):
    return config['evaluation']['seed']+1000000+root_index*1000+repetition


def preferences(config, planner, executor, output):
    target = output/'preferences.json'
    if target.exists():
        return json.loads(target.read_text())
    scored = []
    diagnostic = deepcopy(config)
    diagnostic['root_manifest'] = config.get('diagnostic_root_manifest', config['root_manifest'])
    for group, saved_gate, candidates, _ in load_training_groups(diagnostic):
        gate = planner.score_modes(saved_gate['observation'])
        if gate['prefix_ids'] != saved_gate['prefix_ids']:
            raise ValueError('Preference comparison requires the same public prompt')
        entries = []
        for index, candidate in enumerate(candidates[1:], 1):
            prefix, body = candidate['body_input_ids'], candidate['body_output_ids']
            with active_adapter(planner.model, planner.adapter_name), torch.inference_mode(), planner.llm._autocast():
                scores = sequence_log_probs(planner.model, [prefix], [body], planner.tokenizer.pad_token_id)[0]
            values = scores.float().cpu().tolist()
            entries.append({'candidate': index, 'training_q': group['q_values'][index],
                'body_log_probability': sum(values), 'mean_token_log_probability': sum(values)/len(values),
                'token_log_probabilities': values})
            planner.cost.update(input_tokens=len(prefix), output_tokens=len(body),
                                body_scoring_batches=1, model_forward_calls=1)
        scored.append({'root_id': group['root_id'], 'mode_probabilities': gate['mode_probabilities'],
                       'mode_log_probs': gate['mode_log_probs'], 'bodies': entries})
    result = {'status': 'COMPLETE', 'roots': scored,
        'meaning': 'Exact scores of fixed complete token sequences. These are not marginal probabilities of semantic plan classes; fresh conditional sampling estimates class frequencies separately.'}
    write_json(target, result)
    return result


def conditional_probe(config, framework, planner, executor, output, rows, tests):
    settings = config['evaluation']; row = rows[settings['focus_root']]
    definition = json.loads(Path(row['fixture_path']).read_text())
    variants = {plan_key(a, definition): name for name, a in plans(definition).items()}
    saved_gate = json.loads((Path(config['output_dir'])/'groups'/row['root_id']/'gate.json').read_text())
    gate = planner.score_modes(saved_gate['observation'])
    folder = output/'conditional_probe'; folder.mkdir(exist_ok=True)
    write_json(folder/'gate.json', gate)
    records = []
    for index in range(settings['conditional_split_samples']):
        base = folder/f'sample_{index:02d}'; result_path = base/'result.json'
        if result_path.exists():
            records.append(json.loads(result_path.read_text())); continue
        base.mkdir(exist_ok=True)
        seed = settings['seed']+2000000+index
        set_seed(seed)
        candidate = planner.sample_split(gate)
        write_json(base/'candidate.json', candidate)
        policy = SimpleNamespace(act=lambda obs, c=candidate: planner.as_output(gate, c))
        runtime, _ = runtime_for(row, tests, framework, policy, base/'registration')
        event = runtime.step()
        registered = event.host_result.code == 'PLAN_REGISTERED'
        key = plan_key(candidate['parsed_action'], definition) if registered else None
        record = {'sample': index, 'seed': seed, 'parse_valid': candidate['parse_error'] is None,
            'registered': registered, 'host_code': event.host_result.code,
            'oracle_backed': registered and key is not None,
            'reviewed_variant': variants.get(key) if registered else None,
            'source': 'FRESH_CONDITIONAL_PLAN_PROBE_NO_CONTINUATION_OR_REWARD'}
        write_json(result_path, record); records.append(record)
        print(json.dumps({'probe': row['root_id'], **record}), flush=True)
    result = {'status': 'COMPLETE', 'samples': len(records), 'root_id': row['root_id'],
        'registered': sum(r['registered'] for r in records), 'oracle_backed': sum(r['oracle_backed'] for r in records),
        'reviewed_variants': dict(Counter(r['reviewed_variant'] or 'UNMATCHED' for r in records)), 'records': records}
    write_json(folder/'result.json', result)
    return result


def evaluate(config, framework, method):
    output = Path(config['output_dir'])/'evaluation'/method; output.mkdir(parents=True, exist_ok=True)
    adapter = config['initial_adapter'] if method == 'B0' else str(Path(config['output_dir'])/method/'final_adapter')
    request = {'config': config, 'method': method, 'adapter': adapter,
               'adapter_sha256': file_hash(Path(adapter)/'adapter_model.safetensors')}
    path = output/'request.json'
    if path.exists() and json.loads(path.read_text()) != request:
        raise ValueError('Changed resumed evaluation')
    write_json(path, request)
    planner, executor = load_policies(framework, adapter, config['executor_adapter'])
    before = {name: adapter_fingerprint(planner.model, name) for name in ('default', 'executor')}
    rows = load_root_manifest(config['root_manifest'])
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    if set(rows) != set(config['evaluation']['natural_repetitions']) or any(r['split'] != 'TRAIN' for r in rows.values()):
        raise ValueError('The fresh evaluation is the fixed three-root TRAIN diagnostic')
    pref = preferences(config, planner, executor, output)
    probe = conditional_probe(config, framework, planner, executor, output, rows, tests)
    cases = []
    for index, (rid, row) in enumerate(sorted(rows.items())):
        for repetition in range(config['evaluation']['natural_repetitions'][rid]):
            cases.append(run_case(row, tests, framework, config, planner, executor,
                output/'natural'/rid/f'repeat_{repetition:02d}', natural_seed(config, index, repetition)))
    if any(c['status'] != 'COMPLETE' for c in cases):
        raise ValueError('Incomplete natural evaluation')
    after = {name: adapter_fingerprint(planner.model, name) for name in before}
    if before != after:
        raise RuntimeError('Evaluation changed model parameters')
    result = {'status': 'COMPLETE', 'method': method, 'scope': 'FRESH_SEED_TRAIN_DIAGNOSTIC',
        'natural': case_summary(cases), 'by_root': {rid: case_summary([c for c in cases if c['root_id'] == rid]) for rid in rows},
        'preferences': pref, 'conditional_probe': probe, 'frozen_before': before, 'frozen_after': after,
        'actual_model_cost': dict(costs(planner, executor)), 'optimizer_updates_from_evaluation': 0}
    write_json(output/'result.json', result)
    print(json.dumps({'method': method, 'status': 'COMPLETE', 'natural_passes': result['natural']['final_successes'],
                      'probe_variants': probe['reviewed_variants']}), flush=True)


def evaluation_summary(config, evaluations, methods=METHODS):
    output = Path(config['output_dir'])
    natural = {}; changes = {}; focus = config['evaluation']['focus_root']
    for method, evaluation in evaluations.items():
        if (evaluation['status'] != 'COMPLETE' or evaluation['frozen_before'] != evaluation['frozen_after']
                or evaluation['natural']['runs'] != sum(config['evaluation']['natural_repetitions'].values())
                or evaluation['conditional_probe']['samples'] != config['evaluation']['conditional_split_samples']):
            raise ValueError('Incomplete evaluation')
        natural[method] = {(c['root_id'], c['seed']): c for p in (output/'evaluation'/method/'natural').glob('*/repeat_*/case_result.json')
                           for c in [json.loads(p.read_text())]}
        now = next(r for r in evaluation['preferences']['roots'] if r['root_id'] == focus)
        base = next(r for r in evaluations['B0']['preferences']['roots'] if r['root_id'] == focus)
        delta = {str(r['candidate']): r['body_log_probability']-b['body_log_probability'] for r, b in zip(now['bodies'], base['bodies'])}
        changes[method] = {'focus_p_split': now['mode_probabilities']['SPLIT'],
            'focus_p_split_change': now['mode_probabilities']['SPLIT']-base['mode_probabilities']['SPLIT'],
            'fixed_body_log_probability_changes': delta,
            'good_vs_fused_log_odds_change': delta['2']-delta['3'],
            'fresh_parent_aggregate_count': evaluation['conditional_probe']['reviewed_variants'].get('parent_aggregate', 0)}
    paired = {}
    for method in methods[1:]:
        if natural[method].keys() != natural['B0'].keys():
            raise ValueError('Natural root/seed pairs differ')
        pairs = [(natural['B0'][key]['reward']['y'], natural[method][key]['reward']['y']) for key in natural['B0']]
        paired[method] = {'wins': sum(b > a for a, b in pairs), 'losses': sum(b < a for a, b in pairs),
                          'ties': sum(a == b for a, b in pairs)}
    return {'evaluation': evaluations, 'changes': changes, 'paired_natural_vs_B0': paired,
        'evaluation_natural_runs': sum(e['natural']['runs'] for e in evaluations.values()),
        'evaluation_conditional_plan_samples': sum(e['conditional_probe']['samples'] for e in evaluations.values())}


def report(config):
    output = Path(config['output_dir'])
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in METHODS}
    updates = {m: json.loads((output/m/'training_report.json').read_text()) for m in METHODS[1:]}
    data_hashes = [json.loads((output/m/'request.json').read_text())['data_sha256'] for m in METHODS[1:]]
    if data_hashes[0] != data_hashes[1]:
        raise ValueError('The two methods did not use the same raw groups')
    expected_steps = config['update']['epochs']*config['root_limit']//config['update']['gradient_accumulation_groups']
    for update in updates.values():
        if update['status'] != 'COMPLETE' or update['optimizer_steps'] != expected_steps or update['frozen_before'] != update['frozen_after']:
            raise ValueError('Missing or changed RL update')
    result = {'status': 'COMPLETE', 'scope': 'ONE_FINITE_PR1C_RL_ROUND', 'updates': updates,
        **evaluation_summary(config, evaluations),
        'data_use': 'One shared old-policy TRAIN pool, two matched update epochs, fresh fixed evaluation seeds. No separate SFT phase; the common SFT retention term remains in the RL loss. No held-out task use or reuse of evaluation rewards for optimization.'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'changes': result['changes'], 'paired': result['paired_natural_vs_B0']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'report'])
    parser.add_argument('--config', default='configs/refinement_curriculum_rl.yaml')
    parser.add_argument('--method', choices=METHODS, default='B0')
    args = parser.parse_args(); config, framework = load_config(args.config)
    if args.command == 'evaluate':
        evaluate(config, framework, args.method)
    else:
        report(config)


if __name__ == '__main__':
    main()
