"""Frozen APPS comparison: direct code, AuditTrainer ATOMIC, and free planning.

The existing three-seed ATOMIC rollouts are reused. Direct and free arms use
the same LARGE executor, task split, seeds, decoding, and private scorer.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

from . import apps_fresh_select as fresh, apps_passk_select, apps_scale
from .artifacts import ArtifactStore, SourceError
from .dataset import file_hash
from .evaluate import load_task, score_program, write_json
from .refinement_policy import active_adapter
from .sandbox import ExecutionLimits, Sandbox
from .schema import Contract


ROOT = Path('runs/refinement/apps_fresh_system_compare')
ARMS = ('DIRECT', 'FREE')
SYSTEM_PROMPT = (
    'You are solving a competitive programming problem. Return only Python source code '
    'for exactly one public function def solve(text): which receives the entire standard '
    'input as a string and returns the required standard output as a string. '
    'Private helpers must start with _. Use the Python standard library. '
    'Do not read stdin, print to host stdout, access files or network, or include Markdown or explanations.'
)
USER_PREFIX = 'Problem statement:\n'


def _framework():
    return json.loads((apps_scale.OLD_RUN/'request.json').read_text())['framework']


def _paths():
    return {
        'fresh_request': fresh.ROOT/'request.json',
        'fresh_result': fresh.ROOT/'result.json',
        'fresh_selection': fresh.DATA/'selection.json',
        'fresh_roots': fresh.DATA/'roots.jsonl',
        'fresh_tests': fresh.DATA/'tests.jsonl',
        'scale_request': apps_scale.ROOT/'request.json',
        'old_request': apps_scale.OLD_RUN/'request.json',
        'large_adapter': apps_scale.ROOT/'LARGE/training/final_adapter/adapter_model.safetensors',
    }


def _messages(spec):
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PREFIX + spec}]


def _prompt(tokenizer, spec, framework):
    messages = _messages(spec)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    reserve = framework['runtime']['generation_tokens']['write_code']
    if len(ids) + reserve > framework['runtime']['context_tokens']:
        raise ValueError('Direct problem and code allowance exceed the frozen context')
    return messages, prompt, ids


def _source(raw):
    """Accept plain code, a single Python fence, or the executor's trained action format."""
    value = raw.strip()
    if value.startswith('{'):
        try:
            action = json.loads(value)
            if (action.get('action') == 'write_code'
                    and action.get('args', {}).get('contract_version') == 1
                    and action['args'].get('declared_dependencies') == []):
                value = action['args']['source']
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    else:
        fence = re.search(r'```(?:python)?\s*\n(.*?)```', value, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            value = fence.group(1).strip()
    return value


def _index():
    return {row['root_id']: i for i, row in enumerate(fresh._rows())}


def _case_path(arm, rid, seed):
    return ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'APPS system comparison already exists: {ROOT}')
    fresh.check()
    source = json.loads((fresh.ROOT/'result.json').read_text())
    framework = _framework()
    if source['status'] != 'COMPLETE' or len(source['by_root']) != 48:
        raise ValueError('Three-seed ATOMIC source is incomplete')
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'roots': 48, 'candidates_per_root': 3,
        'new_arms': list(ARMS), 'reused_arm': 'ATOMIC',
        'source_sha256': {key: file_hash(path) for key, path in _paths().items()},
        'seed_base': fresh.SEED_BASE, 'framework': framework,
        'direct_prompt_sha256': hashlib.sha256((SYSTEM_PROMPT + '\n' + USER_PREFIX).encode()).hexdigest(),
        'direct_generation_tokens': framework['runtime']['generation_tokens']['write_code'],
        'selection': 'public example pass, returned status, program exists, then earliest seed',
        'metric': 'exact held-out APPS I/O and matched public example when available',
        'gpu_devices': list(fresh.GPU_IDS), 'training_steps': 0,
    })
    write_json(ROOT/'gpu_resources.json', {'devices': list(fresh.GPU_IDS),
                                           'min_free_mib': {'collect': 16000}})
    check()


def check():
    fresh.check()
    request = json.loads((ROOT/'request.json').read_text())
    resources = json.loads((ROOT/'gpu_resources.json').read_text())
    source = json.loads((fresh.ROOT/'result.json').read_text())
    framework = _framework()
    rows = fresh._rows()
    if (request['status'] != 'PREPARED' or request['roots'] != 48
            or request['candidates_per_root'] != 3 or request['new_arms'] != list(ARMS)
            or request['reused_arm'] != 'ATOMIC'
            or request['source_sha256'] != {key: file_hash(path) for key, path in _paths().items()}
            or request['seed_base'] != fresh.SEED_BASE or request['framework'] != framework
            or request['direct_prompt_sha256'] != hashlib.sha256((SYSTEM_PROMPT + '\n' + USER_PREFIX).encode()).hexdigest()
            or request['direct_generation_tokens'] != framework['runtime']['generation_tokens']['write_code']
            or request['selection'] != 'public example pass, returned status, program exists, then earliest seed'
            or request['metric'] != 'exact held-out APPS I/O and matched public example when available'
            or request['gpu_devices'] != list(fresh.GPU_IDS) or request['training_steps'] != 0
            or resources != {'devices': list(fresh.GPU_IDS), 'min_free_mib': {'collect': 16000}}
            or source['status'] != 'COMPLETE' or set(source['by_root']) != {row['root_id'] for row in rows}):
        raise ValueError('Frozen APPS system comparison design or launch configuration changed')
    if (framework['inference']['decoding'] != {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072):
        raise ValueError('Expected the existing LARGE inference allowance')
    scale = json.loads((apps_scale.ROOT/'request.json').read_text())
    for role in ('internal_gate', 'root_gate', 'body'):
        if file_hash(Path(scale['adapters'][role])/'adapter_model.safetensors') != scale['adapter_sha256'][role]:
            raise ValueError(f'Frozen planner adapter changed: {role}')
    for index, row in enumerate(rows):
        rid = row['root_id']
        item = source['by_root'][rid]
        if len(item['candidate_signals']) != 3 or item['selected_index_by_k']['3'] != apps_passk_select.choose(
                item['candidate_signals'], 3, item['group'] == 'sample'):
            raise ValueError(f'ATOMIC selection differs: {rid}')
        for repetition, signal in enumerate(item['candidate_signals']):
            case = json.loads(Path(signal['case_path']).read_text())
            if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                    or case['seed'] != request['seed_base'] + 100*index + repetition
                    or case['registered_plan_modes'] != {'ATOMIC': 1}
                    or case['model_identity']['executor']['adapter_weights_sha256']
                    != request['source_sha256']['large_adapter']):
                raise ValueError(f'ATOMIC source candidate differs: {rid}/{repetition}')
    # Tokenize all direct prompts before launching any GPU worker.
    from transformers import AutoTokenizer
    adapter = apps_scale.ROOT/'LARGE/training/final_adapter'
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    prompt_lengths = []
    for row in rows:
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        prompt_lengths.append(len(_prompt(tokenizer, spec, framework)[2]))
    ready = {'status': 'READY', 'roots': len(rows), 'arms': ['DIRECT', 'ATOMIC', 'FREE'],
             'reused_atomic_rollouts': len(rows)*3, 'new_rollouts': len(rows)*3*len(ARMS),
             'new_rollouts_per_gpu': len(rows)*3*len(ARMS)//len(fresh.GPU_IDS),
             'seeds_per_root': 3, 'decoding': framework['inference']['decoding'],
             'context_tokens': framework['runtime']['context_tokens'],
             'per_code_call_output_tokens': request['direct_generation_tokens'],
             'max_total_tokens_per_trajectory': framework['runtime']['max_total_tokens'],
             'direct_prompt_tokens_max': max(prompt_lengths),
             'gpu_devices': list(fresh.GPU_IDS), 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _generate_direct(executor, spec, framework, seed):
    import torch
    from transformers import set_seed

    llm = executor.llm
    messages, prompt, ids = _prompt(llm.tokenizer, spec, framework)
    set_seed(seed)
    generation = deepcopy(llm.model.generation_config)
    generation.update(**llm.decoding, max_new_tokens=framework['runtime']['generation_tokens']['write_code'],
                      num_return_sequences=1, num_beams=1, return_dict_in_generate=False,
                      output_scores=False, use_cache=True)
    inputs = torch.tensor([ids], dtype=torch.long, device=llm.model.device)
    with active_adapter(llm.model, executor.adapter_name), torch.inference_mode(), llm._autocast():
        sequences = llm.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       generation_config=generation)
    output_ids = sequences[0, len(ids):].tolist()
    eos = generation.eos_token_id
    ended = bool(output_ids and output_ids[-1] in ({eos} if isinstance(eos, int) else set(eos or [])))
    raw = llm.tokenizer.decode(output_ids[:-1] if ended else output_ids,
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {'raw': raw, 'source': _source(raw),
            'truncated': len(output_ids) >= generation.max_new_tokens and not ended,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'messages': messages, 'input_tokens': len(ids), 'output_tokens': len(output_ids),
            'model_identity': llm.identity, 'seed': seed}


def _direct_case(row, tests, framework, executor, directory, seed):
    path = directory/'case_result.json'
    if path.exists():
        return json.loads(path.read_text())
    if directory.exists():
        raise ValueError(f'Unfinished direct candidate exists: {directory}')
    directory.mkdir(parents=True)
    started = time.monotonic()
    definition, _, contract_spec, _, verifier = load_task(row, tests, framework)
    generated = _generate_direct(executor, definition['task']['spec'], framework, seed)
    snapshot, error = None, None
    if generated['truncated']:
        error = 'TRUNCATED_OUTPUT'
    else:
        contract = Contract(**contract_spec.model_dump(), contract_id='C_N0_1', version=1)
        store = ArtifactStore(directory/'final_program')
        try:
            artifact = store.save_initial('N0', contract, generated['source'])
            snapshot = {'path': str(directory/'final_program'), 'root_id': row['root_id'],
                        'artifact': artifact.model_dump(mode='json'),
                        'contract': contract.model_dump(mode='json')}
        except SourceError as exc:
            error = exc.code
    scores, scoring = score_program(snapshot, verifier, row['root_id'])
    status = 'ERROR' if any(block['status'] == 'ERROR' for block in scores.values()) else 'COMPLETE'
    record = {'root_id': row['root_id'], 'family_id': row['family_id'], 'split': row['split'],
              'arm': 'DIRECT', 'status': status, 'seed': seed, 'model_identity': generated['model_identity'],
              'raw_output': generated['raw'], 'prompt_messages': generated['messages'],
              'prompt_sha256': generated['prompt_sha256'],
              'input_tokens': generated['input_tokens'], 'output_tokens': generated['output_tokens'],
              'model_calls': 1, 'parse_error': error, 'final_program': snapshot,
              'final_scores': scores, 'scoring_cost': scoring,
              'wall_seconds': time.monotonic()-started, 'run_dir': str(directory)}
    write_json(path, record)
    print(json.dumps({'root': row['root_id'], 'arm': 'DIRECT', 'seed': seed,
                      'status': status, 'program': snapshot is not None,
                      'seconds': round(record['wall_seconds'], 2)}, ensure_ascii=False), flush=True)
    return record


def _validate_case(case, arm, rid, seed, adapter_sha):
    identity = case['model_identity'] if arm == 'DIRECT' else case['model_identity']['executor']
    if (case['root_id'] != rid or case['seed'] != seed or case['status'] != 'COMPLETE'
            or identity['adapter_weights_sha256'] != adapter_sha):
        raise ValueError(f'Incomplete or mismatched {arm} candidate: {rid}/{seed}')
    if arm == 'DIRECT' and case['model_calls'] != 1:
        raise ValueError(f'Direct candidate used multiple model calls: {rid}/{seed}')


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire
    from .refinement_rollout import run_case

    if not 0 <= shard < len(fresh.GPU_IDS):
        raise ValueError('Expected shard 0, 1 or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed comparison shard already exists: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    old = json.loads((apps_scale.OLD_RUN/'request.json').read_text())
    rows, tests = fresh._rows(), fresh._tests()
    sandbox = Sandbox(ExecutionLimits(**request['framework']['execution']))
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        planner, executor = apps_scale._context('LARGE')
        completed = {}
        selected_indices = list(range(shard, len(rows), len(fresh.GPU_IDS)))
        with tqdm(total=len(selected_indices)*3*len(ARMS),
                  desc=f'APPS系统对照 GPU {fresh.GPU_IDS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            for index in selected_indices:
                row = rows[index]
                rid = row['root_id']
                sample = fresh._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                candidates = {'DIRECT': [], 'FREE': []}
                for repetition in range(3):
                    seed = request['seed_base'] + 100*index + repetition
                    for arm in ARMS:
                        path = _case_path(arm, rid, seed)
                        if arm == 'DIRECT':
                            case = _direct_case(row, tests, request['framework'], executor, path.parent, seed)
                        else:
                            # A natural planner can reject and retry its first plan. Start from
                            # the same seed; do not require first-step registration.
                            case = run_case(row, tests, old['framework'], old['config'],
                                            planner, executor, path.parent, seed)
                        _validate_case(case, arm, rid, seed, request['source_sha256']['large_adapter'])
                        signal = apps_passk_select._score(path, rid, sample, sandbox)
                        candidates[arm].append(signal)
                        bar.update(1)
                completed[rid] = {'group': 'sample' if sample else 'no_sample',
                                  'candidate_signals': candidates}
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'), 'by_root': completed})
    finally:
        reservation.close()
    print(json.dumps({'status': 'COMPLETE', 'shard': shard, 'roots': len(completed),
                      'new_rollouts': len(completed)*3*len(ARMS)}, ensure_ascii=False), flush=True)


def _root_mode(case):
    path = Path(case['run_dir'])/'runtime/plans/N0.v1.json'
    return json.loads(path.read_text())['mode'] if path.exists() else 'UNREGISTERED'


def report():
    check()
    result_path = ROOT/'result.json'
    if result_path.exists():
        raise FileExistsError(f'Completed comparison report already exists: {result_path}')
    request = json.loads((ROOT/'request.json').read_text())
    atomic = json.loads((fresh.ROOT/'result.json').read_text())['by_root']
    signals = {}
    for shard in range(len(fresh.GPU_IDS)):
        item = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (item['status'] != 'COMPLETE' or item['shard'] != shard
                or item['request_sha256'] != file_hash(ROOT/'request.json')
                or len(item['by_root']) != 16 or set(signals) & set(item['by_root'])):
            raise ValueError(f'Incomplete comparison shard: {shard}')
        signals.update(item['by_root'])
    if set(signals) != set(atomic):
        raise ValueError('Direct/free and ATOMIC roots differ')
    zeros = ('roots', 'private_exact', 'private_and_public', 'oracle_private_exact',
             'oracle_private_and_public', 'model_input_tokens', 'model_output_tokens',
             'generation_calls', 'wall_seconds')
    groups = {group: {arm: {str(k): Counter({key: 0 for key in zeros}) for k in (1, 3)}
                      for arm in ('DIRECT', 'ATOMIC', 'FREE')}
              for group in ('all', 'sample', 'no_sample')}
    paired = {arm: Counter({f'k{k}_{outcome}': 0 for k in (1, 3)
                            for outcome in ('win', 'loss', 'tie')})
              for arm in ('DIRECT_vs_ATOMIC', 'FREE_vs_ATOMIC')}
    modes = Counter()
    mode_outcomes = {}
    by_root = {}
    rows = fresh._rows()
    for index, row in enumerate(rows):
        rid = row['root_id']
        group = atomic[rid]['group']
        if signals[rid]['group'] != group:
            raise ValueError(f'Comparison group differs: {rid}')
        source_signals = {'ATOMIC': atomic[rid]['candidate_signals'],
                          **signals[rid]['candidate_signals']}
        outcomes = {}
        for arm, arm_signals in source_signals.items():
            if len(arm_signals) != 3:
                raise ValueError(f'Missing {arm} candidates: {rid}')
            candidates = []
            for repetition, signal in enumerate(arm_signals):
                expected_seed = request['seed_base'] + 100*index + repetition
                expected_path = (Path(atomic[rid]['candidate_signals'][repetition]['case_path'])
                                 if arm == 'ATOMIC' else _case_path(arm, rid, expected_seed))
                if signal['case_path'] != str(expected_path):
                    raise ValueError(f'{arm} candidate path differs: {rid}/{repetition}')
                case = json.loads(expected_path.read_text())
                _validate_case(case, arm, rid, expected_seed, request['source_sha256']['large_adapter'])
                private = apps_scale._valid_private(case)
                public = private and (group == 'no_sample' or bool(signal['sample_pass']))
                cost = (case['actual_model_cost'] if arm != 'DIRECT'
                        else {'input_tokens': case['input_tokens'], 'output_tokens': case['output_tokens'],
                              'generation_calls': case['model_calls']})
                candidates.append({'private_exact': private, 'private_and_public': public,
                                   'has_program': bool(case['final_program']),
                                   'root_mode': _root_mode(case) if arm == 'FREE' else arm,
                                   'model_input_tokens': cost.get('input_tokens', 0),
                                   'model_output_tokens': cost.get('output_tokens', 0),
                                   'generation_calls': cost.get('generation_calls', 0),
                                   'wall_seconds': case['wall_seconds']})
                if arm == 'FREE':
                    mode = candidates[-1]['root_mode']
                    modes[mode] += 1
                    counts = mode_outcomes.setdefault(mode, Counter())
                    counts['candidates'] += 1
                    counts['private_exact'] += private
                    counts['private_and_public'] += public
            outcomes[arm] = candidates
            for k in (1, 3):
                chosen = apps_passk_select.choose(arm_signals, k, group == 'sample')
                if arm == 'ATOMIC' and chosen != atomic[rid]['selected_index_by_k'][str(k)]:
                    raise ValueError(f'ATOMIC selector changed: {rid}/{k}')
                for label in ('all', group):
                    count = groups[label][arm][str(k)]
                    count['roots'] += 1
                    count['private_exact'] += candidates[chosen]['private_exact']
                    count['private_and_public'] += candidates[chosen]['private_and_public']
                    count['oracle_private_exact'] += any(c['private_exact'] for c in candidates[:k])
                    count['oracle_private_and_public'] += any(c['private_and_public'] for c in candidates[:k])
                    count['model_input_tokens'] += sum(c['model_input_tokens'] for c in candidates[:k])
                    count['model_output_tokens'] += sum(c['model_output_tokens'] for c in candidates[:k])
                    count['generation_calls'] += sum(c['generation_calls'] for c in candidates[:k])
                    count['wall_seconds'] += sum(c['wall_seconds'] for c in candidates[:k])
            if arm != 'ATOMIC':
                for k in (1, 3):
                    other = apps_passk_select.choose(arm_signals, k, group == 'sample')
                    base = apps_passk_select.choose(source_signals['ATOMIC'], k, group == 'sample')
                    a, b = candidates[other]['private_and_public'], outcomes['ATOMIC'][base]['private_and_public']
                    paired[f'{arm}_vs_ATOMIC'][f'k{k}_' + ('win' if a and not b else 'loss' if b and not a else 'tie')] += 1
        by_root[rid] = {'group': group, 'signals': source_signals, 'outcomes': outcomes}
    summary = {group: {arm: {k: dict(counts) for k, counts in ks.items()}
                       for arm, ks in arms.items()} for group, arms in groups.items()}
    write_json(result_path, {'status': 'COMPLETE',
                             'request_sha256': file_hash(ROOT/'request.json'),
                             'summary': summary, 'paired': {arm: dict(counts) for arm, counts in paired.items()},
                             'free_root_modes': dict(modes),
                             'free_mode_outcomes': {mode: dict(counts) for mode, counts in mode_outcomes.items()},
                             'by_root': by_root,
                             'budget_note': 'All arms share the same model context and configured total token ceiling; '
                                            'direct uses one code call, while AuditTrainer may use more calls. '
                                            'Actual model tokens and generation calls are reported.'})
    print(json.dumps({'status': 'COMPLETE', 'summary': summary,
                      'paired': {arm: dict(counts) for arm, counts in paired.items()},
                      'free_root_modes': dict(modes),
                      'free_mode_outcomes': {mode: dict(counts) for mode, counts in mode_outcomes.items()}},
                     ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'evaluate', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'evaluate':
        evaluate(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
