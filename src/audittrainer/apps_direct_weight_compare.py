"""Direct APPS baseline: bare Qwen3-8B versus two frozen executor adapters.

The LARGE direct trajectories are reused. BASE and COMPETITION use the exact
same direct prompt, seeds, decoding, sandbox, selector, and held-out tasks.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

from . import apps_competition_sft as competition, apps_fresh_select as fresh
from . import apps_fresh_system_compare as earlier, apps_passk_select, apps_scale
from .artifacts import ArtifactStore, SourceError
from .dataset import file_hash
from .evaluate import load_task, score_program, write_json
from .policy import LLMPolicy, load_adapter_metadata
from .refinement_policy import active_adapter
from .sandbox import ExecutionLimits, Sandbox
from .schema import Contract


ROOT = Path('runs/refinement/apps_direct_weight_compare')
ARMS = ('BASE', 'COMPETITION')
GPUS = ('0', '1', '2')


def _paths():
    return {
        'model_manifest': Path('data/manifests/model.json'),
        'old_request': apps_scale.OLD_RUN/'request.json',
        'fresh_request': fresh.ROOT/'request.json',
        'fresh_selection': fresh.DATA/'selection.json',
        'fresh_roots': fresh.DATA/'roots.jsonl',
        'fresh_tests': fresh.DATA/'tests.jsonl',
        'large_direct_request': earlier.ROOT/'request.json',
        'large_direct_result': earlier.ROOT/'result.json',
        'large_adapter': apps_scale.ROOT/'LARGE/training/final_adapter/adapter_model.safetensors',
        'competition_request': competition.ROOT/'request.json',
        'competition_result': competition.ROOT/'result.json',
        'competition_adapter': competition.ROOT/'training/final_adapter/adapter_model.safetensors',
    }


def _case_path(arm, rid, seed):
    return ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'Direct weight comparison already exists: {ROOT}')
    earlier.check()
    competition.check(trained=True)
    prior = json.loads((earlier.ROOT/'result.json').read_text())
    trained = json.loads((competition.ROOT/'result.json').read_text())
    if prior['status'] != 'COMPLETE' or trained['status'] != 'COMPLETE':
        raise ValueError('Required completed APPS comparisons are missing')
    framework = earlier._framework()
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arms': list(ARMS), 'reused_arm': 'LARGE',
        'roots': 48, 'candidates_per_root': 3, 'seed_base': fresh.SEED_BASE,
        'framework': framework, 'source_sha256': {name: file_hash(path) for name, path in _paths().items()},
        'direct_prompt_sha256': hashlib.sha256((earlier.SYSTEM_PROMPT + '\n' + earlier.USER_PREFIX).encode()).hexdigest(),
        'selection': 'public example pass, returned status, program exists, then earliest seed',
        'metric': 'all held-out APPS I/O exact and public example when available',
        'gpu_devices': list(GPUS), 'training_steps': 0,
    })
    write_json(ROOT/'gpu_resources.json', {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check():
    earlier.check()
    competition.check(trained=True)
    request = json.loads((ROOT/'request.json').read_text())
    prior = json.loads((earlier.ROOT/'result.json').read_text())
    framework = earlier._framework()
    if (request['status'] != 'PREPARED' or request['arms'] != list(ARMS)
            or request['reused_arm'] != 'LARGE' or request['roots'] != 48
            or request['candidates_per_root'] != 3 or request['seed_base'] != fresh.SEED_BASE
            or request['framework'] != framework
            or request['source_sha256'] != {name: file_hash(path) for name, path in _paths().items()}
            or request['direct_prompt_sha256'] != hashlib.sha256(
                (earlier.SYSTEM_PROMPT + '\n' + earlier.USER_PREFIX).encode()).hexdigest()
            or request['selection'] != 'public example pass, returned status, program exists, then earliest seed'
            or request['metric'] != 'all held-out APPS I/O exact and public example when available'
            or request['gpu_devices'] != list(GPUS) or request['training_steps'] != 0
            or json.loads((ROOT/'gpu_resources.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}
            or prior['status'] != 'COMPLETE'):
        raise ValueError('Frozen direct comparison design or launch configuration changed')
    if (framework['model']['load_in_4bit'] is not True
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072
            or framework['inference']['decoding'] !=
               {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}):
        raise ValueError('Expected the frozen direct inference configuration')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(framework['model']['local_path'], local_files_only=True)
    _, adapter_identity = load_adapter_metadata(
        competition.ROOT/'training/final_adapter', framework['model'], tokenizer)
    if adapter_identity['adapter_weights_sha256'] != request['source_sha256']['competition_adapter']:
        raise ValueError('Competition executor adapter differs from the completed training')
    rows = fresh._rows()
    if set(prior['by_root']) != {row['root_id'] for row in rows}:
        raise ValueError('LARGE direct roots differ from frozen fresh APPS roots')
    lengths = []
    for index, row in enumerate(rows):
        rid = row['root_id']
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        _, prompt, ids = earlier._prompt(tokenizer, spec, framework)
        lengths.append(len(ids))
        old_signals = prior['by_root'][rid]['signals']['DIRECT']
        if len(old_signals) != 3:
            raise ValueError(f'Missing reused LARGE direct candidates: {rid}')
        for repetition, signal in enumerate(old_signals):
            case = json.loads(Path(signal['case_path']).read_text())
            if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                    or case['seed'] != fresh.SEED_BASE+100*index+repetition
                    or case['model_calls'] != 1
                    or case['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or case['model_identity']['adapter_weights_sha256'] !=
                       request['source_sha256']['large_adapter']):
                raise ValueError(f'Reused LARGE direct candidate differs: {rid}/{repetition}')
    ready = {'status': 'READY', 'new_arms': list(ARMS), 'reused_arm': 'LARGE',
             'roots': 48, 'candidates_per_root': 3, 'new_rollouts': 288,
             'new_rollouts_per_gpu': 96, 'gpu_devices': list(GPUS),
             'model': framework['model']['name_or_path'],
             'context_tokens': 8192, 'output_cap': 3072,
             'max_prompt_tokens': max(lengths), 'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def _generate(llm, spec, framework, seed):
    import torch
    from transformers import set_seed

    messages, prompt, ids = earlier._prompt(llm.tokenizer, spec, framework)
    set_seed(seed)
    generation = deepcopy(llm.model.generation_config)
    generation.update(**llm.decoding, max_new_tokens=framework['runtime']['generation_tokens']['write_code'],
                      num_return_sequences=1, num_beams=1, return_dict_in_generate=False,
                      output_scores=False, use_cache=True)
    inputs = torch.tensor([ids], dtype=torch.long, device=llm.model.device)
    adapter = nullcontext() if llm.identity['adapter'] == 'none' else active_adapter(llm.model, 'default')
    with adapter, torch.inference_mode(), llm._autocast():
        sequences = llm.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       generation_config=generation)
    output_ids = sequences[0, len(ids):].tolist()
    eos = generation.eos_token_id
    ended = bool(output_ids and output_ids[-1] in ({eos} if isinstance(eos, int) else set(eos or [])))
    raw = llm.tokenizer.decode(output_ids[:-1] if ended else output_ids,
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {'raw': raw, 'source': earlier._source(raw),
            'truncated': len(output_ids) >= generation.max_new_tokens and not ended,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'messages': messages, 'input_tokens': len(ids), 'output_tokens': len(output_ids),
            'model_identity': llm.identity, 'seed': seed}


def _direct_case(row, tests, framework, llm, arm, directory, seed):
    path = directory/'case_result.json'
    if path.exists():
        return json.loads(path.read_text())
    if directory.exists():
        raise ValueError(f'Unfinished direct candidate exists: {directory}')
    directory.mkdir(parents=True)
    started = time.monotonic()
    definition, _, contract_spec, _, verifier = load_task(row, tests, framework)
    generated = _generate(llm, definition['task']['spec'], framework, seed)
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
              'arm': arm, 'status': status, 'seed': seed, 'model_identity': generated['model_identity'],
              'raw_output': generated['raw'], 'prompt_messages': generated['messages'],
              'prompt_sha256': generated['prompt_sha256'],
              'input_tokens': generated['input_tokens'], 'output_tokens': generated['output_tokens'],
              'model_calls': 1, 'parse_error': error, 'final_program': snapshot,
              'final_scores': scores, 'scoring_cost': scoring,
              'wall_seconds': time.monotonic()-started, 'run_dir': str(directory)}
    write_json(path, record)
    return record


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire

    if not 0 <= shard < len(GPUS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        raise FileExistsError(f'Completed direct comparison shard already exists: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    rows, tests = fresh._rows(), fresh._tests()
    selected_indices = list(range(shard, len(rows), len(GPUS)))
    sandbox = Sandbox(ExecutionLimits(**request['framework']['execution']))
    completed = {rows[i]['root_id']: {'group': None, 'signals': {}} for i in selected_indices}
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        with tqdm(total=len(selected_indices)*3*len(ARMS),
                  desc=f'APPS直接基线 GPU {GPUS[shard]}', unit='轨迹', position=shard,
                  dynamic_ncols=True) as bar:
            for arm in ARMS:
                adapter = None if arm == 'BASE' else competition.ROOT/'training/final_adapter'
                llm = LLMPolicy.from_pretrained(request['framework'], adapter_path=adapter)
                expected_adapter = request['source_sha256']['competition_adapter'] if adapter else None
                if (llm.identity.get('adapter_weights_sha256') != expected_adapter
                        or llm.identity['quantization'] != 'nf4_double'):
                    raise ValueError(f'Loaded model differs from frozen {arm} arm')
                for index in selected_indices:
                    row = rows[index]
                    rid = row['root_id']
                    sample = fresh._sample(json.loads(Path(row['fixture_path']).read_text())['task']['spec'])
                    item = completed[rid]
                    item['group'] = 'sample' if sample else 'no_sample'
                    signals = []
                    for repetition in range(3):
                        seed = request['seed_base']+100*index+repetition
                        path = _case_path(arm, rid, seed)
                        case = _direct_case(row, tests, request['framework'], llm, arm, path.parent, seed)
                        if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                                or case['seed'] != seed or case['model_calls'] != 1):
                            raise ValueError(f'Incomplete {arm} direct candidate: {rid}/{repetition}')
                        signals.append(apps_passk_select._score(path, rid, sample, sandbox))
                        bar.update(1)
                    item['signals'][arm] = signals
                del llm
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        write_json(done, {'status': 'COMPLETE', 'shard': shard,
                          'request_sha256': file_hash(ROOT/'request.json'), 'by_root': completed})
    finally:
        reservation.close()


def report():
    check()
    request = json.loads((ROOT/'request.json').read_text())
    prior = json.loads((earlier.ROOT/'result.json').read_text())['by_root']
    shards = [json.loads((ROOT/f'shard_{i}.json').read_text()) for i in range(len(GPUS))]
    signals = {}
    for index, shard in enumerate(shards):
        if (shard['status'] != 'COMPLETE' or shard['shard'] != index
                or shard['request_sha256'] != file_hash(ROOT/'request.json')
                or len(shard['by_root']) != 16 or set(signals) & set(shard['by_root'])):
            raise ValueError(f'Incomplete comparison shard: {index}')
        signals.update(shard['by_root'])
    rows = fresh._rows()
    if set(signals) != {row['root_id'] for row in rows}:
        raise ValueError('Missing fresh APPS direct roots')
    summary = {arm: {str(k): Counter() for k in (1, 3)}
               for arm in ('BASE', 'LARGE', 'COMPETITION')}
    paired = {arm: Counter() for arm in ('BASE', 'COMPETITION')}
    for index, row in enumerate(rows):
        rid = row['root_id']
        group = prior[rid]['group']
        if signals[rid]['group'] != group:
            raise ValueError(f'Public sample group changed: {rid}')
        all_signals = {'BASE': signals[rid]['signals']['BASE'],
                       'LARGE': prior[rid]['signals']['DIRECT'],
                       'COMPETITION': signals[rid]['signals']['COMPETITION']}
        strict_by_arm = {}
        for arm, candidates in all_signals.items():
            if len(candidates) != 3:
                raise ValueError(f'Missing direct candidates: {arm}/{rid}')
            cases = [json.loads(Path(signal['case_path']).read_text()) for signal in candidates]
            expected_sha = (None if arm == 'BASE' else request['source_sha256'][
                'large_adapter' if arm == 'LARGE' else 'competition_adapter'])
            if any(case['root_id'] != rid or case['seed'] != request['seed_base']+100*index+i
                   or case['model_calls'] != 1 or case['status'] != 'COMPLETE'
                   or case['model_identity'].get('adapter_weights_sha256') != expected_sha
                   for i, case in enumerate(cases)):
                raise ValueError(f'Direct model, seed, or status changed: {arm}/{rid}')
            private = [apps_scale._valid_private(case) for case in cases]
            strict = [passed and (group == 'no_sample' or bool(signal['sample_pass']))
                      for passed, signal in zip(private, candidates)]
            strict_by_arm[arm] = strict
            for k in (1, 3):
                selected = apps_passk_select.choose(candidates, k, group == 'sample')
                count = summary[arm][str(k)]
                count['roots'] += 1
                count['strict'] += strict[selected]
                count['private'] += private[selected]
                count['oracle_strict'] += any(strict[:k])
                count['program_candidates'] += sum(bool(case['final_program']) for case in cases[:k])
                count['public_sample_pass_candidates'] += sum(bool(s['sample_pass']) for s in candidates[:k])
                count['input_tokens'] += sum(case['input_tokens'] for case in cases[:k])
                count['output_tokens'] += sum(case['output_tokens'] for case in cases[:k])
                count['model_calls'] += k
        large_index = apps_passk_select.choose(all_signals['LARGE'], 3, group == 'sample')
        for arm in ('BASE', 'COMPETITION'):
            chosen = apps_passk_select.choose(all_signals[arm], 3, group == 'sample')
            a, b = strict_by_arm['LARGE'][large_index], strict_by_arm[arm][chosen]
            paired[arm]['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'summary': {arm: {k: dict(v) for k, v in ks.items()} for arm, ks in summary.items()},
              'paired_vs_LARGE_k3': {arm: dict(count) for arm, count in paired.items()},
              'metric': request['metric'], 'selection': request['selection']}
    write_json(ROOT/'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


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
