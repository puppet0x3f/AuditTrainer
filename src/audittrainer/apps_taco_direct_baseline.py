"""Direct-code baseline for the frozen 100-task TACO transfer benchmark."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import apps_direct_weight_compare as direct
from . import apps_diverse_sft as diverse, apps_fresh_system_compare as earlier
from . import apps_taco_sft as taco, apps_taco_transfer as transfer
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .policy import LLMPolicy, load_adapter_metadata


ROOT = Path('runs/refinement/apps_taco_direct_baseline')
ARMS = ('BASE', 'DIVERSE', 'TACO')
GPUS = ('0', '1', '2')


def _sources():
    return {
        'transfer_request': transfer.ROOT/'request.json',
        'transfer_result': transfer.ROOT/'result.json',
        'roots': transfer.DATA/'roots.jsonl',
        'tests': transfer.DATA/'tests.jsonl',
        'selected': transfer.DATA/'selected_records.jsonl',
        'model_manifest': Path('data/manifests/model.json'),
        'diverse_adapter': diverse.ROOT/'training/final_adapter/adapter_model.safetensors',
        'taco_adapter': taco.ROOT/'training/final_adapter/adapter_model.safetensors',
    }


def _case_path(arm, rid, seed):
    return ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def _atomic_path(arm, rid, seed):
    return transfer.ROOT/'evaluation'/arm/rid/f'seed_{seed}'/'case_result.json'


def _adapter(arm):
    if arm == 'DIVERSE':
        return diverse.ROOT/'training/final_adapter'
    if arm == 'TACO':
        return taco.ROOT/'training/final_adapter'
    return None


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'TACO direct baseline already prepared: {ROOT}')
    transfer.check()
    source = json.loads((transfer.ROOT/'result.json').read_text())
    if source['status'] != 'COMPLETE' or source['roots'] != 100 or source['request_sha256'] != file_hash(transfer.ROOT/'request.json'):
        raise ValueError('TACO transfer result is incomplete')
    framework = json.loads((transfer.ROOT/'request.json').read_text())['framework']
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arms': list(ARMS), 'gpu_devices': list(GPUS),
        'roots': 100, 'candidates_per_root': 1, 'seed_base': transfer.EVAL_SEED,
        'framework': framework,
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'direct_prompt_sha256': hashlib.sha256(
            (earlier.SYSTEM_PROMPT+'\n'+earlier.USER_PREFIX).encode()).hexdigest(),
        'metric': 'visible and nine private TACO cases both pass',
        'comparison': 'same first seed and executor weights as frozen ATOMIC',
        'training_steps': 0,
    })
    write_json(ROOT/'gpu_resources.json',
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check():
    from transformers import AutoTokenizer

    transfer.check()
    request = json.loads((ROOT/'request.json').read_text())
    source = json.loads((transfer.ROOT/'result.json').read_text())
    framework = json.loads((transfer.ROOT/'request.json').read_text())['framework']
    expected_sources = {name: file_hash(path) for name, path in _sources().items()}
    if (request['status'] != 'PREPARED' or request['arms'] != list(ARMS)
            or request['gpu_devices'] != list(GPUS) or request['roots'] != 100
            or request['candidates_per_root'] != 1
            or request['seed_base'] != transfer.EVAL_SEED
            or request['framework'] != framework
            or request['source_sha256'] != expected_sources
            or request['direct_prompt_sha256'] != hashlib.sha256(
                (earlier.SYSTEM_PROMPT+'\n'+earlier.USER_PREFIX).encode()).hexdigest()
            or request['metric'] != 'visible and nine private TACO cases both pass'
            or request['comparison'] != 'same first seed and executor weights as frozen ATOMIC'
            or request['training_steps'] != 0
            or json.loads((ROOT/'gpu_resources.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}
            or source['status'] != 'COMPLETE' or source['roots'] != 100
            or source['request_sha256'] != expected_sources['transfer_request']):
        raise ValueError('TACO direct baseline design or startup configuration changed')
    if (framework['model']['load_in_4bit'] is not True
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072
            or framework['inference']['decoding'] !=
               {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}):
        raise ValueError('Expected frozen ATOMIC model and decoding configuration')
    tokenizer = AutoTokenizer.from_pretrained(framework['model']['local_path'], local_files_only=True)
    for arm in ARMS[1:]:
        _, identity = load_adapter_metadata(_adapter(arm), framework['model'], tokenizer)
        if identity['adapter_weights_sha256'] != expected_sources[arm.lower()+'_adapter']:
            raise ValueError(f'{arm} adapter differs from frozen ATOMIC')
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    if len(rows) != 100:
        raise ValueError('TACO transfer roots changed')
    prompt_lengths = []
    for index, row in enumerate(rows):
        rid = row['root_id']
        seed = request['seed_base']+100*index
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        prompt_lengths.append(len(earlier._prompt(tokenizer, spec, framework)[2]))
        for arm in ARMS[1:]:
            case = json.loads(_atomic_path(arm, rid, seed).read_text())
            if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                    or case['seed'] != seed or case['registered_plan_modes'] != {'ATOMIC': 1}
                    or case['model_identity']['executor']['adapter_weights_sha256']
                       != expected_sources[arm.lower()+'_adapter']):
                raise ValueError(f'Frozen ATOMIC first candidate differs: {arm}/{rid}')
    ready = {'status': 'READY', 'roots': 100, 'new_rollouts': 300,
             'arms': list(ARMS), 'gpu_devices': list(GPUS),
             'rollouts_per_gpu': 100, 'seed_base': request['seed_base'],
             'max_direct_prompt_tokens': max(prompt_lengths),
             'context_tokens': 8192, 'output_cap': 3072,
             'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire

    if not 0 <= shard < len(ARMS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check()
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if (record['status'] == 'COMPLETE' and record['arm'] == ARMS[shard]
                and record['request_sha256'] == file_hash(ROOT/'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete TACO direct shard: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA/'tests.jsonl')}
    arm = ARMS[shard]
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        llm = LLMPolicy.from_pretrained(request['framework'], adapter_path=_adapter(arm))
        expected = None if arm == 'BASE' else request['source_sha256'][arm.lower()+'_adapter']
        if (llm.identity.get('adapter_weights_sha256') != expected
                or llm.identity['quantization'] != 'nf4_double'):
            raise ValueError(f'Loaded {arm} model differs from frozen request')
        with tqdm(total=len(rows), desc=f'TACO直接解题 {arm} GPU {GPUS[shard]}',
                  unit='题', position=shard, dynamic_ncols=True) as bar:
            for index, row in enumerate(rows):
                rid = row['root_id']
                seed = request['seed_base']+100*index
                case = direct._direct_case(
                    row, tests, request['framework'], llm, arm,
                    _case_path(arm, rid, seed).parent, seed)
                if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                        or case['seed'] != seed or case['model_calls'] != 1
                        or case['model_identity'].get('adapter_weights_sha256') != expected):
                    raise RuntimeError(f'Incomplete TACO direct candidate: {arm}/{rid}')
                bar.update(1)
        write_json(done, {'status': 'COMPLETE', 'shard': shard, 'arm': arm,
                          'request_sha256': file_hash(ROOT/'request.json'),
                          'roots': [row['root_id'] for row in rows]})
    finally:
        reservation.close()


def _signals(case, *, direct_case, first=False):
    scores = case['first_scores'] if first else case['final_scores']
    visible = scores['visible']['status'] == 'PASS'
    private = scores['private']['status'] == 'PASS'
    result = {'strict': int(visible and private), 'visible': int(visible),
              'private': int(private),
              'program': int((case['first_program'] if first else case['final_program']) is not None)}
    if first:
        return result
    if direct_case:
        result.update(input_tokens=case['input_tokens'], output_tokens=case['output_tokens'],
                      model_calls=case['model_calls'])
    else:
        result.update(input_tokens=case['actual_model_cost'].get('input_tokens', 0),
                      output_tokens=case['actual_model_cost'].get('output_tokens', 0),
                      model_calls=case['role_calls'].get('executor', 0))
    return result


def report():
    from tqdm import tqdm

    check()
    if (ROOT/'result.json').exists():
        raise FileExistsError(f'TACO direct result already exists: {ROOT/"result.json"}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    difficulties = {f'taco_test_{r["id"]:04d}': r['difficulty']
                    for r in read_jsonl(transfer.DATA/'selected_records.jsonl')}
    for shard, arm in enumerate(ARMS):
        record = json.loads((ROOT/f'shard_{shard}.json').read_text())
        if (record['status'] != 'COMPLETE' or record['shard'] != shard
                or record['arm'] != arm
                or record['request_sha256'] != file_hash(ROOT/'request.json')
                or record['roots'] != [row['root_id'] for row in rows]):
            raise ValueError(f'Direct shard incomplete: {arm}')
    summary = {'DIRECT': {arm: Counter() for arm in ARMS},
               'ATOMIC_FIRST': {arm: Counter() for arm in ARMS[1:]},
               'ATOMIC_FINAL': {arm: Counter() for arm in ARMS[1:]}}
    by_difficulty = {difficulty: {mode: {arm: Counter() for arm in arms}
                                  for mode, arms in summary.items()}
                     for difficulty in transfer.QUOTAS}
    paired_atomic = {mode: {arm: Counter() for arm in ARMS[1:]}
                     for mode in ('ATOMIC_FIRST', 'ATOMIC_FINAL')}
    paired_base = {arm: Counter() for arm in ARMS[1:]}
    parse_errors = {arm: Counter() for arm in ARMS}
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='TACO直接解题配对汇总', unit='题',
                                     dynamic_ncols=True)):
        rid = row['root_id']
        seed = request['seed_base']+100*index
        difficulty = difficulties[rid]
        values = {mode: {} for mode in summary}
        for mode, arms in (('DIRECT', ARMS), ('ATOMIC_FIRST', ARMS[1:]),
                           ('ATOMIC_FINAL', ARMS[1:])):
            for arm in arms:
                path = (_case_path(arm, rid, seed) if mode == 'DIRECT'
                        else _atomic_path(arm, rid, seed))
                case = json.loads(path.read_text())
                identity = (case['model_identity'] if mode == 'DIRECT'
                            else case['model_identity']['executor'])
                expected = (None if arm == 'BASE' else
                            request['source_sha256'][arm.lower()+'_adapter'])
                if (case['status'] != 'COMPLETE' or case['root_id'] != rid
                        or case['seed'] != seed
                        or identity.get('adapter_weights_sha256') != expected):
                    raise ValueError(f'Candidate identity differs: {mode}/{arm}/{rid}')
                signal = _signals(case, direct_case=mode == 'DIRECT',
                                  first=mode == 'ATOMIC_FIRST')
                values[mode][arm] = signal
                for counter in (summary[mode][arm], by_difficulty[difficulty][mode][arm]):
                    counter['roots'] += 1
                    counter.update(signal)
                if mode == 'DIRECT':
                    parse_errors[arm][case['parse_error'] or 'NONE'] += 1
        for arm in ARMS[1:]:
            b = values['DIRECT'][arm]['strict']
            for mode in paired_atomic:
                a = values[mode][arm]['strict']
                paired_atomic[mode][arm][
                    'win' if b and not a else 'loss' if a and not b else 'tie'] += 1
            a = values['DIRECT']['BASE']['strict']
            paired_base[arm]['win' if b and not a else 'loss' if a and not b else 'tie'] += 1
        by_root[rid] = {'difficulty': difficulty, **values}
    result = {'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
              'roots': len(rows), 'summary': {mode: {arm: dict(c) for arm, c in arms.items()}
                                             for mode, arms in summary.items()},
              'by_difficulty': {d: {mode: {arm: dict(c) for arm, c in arms.items()}
                                   for mode, arms in modes.items()}
                                for d, modes in by_difficulty.items()},
              'paired_DIRECT_vs_ATOMIC': {
                  mode: {arm: dict(c) for arm, c in arms.items()}
                  for mode, arms in paired_atomic.items()},
              'paired_DIRECT_vs_BASE': {arm: dict(c) for arm, c in paired_base.items()},
              'parse_errors': {arm: dict(c) for arm, c in parse_errors.items()},
              'by_root': by_root}
    write_json(ROOT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'by_root'},
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
