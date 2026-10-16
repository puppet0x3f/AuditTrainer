"""Fair three-candidate direct baseline for the frozen TACO transfer set."""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

from . import apps_direct_weight_compare as direct
from . import apps_diverse_sft as diverse
from . import apps_taco_io_replay as interface
from . import apps_taco_sft as taco
from . import apps_taco_transfer as transfer
from . import apps_taco_direct_baseline as baseline
from .dataset import file_hash, read_jsonl
from .evaluate import load_task, score_program, write_json
from .policy import LLMPolicy, load_adapter_metadata
from .schema import Contract


ROOT = Path('runs/refinement/apps_taco_direct_k3')
ARMS = ('BASE', 'DIVERSE', 'TACO')
GPUS = ('0', '1', '2')
REPETITIONS = 3
NEW_REPETITIONS = (1, 2)


def _sources():
    return {
        'baseline_request': baseline.ROOT/'request.json',
        'baseline_result': baseline.ROOT/'result.json',
        'transfer_request': transfer.ROOT/'request.json',
        'transfer_result': transfer.ROOT/'result.json',
        'roots': transfer.DATA/'roots.jsonl',
        'tests': transfer.DATA/'tests.jsonl',
        'selected': transfer.DATA/'selected_records.jsonl',
        'model_manifest': Path('data/manifests/model.json'),
        'diverse_adapter': diverse.ROOT/'training/final_adapter/adapter_model.safetensors',
        'taco_adapter': taco.ROOT/'training/final_adapter/adapter_model.safetensors',
        'bridge_module': Path(interface.__file__),
    }


def _seed(index, repetition):
    return transfer.EVAL_SEED + 100*index + repetition


def _adapter(arm):
    return baseline._adapter(arm)


def _original_path(arm, rid, seed, repetition):
    if repetition == 0:
        return baseline._case_path(arm, rid, seed)
    return ROOT/'evaluation/original'/arm/rid/f'seed_{seed}'/'case_result.json'


def _bridged_path(arm, rid, seed):
    return ROOT/'evaluation/bridged'/arm/rid/f'seed_{seed}'/'case_result.json'


def _identity_ok(case, arm, seed, expected_adapter):
    return (case['status'] == 'COMPLETE' and case['arm'] == arm
            and case['seed'] == seed and case['model_calls'] == 1
            and case['model_identity'].get('adapter_weights_sha256') == expected_adapter)


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'TACO direct k3 comparison already exists: {ROOT}')
    transfer.check()
    baseline.check()
    transfer_result = json.loads((transfer.ROOT/'result.json').read_text())
    if (transfer_result['status'] != 'COMPLETE' or transfer_result['roots'] != 100
            or transfer_result['request_sha256'] != file_hash(transfer.ROOT/'request.json')):
        raise ValueError('Complete TACO transfer result is required')
    framework = json.loads((baseline.ROOT/'request.json').read_text())['framework']
    ROOT.mkdir(parents=True)
    write_json(ROOT/'request.json', {
        'status': 'PREPARED', 'arms': list(ARMS), 'roots': 100,
        'candidates_per_root': REPETITIONS, 'reused_repetitions': [0],
        'new_repetitions': list(NEW_REPETITIONS), 'seed_base': transfer.EVAL_SEED,
        'gpu_devices': list(GPUS), 'cpu_shards': 3,
        'framework': framework,
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'bridge_sha256': hashlib.sha256(interface.BRIDGE.encode()).hexdigest(),
        'selection': 'first visible fixture pass, then earliest candidate',
        'metric': 'visible and nine private TACO cases both pass',
        'comparison': 'same three seeds and public selector as frozen ATOMIC transfer',
        'new_model_calls': len(ARMS)*100*len(NEW_REPETITIONS),
        'training_steps': 0,
    })
    write_json(ROOT/'gpu_resources.json',
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}})
    return check()


def check():
    transfer.check()
    baseline.check()
    request = json.loads((ROOT/'request.json').read_text())
    baseline_request = json.loads((baseline.ROOT/'request.json').read_text())
    expected_sources = {name: file_hash(path) for name, path in _sources().items()}
    if (request['status'] != 'PREPARED' or request['arms'] != list(ARMS)
            or request['roots'] != 100 or request['candidates_per_root'] != REPETITIONS
            or request['reused_repetitions'] != [0]
            or request['new_repetitions'] != list(NEW_REPETITIONS)
            or request['seed_base'] != transfer.EVAL_SEED
            or request['gpu_devices'] != list(GPUS) or request['cpu_shards'] != 3
            or request['framework'] != baseline_request['framework']
            or request['source_sha256'] != expected_sources
            or request['bridge_sha256'] != hashlib.sha256(interface.BRIDGE.encode()).hexdigest()
            or request['selection'] != 'first visible fixture pass, then earliest candidate'
            or request['metric'] != 'visible and nine private TACO cases both pass'
            or request['comparison'] != 'same three seeds and public selector as frozen ATOMIC transfer'
            or request['new_model_calls'] != 600 or request['training_steps'] != 0
            or json.loads((ROOT/'gpu_resources.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('Frozen TACO direct k3 design or startup configuration changed')
    framework = request['framework']
    if (framework['model']['load_in_4bit'] is not True
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072
            or framework['inference']['decoding'] !=
               {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}):
        raise ValueError('Expected frozen direct inference configuration')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(framework['model']['local_path'], local_files_only=True)
    for arm in ARMS[1:]:
        _, identity = load_adapter_metadata(_adapter(arm), framework['model'], tokenizer)
        if identity['adapter_weights_sha256'] != expected_sources[arm.lower()+'_adapter']:
            raise ValueError(f'{arm} adapter differs from frozen request')
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    if len(rows) != 100:
        raise ValueError('TACO transfer roots changed')
    expected_adapters = {arm: None if arm == 'BASE' else
                         expected_sources[arm.lower()+'_adapter'] for arm in ARMS}
    existing = 0
    new_complete = 0
    for index, row in enumerate(rows):
        rid = row['root_id']
        seed = _seed(index, 0)
        for arm in ARMS:
            path = _original_path(arm, rid, seed, 0)
            case = json.loads(path.read_text())
            if not _identity_ok(case, arm, seed, expected_adapters[arm]):
                raise ValueError(f'Reused seed-zero candidate differs: {arm}/{rid}')
            existing += 1
            for repetition in NEW_REPETITIONS:
                candidate = _original_path(arm, rid, _seed(index, repetition), repetition)
                if candidate.exists():
                    saved = json.loads(candidate.read_text())
                    if not _identity_ok(saved, arm, _seed(index, repetition), expected_adapters[arm]):
                        raise ValueError(f'Saved new candidate differs: {arm}/{rid}/{repetition}')
                    new_complete += 1
    ready = {'status': 'READY', 'roots': len(rows), 'arms': list(ARMS),
             'reused_candidates': existing, 'new_rollouts': 600,
             'new_rollouts_per_gpu': 200, 'candidates_to_bridge': 900,
             'gpu_devices': list(GPUS), 'cpu_shards': 3,
             'decoding': framework['inference']['decoding'],
             'context_tokens': framework['runtime']['context_tokens'],
             'output_cap': framework['runtime']['generation_tokens']['write_code'],
             'new_candidates_already_complete': new_complete,
             'training_steps': 0}
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire

    if not 0 <= shard < len(ARMS):
        raise ValueError('Expected GPU shard 0, 1, or 2')
    check()
    arm = ARMS[shard]
    done = ROOT/f'shard_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if (record['status'] == 'COMPLETE' and record['shard'] == shard
                and record['arm'] == arm
                and record['request_sha256'] == file_hash(ROOT/'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete TACO direct k3 shard: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA/'tests.jsonl')}
    expected = None if arm == 'BASE' else request['source_sha256'][arm.lower()+'_adapter']
    reservation = acquire(ROOT/'gpu_resources.json', 'collect')
    try:
        llm = LLMPolicy.from_pretrained(request['framework'], adapter_path=_adapter(arm))
        if (llm.identity.get('adapter_weights_sha256') != expected
                or llm.identity['quantization'] != 'nf4_double'):
            raise ValueError(f'Loaded {arm} model differs from frozen request')
        with tqdm(total=len(rows)*len(NEW_REPETITIONS),
                  desc=f'TACO直接 k3 {arm} GPU {GPUS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            for repetition in NEW_REPETITIONS:
                for index, row in enumerate(rows):
                    rid = row['root_id']
                    seed = _seed(index, repetition)
                    path = _original_path(arm, rid, seed, repetition)
                    case = direct._direct_case(row, tests, request['framework'], llm,
                                               arm, path.parent, seed)
                    if not _identity_ok(case, arm, seed, expected):
                        raise RuntimeError(f'Incomplete direct k3 candidate: {arm}/{rid}/{repetition}')
                    bar.update(1)
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        write_json(done, {'status': 'COMPLETE', 'shard': shard, 'arm': arm,
                          'request_sha256': file_hash(ROOT/'request.json'),
                          'roots': [row['root_id'] for row in rows],
                          'new_repetitions': list(NEW_REPETITIONS)})
    finally:
        reservation.close()


def bridge(shard):
    from tqdm import tqdm

    if not 0 <= shard < len(ARMS):
        raise ValueError('Expected CPU bridge shard 0, 1, or 2')
    check()
    arm = ARMS[shard]
    done = ROOT/f'bridge_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if (record['status'] == 'COMPLETE' and record['shard'] == shard
                and record['arm'] == arm
                and record['request_sha256'] == file_hash(ROOT/'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete TACO direct k3 bridge shard: {done}')
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA/'tests.jsonl')}
    with tqdm(total=len(rows)*REPETITIONS,
              desc=f'TACO直接 k3 接口复评 {arm}', unit='候选', position=shard,
              dynamic_ncols=True) as bar:
        for index, row in enumerate(rows):
            rid = row['root_id']
            for repetition in range(REPETITIONS):
                seed = _seed(index, repetition)
                original_path = _original_path(arm, rid, seed, repetition)
                target = _bridged_path(arm, rid, seed)
                if target.exists():
                    saved = json.loads(target.read_text())
                    if (saved['status'] != 'COMPLETE' or saved['arm'] != arm
                            or saved['root_id'] != rid or saved['seed'] != seed
                            or saved['original_case_sha256'] != file_hash(original_path)
                            or saved['request_sha256'] != file_hash(ROOT/'request.json')):
                        raise ValueError(f'Saved bridge candidate differs: {arm}/{rid}/{repetition}')
                    bar.update(1)
                    continue
                if target.parent.exists():
                    raise ValueError(f'Unfinished bridge candidate exists: {target.parent}')
                original = json.loads(original_path.read_text())
                target.parent.mkdir(parents=True)
                snapshot = original['final_program']
                if snapshot is None:
                    adapted, scores, cost = None, original['final_scores'], original['scoring_cost']
                    applied = False
                else:
                    _, _, contract_spec, _, verifier = load_task(row, tests, request['framework'])
                    contract = Contract(**contract_spec.model_dump(),
                                        contract_id='C_N0_1', version=1)
                    source = (Path(snapshot['path'])/
                              snapshot['artifact']['module_path']).read_text()
                    store = interface.ArtifactStore(target.parent/'final_program')
                    artifact = store.save_initial('N0', contract, interface.bridge_source(source))
                    adapted = {'path': str(target.parent/'final_program'), 'root_id': rid,
                               'artifact': artifact.model_dump(mode='json'),
                               'contract': contract.model_dump(mode='json')}
                    scores, cost = score_program(adapted, verifier, rid)
                    applied = True
                status = ('ERROR' if any(value['status'] == 'ERROR' for value in scores.values())
                          else 'COMPLETE')
                if status != 'COMPLETE':
                    raise RuntimeError(f'Bridge scoring error: {arm}/{rid}/{repetition}')
                write_json(target, {
                    'status': status, 'root_id': rid, 'arm': arm, 'seed': seed,
                    'request_sha256': file_hash(ROOT/'request.json'),
                    'original_case_path': str(original_path),
                    'original_case_sha256': file_hash(original_path),
                    'bridge_applied': applied, 'final_program': adapted,
                    'final_scores': scores, 'scoring_cost': cost,
                    'run_dir': str(target.parent),
                })
                bar.update(1)
    write_json(done, {'status': 'COMPLETE', 'shard': shard, 'arm': arm,
                      'request_sha256': file_hash(ROOT/'request.json'),
                      'roots': [row['root_id'] for row in rows],
                      'repetitions': list(range(REPETITIONS))})


def _signal(case):
    visible = case['final_scores']['visible']['status'] == 'PASS'
    private = case['final_scores']['private']['status'] == 'PASS'
    return {'strict': int(visible and private), 'visible': int(visible),
            'private': int(private), 'program': int(case['final_program'] is not None),
            'candidate_exception': int(case['final_scores']['visible']['reason'] ==
                                       'CANDIDATE_EXCEPTION')}


def _selected(signals, k):
    return next((index for index, signal in enumerate(signals[:k])
                 if signal['visible']), 0)


def _empty_counts():
    return Counter()


def _add_counts(counter, signal):
    counter['roots'] += 1
    for key in ('strict', 'visible', 'private', 'program', 'candidate_exception'):
        counter[key] += signal[key]


def _case(path):
    return json.loads(path.read_text())


def report():
    from tqdm import tqdm

    check()
    request = json.loads((ROOT/'request.json').read_text())
    rows = read_jsonl(transfer.DATA/'roots.jsonl')
    records = {f'taco_test_{r["id"]:04d}': r
               for r in read_jsonl(transfer.DATA/'selected_records.jsonl')}
    for shard, arm in enumerate(ARMS):
        gpu = json.loads((ROOT/f'shard_{shard}.json').read_text())
        cpu = json.loads((ROOT/f'bridge_{shard}.json').read_text())
        for saved, label in ((gpu, 'GPU'), (cpu, 'bridge')):
            if (saved['status'] != 'COMPLETE' or saved['shard'] != shard
                    or saved['arm'] != arm
                    or saved['request_sha256'] != file_hash(ROOT/'request.json')
                    or saved['roots'] != [row['root_id'] for row in rows]):
                raise ValueError(f'Incomplete direct k3 {label} shard: {arm}')
    modes = {
        'DIRECT_ORIGINAL': {arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS},
        'DIRECT_BRIDGED': {arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS},
        'ATOMIC': {arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS[1:]},
    }
    by_difficulty = {
        difficulty: {
            mode: {arm: {str(k): Counter() for k in (1, 3)}
                   for arm in arms}
            for mode, arms in modes.items()
        }
        for difficulty in transfer.QUOTAS
    }
    paired = {
        'DIRECT_BRIDGED_vs_ATOMIC': {
            arm: {str(k): Counter() for k in (1, 3)} for arm in ARMS[1:]},
        'DIRECT_BRIDGED_vs_ATOMIC_ORACLE': {
            arm: Counter() for arm in ARMS[1:]},
    }
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='TACO直接 k3 配对汇总', unit='题',
                                     dynamic_ncols=True)):
        rid = row['root_id']
        difficulty = records[rid]['difficulty']
        root = {'difficulty': difficulty, 'direct_original': {},
                'direct_bridged': {}, 'atomic': {}}
        atomic_by_arm = {}
        for arm in ARMS:
            originals = [_case(_original_path(arm, rid, _seed(index, rep), rep))
                         for rep in range(REPETITIONS)]
            bridged = [_case(_bridged_path(arm, rid, _seed(index, rep)))
                       for rep in range(REPETITIONS)]
            expected = None if arm == 'BASE' else request['source_sha256'][arm.lower()+'_adapter']
            for repetition, case in enumerate(originals):
                if not _identity_ok(case, arm, _seed(index, repetition), expected):
                    raise ValueError(f'Direct identity changed: {arm}/{rid}/{repetition}')
            for repetition, case in enumerate(bridged):
                original_path = _original_path(arm, rid, _seed(index, repetition), repetition)
                if (case['status'] != 'COMPLETE' or case['arm'] != arm
                        or case['root_id'] != rid or case['seed'] != _seed(index, repetition)
                        or case['request_sha256'] != file_hash(ROOT/'request.json')
                        or case['original_case_path'] != str(original_path)
                        or case['original_case_sha256'] != file_hash(original_path)):
                    raise ValueError(f'Bridged identity or provenance changed: {arm}/{rid}/{repetition}')
            original_signals = [_signal(case) for case in originals]
            bridged_signals = [_signal(case) for case in bridged]
            root['direct_original'][arm] = {'candidates': original_signals,
                                             'selected': {str(k): _selected(original_signals, k)
                                                          for k in (1, 3)}}
            root['direct_bridged'][arm] = {'candidates': bridged_signals,
                                           'selected': {str(k): _selected(bridged_signals, k)
                                                        for k in (1, 3)}}
            for mode, signals in (('DIRECT_ORIGINAL', original_signals),
                                  ('DIRECT_BRIDGED', bridged_signals)):
                for k in (1, 3):
                    selected = signals[_selected(signals, k)]
                    _add_counts(modes[mode][arm][str(k)], selected)
                    _add_counts(by_difficulty[difficulty][mode][arm][str(k)], selected)
            if arm != 'BASE':
                atomic = [_case(transfer.ROOT/'evaluation'/arm/rid/
                                f'seed_{_seed(index, rep)}'/'case_result.json')
                          for rep in range(REPETITIONS)]
                atomic_signals = [_signal(case) for case in atomic]
                atomic_by_arm[arm] = atomic_signals
                root['atomic'][arm] = {'candidates': atomic_signals,
                                       'selected': {str(k): _selected(atomic_signals, k)
                                                    for k in (1, 3)}}
                for k in (1, 3):
                    selected = atomic_signals[_selected(atomic_signals, k)]
                    _add_counts(modes['ATOMIC'][arm][str(k)], selected)
                    _add_counts(by_difficulty[difficulty]['ATOMIC'][arm][str(k)], selected)
                    direct_selected = bridged_signals[_selected(bridged_signals, k)]
                    atomic_selected = selected
                    paired['DIRECT_BRIDGED_vs_ATOMIC'][arm][str(k)][
                        'win' if direct_selected['strict'] and not atomic_selected['strict']
                        else 'loss' if atomic_selected['strict'] and not direct_selected['strict']
                        else 'tie'] += 1
                direct_oracle = any(signal['strict'] for signal in bridged_signals)
                atomic_oracle = any(signal['strict'] for signal in atomic_signals)
                paired['DIRECT_BRIDGED_vs_ATOMIC_ORACLE'][arm][
                    'win' if direct_oracle and not atomic_oracle
                    else 'loss' if atomic_oracle and not direct_oracle
                    else 'tie'] += 1
        by_root[rid] = root
    result = {
        'status': 'COMPLETE', 'request_sha256': file_hash(ROOT/'request.json'),
        'roots': len(rows), 'candidates_per_root': REPETITIONS,
        'summary': {mode: {arm: {k: dict(value) for k, value in arms.items()}
                           for arm, arms in values.items()}
                    for mode, values in modes.items()},
        'by_difficulty': {
            difficulty: {mode: {arm: {k: dict(value) for k, value in arms.items()}
                                for arm, arms in values.items()}
                         for mode, values in modes.items()}
            for difficulty, modes in by_difficulty.items()},
        'paired_DIRECT_BRIDGED_vs_ATOMIC': {
            arm: {k: dict(value) for k, value in ks.items()}
            for arm, ks in paired['DIRECT_BRIDGED_vs_ATOMIC'].items()},
        'paired_DIRECT_BRIDGED_vs_ATOMIC_ORACLE': {
            arm: dict(value) for arm, value in paired['DIRECT_BRIDGED_vs_ATOMIC_ORACLE'].items()},
        'by_root': by_root,
    }
    write_json(ROOT/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'by_root'},
                     ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'evaluate', 'bridge', 'report'))
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check()
    elif args.command == 'evaluate':
        evaluate(args.shard)
    elif args.command == 'bridge':
        bridge(args.shard)
    else:
        report()


if __name__ == '__main__':
    main()
