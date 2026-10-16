"""Evaluate the fixed-budget DIRECT_REPLAY adapter on the frozen 100-task TACO transfer set."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import symtable

from . import apps_direct_retention_sft as replay_sft
from . import apps_direct_weight_compare as direct
from . import apps_fresh_system_compare as fresh
from . import apps_taco_direct_k3 as frozen
from . import apps_taco_io_replay as interface
from . import apps_taco_transfer as transfer
from .dataset import file_hash, read_jsonl
from .evaluate import load_task, score_program, write_json
from .policy import LLMPolicy, load_adapter_metadata
from .schema import Contract


ROOT = Path('runs/refinement/apps_taco_direct_replay_k3')
ARM = 'DIRECT_REPLAY'
GPUS = ('0', '1', '2')
SHARDS = len(GPUS)
REPETITIONS = 3
NEW_MODEL_CALLS = 100 * REPETITIONS
ADAPTER = replay_sft.ROOT / 'training' / 'final_adapter'
ROOT_PARTITION = f'index modulo {SHARDS} shards'
EXPECTED_CANDIDATE_ERRORS = frozenset({'TIMEOUT', 'OUTPUT_LIMIT'})


def _sources():
    return {
        'frozen_request': frozen.ROOT / 'request.json',
        'frozen_result': frozen.ROOT / 'result.json',
        'transfer_request': transfer.ROOT / 'request.json',
        'transfer_result': transfer.ROOT / 'result.json',
        'roots': transfer.DATA / 'roots.jsonl',
        'tests': transfer.DATA / 'tests.jsonl',
        'selected': transfer.DATA / 'selected_records.jsonl',
        'model_manifest': Path('data/manifests/model.json'),
        'replay_sft_request': replay_sft.ROOT / 'request.json',
        'replay_sft_report': replay_sft.ROOT / 'training' / 'training_report.json',
        'replay_adapter': ADAPTER / 'adapter_model.safetensors',
        'replay_adapter_config': ADAPTER / 'adapter_config.json',
        'bridge_module': Path(interface.__file__),
        'evaluation_module': Path(__file__),
    }


def _framework():
    return json.loads((frozen.ROOT / 'request.json').read_text())['framework']


def _seed(index, repetition):
    return frozen._seed(index, repetition)


def _original_path(root_id, seed, repetition):
    return ROOT / 'evaluation' / 'original' / ARM / root_id / f'seed_{seed}' / 'case_result.json'


def _bridged_path(root_id, seed):
    return ROOT / 'evaluation' / 'bridged' / ARM / root_id / f'seed_{seed}' / 'case_result.json'


def _selected_indices(shard):
    return list(range(shard, 100, SHARDS))


def _case_is_recorded(case):
    """Accept resource-limited candidates as scored outcomes, not infra failures."""
    if case.get('status') == 'COMPLETE':
        return True
    errors = [score for score in case.get('final_scores', {}).values()
              if score.get('status') == 'ERROR']
    return bool(errors) and all(score.get('reason') in EXPECTED_CANDIDATE_ERRORS
                                for score in errors)


def _adapter_identity(framework):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        framework['model']['local_path'], local_files_only=True,
    )
    _, identity = load_adapter_metadata(ADAPTER, framework['model'], tokenizer)
    return identity


def prepare():
    if ROOT.exists():
        raise FileExistsError(f'DIRECT_REPLAY TACO k3 output already exists: {ROOT}')
    frozen.check()
    replay_sft.check(trained=True)
    framework = _framework()
    direct_request = json.loads((replay_sft.ROOT / 'request.json').read_text())
    if direct_request['framework'] != framework:
        raise ValueError('DIRECT_REPLAY adapter was not trained from the frozen TACO framework')
    identity = _adapter_identity(framework)
    expected_adapter = file_hash(ADAPTER / 'adapter_model.safetensors')
    if identity.get('adapter_weights_sha256') != expected_adapter:
        raise ValueError('DIRECT_REPLAY adapter identity does not match its saved weights')
    ROOT.mkdir(parents=True)
    write_json(ROOT / 'request.json', {
        'status': 'PREPARED', 'arm': ARM, 'roots': 100,
        'candidates_per_root': REPETITIONS, 'seed_base': transfer.EVAL_SEED,
        'root_partition': ROOT_PARTITION, 'gpu_devices': list(GPUS),
        'cpu_shards': SHARDS, 'framework': framework,
        'adapter_path': str(ADAPTER), 'adapter_sha256': expected_adapter,
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'direct_prompt_sha256': hashlib.sha256(
            (fresh.SYSTEM_PROMPT + '\n' + fresh.USER_PREFIX).encode()).hexdigest(),
        'decoding': framework['inference']['decoding'],
        'selection': 'first visible fixture pass, then earliest candidate',
        'metric': 'visible and nine private TACO cases both pass',
        'comparison': 'same frozen 100 roots, three seeds, bridge and selector as direct k3',
        'new_model_calls': NEW_MODEL_CALLS, 'training_steps': 0,
    })
    write_json(ROOT / 'gpu_resources.json', {
        'devices': list(GPUS), 'min_free_mib': {'collect': 16000},
    })
    return check()


def check():
    frozen.check()
    replay_sft.check(trained=True)
    request = json.loads((ROOT / 'request.json').read_text())
    framework = _framework()
    expected_sources = {name: file_hash(path) for name, path in _sources().items()}
    if (request['status'] != 'PREPARED' or request['arm'] != ARM
            or request['roots'] != 100 or request['candidates_per_root'] != REPETITIONS
            or request['seed_base'] != transfer.EVAL_SEED
            or request['root_partition'] != ROOT_PARTITION
            or request['gpu_devices'] != list(GPUS) or request['cpu_shards'] != SHARDS
            or request['framework'] != framework
            or request['adapter_path'] != str(ADAPTER)
            or request['adapter_sha256'] != file_hash(ADAPTER / 'adapter_model.safetensors')
            or request['source_sha256'] != expected_sources
            or request['direct_prompt_sha256'] != hashlib.sha256(
                (fresh.SYSTEM_PROMPT + '\n' + fresh.USER_PREFIX).encode()).hexdigest()
            or request['decoding'] != framework['inference']['decoding']
            or request['selection'] != 'first visible fixture pass, then earliest candidate'
            or request['metric'] != 'visible and nine private TACO cases both pass'
            or request['comparison'] !=
               'same frozen 100 roots, three seeds, bridge and selector as direct k3'
            or request['new_model_calls'] != NEW_MODEL_CALLS
            or request['training_steps'] != 0
            or json.loads((ROOT / 'gpu_resources.json').read_text()) !=
               {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('DIRECT_REPLAY TACO k3 design or startup configuration changed')
    if (framework['model']['load_in_4bit'] is not True
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072
            or framework['inference']['decoding'] !=
               {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}):
        raise ValueError('Expected the frozen direct inference configuration')
    identity = _adapter_identity(framework)
    # ``load_adapter_metadata`` only verifies the adapter files.  Quantization
    # is checked after the full model is loaded in ``evaluate`` below.
    if identity.get('adapter_weights_sha256') != request['adapter_sha256']:
        raise ValueError('Loaded DIRECT_REPLAY adapter differs from the prepared request')
    rows = read_jsonl(transfer.DATA / 'roots.jsonl')
    if len(rows) != 100:
        raise ValueError('TACO transfer roots changed')
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        framework['model']['local_path'], local_files_only=True,
    )
    prompt_lengths = []
    for row in rows:
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        prompt_lengths.append(len(fresh._prompt(tokenizer, spec, framework)[2]))
    ready = {
        'status': 'READY', 'arm': ARM, 'roots': len(rows),
        'candidates_per_root': REPETITIONS, 'new_rollouts': NEW_MODEL_CALLS,
        'rollouts_per_gpu': NEW_MODEL_CALLS // SHARDS, 'gpu_devices': list(GPUS),
        'cpu_shards': SHARDS, 'decoding': framework['inference']['decoding'],
        'context_tokens': framework['runtime']['context_tokens'],
        'output_cap': framework['runtime']['generation_tokens']['write_code'],
        'max_prompt_tokens': max(prompt_lengths), 'training_steps': 0,
    }
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def evaluate(shard):
    from tqdm import tqdm
    from .gpu_resource import acquire

    if not 0 <= shard < SHARDS:
        raise ValueError(f'Expected GPU shard 0, 1, or 2, got {shard}')
    check()
    done = ROOT / f'shard_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if (record['status'] == 'COMPLETE' and record['shard'] == shard
                and record['arm'] == ARM
                and record['request_sha256'] == file_hash(ROOT / 'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete DIRECT_REPLAY shard: {done}')
    request = json.loads((ROOT / 'request.json').read_text())
    rows = read_jsonl(transfer.DATA / 'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA / 'tests.jsonl')}
    selected = _selected_indices(shard)
    reservation = acquire(ROOT / 'gpu_resources.json', 'collect')
    try:
        llm = LLMPolicy.from_pretrained(request['framework'], adapter_path=ADAPTER)
        if (llm.identity.get('adapter_weights_sha256') != request['adapter_sha256']
                or llm.identity.get('quantization') != 'nf4_double'):
            raise ValueError('Loaded DIRECT_REPLAY model differs from prepared identity')
        with tqdm(total=len(selected) * REPETITIONS,
                  desc=f'TACO DIRECT_REPLAY GPU {GPUS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            for index in selected:
                row = rows[index]
                rid = row['root_id']
                for repetition in range(REPETITIONS):
                    seed = _seed(index, repetition)
                    path = _original_path(rid, seed, repetition)
                    case = direct._direct_case(
                        row, tests, request['framework'], llm, ARM, path.parent, seed,
                    )
                    if (not _case_is_recorded(case) or case['root_id'] != rid
                            or case['seed'] != seed or case['model_calls'] != 1
                            or case['model_identity'].get('adapter_weights_sha256')
                               != request['adapter_sha256']):
                        raise RuntimeError(f'Incomplete DIRECT_REPLAY candidate: {rid}/{repetition}')
                    bar.update(1)
        write_json(done, {
            'status': 'COMPLETE', 'shard': shard, 'arm': ARM,
            'request_sha256': file_hash(ROOT / 'request.json'),
            'indices': selected, 'roots': [rows[i]['root_id'] for i in selected],
            'repetitions': list(range(REPETITIONS)),
        })
    finally:
        reservation.close()


def _bridge_source(source):
    """Use the frozen stdio wrapper, resolving solve references by Python scope."""
    tree = ast.parse(source)
    solves = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == 'solve']
    if len(solves) != 1:
        raise ValueError('Expected one nonrecursive top-level solve function')
    root = solves[0]
    table = symtable.symtable(source, '<taco_stdio_bridge>', 'exec')
    scopes = [child for child in table.get_children()
              if child.get_name() == 'solve' and child.get_lineno() == root.lineno]
    # A solve defined inside _run is a local binding.  Only references to the
    # global solve would change meaning when the public function is renamed.
    while scopes:
        scope = scopes.pop()
        if 'solve' in scope.get_identifiers():
            symbol = scope.lookup('solve')
            if symbol.is_global() and symbol.is_referenced():
                raise ValueError('Expected one nonrecursive top-level solve function')
        scopes.extend(scope.get_children())
    root.name = '_original_solve'
    adapted = ast.unparse(tree) + '\n' + interface.BRIDGE
    compile(adapted, '<taco_stdio_bridge>', 'exec')
    return adapted


def _bridge_case(original_path, target_path, row, tests, framework, request):
    original = json.loads(original_path.read_text())
    if target_path.exists():
        saved = json.loads(target_path.read_text())
        if (_case_is_recorded(saved) and saved['arm'] == ARM
                and saved['root_id'] == row['root_id']
                and saved['request_sha256'] == file_hash(ROOT / 'request.json')
                and saved['original_case_sha256'] == file_hash(original_path)):
            return saved
        raise ValueError(f'Saved bridge candidate differs: {target_path}')
    if target_path.parent.exists():
        raise ValueError(f'Unfinished bridge candidate exists: {target_path.parent}')
    target_path.parent.mkdir(parents=True)
    snapshot = original['final_program']
    if snapshot is None:
        adapted, scores, cost = None, original['final_scores'], original['scoring_cost']
        applied = False
    else:
        _, _, contract_spec, _, verifier = load_task(row, tests, framework)
        contract = Contract(**contract_spec.model_dump(), contract_id='C_N0_1', version=1)
        source = (Path(snapshot['path']) / snapshot['artifact']['module_path']).read_text()
        store = interface.ArtifactStore(target_path.parent / 'final_program')
        artifact = store.save_initial('N0', contract, _bridge_source(source))
        adapted = {
            'path': str(target_path.parent / 'final_program'), 'root_id': row['root_id'],
            'artifact': artifact.model_dump(mode='json'),
            'contract': contract.model_dump(mode='json'),
        }
        scores, cost = score_program(adapted, verifier, row['root_id'])
        applied = True
    status = ('ERROR' if any(value['status'] == 'ERROR' for value in scores.values())
              else 'COMPLETE')
    if status == 'ERROR' and not _case_is_recorded({
            'status': status, 'final_scores': scores}):
        raise RuntimeError(f'Bridge scoring error: {row["root_id"]}')
    saved = {
        'status': status, 'root_id': row['root_id'], 'arm': ARM,
        'seed': original['seed'], 'request_sha256': file_hash(ROOT / 'request.json'),
        'original_case_path': str(original_path),
        'original_case_sha256': file_hash(original_path), 'bridge_applied': applied,
        'final_program': adapted, 'final_scores': scores, 'scoring_cost': cost,
        'run_dir': str(target_path.parent),
    }
    write_json(target_path, saved)
    return saved


def bridge(shard):
    from tqdm import tqdm

    if not 0 <= shard < SHARDS:
        raise ValueError(f'Expected CPU bridge shard 0, 1, or 2, got {shard}')
    check()
    done = ROOT / f'bridge_{shard}.json'
    if done.exists():
        record = json.loads(done.read_text())
        if (record['status'] == 'COMPLETE' and record['shard'] == shard
                and record['arm'] == ARM
                and record['request_sha256'] == file_hash(ROOT / 'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete DIRECT_REPLAY bridge shard: {done}')
    request = json.loads((ROOT / 'request.json').read_text())
    rows = read_jsonl(transfer.DATA / 'roots.jsonl')
    tests = {row['root_id']: row for row in read_jsonl(transfer.DATA / 'tests.jsonl')}
    selected = _selected_indices(shard)
    with tqdm(total=len(selected) * REPETITIONS,
              desc=f'TACO DIRECT_REPLAY bridge shard {shard}', unit='候选',
              position=shard, dynamic_ncols=True) as bar:
        for index in selected:
            row = rows[index]
            rid = row['root_id']
            for repetition in range(REPETITIONS):
                seed = _seed(index, repetition)
                _bridge_case(
                    _original_path(rid, seed, repetition), _bridged_path(rid, seed),
                    row, tests, request['framework'], request,
                )
                bar.update(1)
    write_json(done, {
        'status': 'COMPLETE', 'shard': shard, 'arm': ARM,
        'request_sha256': file_hash(ROOT / 'request.json'),
        'indices': selected, 'roots': [rows[i]['root_id'] for i in selected],
        'repetitions': list(range(REPETITIONS)),
    })


def _signal(case):
    visible = case['final_scores']['visible']['status'] == 'PASS'
    private = case['final_scores']['private']['status'] == 'PASS'
    reasons = [score.get('reason') for score in case['final_scores'].values()]
    return {
        'strict': int(visible and private), 'visible': int(visible),
        'private': int(private), 'program': int(case['final_program'] is not None),
        'candidate_exception': int(case['final_scores']['visible']['reason']
                                   == 'CANDIDATE_EXCEPTION'),
        'timeout': int('TIMEOUT' in reasons),
        'output_limit': int('OUTPUT_LIMIT' in reasons),
    }


def _selected(signals, k):
    return next((index for index, signal in enumerate(signals[:k])
                 if signal['visible']), 0)


def _add_counts(counter, signal):
    counter['roots'] += 1
    for key in ('strict', 'visible', 'private', 'program', 'candidate_exception',
                'timeout', 'output_limit'):
        counter[key] += signal[key]


def report():
    from tqdm import tqdm

    check()
    request = json.loads((ROOT / 'request.json').read_text())
    rows = read_jsonl(transfer.DATA / 'roots.jsonl')
    records = {f'taco_test_{r["id"]:04d}': r
               for r in read_jsonl(transfer.DATA / 'selected_records.jsonl')}
    for shard in range(SHARDS):
        for prefix in ('shard', 'bridge'):
            saved = json.loads((ROOT / f'{prefix}_{shard}.json').read_text())
            expected = [rows[i]['root_id'] for i in _selected_indices(shard)]
        if (saved['status'] != 'COMPLETE' or saved['shard'] != shard
                    or saved['arm'] != ARM
                    or saved['request_sha256'] != file_hash(ROOT / 'request.json')
                    or saved['roots'] != expected
                    or saved['repetitions'] != list(range(REPETITIONS))):
                raise ValueError(f'Incomplete DIRECT_REPLAY {prefix} shard: {shard}')
    modes = {mode: {str(k): Counter() for k in (1, 3)}
             for mode in ('DIRECT_ORIGINAL', 'DIRECT_BRIDGED')}
    by_difficulty = {
        difficulty: {
            mode: {str(k): Counter() for k in (1, 3)}
            for mode in modes
        }
        for difficulty in transfer.QUOTAS
    }
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='TACO DIRECT_REPLAY k3 汇总',
                                      unit='题', dynamic_ncols=True)):
        rid = row['root_id']
        difficulty = records[rid]['difficulty']
        originals = [json.loads(_original_path(rid, _seed(index, rep), rep).read_text())
                     for rep in range(REPETITIONS)]
        bridged = [json.loads(_bridged_path(rid, _seed(index, rep)).read_text())
                   for rep in range(REPETITIONS)]
        for repetition, case in enumerate(originals):
            if (not _case_is_recorded(case) or case['arm'] != ARM
                    or case['root_id'] != rid or case['seed'] != _seed(index, repetition)
                    or case['model_identity'].get('adapter_weights_sha256')
                       != request['adapter_sha256']):
                raise ValueError(f'DIRECT_REPLAY original identity changed: {rid}/{repetition}')
        for repetition, case in enumerate(bridged):
            if (not _case_is_recorded(case) or case['arm'] != ARM
                    or case['root_id'] != rid or case['seed'] != _seed(index, repetition)
                    or case['request_sha256'] != file_hash(ROOT / 'request.json')):
                raise ValueError(f'DIRECT_REPLAY bridge identity changed: {rid}/{repetition}')
        original_signals = [_signal(case) for case in originals]
        bridged_signals = [_signal(case) for case in bridged]
        root = {
            'difficulty': difficulty,
            'direct_original': {
                'candidates': original_signals,
                'selected': {str(k): _selected(original_signals, k) for k in (1, 3)},
                'oracle_strict': int(any(signal['strict'] for signal in original_signals)),
            },
            'direct_bridged': {
                'candidates': bridged_signals,
                'selected': {str(k): _selected(bridged_signals, k) for k in (1, 3)},
                'oracle_strict': int(any(signal['strict'] for signal in bridged_signals)),
            },
        }
        by_root[rid] = root
        for mode, signals in (('DIRECT_ORIGINAL', original_signals),
                              ('DIRECT_BRIDGED', bridged_signals)):
            for k in (1, 3):
                selected = signals[_selected(signals, k)]
                _add_counts(modes[mode][str(k)], selected)
                _add_counts(by_difficulty[difficulty][mode][str(k)], selected)
    frozen_result = json.loads((frozen.ROOT / 'result.json').read_text())
    result = {
        'status': 'COMPLETE', 'request_sha256': file_hash(ROOT / 'request.json'),
        'roots': len(rows), 'candidates_per_root': REPETITIONS,
        'summary': {mode: {str(k): dict(value) for k, value in ks.items()}
                    for mode, ks in modes.items()},
        'by_difficulty': {
            difficulty: {
                mode: {str(k): dict(value) for k, value in ks.items()}
                for mode, ks in values.items()
            }
            for difficulty, values in by_difficulty.items()
        },
        'oracle': {
            mode: sum(root[mode.lower()]['oracle_strict'] for root in by_root.values())
            for mode in ('DIRECT_ORIGINAL', 'DIRECT_BRIDGED')
        },
        'frozen_comparison_reference': {
            'result_path': str(frozen.ROOT / 'result.json'),
            'result_sha256': file_hash(frozen.ROOT / 'result.json'),
            'summary': frozen_result['summary'],
        },
        'by_root': by_root,
    }
    write_json(ROOT / 'result.json', result)
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
