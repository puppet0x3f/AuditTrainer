"""Evaluate the fixed-budget replay adapter on the frozen APPS retention set."""

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

from . import apps_direct_retention_sft as replay_sft
from . import apps_direct_weight_compare as direct
from . import apps_fresh_system_compare as frozen
from . import apps_passk_select
from . import apps_scale
from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .policy import LLMPolicy, load_adapter_metadata
from .sandbox import ExecutionLimits, Sandbox


ROOT = Path('runs/refinement/apps_direct_replay_retention')
ARM = 'REPLAY'
GPUS = ('0', '1', '2')
SHARDS = len(GPUS)
REPETITIONS = 3
ADAPTER = replay_sft.ROOT / 'training' / 'final_adapter'
EXPECTED_CANDIDATE_ERRORS = frozenset({'TIMEOUT', 'OUTPUT_LIMIT'})


def _sources():
    return {
        'frozen_request': frozen.ROOT / 'request.json',
        'frozen_result': frozen.ROOT / 'result.json',
        'fresh_roots': frozen.fresh.DATA / 'roots.jsonl',
        'fresh_tests': frozen.fresh.DATA / 'tests.jsonl',
        'scale_request': apps_scale.ROOT / 'request.json',
        'large_adapter': apps_scale.ROOT / 'LARGE/training/final_adapter/adapter_model.safetensors',
        'replay_sft_request': replay_sft.ROOT / 'request.json',
        'replay_sft_report': replay_sft.ROOT / 'training/training_report.json',
        'replay_adapter': ADAPTER / 'adapter_model.safetensors',
        'replay_adapter_config': ADAPTER / 'adapter_config.json',
        'evaluation_module': Path(__file__),
    }


def _framework():
    return frozen._framework()


def _rows():
    return frozen.fresh._rows()


def _tests():
    return frozen.fresh._tests()


def _seed(index, repetition):
    return frozen.fresh.SEED_BASE + 100 * index + repetition


def _selected_indices(shard):
    return list(range(shard, len(_rows()), SHARDS))


def _case_path(root_id, seed):
    return ROOT / 'evaluation' / ARM / root_id / f'seed_{seed}' / 'case_result.json'


def _case_is_recorded(case):
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
        raise FileExistsError(f'APPS replay retention output already exists: {ROOT}')
    frozen.check()
    replay_sft.check(trained=True)
    framework = _framework()
    identity = _adapter_identity(framework)
    adapter_sha = file_hash(ADAPTER / 'adapter_model.safetensors')
    if identity.get('adapter_weights_sha256') != adapter_sha:
        raise ValueError('Replay adapter identity does not match its saved weights')
    ROOT.mkdir(parents=True)
    write_json(ROOT / 'request.json', {
        'status': 'PREPARED', 'arm': ARM, 'roots': len(_rows()),
        'candidates_per_root': REPETITIONS, 'seed_base': frozen.fresh.SEED_BASE,
        'root_partition': f'index modulo {SHARDS} shards',
        'gpu_devices': list(GPUS), 'framework': framework,
        'adapter_path': str(ADAPTER), 'adapter_sha256': adapter_sha,
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'direct_prompt_sha256': hashlib.sha256(
            (frozen.SYSTEM_PROMPT + '\n' + frozen.USER_PREFIX).encode()).hexdigest(),
        'decoding': framework['inference']['decoding'],
        'selection': 'public example pass, returned status, program exists, then earliest candidate',
        'metric': 'exact held-out APPS I/O and matched public example when available',
        'training_steps': 0,
        'comparison': 'same 48 APPS roots, three seeds, prompt, selector and scorer as frozen LARGE direct arm',
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
    resources = json.loads((ROOT / 'gpu_resources.json').read_text())
    if (request['status'] != 'PREPARED' or request['arm'] != ARM
            or request['roots'] != len(_rows()) or request['candidates_per_root'] != REPETITIONS
            or request['seed_base'] != frozen.fresh.SEED_BASE
            or request['root_partition'] != f'index modulo {SHARDS} shards'
            or request['gpu_devices'] != list(GPUS) or request['framework'] != framework
            or request['adapter_path'] != str(ADAPTER)
            or request['adapter_sha256'] != file_hash(ADAPTER / 'adapter_model.safetensors')
            or request['source_sha256'] != expected_sources
            or request['direct_prompt_sha256'] != hashlib.sha256(
                (frozen.SYSTEM_PROMPT + '\n' + frozen.USER_PREFIX).encode()).hexdigest()
            or request['decoding'] != framework['inference']['decoding']
            or request['selection'] !=
               'public example pass, returned status, program exists, then earliest candidate'
            or request['metric'] !=
               'exact held-out APPS I/O and matched public example when available'
            or request['training_steps'] != 0
            or resources != {'devices': list(GPUS), 'min_free_mib': {'collect': 16000}}):
        raise ValueError('APPS replay retention design or launch configuration changed')
    if (framework['model']['load_in_4bit'] is not True
            or framework['runtime']['context_tokens'] != 8192
            or framework['runtime']['generation_tokens']['write_code'] != 3072
            or framework['inference']['decoding'] !=
               {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}):
        raise ValueError('Expected the frozen APPS direct inference configuration')
    identity = _adapter_identity(framework)
    if identity.get('adapter_weights_sha256') != request['adapter_sha256']:
        raise ValueError('Loaded replay adapter differs from the prepared request')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(framework['model']['local_path'], local_files_only=True)
    prompt_lengths = []
    for row in _rows():
        spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
        prompt_lengths.append(len(frozen._prompt(tokenizer, spec, framework)[2]))
    ready = {
        'status': 'READY', 'arm': ARM, 'roots': len(_rows()),
        'candidates_per_root': REPETITIONS, 'new_rollouts': len(_rows()) * REPETITIONS,
        'rollouts_per_gpu': len(_rows()) * REPETITIONS // SHARDS,
        'gpu_devices': list(GPUS), 'decoding': framework['inference']['decoding'],
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
        saved = json.loads(done.read_text())
        if (saved['status'] == 'COMPLETE' and saved['shard'] == shard
                and saved['request_sha256'] == file_hash(ROOT / 'request.json')):
            print(json.dumps({'shard': shard, 'status': 'ALREADY_COMPLETE'}), flush=True)
            return
        raise ValueError(f'Incomplete APPS replay retention shard: {done}')
    request = json.loads((ROOT / 'request.json').read_text())
    rows, tests = _rows(), _tests()
    selected = _selected_indices(shard)
    sandbox = Sandbox(ExecutionLimits(**request['framework']['execution']))
    reservation = acquire(ROOT / 'gpu_resources.json', 'collect')
    try:
        llm = LLMPolicy.from_pretrained(request['framework'], adapter_path=ADAPTER)
        if (llm.identity.get('adapter_weights_sha256') != request['adapter_sha256']
                or llm.identity.get('quantization') != 'nf4_double'):
            raise ValueError('Loaded replay model differs from prepared identity')
        completed = {}
        with tqdm(total=len(selected) * REPETITIONS,
                  desc=f'APPS replay retention GPU {GPUS[shard]}', unit='轨迹',
                  position=shard, dynamic_ncols=True) as bar:
            for index in selected:
                row = rows[index]
                rid = row['root_id']
                spec = json.loads(Path(row['fixture_path']).read_text())['task']['spec']
                sample = frozen.fresh._sample(spec)
                signals = []
                for repetition in range(REPETITIONS):
                    seed = _seed(index, repetition)
                    path = _case_path(rid, seed)
                    case = direct._direct_case(
                        row, tests, request['framework'], llm, ARM, path.parent, seed,
                    )
                    if (not _case_is_recorded(case) or case['root_id'] != rid
                            or case['seed'] != seed or case['model_calls'] != 1
                            or case['model_identity'].get('adapter_weights_sha256')
                               != request['adapter_sha256']):
                        raise RuntimeError(f'Incomplete APPS replay candidate: {rid}/{repetition}')
                    signals.append(apps_passk_select._score(path, rid, sample, sandbox))
                    bar.update(1)
                completed[rid] = {
                    'group': 'sample' if sample else 'no_sample',
                    'candidate_signals': signals,
                }
        write_json(done, {
            'status': 'COMPLETE', 'shard': shard,
            'request_sha256': file_hash(ROOT / 'request.json'),
            'indices': selected, 'roots': [rows[i]['root_id'] for i in selected],
            'repetitions': list(range(REPETITIONS)), 'by_root': completed,
        })
    finally:
        reservation.close()
        try:
            del llm
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except UnboundLocalError:
            pass


def _summary(candidates, group_by_root, *, include_candidate_counts=True):
    out = Counter({'roots': 0, 'strict': 0, 'private': 0, 'oracle_strict': 0,
                   'program_candidates': 0, 'public_sample_pass_candidates': 0,
                   'input_tokens': 0, 'output_tokens': 0, 'model_calls': 0})
    for rid, signals in candidates.items():
        group = group_by_root[rid]
        has_sample = group == 'sample'
        for k in (1, 3):
            chosen = apps_passk_select.choose(signals, k, has_sample)
            cases = [json.loads(Path(item['case_path']).read_text()) for item in signals[:k]]
            strict = [apps_scale._valid_private(case)
                      and (not has_sample or bool(signal['sample_pass']))
                      for case, signal in zip(cases, signals[:k])]
            if k == 1:
                out['roots'] += 1
                out['strict'] += strict[chosen]
                out['private'] += apps_scale._valid_private(cases[chosen])
            out['oracle_strict'] += any(strict)
            out['program_candidates'] += sum(bool(case['final_program']) for case in cases)
            out['public_sample_pass_candidates'] += sum(bool(s['sample_pass']) for s in signals[:k])
            out['input_tokens'] += sum(case['input_tokens'] for case in cases)
            out['output_tokens'] += sum(case['output_tokens'] for case in cases)
            out['model_calls'] += k
    # k=1 and k=3 are stored separately by the caller; this function is kept
    # local so selection always remains public-only.
    return dict(out)


def report():
    from tqdm import tqdm

    check()
    request = json.loads((ROOT / 'request.json').read_text())
    rows = _rows()
    frozen_result = json.loads((frozen.ROOT / 'result.json').read_text())
    signals = {}
    for shard in range(SHARDS):
        saved = json.loads((ROOT / f'shard_{shard}.json').read_text())
        expected = [rows[i]['root_id'] for i in _selected_indices(shard)]
        if (saved['status'] != 'COMPLETE' or saved['shard'] != shard
                or saved['request_sha256'] != file_hash(ROOT / 'request.json')
                or saved['roots'] != expected
                or set(signals) & set(saved['by_root'])):
            raise ValueError(f'Incomplete APPS replay retention shard: {shard}')
        signals.update(saved['by_root'])
    if set(signals) != {row['root_id'] for row in rows}:
        raise ValueError('APPS replay retention roots differ from frozen validation')
    groups = {rid: item['group'] for rid, item in signals.items()}
    summaries = {str(k): Counter() for k in (1, 3)}
    paired = {str(k): Counter() for k in (1, 3)}
    by_root = {}
    for index, row in enumerate(tqdm(rows, desc='APPS replay retention 汇总', unit='题', dynamic_ncols=True)):
        rid = row['root_id']
        item = signals[rid]
        if len(item['candidate_signals']) != REPETITIONS:
            raise ValueError(f'Missing replay candidates: {rid}')
        replay_signals = item['candidate_signals']
        old_signals = frozen_result['by_root'][rid]['signals']['DIRECT']
        if [s['case_path'] for s in old_signals] != [
                frozen_result['by_root'][rid]['signals']['DIRECT'][i]['case_path']
                for i in range(REPETITIONS)]:
            raise ValueError(f'Frozen LARGE candidate order changed: {rid}')
        choices = {}
        for k in (1, 3):
            replay_choice = apps_passk_select.choose(replay_signals, k, item['group'] == 'sample')
            old_choice = apps_passk_select.choose(old_signals, k, item['group'] == 'sample')
            replay_cases = [json.loads(Path(s['case_path']).read_text()) for s in replay_signals[:k]]
            old_cases = [json.loads(Path(s['case_path']).read_text()) for s in old_signals[:k]]
            replay_strict = [apps_scale._valid_private(case)
                             and (item['group'] == 'no_sample' or bool(signal['sample_pass']))
                             for case, signal in zip(replay_cases, replay_signals[:k])]
            old_strict = [apps_scale._valid_private(case)
                          and (item['group'] == 'no_sample' or bool(signal['sample_pass']))
                          for case, signal in zip(old_cases, old_signals[:k])]
            selected_replay = replay_strict[replay_choice]
            selected_old = old_strict[old_choice]
            count = summaries[str(k)]
            count['roots'] += 1
            count['strict'] += selected_replay
            count['private'] += apps_scale._valid_private(replay_cases[replay_choice])
            count['oracle_strict'] += any(replay_strict)
            count['program_candidates'] += sum(bool(case['final_program']) for case in replay_cases)
            count['public_sample_pass_candidates'] += sum(
                bool(signal['sample_pass']) for signal in replay_signals[:k])
            count['input_tokens'] += sum(case['input_tokens'] for case in replay_cases)
            count['output_tokens'] += sum(case['output_tokens'] for case in replay_cases)
            count['model_calls'] += k
            outcome = ('win' if selected_replay and not selected_old
                       else 'loss' if selected_old and not selected_replay else 'tie')
            paired[str(k)][outcome] += 1
            choices[str(k)] = {
                'replay_index': replay_choice, 'large_index': old_choice,
                'replay_strict': bool(selected_replay), 'large_strict': bool(selected_old),
            }
        by_root[rid] = {
            'group': item['group'], 'candidate_signals': replay_signals, 'choices': choices,
        }
    result = {
        'status': 'COMPLETE', 'request_sha256': file_hash(ROOT / 'request.json'),
        'roots': len(rows), 'candidates_per_root': REPETITIONS,
        'summary': {k: dict(v) for k, v in summaries.items()},
        'paired_vs_LARGE': {k: dict(v) for k, v in paired.items()},
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
