"""Prepare and train a direct-code SFT adapter for free execution."""

import argparse
import ast
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from . import apps_diverse_sft as diverse
from . import apps_fresh_system_compare as direct
from . import apps_taco_sft as taco
from .dataset import file_hash, read_jsonl
from .evaluate import write_json


DATA = Path('data/refinement/apps_direct_sft')
ROOT = Path('runs/refinement/apps_direct_sft')
APPS_TRAINING = diverse.DATA / 'training.jsonl'
TACO_TRAINING = taco.DATA / 'training.jsonl'
FRAMEWORK_CONFIG = Path('configs/frozen_mechanism.yaml')
SYSTEM_PROMPT = direct.SYSTEM_PROMPT
USER_PREFIX = direct.USER_PREFIX
SEED = 620020713
MAX_LENGTH = 8192
ACCUMULATION = 16
LEARNING_RATE = 2e-5
GPUS = ('0', '2')
LORA = {
    'r': 16, 'alpha': 32, 'dropout': 0.05,
    'target_modules': 'all-linear', 'bias': 'none', 'task_type': 'CAUSAL_LM',
}


def _sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _target_rows():
    """Use only verified TRAIN targets from the frozen APPS and TACO exports."""
    apps = [row for row in read_jsonl(APPS_TRAINING)
            if row['kind'] == 'target' and row['split'] == 'TRAIN']
    taco_rows = [row for row in read_jsonl(TACO_TRAINING)
                 if row['kind'] == 'target' and row['split'] == 'TRAIN'
                 and int(row['root_id'].rsplit('_', 1)[-1]) >= 100000]
    if len(apps) != 1444 or len(taco_rows) != 2048:
        raise ValueError(f'Expected 1444 APPS and 2048 TACO verified targets, got '
                         f'{len(apps)} and {len(taco_rows)}')
    rows = []
    seen = set()
    for source, values in (('APPS', apps), ('TACO', taco_rows)):
        for row in values:
            if row['root_id'] in seen:
                raise ValueError(f'Duplicate direct SFT root: {row["root_id"]}')
            seen.add(row['root_id'])
            action = row.get('target_action', {})
            args = action.get('args', {})
            source_code = args.get('source')
            if (action.get('action') != 'write_code' or args.get('contract_version') != 1
                    or args.get('declared_dependencies') != []
                    or not isinstance(source_code, str) or not source_code.strip()):
                raise ValueError(f'Invalid verified direct target: {row["root_id"]}')
            try:
                tree = ast.parse(source_code)
                solves = [node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'solve']
                if len(solves) != 1:
                    raise ValueError('Expected exactly one top-level solve')
                compile(source_code, '<direct_sft_target>', 'exec')
            except (SyntaxError, ValueError) as error:
                raise ValueError(f'Invalid direct target source: {row["root_id"]}: {error}') from error
            spec = row['observation'].get('root_spec')
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f'Missing problem statement: {row["root_id"]}')
            rows.append({
                'sample_id': f'direct_{source.lower()}_{row["root_id"]}',
                'root_id': row['root_id'], 'source': source,
                'difficulty': row['observation'].get('difficulty', 'UNKNOWN'),
                'split': 'TRAIN', 'spec': spec, 'target': source_code,
                'target_source_sha256': _sha256_text(source_code),
                'teacher_source_sha256': row.get('source_sha256'),
            })
    rows.sort(key=lambda row: hashlib.sha256(
        f'{SEED}:{row["source"]}:{row["root_id"]}'.encode()).hexdigest())
    return rows


def _messages(spec):
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PREFIX + spec},
    ]


def _tokenized_rows(rows, tokenizer):
    encoded = []
    prompt_hash = _sha256_text(SYSTEM_PROMPT + '\n' + USER_PREFIX)
    for row in rows:
        messages = _messages(row['spec'])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prefix = list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        ))
        target = tokenizer.encode(row['target'], add_special_tokens=False)
        if tokenizer.eos_token_id is None or not prefix or not target:
            raise ValueError(f'Empty direct SFT prompt or target: {row["root_id"]}')
        target = target + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > MAX_LENGTH:
            raise ValueError(f'Direct SFT example exceeds context: {row["root_id"]}')
        encoded.append({
            **row, 'messages': messages,
            'prompt_sha256': _sha256_text(prompt),
            'prompt_template_sha256': prompt_hash,
            'prompt_tokens': len(prefix), 'target_tokens': len(target),
            'total_tokens': len(prefix) + len(target),
        })
    return encoded


def _sources():
    return {
        'apps_training': APPS_TRAINING,
        'taco_training': TACO_TRAINING,
        'apps_request': diverse.ROOT / 'request.json',
        'taco_request': taco.ROOT / 'request.json',
        'framework_config': FRAMEWORK_CONFIG,
        'model_manifest': Path('data/manifests/model.json'),
        'prompt_module': Path(direct.__file__),
        'training_module': Path(__file__),
    }


def _train_config(request):
    return {
        'stage': 'DIRECT_CODE_SFT',
        'framework_config': str(FRAMEWORK_CONFIG),
        'model': {
            'base': request['model']['name_or_path'],
            'revision': request['model']['revision'],
            'local_path': request['model']['local_path'],
            'enable_thinking': False, 'load_in_4bit': True,
            'quant_type': 'nf4', 'double_quant': True,
            'compute_dtype': 'bfloat16', 'device_map': 'single_visible_gpu',
        },
        'lora': LORA,
        'data': {'path': str(DATA / 'training.jsonl'), 'allowed_splits': ['TRAIN']},
        'training': {
            'max_length': MAX_LENGTH, 'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': ACCUMULATION,
            'learning_rate': LEARNING_RATE, 'weight_decay': 0.0,
            'warmup_ratio': 0.03, 'lr_scheduler_type': 'cosine',
            'max_grad_norm': 1.0, 'gradient_checkpointing': True,
            'packing': False, 'loss_normalization': 'per_example_then_batch',
            'seed': SEED, 'report_to': 'none', 'epoch_limit': 1,
            'output_dir': str(ROOT / 'training'), 'logging_steps': 1,
            'save_strategy': 'final_only', 'evaluation_strategy': 'no_effect_gate',
        },
    }


def prepare():
    if DATA.exists() or ROOT.exists():
        raise FileExistsError(f'Direct SFT output already exists: {DATA} or {ROOT}')
    diverse.check(trained=True)
    taco.check(trained=True)
    taco_request = json.loads((taco.ROOT / 'request.json').read_text())
    framework = taco_request['framework']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        framework['model']['local_path'], local_files_only=True,
    )
    rows = _tokenized_rows(_target_rows(), tokenizer)
    DATA.mkdir(parents=True)
    ROOT.mkdir(parents=True)
    (DATA / 'training.jsonl').write_text(''.join(
        # ``read_jsonl`` uses ``splitlines()``; escaping non-ASCII keeps source
        # strings containing Unicode line separators on one physical JSONL line.
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n'
        for row in rows
    ))
    counts = Counter(row['source'] for row in rows)
    steps = math.ceil(len(rows) / ACCUMULATION)
    request = {
        'status': 'PREPARED', 'framework': framework,
        'initial_adapter': None, 'base_model': framework['model'],
        'data_path': str(DATA / 'training.jsonl'),
        'source_sha256': {name: file_hash(path) for name, path in _sources().items()},
        'training_sha256': file_hash(DATA / 'training.jsonl'),
        'direct_prompt_sha256': _sha256_text(SYSTEM_PROMPT + '\n' + USER_PREFIX),
        'source_counts': dict(counts), 'examples': len(rows),
        'steps': steps, 'target_exposures': len(rows),
        'gradient_accumulation_steps': ACCUMULATION,
        'learning_rate': LEARNING_RATE, 'seed': SEED,
        'max_length': MAX_LENGTH, 'lora': LORA,
        'train_gpus': list(GPUS), 'training_device_policy': 'one visible GPU per QLoRA process',
        'objective': 'Direct-code SFT: exact free-execution prompt followed by a verified solve(text) program; one exposure for each distinct APPS/TACO target; no planner actions or retention protocol examples.',
        'evaluation_plan': 'Use the frozen 100-task TACO direct k3 benchmark with the same seeds, bridge, selector and private scorer; compare BASE, current DIVERSE/TACO, and this adapter.',
    }
    write_json(ROOT / 'request.json', request)
    write_json(ROOT / 'gpu_resources_train.json', {
        'devices': list(GPUS), 'min_free_mib': {'train': 22000},
    })
    return check()


def check(*, trained=False):
    request = json.loads((ROOT / 'request.json').read_text())
    expected_sources = {name: file_hash(path) for name, path in _sources().items()}
    if (request['status'] != 'PREPARED'
            or request['source_sha256'] != expected_sources
            or request['training_sha256'] != file_hash(DATA / 'training.jsonl')
            or request['direct_prompt_sha256'] != _sha256_text(SYSTEM_PROMPT + '\n' + USER_PREFIX)
            or request['examples'] != 3492
            or request['source_counts'] != {'APPS': 1444, 'TACO': 2048}
            or request['steps'] != math.ceil(3492 / ACCUMULATION)
            or request['target_exposures'] != 3492
            or request['gradient_accumulation_steps'] != ACCUMULATION
            or request['learning_rate'] != LEARNING_RATE
            or request['seed'] != SEED or request['max_length'] != MAX_LENGTH
            or request['lora'] != LORA or request['train_gpus'] != list(GPUS)
            or request['initial_adapter'] is not None
            or json.loads((ROOT / 'gpu_resources_train.json').read_text()) != {
                'devices': list(GPUS), 'min_free_mib': {'train': 22000}}):
        raise ValueError('Direct SFT design or launch configuration changed')
    rows = read_jsonl(DATA / 'training.jsonl')
    if len(rows) != 3492 or len({row['sample_id'] for row in rows}) != 3492:
        raise ValueError('Direct SFT examples are missing or duplicated')
    counts = Counter(row['source'] for row in rows)
    if counts != Counter({'APPS': 1444, 'TACO': 2048}):
        raise ValueError(f'Unexpected direct SFT source counts: {counts}')
    for row in rows:
        if (row['split'] != 'TRAIN' or row['messages'][0]['content'] != SYSTEM_PROMPT
                or row['messages'][1]['content'] != USER_PREFIX + row['spec']
                or row['prompt_template_sha256'] != request['direct_prompt_sha256']
                or row['prompt_tokens'] + row['target_tokens'] > MAX_LENGTH
                or not row['target'].startswith('def solve(')
                or _sha256_text(row['target']) != row['target_source_sha256']):
            raise ValueError(f'Invalid direct SFT example: {row.get("sample_id")}')
    if trained:
        report = json.loads((ROOT / 'training' / 'training_report.json').read_text())
        adapter = ROOT / 'training' / 'final_adapter'
        if (report['status'] != 'COMPLETE' or report['optimizer_steps'] != request['steps']
                or report['adapter_sha256'] != file_hash(adapter / 'adapter_model.safetensors')):
            raise ValueError('Direct SFT training is incomplete or changed')
    ready = {
        'status': 'READY', 'examples': len(rows), 'source_counts': dict(counts),
        'steps': request['steps'], 'target_exposures': request['target_exposures'],
        'train_gpus': list(GPUS), 'single_visible_gpu_per_process': True,
        'max_prompt_plus_target_tokens': max(row['total_tokens'] for row in rows),
        'max_prompt_tokens': max(row['prompt_tokens'] for row in rows),
        'max_target_tokens': max(row['target_tokens'] for row in rows),
        'trained': trained,
    }
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    return ready


def train():
    from .gpu_resource import acquire

    check()
    reservation = acquire(ROOT / 'gpu_resources_train.json', 'train')
    try:
        _train_on_reserved_gpu()
    finally:
        reservation.close()


def _train_on_reserved_gpu():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from .tokenize_sft import ActionCollator
    from .train_sft import (ActionSFTTrainer, GradientCheck, changed_tensors,
                             load_identity_and_tokenizer, load_quantized_base,
                             make_training_arguments, prefix_forward,
                             tensor_hash, trainable_snapshots)

    request = json.loads((ROOT / 'request.json').read_text())
    config = _train_config(request['framework'])
    identity, tokenizer = load_identity_and_tokenizer(config)
    rows = read_jsonl(DATA / 'training.jsonl')
    encoded = []
    for row in rows:
        prefix = list(tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False,
        ))
        target = tokenizer.encode(row['target'], add_special_tokens=False)
        target = target + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > MAX_LENGTH:
            raise ValueError(f'Encoded direct SFT example overflow: {row["sample_id"]}')
        encoded.append({
            'input_ids': prefix + target,
            'attention_mask': [1] * (len(prefix) + len(target)),
            'labels': [-100] * len(prefix) + target,
        })
    out = ROOT / 'training'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'encoded_summary.json', {
        'examples': len(encoded), 'training_sha256': file_hash(DATA / 'training.jsonl'),
        'max_tokens': max(len(row['input_ids']) for row in encoded),
        'tokenizer_sha256': identity['tokenizer_sha256'],
    })
    set_seed(SEED)
    model = load_quantized_base(config)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    model = get_peft_model(model, LoraConfig(
        r=LORA['r'], lora_alpha=LORA['alpha'], lora_dropout=LORA['dropout'],
        target_modules=LORA['target_modules'], bias=LORA['bias'],
        task_type=LORA['task_type'], revision=request['framework']['model']['revision'],
    ))
    model.config.use_cache = False
    before, trainable = trainable_snapshots(model)
    args = make_training_arguments(config, request['steps'])
    args.disable_tqdm = False
    callback = GradientCheck()
    trainer = ActionSFTTrainer(
        model=model, args=args, train_dataset=encoded,
        data_collator=ActionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer, callbacks=[callback],
    )
    result = trainer.train()
    if trainer.state.global_step != request['steps'] or len(callback.steps) != request['steps']:
        raise RuntimeError('Direct SFT did not complete the requested optimizer steps')
    changes = changed_tensors(model, before)
    final = out / 'final_adapter'
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    trainer.save_state()
    prefix = list(tokenizer.apply_chat_template(
        rows[0]['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    original_logits = prefix_forward(model, prefix)
    peak_bytes = torch.cuda.max_memory_allocated()
    history = deepcopy(trainer.state.log_history)
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_tokenizer = tokenizer.__class__.from_pretrained(final, local_files_only=True)
    reloaded_prefix = list(reloaded_tokenizer.apply_chat_template(
        rows[0]['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    if reloaded_prefix != prefix:
        raise RuntimeError('Saved direct adapter tokenizer changed the inference prefix')
    base = prepare_model_for_kbit_training(
        load_quantized_base(config), use_gradient_checkpointing=False,
    )
    reloaded = PeftModel.from_pretrained(base, final, local_files_only=True, is_trainable=False)
    restored_logits = prefix_forward(reloaded, prefix)
    delta = float((restored_logits - original_logits).abs().max())
    if not torch.isfinite(restored_logits).all() or not torch.allclose(
            restored_logits, original_logits, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f'Saved direct adapter differs after reload: max_abs={delta}')
    report = {
        'status': 'COMPLETE', 'optimizer_steps': request['steps'],
        'target_exposures': len(rows), 'retention_exposures': 0,
        'trainable_parameters': trainable, 'parameter_changes': changes,
        'gradient_checks': callback.steps, 'metrics': result.metrics,
        'history': history, 'peak_cuda_allocated_bytes': peak_bytes,
        'reload': {'fresh_quantized_base': True, 'prefix_identical': True,
                   'max_abs_logit_difference': delta, 'atol': 1e-5, 'rtol': 1e-5},
        'adapter': str(final),
        'adapter_sha256': file_hash(final / 'adapter_model.safetensors'),
        'adapter_config_sha256': file_hash(final / 'adapter_config.json'),
        'training_sha256': file_hash(DATA / 'training.jsonl'),
        'implementation_sha256': {
            'module': file_hash(Path(__file__)),
            'trainer': file_hash(Path(__file__).with_name('train_sft.py')),
            'tokenizer': file_hash(Path(__file__).with_name('tokenize_sft.py')),
        },
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')},
    }
    write_json(out / 'training_report.json', report)
    print(json.dumps({'status': 'COMPLETE', 'steps': request['steps'],
                      'adapter': str(final), 'reload_max_abs_difference': delta}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'train'))
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        check(trained=args.trained)
    else:
        train()


if __name__ == '__main__':
    main()
