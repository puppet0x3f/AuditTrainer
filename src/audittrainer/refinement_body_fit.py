"""One bounded, unconstrained body-SFT learnability diagnostic."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in {'train', 'probe'}:
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_body_fit.yaml'
    _settings = yaml.safe_load(Path(_path).read_text())
    _reservation = acquire(Path(_settings['output_dir'])/'gpu_resources.json',
                           'train' if sys.argv[1] == 'train' else 'collect')

import torch
import torch.nn.functional as F
from transformers import set_seed

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_curriculum_rl import conditional_probe, plan_key
from .refinement_generalization import guarded_json
from .refinement_parent_validation import presentation_identity
from .refinement_policy import active_adapter, adapter_fingerprint, load_policies, mode_log_probs, MODES
from .refinement_pr1 import load_config, causal_logits, enable_planner
from .train_refinement import encode_demo
from .train_sft import changed_tensors, prefix_forward, trainable_snapshots


def prepare(study):
    scored = json.loads(Path(study['scoring_result']).read_text())
    if scored['status'] != 'COMPLETE' or scored['errors']:
        raise ValueError('Corrected calibration must finish before body fitting')
    quality = {name: block['new_successes']/block['runs'] for name, block in scored['groups'].items()
               if name.startswith('fixed/')}
    if quality['fixed/parent_aggregate'] <= max(v for k, v in quality.items() if k != 'fixed/parent_aggregate'):
        raise ValueError('The original teaching target lost its calibrated advantage')
    config, framework = load_config(study['source_config'])
    config.update(root_manifest=scored['config']['root_manifest'], test_manifest=scored['config']['test_manifest'])
    demo = json.loads(Path(study['target']).read_text())
    if demo['split'] != 'TRAIN' or demo['target_action']['args']['mode'] != 'SPLIT':
        raise ValueError('Expected one TRAIN SPLIT target')
    output = Path(study['output_dir']); output.mkdir(parents=True, exist_ok=True)
    guarded_json(output/'request.json', {'study': study, 'config': config, 'framework': framework,
        'target': demo, 'corrected_calibration': quality, 'scoring_sha256': file_hash(study['scoring_result']),
        'presentation': presentation_identity(),
        'objective': 'mean body-token CE only; no gate target, KL, retention, or RL',
        'adapters': {k: file_hash(Path(config[k])/'adapter_model.safetensors') for k in ('initial_adapter', 'executor_adapter')}})
    guarded_json(output/'gpu_resources.json', {'devices': ['0', '1', '2'], 'min_free_mib': {'collect': 16000, 'train': 22000}})
    return config, framework, demo


def train(study):
    config, framework, demo = prepare(study)
    output = Path(study['output_dir'])/'PURE_BODY'; output.mkdir(exist_ok=False)
    set_seed(study['seed'])
    planner, _ = load_policies(framework, config['initial_adapter'], config['executor_adapter'])
    model, tok = planner.model, planner.tokenizer
    frozen = adapter_fingerprint(model, 'executor')
    example = encode_demo(demo, tok, framework)
    saved_gate = json.loads((Path(config['output_dir'])/'groups'/demo['root_id']/'gate.json').read_text())
    if example['prefix'] != saved_gate['prefix_ids']:
        raise ValueError('Teaching and sampling prefixes differ')
    write_json(output/'encoded_target.json', example)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    model.peft_config['default'].lora_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False
    enable_planner(model); before, trainable = trainable_snapshots(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=study['learning_rate'], weight_decay=0.)
    labels = torch.tensor(example['body_target'], device=model.device)
    history = []; start = time.monotonic()
    for step in range(study['optimizer_steps']):
        optimizer.zero_grad(set_to_none=True); enable_planner(model)
        with planner.llm._autocast():
            logits = causal_logits(model, example['body_prefix'], example['body_target'])
            loss = F.cross_entropy(logits.float(), labels, reduction='mean')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, study['max_grad_norm'])
        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError('Invalid body learning gradient')
        optimizer.step()
        record = {'step': step+1, 'body_ce_before_update': float(loss.detach()), 'gradient_norm': float(norm)}
        history.append(record); print(json.dumps(record), flush=True)
        del logits, loss
    final = output/'final_adapter'
    model.save_pretrained(final, selected_adapters=['default']); tok.save_pretrained(final)
    model.eval()
    with torch.no_grad(), planner.llm._autocast():
        final_ce = float(F.cross_entropy(causal_logits(model, example['body_prefix'], example['body_target']).float(), labels))
    changes = changed_tensors(model, before)
    after = adapter_fingerprint(model, 'executor')
    old = prefix_forward(model, example['prefix'])
    model.load_adapter(str(final), adapter_name='reload_check', is_trainable=False, local_files_only=True)
    with active_adapter(model, 'reload_check'): restored = prefix_forward(model, example['prefix'])
    delta = float((old-restored).abs().max())
    if frozen != after or delta > 1e-5:
        raise ValueError('Executor freeze or adapter reload failed')
    report = {'status': 'COMPLETE', 'scope': 'SINGLE_TRAIN_STATE_PURE_BODY_DIAGNOSTIC',
        'optimizer_steps': len(history), 'previous_optimizer_steps': 0, 'rl_updates': 0,
        'learning_rate': study['learning_rate'], 'objective_scale': 1.,
        'gate_weight': 0., 'body_ce_weight': 1., 'kl_weight': 0., 'retention_weight': 0.,
        'history': history, 'final_body_ce': final_ce, 'body_target_tokens': len(example['body_target']),
        'trainable_parameters': trainable, 'parameter_changes': changes,
        'executor_before': frozen, 'executor_after': after, 'reload_max_abs_difference': delta,
        'adapter': str(final), 'adapter_sha256': file_hash(final/'adapter_model.safetensors'),
        'adapter_config_sha256': file_hash(final/'adapter_config.json'), 'train_seconds': time.monotonic()-start}
    write_json(output/'training_report.json', report)
    return report


def probe(study, method):
    config, framework, demo = prepare(study)
    output = Path(study['output_dir'])/'evaluation'/method; output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists(): return json.loads((output/'result.json').read_text())
    adapter = config['initial_adapter'] if method == 'B0' else str(Path(study['output_dir'])/'PURE_BODY/final_adapter')
    planner, executor = load_policies(framework, adapter, config['executor_adapter'])
    before = {name: adapter_fingerprint(planner.model, name) for name in ('default', 'executor')}
    config['evaluation'] = {'seed': study['probe_seed'], 'focus_root': demo['root_id'],
                            'conditional_split_samples': study['conditional_split_samples']}
    rows = load_root_manifest(config['root_manifest'])
    tests = {r['root_id']: r for r in read_jsonl(config['test_manifest'])}
    result = conditional_probe(config, framework, planner, executor, output, rows, tests)
    definition = json.loads(Path(rows[demo['root_id']]['fixture_path']).read_text())
    target = json.loads(plan_key(demo['target_action'], definition))
    mismatches = Counter()
    for record in result['records']:
        candidate = json.loads((output/'conditional_probe'/f"sample_{record['sample']:02d}"/'candidate.json').read_text())
        if not record['registered']:
            mismatches[record['host_code']] += 1; continue
        key = plan_key(candidate['parsed_action'], definition)
        if key is None: mismatches['unmatched_child_contract'] += 1; continue
        actual = json.loads(key)
        for field in target:
            if target[field] != actual[field]: mismatches['different_'+field] += 1
    gate = json.loads((output/'conditional_probe/gate.json').read_text())
    after = {name: adapter_fingerprint(planner.model, name) for name in before}
    if before != after: raise ValueError('Probe modified weights')
    result.update(method=method, p_split=gate['mode_probabilities']['SPLIT'], mismatches=dict(mismatches),
                  frozen_before=before, frozen_after=after, actual_model_cost=dict(planner.cost),
                  optimizer_steps=0, full_execution_continuations=0)
    write_json(output/'result.json', result)
    return result


def report(study):
    output = Path(study['output_dir'])
    training = json.loads((output/'PURE_BODY/training_report.json').read_text())
    evaluations = {m: json.loads((output/'evaluation'/m/'result.json').read_text()) for m in ('B0', 'PURE_BODY')}
    if training['optimizer_steps'] != study['optimizer_steps'] or any(e['samples'] != study['conditional_split_samples'] for e in evaluations.values()):
        raise ValueError('Incomplete finite budget')
    counts = {m: e['reviewed_variants'].get('parent_aggregate', 0) for m, e in evaluations.items()}
    result = {'status': 'COMPLETE', 'training': training, 'evaluations': evaluations, 'good_plan_counts': counts,
        'reference_target_reached': counts['PURE_BODY'] >= study['reference_target_count'],
        'good_plan_gain': counts['PURE_BODY']-counts['B0'],
        'interpretation': 'Free-generation capability on one known TRAIN state; no natural-return or generalization claim.'}
    write_json(output/'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'good_plan_counts': counts, 'reference_target_reached': result['reference_target_reached']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'probe', 'report'])
    parser.add_argument('--config', default='configs/refinement_body_fit.yaml')
    parser.add_argument('--model', choices=['B0', 'PURE_BODY'], default='B0')
    args = parser.parse_args(); study = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'prepare': prepare(study)
    elif args.command == 'train': train(study)
    elif args.command == 'probe': probe(study, args.model)
    else: report(study)
