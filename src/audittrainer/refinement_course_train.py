"""Finite multi-state responsibility SFT from the original common adapter."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

import yaml

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'train':
    from .gpu_resource import acquire
    _path = sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv else 'configs/refinement_responsibility_course.yaml'
    _study = yaml.safe_load(Path(_path).read_text())
    _reservation = acquire(Path(_study['output_dir'])/'gpu_resources.json', 'train')

import torch
from transformers import set_seed

from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_policy import active_adapter, adapter_fingerprint, load_policies
from .refinement_pr1 import enable_planner
from .train_refinement import encode_demo, representation_loss
from .train_sft import changed_tensors, prefix_forward, trainable_snapshots


def schedule(course, replay, count, seed):
    """Root-balanced new states, rotating internal states, 20% old SFT replay."""
    if count % 5: raise ValueError('The finite exposure count must admit 20% replay')
    pools = defaultdict(list)
    for i, example in enumerate(course):
        if example['split'] != 'TRAIN': raise ValueError('Non-TRAIN course state')
        pools[example['root_id']].append(i)
    if not pools or not replay or any(e['split'] != 'TRAIN' for e in replay):
        raise ValueError('Both course and TRAIN replay are required')
    rng = random.Random(seed); roots = sorted(pools); rng.shuffle(roots)
    replay_ids = list(range(len(replay))); rng.shuffle(replay_ids)
    visits = Counter(); result = []; new_index = old_index = 0
    for index in range(count):
        if index % 5 == 4:
            result.append(('replay', replay_ids[old_index % len(replay_ids)])); old_index += 1
        else:
            root = roots[new_index % len(roots)]; states = pools[root]
            result.append(('course', states[visits[root] % len(states)]))
            visits[root] += 1; new_index += 1
    return result


def train(study):
    folder = Path(study['output_dir'])
    reference = json.loads((folder/'reference_result.json').read_text())
    diagnostic = json.loads(Path('runs/refinement/body_fit/result.json').read_text())
    if reference['status'] != 'COMPLETE' or not diagnostic['reference_target_reached']:
        raise ValueError('Complete executable references and successful body diagnostic are required')
    request = json.loads((folder/'request.json').read_text())
    config, framework = request['config'], request['framework']
    output = folder/'training'; output.mkdir(exist_ok=False)
    rows = read_jsonl(folder/'demonstrations.jsonl')
    replay_path = Path('runs/refinement/PR1C_support/adaptation/examples.jsonl')
    replay = read_jsonl(replay_path)
    count = study['optimizer_steps']*study['gradient_accumulation_steps']
    order = schedule(rows, replay, count, study['seed'])
    guarded_json(output/'request.json', {'study': study, 'initial_adapter': config['initial_adapter'],
        'initial_adapter_sha256': file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors'),
        'demonstrations_sha256': file_hash(folder/'demonstrations.jsonl'), 'replay_sha256': file_hash(replay_path),
        'diagnostic_sha256': file_hash('runs/refinement/body_fit/result.json'),
        'objective': 'mode CE + mean body CE; 20% old structure SFT replay; no KL, RL, or boundary weighting',
        'schedule': [{'source': source, 'sample_id': (rows if source == 'course' else replay)[i]['sample_id']}
                     for source, i in order]})
    set_seed(study['seed'])
    planner, _ = load_policies(framework, config['initial_adapter'], config['executor_adapter'])
    model, tok = planner.model, planner.tokenizer
    course = [encode_demo(r, tok, framework) for r in rows]
    # Historical encoding is retained, but the new objective has no boundary multiplier.
    for example in replay:
        example.pop('boundary_tokens', None); example.pop('boundary_weight', None)
    examples = [(course if source == 'course' else replay)[i] for source, i in order]
    (output/'examples.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in examples))
    frozen = adapter_fingerprint(model, 'executor')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    model.peft_config['default'].lora_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False
    enable_planner(model); before, trainable = trainable_snapshots(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=study['learning_rate'], weight_decay=0.)
    history = []; accumulation = study['gradient_accumulation_steps']; started = time.monotonic()
    for step in range(study['optimizer_steps']):
        optimizer.zero_grad(set_to_none=True); enable_planner(model)
        losses = Counter()
        for example in examples[step*accumulation:(step+1)*accumulation]:
            with planner.llm._autocast(), torch.autograd.graph.save_on_cpu(pin_memory=False):
                loss, gate, body = representation_loss(model, example, tok.pad_token_id)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite course loss')
            (loss/accumulation).backward()
            losses.update(loss=float(loss.detach())/accumulation, gate_ce=float(gate.detach())/accumulation,
                          body_ce=float(body.detach())/accumulation)
            del loss, gate, body
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm) or norm <= 0: raise FloatingPointError('Invalid course gradient')
        optimizer.step()
        record = {'step': step+1, **losses, 'gradient_norm': float(norm)}
        history.append(record); print(json.dumps(record), flush=True)
    final = output/'final_adapter'
    model.save_pretrained(final, selected_adapters=['default']); tok.save_pretrained(final)
    changes = changed_tensors(model, before); after = adapter_fingerprint(model, 'executor')
    old = prefix_forward(model, examples[0]['prefix'])
    model.load_adapter(str(final), adapter_name='reload_check', is_trainable=False, local_files_only=True)
    with active_adapter(model, 'reload_check'): restored = prefix_forward(model, examples[0]['prefix'])
    delta = float((old-restored).abs().max())
    if frozen != after or delta > 1e-5: raise ValueError('Executor freeze or adapter reload failed')
    report = {'status': 'COMPLETE', 'scope': 'MULTI_STATE_STRUCTURAL_SFT_NOT_REWARD_OPTIMIZATION',
        'initial_adapter': config['initial_adapter'], 'optimizer_steps': len(history), 'rl_updates': 0,
        'examples': len(examples), 'by_source': dict(Counter(s for s, _ in order)),
        'by_mode': dict(Counter(e['mode'] for e in examples)), 'course_unique_states': len(course),
        'course_roots': len({e['root_id'] for e in course}),
        'course_internal_exposures': sum(source == 'course' and rows[i]['node_id'] != 'N0' for source, i in order),
        'history': history, 'trainable_parameters': trainable, 'parameter_changes': changes,
        'executor_before': frozen, 'executor_after': after, 'reload_max_abs_difference': delta,
        'adapter': str(final), 'adapter_sha256': file_hash(final/'adapter_model.safetensors'),
        'adapter_config_sha256': file_hash(final/'adapter_config.json'), 'train_seconds': time.monotonic()-started}
    write_json(output/'training_report.json', report)
    print(json.dumps({'status': 'COMPLETE', 'steps': len(history), 'examples': len(examples)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train'])
    parser.add_argument('--config', default='configs/refinement_responsibility_course.yaml')
    args = parser.parse_args(); train(yaml.safe_load(Path(args.config).read_text()))
