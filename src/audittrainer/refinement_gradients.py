"""Replay one saved B3 accumulation window without updating any parameters."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from transformers import set_seed

from .dataset import file_hash, read_jsonl
from .evaluate import write_json
from .refinement_policy import MODES, active_adapter, adapter_fingerprint, load_policies, mode_log_probs
from .refinement_pr1 import (causal_logits, conditional_token_terms, credit_terms,
                             enable_planner, load_config, load_training_groups, mode_surrogate)
from .train_refinement import representation_loss


def flat_gradient(loss, parameters, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().reshape(-1)
                      for p, g in zip(parameters, gradients)])


def geometry(vectors):
    norms = {name: float(value.norm()) for name, value in vectors.items()}
    pairs = {}
    for i, a in enumerate(vectors):
        for b in list(vectors)[i+1:]:
            dot = float(torch.dot(vectors[a], vectors[b]))
            denominator = norms[a]*norms[b]
            pairs[f'{a}:{b}'] = {'dot': dot, 'cosine': dot/denominator if denominator else None}
    return {'norms': norms, 'pairs': pairs}


def diagnose(config, framework, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = load_training_groups(config)
    settings = config['update']
    accumulation = settings['gradient_accumulation_groups']
    if len(groups) != accumulation:
        raise ValueError('This diagnostic replays exactly one complete accumulation window')
    write_json(output/'request.json', {'config': config, 'epoch': 0,
        'optimizer_steps': 0, 'new_rollouts': 0,
        'sampling_adapter_sha256': file_hash(Path(config['initial_adapter'])/'adapter_model.safetensors')})
    torch.set_num_threads(8)
    set_seed(config['seed'])
    planner, _ = load_policies(framework, config['initial_adapter'], config['executor_adapter'])
    model, tokenizer = planner.model, planner.tokenizer
    model.load_adapter(config['reference_adapter'], adapter_name='reference', is_trainable=False,
                       local_files_only=True)
    frozen = {n: adapter_fingerprint(model, n) for n in ('default', 'reference', 'executor')}
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.peft_config['default'].lora_dropout = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    enable_planner(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in parameters)
    components = {name: torch.zeros(size) for name in ('mode_reward', 'body_reward', 'mode_kl', 'body_kl', 'sft')}
    mode_gradients, records = {}, []
    demos = read_jsonl(Path(config['adaptation']['output_dir'])/'examples.jsonl')
    eps, beta, eta, scale = (settings[k] for k in ('clip_epsilon', 'kl_weight', 'sft_weight', 'objective_scale'))
    order = list(range(len(groups)))
    random.Random(config['seed']).shuffle(order)
    model.zero_grad(set_to_none=True)
    started = time.monotonic()
    for position, index in enumerate(order):
        group, gate, candidates, baseline = groups[index]
        rid = group['root_id']
        mode_a, body_a = credit_terms(group, 'B3', baseline)
        old = torch.tensor([gate['mode_log_probs'][m] for m in MODES], device=model.device)
        with active_adapter(model, 'reference'), torch.no_grad(), planner.llm._autocast():
            ref_mode = mode_log_probs(model, gate['prefix_ids'], gate['mode_token_ids'], MODES,
                                      tokenizer.pad_token_id)[0].detach()
        enable_planner(model)
        with planner.llm._autocast():
            current, _ = mode_log_probs(model, gate['prefix_ids'], gate['mode_token_ids'], MODES,
                                        tokenizer.pad_token_id)
            objective = mode_surrogate(current, old, mode_a, body_a, 'B3', eps)
            reward = -scale*objective/accumulation
            kl = scale*beta*(current.exp()*(current-ref_mode)).sum()/accumulation
            joint = scale*(-objective+beta*(current.exp()*(current-ref_mode)).sum())/accumulation
        initial_delta = float((current.detach()-old).abs().max())
        if initial_delta > .05:
            raise ValueError(f'Wrong sampling checkpoint: {initial_delta}')
        # Gate sensitivity distinguishes cancellation from a small norm alone.
        sensitivity = flat_gradient(current[1]-current[0], parameters, retain_graph=True)
        mode_gradients[rid] = sensitivity
        g = flat_gradient(reward, parameters, retain_graph=True)
        components['mode_reward'] += g
        mode_norm = float(g.norm())
        components['mode_kl'] += flat_gradient(kl, parameters, retain_graph=True)
        joint.backward()
        record = {'root_id': rid, 'q_values': group['q_values'], 'baseline': baseline,
                  'split_minus_atomic_q': sum(group['q_values'][1:])/len(body_a)-group['q_values'][0],
                  'mode_advantages': mode_a, 'body_advantages': body_a,
                  'p_split': float(current[1].detach().exp()), 'mode_reward_norm': mode_norm,
                  'max_initial_gate_logp_difference': initial_delta, 'bodies': []}
        del current, objective, reward, kl, joint, g, sensitivity
        for candidate_index, (candidate, advantage) in enumerate(zip(candidates[1:], body_a), start=1):
            prefix, target = candidate['body_input_ids'], candidate['body_output_ids']
            with active_adapter(model, 'reference'), torch.no_grad(), planner.llm._autocast():
                reference = causal_logits(model, prefix, target).detach()
            scoring_path = Path(config['output_dir'])/'B3/scoring_probabilities'/f'{rid}_{candidate_index}.json'
            old_scoring = json.loads(scoring_path.read_text())['old_teacher_forced_log_probs']
            enable_planner(model)
            with planner.llm._autocast():
                logits = causal_logits(model, prefix, target)
                objective, kl, chosen = conditional_token_terms(logits, reference, target, old_scoring, advantage, eps)
                weight = scale*old[1].exp().detach()/len(body_a)/accumulation
                reward, penalty = -weight*objective, weight*beta*kl
                joint = scale*old[1].exp().detach()/len(body_a)*(-objective+beta*kl)/accumulation
            delta = float((chosen.detach()-torch.tensor(old_scoring, device=model.device)).abs().max())
            if delta > 1e-4:
                raise ValueError(f'Wrong body scoring checkpoint: {delta}')
            g = flat_gradient(reward, parameters, retain_graph=True)
            components['body_reward'] += g
            body_norm = float(g.norm())
            components['body_kl'] += flat_gradient(penalty, parameters, retain_graph=True)
            joint.backward()
            record['bodies'].append({'candidate': candidate_index, 'tokens': len(target),
                'advantage': advantage, 'actual_weight': float(weight), 'weighted_reward_gradient_norm': body_norm,
                'max_initial_logp_difference': delta})
            del reference, logits, objective, kl, chosen, reward, penalty, joint, g
        enable_planner(model)
        retention_index = config.get('retention_offset', 0)+position
        with planner.llm._autocast():
            retention, _, _ = representation_loss(model, demos[retention_index % len(demos)], tokenizer.pad_token_id)
            regularizer = scale*eta*retention/accumulation
        components['sft'] += flat_gradient(regularizer, parameters, retain_graph=True)
        regularizer.backward()
        del regularizer, retention
        record['retention_index'] = retention_index
        records.append(record)
        write_json(output/'progress.json', {'completed_states': len(records), 'records': records})
        print(json.dumps({'completed_states': len(records), 'root_id': rid, 'mode_norm': mode_norm,
                          'seconds': time.monotonic()-started}), flush=True)
    actual = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().float().cpu().reshape(-1)
                        for p in parameters])
    combined = sum(components.values())
    for record in records:
        sensitivity = mode_gradients[record['root_id']]
        record['split_log_odds_directional_derivatives'] = {
            name: -float(torch.dot(sensitivity, gradient)) for name, gradient in components.items()}
        record['split_log_odds_directional_derivatives']['actual_combined'] = -float(torch.dot(sensitivity, actual))
        record['split_log_odds_directional_derivatives']['mode_weight_4'] = -float(torch.dot(
            sensitivity, actual+3*components['mode_reward']))
    after = {n: adapter_fingerprint(model, n) for n in frozen}
    if frozen != after:
        raise RuntimeError('Diagnostic changed model weights')
    result = {'status': 'COMPLETE', 'optimizer_steps': 0, 'new_rollouts': 0, 'accumulated_states': len(groups),
              'source_config': config, 'parameters': size, 'records': records,
              **geometry(components), 'actual_combined_norm': float(actual.norm()),
              'component_sum_norm': float(combined.norm()),
              'relative_component_sum_residual': float((combined-actual).norm()/actual.norm()),
              'clip_factor': min(1., settings['max_grad_norm']/(float(actual.norm())+1e-6)),
              'frozen_before': frozen, 'frozen_after': after, 'seconds': time.monotonic()-started,
              'interpretation': 'Loss gradients after all weights and the full first six-state accumulation. '
                  'Directional derivatives are infinitesimal SGD directions, not Adam updates or measured probability changes. '
                  'Independent component backpropagation can differ slightly from combined backpropagation under BF16; residual reported.'}
    write_json(output/'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'norms', 'pairs', 'actual_combined_norm',
                     'relative_component_sum_residual', 'seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='runs/refinement/B3_variance/arms/repeat/config.yaml')
    parser.add_argument('--output', default='runs/refinement/B3_gradients/diagnostic')
    args = parser.parse_args()
    config, framework = load_config(args.config)
    diagnose(config, framework, args.output)


if __name__ == '__main__':
    main()
