"""Read-only gate precision check on the saved before/after repeat checkpoints."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audittrainer.evaluate import write_json
from audittrainer.refinement_policy import MODES, active_adapter, adapter_fingerprint, load_policies, mode_log_probs
from audittrainer.refinement_pr1 import load_config, load_training_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-adapter', default='runs/refinement/B3_variance/arms/repeat/B3/final_adapter')
    parser.add_argument('--output', default='runs/refinement/B3_gradients/precision.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, framework = load_config('runs/refinement/B3_variance/arms/repeat/config.yaml')
    groups = load_training_groups(config)
    planner, _ = load_policies(framework, config['initial_adapter'], config['executor_adapter'])
    model = planner.model
    model.load_adapter(args.after_adapter, adapter_name='after',
                       is_trainable=False, local_files_only=True)
    before = {n: adapter_fingerprint(model, n) for n in ('default', 'after', 'executor')}
    head = model.get_output_embeddings()
    records = []
    for adapter in ('default', 'after'):
        for group, gate, _, _ in groups:
            selected = [gate['mode_token_ids'][m][0] for m in MODES]
            captured = {}

            def inspect_head(module, inputs, output):
                # Keep the backbone and hidden states unchanged. Only redo its
                # final linear projection in FP32 to inspect rounding loss.
                with torch.autocast('cuda', enabled=False):
                    full = F.linear(inputs[0].float(), module.weight.float(), module.bias)
                captured['bf16_first_logits'] = output[0, 0, selected].float().cpu().tolist()
                captured['fp32_first_logits'] = full[0, 0, selected].cpu().tolist()
                captured['original_logits_dtype'] = str(output.dtype)
                return full if captured['replace'] else output

            hook = head.register_forward_hook(inspect_head)
            probabilities = {}
            try:
                for precision in ('bf16', 'fp32_head'):
                    captured['replace'] = precision == 'fp32_head'
                    with active_adapter(model, adapter), torch.inference_mode(), planner.llm._autocast():
                        lp, _ = mode_log_probs(model, gate['prefix_ids'], gate['mode_token_ids'], MODES,
                                               planner.tokenizer.pad_token_id)
                    probabilities[precision] = float(lp[1].exp())
            finally:
                hook.remove()
            records.append({'adapter': adapter, 'root_id': group['root_id'], 'p_split': probabilities,
                            **{k: v for k, v in captured.items() if k != 'replace'}})
    after = {n: adapter_fingerprint(model, n) for n in before}
    assert before == after
    result = {'status': 'COMPLETE', 'optimizer_steps': 0, 'new_rollouts': 0,
              'before_adapter': config['initial_adapter'],
              'after_adapter': args.after_adapter,
              'records': records, 'frozen_before': before, 'frozen_after': after,
              'scope': 'Diagnostic only: FP32 final projection on identical backbone hidden states. '
                       'The deployed policy and all previous evaluation results still use the original BF16 projection.'}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
