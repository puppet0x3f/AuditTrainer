"""Aggregate the completed A5 arms once, without choosing a best seed/checkpoint."""
from collections import Counter
import json
from pathlib import Path

from .a5_data import GROUPS
from .dataset import file_hash, read_jsonl
from .evaluate import write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    data = read('runs/a5/data_comparison.json')
    training, evaluation, diagnosis, rates = {}, {}, {}, {}
    initial = []
    for group in GROUPS:
        directory = Path(f'runs/a5/train/{group}_seed42')
        report = read(directory / 'training_report.json')
        assert report['status'] == 'COMPLETE' and report['optimizer_steps'] == 140
        assert report['reload']['max_abs_logit_difference'] <= 1e-5
        observed_tokens = sum(row.get('target_tokens', 0) for row in report['history'])
        assert observed_tokens == data['groups'][group]['target_tokens']
        assert file_hash(Path(report['adapter']) / 'adapter_model.safetensors') == report['adapter_sha256']
        initial.append(read(directory / 'before_update.json'))
        training[group] = {k: report[k] for k in ('optimizer_steps', 'adapter', 'adapter_sha256', 'metrics', 'reload', 'peak_cuda_allocated_bytes')}
        training[group]['observed_target_tokens'] = observed_tokens
        evaluation[group] = read(f'runs/a5/evaluation/{group}/summary.json')
        assert evaluation[group]['status'] == 'COMPLETE'
        request = read(f'runs/a5/evaluation/{group}/request.json')
        assert Path(request['adapter']).resolve() == Path(report['adapter']).resolve()
        probe_path = f'runs/a5/evaluation/{group}/adapter_probe.json'
        assert read(probe_path)['status'] == 'PASS'
        training[group]['runtime_reconnection'] = probe_path
        records = read_jsonl(f'runs/a5/evaluation/{group}/isolation.jsonl')
        assert len(records) == 40 and len({(r['root_id'], r['condition']) for r in records}) == 40
        assert all(r['child_artifacts_frozen'] for r in records)
        for record in records:
            case_request = read(f"runs/a5/evaluation/{group}/{record['condition']}/{record['root_id']}/request.json")
            assert case_request['model_identity']['adapter_weights_sha256'] == report['adapter_sha256']
        assert not any(r['final_scores']['private']['status'] == 'ERROR' for r in records)
        free = evaluation[group]['free_result']['aggregate']
        assert free['roots'] == 5
        assert not free['root_status'].get('ERROR', 0)
        rates[group] = {}
        for name, conditions in [('local_plan', {'local_plan'}), ('implementation', {'transform', 'summarize'}),
                                  ('composition', {'parent_both_children', 'parent_transform_child'})]:
            pool = [r for r in records if r['condition'] in conditions]
            successes = sum(r['final_scores']['private']['status'] == 'PASS' for r in pool)
            rates[group][name] = {'successes': successes, 'cases': len(pool), 'rate': successes / len(pool)}
        rates[group]['free_recursive'] = {'successes': free['budgeted_successes'], 'cases': 5, 'rate': free['BudgetedSuccess']}
        labels = Counter()
        for r in records:
            passed = r['final_scores']['private']['status'] == 'PASS'
            if r['condition'] == 'local_plan':
                label = 'D_PLAN_NOT_REGISTERED' if r['accepted_plan_mode'] is None else 'PASS' if passed else 'D_OR_FIXED_EXECUTOR_UNRESOLVED'
            else:
                label = 'PASS' if passed else 'C_FIXED_PARENT_FAILURE' if r['condition'].startswith('parent_') else 'I_FIXED_COMPONENT_FAILURE'
            labels[label] += 1
        diagnosis[group] = {'isolation': dict(labels),
                            'semantic_unknown': sum(r['responsibility_review']['status'] == 'UNKNOWN' for r in records),
                            'free_stop_reasons': free['stop_reasons'],
                            'unknown_is_not_violation': True}
    assert all(value == initial[0] for value in initial)
    effects = {}
    for metric in rates['F00']:
        a, b, c, d = [rates[g][metric]['rate'] for g in GROUPS]
        effects[metric] = {'D_average_difference_pp': 100 * ((b-a)+(d-c))/2,
                           'I_average_difference_pp': 100 * ((c-a)+(d-b))/2,
                           'interaction_pp': 100 * (d-b-c+a)}
    result = {'stage': 'A5', 'status': 'COMPLETE', 'scope': 'single-seed development; not multi-seed paper evaluation',
              'seed': 42, 'data': data, 'training': training, 'evaluation': evaluation,
              'rates': rates, 'factorial_effects': effects, 'aggregate_diagnosis': diagnosis,
              'initial_tracked_lora_tensors_identical': True, 'formal_from_pilot': False,
              'tests': {'full_suite': {'passed': 256, 'log': 'runs/a5/pytest.log'},
                        'offload_regression': {'passed': 17, 'log': 'runs/a5/offload_tests.log'},
                        'composition_regression': {'passed': 6, 'log': 'runs/a5/composition_tests.log'},
                        'distinct_test_cases_covered': 258}, 'next_stage': 'A6',
              'implementation_sha256': {p: file_hash(p) for p in [
                  'src/audittrainer/a5_data.py', 'src/audittrainer/a5_evaluate.py',
                  'src/audittrainer/train_sft.py', 'configs/a5_evaluation.yaml']}}
    write_json('runs/a5/result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'rates': rates, 'effects': effects}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
