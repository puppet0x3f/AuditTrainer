import json
import os
from pathlib import Path

from audittrainer import gpu_resource


def settings(tmp_path):
    p = tmp_path/'gpu_resources.json'
    p.write_text(json.dumps({'devices': ['0', '1'], 'min_free_mib': {'train': 22000, 'collect': 16000}}))
    return p


def test_reservation_excludes_another_worker_even_before_it_loads_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    p = settings(tmp_path)
    probe = lambda scope: ({'0': 33000, '1': 33000}, set())
    first = gpu_resource.acquire(p, 'train', probe)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    second = gpu_resource.acquire(p, 'train', probe)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    first.close(); second.close()


def test_waits_for_capacity_and_does_not_overlap_a_preexisting_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    p = settings(tmp_path); waited = []
    def probe(scope):
        return {'0': 33000, '1': 33000 if waited else 15000}, {'0'}
    monkeypatch.setattr(gpu_resource.time, 'sleep', lambda seconds: waited.append(seconds))
    handle = gpu_resource.acquire(p, 'train', probe)
    assert waited == [10] and os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    handle.close()


def test_preexisting_gpu_process_detection_keeps_unrelated_jobs_as_memory_usage(monkeypatch):
    def query(arguments, text):
        if '--query-gpu=index,uuid,memory.free' in arguments:
            return '0, GPU-a, 33000\n1, GPU-b, 15000\n'
        return 'GPU-a, 101\nGPU-b, 102\n'
    monkeypatch.setattr(gpu_resource.subprocess, 'check_output', query)
    def cmdline(path):
        if str(path) == '/proc/101/cmdline':
            return b'python\0-m\0audittrainer.refinement_pr1\0train\0--config\0runs/refinement/B3_generalization/round/config.yaml'
        return b'python\0unrelated_training.py'
    monkeypatch.setattr(Path, 'read_bytes', cmdline)
    assert gpu_resource.available_gpus('B3_generalization') == ({'0': 33000, '1': 15000}, {'0'})
