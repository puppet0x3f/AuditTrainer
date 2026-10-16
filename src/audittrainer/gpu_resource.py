"""Reserve one sufficiently free GPU for a configured local experiment stage."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def available_gpus(scope):
    def query(arguments):
        return list(csv.reader(subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'], text=True).splitlines()))
    rows = query(['--query-gpu=index,uuid,memory.free'])
    free = {index.strip(): int(memory) for index, uuid, memory in rows}
    indices = {uuid.strip(): index.strip() for index, uuid, memory in rows}
    occupied = set()
    # Include workers that started before reservations were enabled.
    for uuid, pid in query(['--query-compute-apps=gpu_uuid,pid']):
        try:
            command = (Path('/proc')/pid.strip()/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if 'audittrainer.refinement_pr1' in command and scope in command:
            occupied.add(indices[uuid.strip()])
    return free, occupied


def acquire(settings_path, command, probe=available_gpus):
    settings_path = Path(settings_path)
    settings = json.loads(settings_path.read_text())
    locks = settings_path.parent/'.gpu_locks'; locks.mkdir(exist_ok=True)
    preferred = os.environ.get('CUDA_VISIBLE_DEVICES')
    required = settings['min_free_mib'][command]
    last_notice = -float('inf')
    while True:
        free, occupied = probe(settings_path.parent.name)
        devices = sorted(settings['devices'], key=lambda d: (d != preferred, -free[d]))
        for device in devices:
            handle = (locks/f'{device}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close(); continue
            if free[device] < required or device in occupied:
                handle.close(); continue
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            print(json.dumps({'gpu_allocated': device, 'free_mib': free[device], 'required_mib': required, 'stage': command}), flush=True)
            return handle  # Held until process exit, including loading and cleanup.
        if time.monotonic()-last_notice >= 60:
            print(json.dumps({'gpu_wait': command, 'required_mib': required, 'free_mib': free,
                              'active_experiment_gpus': sorted(occupied)}), flush=True)
            last_notice = time.monotonic()
        time.sleep(10)


def reserve_cli_gpu():
    if len(sys.argv) < 2 or sys.argv[1] not in {'collect', 'train'} or '--config' not in sys.argv:
        return None
    config_path = Path(sys.argv[sys.argv.index('--config')+1]).resolve()
    for folder in config_path.parents:
        settings_path = folder/'gpu_resources.json'
        if settings_path.exists():
            return acquire(settings_path, sys.argv[1])
    return None
