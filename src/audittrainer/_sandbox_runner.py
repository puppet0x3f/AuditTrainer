"""Trusted, stdlib-only single-case entry point copied into the sandbox.

The request contains input arguments and limits, never expected answers.
"""

import builtins
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import traceback


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def prepare_bundle():
    """Load only the selected closure; count public-function calls by caller module."""
    manifest = json.loads(Path("/work/manifest.json").read_text())
    modules = manifest["modules"]
    file_nodes = {f"/work/nodes/{node_id}.py": node_id for node_id in modules}
    allowed = {node_id: {d["child_id"] for d in module["dependency_bindings"]} for node_id, module in modules.items()}
    counts = {}

    def caller():
        frame = sys._getframe(2)
        while frame:
            node_id = file_nodes.get(frame.f_code.co_filename)
            if node_id:
                return node_id
            frame = frame.f_back
        return None

    def check_import(name):
        if name.startswith("nodes."):
            owner, target = caller(), name.removeprefix("nodes.")
            if owner and target != owner and target not in allowed[owner]:
                raise ImportError("Node may import only its explicitly consumed direct children")

    original_import, original_import_module = builtins.__import__, importlib.import_module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        check_import(name)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        check_import(importlib.util.resolve_name(name, package) if name.startswith(".") else name)
        return original_import_module(name, package)

    def profile(frame, event, arg):
        if event == "call":
            child_id = file_nodes.get(frame.f_code.co_filename)
            consumer = file_nodes.get(frame.f_back.f_code.co_filename) if frame.f_back else None
            if child_id and consumer and consumer != child_id and frame.f_code.co_name == modules[child_id]["public_symbol"]:
                key = (consumer, child_id)
                counts[key] = counts.get(key, 0) + 1

    sys.path.insert(0, "/work")
    builtins.__import__, importlib.import_module = guarded_import, guarded_import_module
    sys.setprofile(profile)
    return original_import_module, profile, counts


def main():
    try:
        request = json.loads(Path("/work/request.json").read_text())
        limits = request["limits"]
        memory = limits["memory_mb"] * 1024 * 1024
        cpu = max(1, math.ceil(limits["per_case_timeout_seconds"]))
        for kind, values in (
            (resource.RLIMIT_AS, (memory, memory)),
            (resource.RLIMIT_CPU, (cpu, cpu + 1)),
            (resource.RLIMIT_FSIZE, (limits["max_output_bytes"], limits["max_output_bytes"])),
            (resource.RLIMIT_NOFILE, (64, 64)),
            (resource.RLIMIT_NPROC, (0, 0)),
            (resource.RLIMIT_CORE, (0, 0)),
        ):
            resource.setrlimit(kind, values)
        args = request["args"]
        before = json.dumps(args, sort_keys=True, allow_nan=False)
        symbol = request["public_symbol"]
    except BaseException:
        emit({"kind": "runner_error", "message": traceback.format_exc(limit=5)})
        return 1

    emit({"kind": "ready"})
    # Keep candidate prints separate from the returned JSON value. Neither stream
    # carries an authoritative verdict; the host compares the value with its oracle.
    response_fd = os.dup(1)
    os.dup2(2, 1)
    response_stream = os.fdopen(response_fd, "w", encoding="utf-8")
    profiler, counts = None, {}
    try:
        if Path("/work/manifest.json").exists():
            load_module, profiler, counts = prepare_bundle()
            module = load_module("entry")
            symbol = "solve" # Fixed adapter points to the contracted public callable.
        else:
            spec = importlib.util.spec_from_file_location("candidate", "/work/source.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        value = getattr(module, symbol)(*args)
        response = {
            "kind": "returned", "value": value,
            "mutated_inputs": before != json.dumps(args, sort_keys=True, allow_nan=False),
        }
        # Check serializability while errors can still be attributed to the candidate.
        json.dumps(response, ensure_ascii=False, allow_nan=False)
    except BaseException:
        response = {"kind": "candidate_error", "message": traceback.format_exc(limit=8)}
    response["call_trace_complete"] = profiler is not None and sys.getprofile() is profiler
    sys.setprofile(None)
    response["call_counts"] = [{"consumer_node": owner, "child_node": child, "count": count}
                               for (owner, child), count in sorted(counts.items())]
    payload = json.dumps(response, ensure_ascii=False, allow_nan=False)
    response_stream.write(payload + "\n")
    response_stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
