"""Linux Bubblewrap executor with bounded output and no host-project mounts."""

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal

from pydantic import Field

from .schema import Record
from .artifacts import Bundle


class ExecutionLimits(Record):
    network: Literal["disabled"] = "disabled"
    per_case_timeout_seconds: float = Field(default=2.0, gt=0)
    memory_mb: int = Field(default=1024, ge=32)
    max_output_bytes: int = Field(default=1048576, ge=128)
    max_processes: Literal[1] = 1


class ExecutionResult(Record):
    status: Literal["RETURNED", "FAIL", "ERROR"]
    reason: str
    value: Any = None
    mutated_inputs: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    elapsed_seconds: float = 0.0
    call_counts: list[dict] = Field(default_factory=list)
    call_trace_complete: bool = False


BLOCKED_SYSCALLS = (
    "clone", "clone3", "fork", "vfork", "unshare", "setns", "mount",
    "umount2", "pivot_root", "ptrace", "bpf", "perf_event_open",
    "open_by_handle_at", "keyctl", "add_key", "request_key",
)


def _seccomp_file():
    """Export a small libseccomp BPF filter for Bubblewrap's target process."""
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = library.seccomp_init(0x7FFF0000) # SCMP_ACT_ALLOW
    if not context:
        raise OSError("Cannot initialize libseccomp")
    stream = tempfile.TemporaryFile()
    try:
        for name in BLOCKED_SYSCALLS:
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall < 0:
                raise OSError(f"libseccomp cannot resolve {name}")
            result = library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, syscall, 0)
            if result != 0:
                raise OSError(f"Cannot install seccomp rule {name}: {result}")
        if library.seccomp_export_bpf(context, stream.fileno()) != 0:
            raise OSError("Cannot export seccomp filter")
        stream.seek(0)
    except BaseException:
        stream.close()
        raise
    finally:
        library.seccomp_release(context)
    return stream


def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Sandbox:
    def __init__(self, limits: ExecutionLimits | None = None, bwrap: str | None = None):
        self.limits = limits or ExecutionLimits()
        self.bwrap = bwrap or shutil.which("bwrap") or "bwrap"
        self.runner = Path(__file__).with_name("_sandbox_runner.py")
        try:
            self.bwrap_version = subprocess.check_output(
                [self.bwrap, "--version"], text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            self.bwrap_version = "UNAVAILABLE"

    def identity(self) -> dict:
        return {
            "backend": "bubblewrap", "bwrap_version": self.bwrap_version,
            "python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "runner_sha256": hashlib.sha256(self.runner.read_bytes()).hexdigest(),
            "sandbox_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "blocked_syscalls": list(BLOCKED_SYSCALLS),
            "execution_config": self.limits.model_dump(),
        }

    def run(self, source: str, public_symbol: str, args: list, *, bundle: Bundle | None = None) -> ExecutionResult:
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="audittrainer-case-") as directory, _seccomp_file() as filter_file:
                work = Path(directory)
                (work / "source.py").write_text(source, encoding="utf-8")
                (work / "runner.py").write_bytes(self.runner.read_bytes())
                if bundle is not None:
                    bundle.materialize(work)
                (work / "request.json").write_text(json.dumps({
                    "public_symbol": public_symbol, "args": args, "limits": self.limits.model_dump(),
                }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                command = [
                    self.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                    "--ro-bind", str(Path(sys.executable).resolve()), "/runtime/bin/python3.11",
                    "--ro-bind", str(Path(sys.base_prefix) / "lib/python3.11"), "/runtime/lib/python3.11",
                    "--tmpfs", "/runtime/lib/python3.11/site-packages",
                    "--remount-ro", "/runtime/lib/python3.11/site-packages",
                    "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                    "--ro-bind", str(work), "/work", "--chdir", "/tmp",
                    "--seccomp", str(filter_file.fileno()),
                    "/runtime/bin/python3.11", "-I", "-S", "-B", "/work/runner.py",
                ]
                # No inherited credentials, Python import path, CUDA visibility, or open host descriptors.
                with subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={"PATH": "/runtime/bin", "HOME": "/tmp", "TMPDIR": "/tmp", "LANG": "C.UTF-8"},
                    start_new_session=True, close_fds=True, pass_fds=(filter_file.fileno(),),
                ) as process:
                    out, err, reason = self._capture(process)
                    elapsed = time.monotonic() - started
                    return self._interpret(out, err, process.returncode, reason, elapsed)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return ExecutionResult(status="ERROR", reason="SANDBOX_ERROR", stderr=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)

    def _capture(self, process):
        streams = {"stdout": bytearray(), "stderr": bytearray()}
        total = 0
        deadline = time.monotonic() + self.limits.per_case_timeout_seconds
        reason = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reason = "TIMEOUT"
                        break
                    for key, _ in selector.select(min(remaining, 0.05)):
                        chunk = os.read(key.fileobj.fileno(), min(65536, self.limits.max_output_bytes - total + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        keep = max(0, self.limits.max_output_bytes - total)
                        streams[key.data].extend(chunk[:keep])
                        total += len(chunk)
                        if total > self.limits.max_output_bytes:
                            reason = "OUTPUT_LIMIT"
                            break
                    if reason:
                        break
                if reason:
                    _kill(process)
                else:
                    try:
                        process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        reason = "TIMEOUT"
                        _kill(process)
        finally:
            if process.poll() is None:
                _kill(process)
        return *(data.decode("utf-8", errors="replace") for data in streams.values()), reason

    @staticmethod
    def _interpret(stdout, stderr, returncode, reason, elapsed):
        fields = dict(stdout=stdout, stderr=stderr, returncode=returncode, elapsed_seconds=elapsed)
        lines = stdout.splitlines()
        ready = bool(lines and lines[0] == '{"kind": "ready"}')
        if reason:
            return ExecutionResult(status="FAIL" if ready else "ERROR", reason=reason, **fields)
        if not ready:
            return ExecutionResult(status="ERROR", reason="RUNNER_START_FAILED", **fields)
        if returncode != 0:
            return ExecutionResult(status="FAIL", reason="CANDIDATE_EXIT", **fields)
        try:
            if len(lines) != 2:
                raise ValueError("Expected one candidate response")
            response = json.loads(lines[1], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            calls = response.get("call_counts", [])
            if not isinstance(calls, list) or any(not isinstance(c, dict) or set(c) != {"consumer_node", "child_node", "count"}
                    or not isinstance(c["consumer_node"], str) or not isinstance(c["child_node"], str)
                    or type(c["count"]) is not int or c["count"] < 1 for c in calls):
                raise ValueError("Malformed call trace")
            fields.update(call_counts=calls, call_trace_complete=response.get("call_trace_complete") is True)
            if response.get("kind") == "candidate_error":
                return ExecutionResult(status="FAIL", reason="CANDIDATE_EXCEPTION", **fields)
            if response.get("kind") != "returned" or "value" not in response or type(response.get("mutated_inputs")) is not bool:
                raise ValueError("Malformed candidate response")
            return ExecutionResult(status="RETURNED", reason="RETURNED", value=response["value"], mutated_inputs=response["mutated_inputs"], **fields)
        except (ValueError, AttributeError):
            return ExecutionResult(status="FAIL", reason="INVALID_RESULT", **fields)
