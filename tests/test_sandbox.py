"""Real subprocess checks: never replace isolation with a mock executor."""

import os
import socket
import textwrap

import pytest

from audittrainer.sandbox import ExecutionLimits, Sandbox


def test_callable_runs_in_selected_python_and_separates_prints():
    source = "import sys\ndef add(a, b):\n    print('candidate output')\n    return [a + b, sys.version_info[:2]]\n"
    result = Sandbox().run(source, "add", [2, 3])
    assert result.status == "RETURNED", result
    assert result.value == [5, [3, 11]]
    assert "candidate output" in result.stderr
    assert "candidate output" not in result.stdout


@pytest.mark.parametrize("source,limits,reason", [
    ("def solve(text):\n    while True: pass", {"per_case_timeout_seconds": 1.0}, "TIMEOUT"),
    ("def solve(text): raise ValueError('fixture failure')", {}, "CANDIDATE_EXCEPTION"),
    ("import os\ndef solve(text): os._exit(4)", {}, "CANDIDATE_EXIT"),
    ("def solve(text): return 'x' * 100000", {"max_output_bytes": 4096}, "OUTPUT_LIMIT"),
    ("def solve(text):\n    while True: print('x' * 1000, flush=True)", {"max_output_bytes": 4096}, "OUTPUT_LIMIT"),
    ("def solve(text): return bytearray(256 * 1024 * 1024)", {"memory_mb": 64}, "CANDIDATE_EXCEPTION"),
])
def test_candidate_failures_are_bounded(source, limits, reason):
    result = Sandbox(ExecutionLimits(**limits)).run(source, "solve", [""])
    assert (result.status, result.reason) == ("FAIL", reason), result
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= limits.get("max_output_bytes", 1048576)
    assert result.elapsed_seconds < 5
    if limits.get("memory_mb"):
        assert "MemoryError" in result.stdout


def test_host_files_credentials_processes_and_gpu_are_not_visible(tmp_path, monkeypatch):
    canary = tmp_path / "host-only.txt"
    canary.write_text("host-only-content")
    monkeypatch.setenv("AUDITTRAINER_TEST_SECRET", "host-secret-value")
    source = textwrap.dedent(f"""
        import os
        from pathlib import Path
        def inspect():
            return {{
                'host_file': Path({str(canary)!r}).exists(),
                'project': Path('/home/lc/AuditTrainer').exists(),
                'secret': os.environ.get('AUDITTRAINER_TEST_SECRET'),
                'host_process': Path('/proc/{os.getpid()}/cmdline').exists(),
                'gpu': any('nvidia' in p.name or p.name == 'dri' for p in Path('/dev').iterdir()),
                'packages': list(Path('/runtime/lib/python3.11/site-packages').iterdir()),
            }}
    """)
    result = Sandbox().run(source, "inspect", [])
    assert result.status == "RETURNED", result
    assert result.value == dict(host_file=False, project=False, secret=None, host_process=False, gpu=False, packages=[])
    assert canary.read_text() == "host-only-content"


def test_source_is_read_only_temporary_directory_writable_and_no_expected_in_request():
    source = textwrap.dedent("""
        import json
        from pathlib import Path
        def inspect():
            try:
                Path('/work/source.py').write_text('changed')
                readonly = False
            except OSError:
                readonly = True
            Path('/tmp/scratch').write_text('ok')
            return [readonly, Path('/tmp/scratch').read_text(), sorted(json.loads(Path('/work/request.json').read_text()))]
    """)
    result = Sandbox().run(source, "inspect", [])
    assert result.status == "RETURNED", result
    assert result.value == [True, "ok", ["args", "limits", "public_symbol"]]


def test_network_cannot_reach_host_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        source = textwrap.dedent(f"""
            import socket
            def inspect():
                try:
                    socket.create_connection(('127.0.0.1', {port}), timeout=0.2).close()
                    return 'connected'
                except OSError:
                    return 'isolated'
        """)
        result = Sandbox().run(source, "inspect", [])
    assert result.status == "RETURNED", result
    assert result.value == "isolated"


def test_process_creation_is_denied():
    source = textwrap.dedent("""
        import os
        def inspect():
            try:
                pid = os.fork()
            except PermissionError:
                return 'denied'
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            return 'forked'
    """)
    result = Sandbox().run(source, "inspect", [])
    assert result.status == "RETURNED", result
    assert result.value == "denied"


def test_missing_backend_reports_infrastructure_error_without_host_fallback(tmp_path):
    marker = tmp_path / "must-not-exist"
    result = Sandbox(bwrap=str(tmp_path / "missing-bwrap")).run(
        f"def solve(text):\n    open({str(marker)!r}, 'w').write('executed')\n    return '0'", "solve", [""],
    )
    assert (result.status, result.reason) == ("ERROR", "SANDBOX_ERROR")
    assert not marker.exists()


def test_broken_runner_is_error(tmp_path):
    runner = tmp_path / "broken-runner.py"
    runner.write_text("raise RuntimeError('runner setup failed')")
    sandbox = Sandbox()
    sandbox.runner = runner
    result = sandbox.run("def solve(text): return '0'", "solve", [""])
    assert (result.status, result.reason) == ("ERROR", "RUNNER_START_FAILED")
