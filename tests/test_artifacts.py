import pytest

from audittrainer.artifacts import SourceError


def test_saving_source_never_executes_module_top_level(make_runtime, tmp_path):
    runtime = make_runtime()
    marker = tmp_path / "must_not_be_created"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef solve(text): return '0'\n"
    artifact = runtime.store.save_initial("N0", runtime.node.incoming_contract, source)
    assert not marker.exists()
    assert (runtime.run_dir / artifact.module_path).read_text() == source


def test_immutable_artifact_and_owned_path(make_runtime):
    runtime = make_runtime()
    source = "def solve(text): return '0'\n"
    artifact = runtime.store.save_initial("N0", runtime.node.incoming_contract, source)
    with pytest.raises(SourceError, match="overwrite"):
        runtime.store.save_initial("N0", runtime.node.incoming_contract, "def solve(text): return '1'\n")
    assert (runtime.run_dir / artifact.module_path).read_text() == source
    with pytest.raises(ValueError):
        runtime.store.save_initial("../outside", runtime.node.incoming_contract, source)


def test_owner_directory_symlink_cannot_escape_run(make_runtime, tmp_path):
    runtime = make_runtime()
    directory = runtime.run_dir / "artifacts"
    directory.mkdir()
    (directory / "N0").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SourceError, match="leaves"):
        runtime.store.save_initial("N0", runtime.node.incoming_contract, "def solve(text): return '0'\n")
    assert not (tmp_path / "v1.py").exists()
