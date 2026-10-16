"""Immutable source versions and explicitly bound dependency workspaces."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from .schema import Artifact, Contract, Dependency, Identifier


ENTRY_TEMPLATE = '''"""Fixed I/O adapter; task parsing and composition belong to the candidate."""
from nodes.{node_id} import {symbol} as solve

if __name__ == "__main__":
    import sys
    sys.stdout.write(solve(sys.stdin.read()))
'''


class SourceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def inspect_links(source: str, dependencies: list[Dependency], artifacts: dict[str, Artifact]) -> list[dict]:
    """Recognize the protocol's explicit `from nodes.ID import symbol as alias`."""
    approved = {d.child_id: d for d in dependencies}
    linked = set()
    for item in ast.walk(ast.parse(source)):
        if isinstance(item, ast.Import) and any(a.name == "nodes" or a.name.startswith("nodes.") for a in item.names):
            raise SourceError("DEPENDENCY_IMPORT_MISMATCH", "Use from nodes.<child_id> import <public_symbol> as <alias>")
        if not isinstance(item, ast.ImportFrom):
            continue
        if item.level or item.module == "nodes" or (item.module or "").startswith("nodes."):
            child_id = (item.module or "").removeprefix("nodes.")
            dependency = approved.get(child_id)
            if item.level or dependency is None or child_id not in artifacts:
                raise SourceError("UNAPPROVED_IMPORT", "Candidate may import only explicitly consumed direct children")
            if len(item.names) != 1 or item.names[0].name != artifacts[child_id].public_symbol or (item.names[0].asname or item.names[0].name) != dependency.alias:
                raise SourceError("DEPENDENCY_IMPORT_MISMATCH", "Imported symbol/alias must match the selected artifact")
            linked.add(child_id)
    return [{"child_id": d.child_id, "artifact_version": d.artifact_version, "artifact_hash": d.artifact_hash,
             "alias": d.alias, "selected": True, "linked": d.child_id in linked} for d in dependencies]


@dataclass
class Bundle:
    entry_node: str
    artifacts: dict[str, Artifact]
    sources: dict[str, str]

    def manifest(self) -> dict:
        return {"entry_node": self.entry_node, "modules": {
            node_id: {**artifact.model_dump(), "workspace_path": f"nodes/{node_id}.py"}
            for node_id, artifact in sorted(self.artifacts.items())
        }}

    def materialize(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "nodes").mkdir()
        (directory / "nodes/__init__.py").write_text("")
        for node_id, source in self.sources.items():
            (directory / f"nodes/{node_id}.py").write_text(source, encoding="utf-8")
        (directory / "entry.py").write_text(ENTRY_TEMPLATE.format(
            node_id=self.entry_node, symbol=self.artifacts[self.entry_node].public_symbol,
        ), encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(self.manifest(), indent=2) + "\n")


class ArtifactStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()

    def save_initial(self, node_id: str, contract: Contract, source: str, dependencies: list[Dependency] | None = None) -> Artifact:
        return self._save(node_id, contract, source, 1, dependencies or [])

    def save_next(self, previous: Artifact, contract: Contract, source: str, dependencies: list[Dependency] | None = None) -> Artifact:
        """Host version storage; policy code changes still require F4 repair authorization."""
        if self.load(previous.node_id, previous.version) != previous or self.latest_version(previous.node_id) != previous.version:
            raise SourceError("STALE_ARTIFACT", "New versions must follow the current stored artifact")
        return self._save(previous.node_id, contract, source, previous.version + 1,
                          previous.dependency_bindings if dependencies is None else dependencies)

    def _save(self, node_id: str, contract: Contract, source: str, version: int, dependencies: list[Dependency]) -> Artifact:
        TypeAdapter(Identifier).validate_python(node_id)
        try:
            tree = ast.parse(source)
            compile(tree, f"{node_id}.py", "exec")
        except (SyntaxError, ValueError) as error:
            raise SourceError("INVALID_SOURCE", str(error)) from error
        functions = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        public = [item for item in functions if not item.name.startswith("_")]
        expected = contract.signature
        if len(public) != 1 or public[0].name != expected.name or isinstance(public[0], ast.AsyncFunctionDef):
            raise SourceError("INTERFACE_MISMATCH", "Expected exactly the contracted public function")
        args = public[0].args
        names = [arg.arg for arg in args.posonlyargs + args.args]
        if names != [p.name for p in expected.parameters] or args.vararg or args.kwarg or args.kwonlyargs:
            raise SourceError("INTERFACE_MISMATCH", "Function parameters do not match the contract")

        referenced = {d.child_id: self.resolve(d) for d in dependencies}
        for child in referenced.values():
            self.bundle(child) # Validate the transitive closure before writing a candidate version.
        if len({d.alias for d in dependencies}) != len(dependencies) or len(referenced) != len(dependencies):
            raise SourceError("DUPLICATE_DEPENDENCY", "Dependency node IDs and aliases must be unique")
        inspect_links(source, dependencies, referenced)
        path = self.run_dir / "artifacts" / node_id / f"v{version}.py"
        if not path.resolve().is_relative_to(self.run_dir):
            raise SourceError("UNAUTHORIZED_PATH", "Artifact path leaves the run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise SourceError("ARTIFACT_EXISTS", "Initial source cannot overwrite a saved version")
        data = source.encode("utf-8")
        artifact = Artifact(
            node_id=node_id, version=version, module_path=str(path.relative_to(self.run_dir)),
            public_symbol=expected.name, source_hash=hashlib.sha256(data).hexdigest(),
            contract_version=contract.version, dependency_bindings=dependencies,
        )
        with path.open("xb") as stream:
            stream.write(data)
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            stream.write(artifact.model_dump_json(indent=2) + "\n")
        return artifact

    def latest_version(self, node_id: str) -> int:
        TypeAdapter(Identifier).validate_python(node_id)
        return max((int(p.stem[1:]) for p in (self.run_dir / "artifacts" / node_id).glob("v*.json")
                    if p.stem[1:].isdigit()), default=0)

    def load(self, node_id: str, version: int) -> Artifact:
        TypeAdapter(Identifier).validate_python(node_id)
        path = self.run_dir / "artifacts" / node_id / f"v{version}.json"
        if not path.resolve().is_relative_to(self.run_dir):
            raise SourceError("UNAUTHORIZED_PATH", "Metadata path leaves run directory")
        artifact = Artifact.model_validate_json(path.read_text())
        if (artifact.node_id, artifact.version, artifact.module_path) != (node_id, version, f"artifacts/{node_id}/v{version}.py"):
            raise SourceError("ARTIFACT_BINDING_MISMATCH", "Stored artifact identity does not match its location")
        return artifact

    def resolve(self, dependency: Dependency) -> Artifact:
        artifact = self.load(dependency.child_id, dependency.artifact_version)
        if artifact.source_hash != dependency.artifact_hash or self.latest_version(dependency.child_id) != dependency.artifact_version:
            raise SourceError("STALE_DEPENDENCY", "Dependency must select the current version and exact source hash")
        return artifact

    def bundle(self, entry: Artifact) -> Bundle:
        artifacts, sources, active = {}, {}, set()

        def visit(artifact):
            if artifact.node_id in active:
                raise SourceError("DEPENDENCY_CYCLE", "Artifact bundles must be acyclic")
            if artifact.node_id in artifacts:
                if artifacts[artifact.node_id] != artifact:
                    raise SourceError("DEPENDENCY_VERSION_CONFLICT", "One node cannot have two versions in one bundle")
                return
            active.add(artifact.node_id)
            path = (self.run_dir / artifact.module_path).resolve()
            if not path.is_relative_to(self.run_dir):
                raise SourceError("UNAUTHORIZED_PATH", "Source path leaves run directory")
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != artifact.source_hash:
                raise SourceError("ARTIFACT_BINDING_MISMATCH", "Stored source does not match its immutable artifact hash")
            children = {d.child_id: self.resolve(d) for d in artifact.dependency_bindings}
            inspect_links(source.decode("utf-8"), artifact.dependency_bindings, children)
            artifacts[artifact.node_id], sources[artifact.node_id] = artifact, source.decode("utf-8")
            for child in children.values():
                visit(child)
            active.remove(artifact.node_id)

        visit(entry)
        return Bundle(entry.node_id, artifacts, sources)
