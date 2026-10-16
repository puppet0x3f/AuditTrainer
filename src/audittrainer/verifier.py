"""Host-owned, explicitly reviewed oracles and version-bound functional evidence."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, model_validator

from .sandbox import Sandbox
from .artifacts import ArtifactStore, inspect_links
from .schema import Artifact, Contract, ContractSpec, Evidence, Record, Text, Version


NO_APPLICABLE_SUITE = "NO_APPLICABLE_SUITE"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class OracleCase(Record):
    case_id: Text
    args: list[JsonValue]
    expected: JsonValue


class Suite(Record):
    suite_id: Text
    root_id: Text
    contract: ContractSpec
    contract_version: Version = 1
    check_kind: Literal["ROOT", "LOCAL"]
    visibility: Literal["VISIBLE", "PRIVATE"]
    oracle_origin: Text
    covered_obligations: list[Text]
    cases: list[OracleCase]
    comparison: Literal["exact_json", "apps_stdio_legacy", "stdio_tokens"] = "exact_json"

    def matches(self, actual, expected):
        if self.comparison == "apps_stdio_legacy":
            from .apps import compare_stdio
            return compare_stdio(actual, expected)
        if self.comparison == "stdio_tokens":
            return isinstance(actual, str) and isinstance(expected, str) and actual.split() == expected.split()
        return canonical(actual) == canonical(expected)

    @model_validator(mode="after")
    def check_cases(self):
        if self.suite_id == NO_APPLICABLE_SUITE:
            raise ValueError("Reserved missing-oracle suite ID")
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("Case IDs must be unique")
        if not set(self.covered_obligations) <= {o.id for o in self.contract.ensures}:
            raise ValueError("Suite coverage must reference its reviewed contract")
        if any(len(case.args) != len(self.contract.signature.parameters) for case in self.cases):
            raise ValueError("Oracle arguments must match the reviewed signature")
        # Reject non-finite numeric oracles before candidate execution.
        canonical(self.model_dump(mode="json"))
        return self

    def applies(self, root_id: str, contract: Contract) -> bool:
        spec = contract.model_dump(exclude={"contract_id", "version"})
        return self.root_id == root_id and self.contract_version == contract.version and self.contract.model_dump() == spec


class Verifier:
    def __init__(self, suites: list[Suite], sandbox: Sandbox | None = None):
        self.suites = {suite.suite_id: suite.model_copy(deep=True) for suite in suites}
        if len(self.suites) != len(suites):
            raise ValueError("Duplicate suite IDs")
        self.sandbox = sandbox or Sandbox()

    def available_suites(self, root_id: str, contract: Contract) -> list[str]:
        visible = [s.suite_id for s in self.suites.values() if s.visibility == "VISIBLE" and s.applies(root_id, contract)]
        return visible or [NO_APPLICABLE_SUITE]

    def _binding(self, run_dir: Path, root_id: str, contract: Contract, artifact: Artifact, suite_id: str):
        path = (run_dir / artifact.module_path).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Artifact path escapes run directory")
        source = path.read_bytes()
        suite = self.suites.get(suite_id)
        bundle, bundle_error = None, None
        if artifact.dependency_bindings:
            try:
                bundle = ArtifactStore(run_dir).bundle(artifact)
            except (OSError, ValueError) as error:
                bundle_error = str(error)
        binding = {
            "root_id": root_id, "contract_id": contract.contract_id, "contract_version": contract.version,
            "contract_hash": digest(contract.model_dump()),
            "artifact": artifact.model_dump(), "source_hash": hashlib.sha256(source).hexdigest(),
            "dependency_bundle_hash": digest(bundle.manifest() if bundle else [d.model_dump() for d in artifact.dependency_bindings]),
            "bundle": bundle.manifest() if bundle else None, "bundle_error": bundle_error,
            "bundle_implementation_hash": hashlib.sha256(Path(__file__).with_name("artifacts.py").read_bytes()).hexdigest(),
            "suite_id": suite_id, "suite_hash": digest(suite.model_dump(mode="json") if suite else None),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runner": self.sandbox.identity(),
        }
        return binding, source, bundle

    def is_current(self, evidence: Evidence, run_dir: Path, root_id: str, contract: Contract, artifact: Artifact) -> bool:
        try:
            binding, _, _ = self._binding(run_dir, root_id, contract, artifact, evidence.suite_id)
            return evidence.fingerprint == digest(binding) and binding["source_hash"] == artifact.source_hash and not binding["bundle_error"]
        except (OSError, ValueError):
            return False

    def verify(
        self, run_dir: Path, root_id: str, contract: Contract, artifact: Artifact,
        suite_id: str, evidence_id: str, *, allow_private: bool = False,
    ) -> Evidence:
        suite = self.suites.get(suite_id)
        if suite and suite.visibility == "PRIVATE" and not allow_private:
            raise ValueError("Private suites are host-only; unavailable to online verification")
        binding, source, bundle = self._binding(run_dir, root_id, contract, artifact, suite_id)
        status, reason, responsibility = "UNKNOWN", "NO_APPLICABLE_ORACLE", "UNKNOWN"
        results = []
        if (binding["source_hash"] != artifact.source_hash or artifact.contract_version != contract.version
                or artifact.public_symbol != contract.signature.name):
            status, reason = "ERROR", "ARTIFACT_BINDING_MISMATCH"
        elif binding["bundle_error"]:
            status, reason = "ERROR", "DEPENDENCY_BINDING_MISMATCH"
        elif suite and not suite.applies(root_id, contract):
            status, reason = "ERROR", "ORACLE_BINDING_MISMATCH"
        elif suite is not None:
            for case in suite.cases:
                execution = self.sandbox.run(source.decode("utf-8"), artifact.public_symbol, case.args, bundle=bundle)
                result = {"case_id": case.case_id, "args": case.args, "expected": case.expected,
                          "execution": execution.model_dump(mode="json")}
                if execution.mutated_inputs and not contract.frame.mutate_inputs:
                    responsibility = "VIOLATION"
                    result["responsibility_violation"] = "frame.mutate_inputs"
                if bundle:
                    unauthorized = [call for call in execution.call_counts if call["consumer_node"] not in bundle.artifacts
                                    or call["child_node"] not in {d.child_id for d in bundle.artifacts[call["consumer_node"]].dependency_bindings}]
                    if unauthorized:
                        responsibility = "VIOLATION"
                        result["unauthorized_calls"] = unauthorized
                if execution.status != "RETURNED":
                    status, reason = execution.status, execution.reason
                elif not suite.matches(execution.value, case.expected):
                    status, reason = "FAIL", "WRONG_OUTPUT"
                else:
                    status, reason = "PASS", "MATCH"
                result.update(status=status, reason=reason)
                results.append(result)
                if status != "PASS":
                    break
            if not results:
                status, reason = "UNKNOWN", "EMPTY_SUITE"
            elif status == "PASS":
                if {o.id for o in contract.ensures} <= set(suite.covered_obligations):
                    reason = "ALL_CASES_PASSED"
                else:
                    status, reason = "UNKNOWN", "INCOMPLETE_OBLIGATION_COVERAGE"
        usage = []
        if bundle:
            usage = inspect_links(source.decode("utf-8"), artifact.dependency_bindings, bundle.artifacts)
            complete = bool(results) and all(r["execution"]["call_trace_complete"] for r in results)
            for item in usage:
                count = sum(call["count"] for r in results for call in r["execution"]["call_counts"]
                            if call["consumer_node"] == artifact.node_id and call["child_node"] == item["child_id"])
                item.update(call_count=count, called=True if count else (False if complete else None),
                            call_trace_complete=complete, dependency_supported="UNKNOWN")
        evidence = Evidence(
            evidence_id=evidence_id, check_kind=suite.check_kind if suite else "UNBOUND",
            target_ref=artifact.module_path, suite_id=suite_id, suite_hash=binding["suite_hash"],
            status=status, coverage=suite.covered_obligations if suite else [],
            visibility=suite.visibility if suite else "VISIBLE", fingerprint=digest(binding), binding=binding,
            responsibility_status=responsibility,
            details={"reason": reason, "oracle_origin": suite.oracle_origin if suite else None,
                     "cases": results, "executed_cases": len(results),
                     "dependency_usage": usage,
                     "elapsed_seconds": sum(r["execution"]["elapsed_seconds"] for r in results)},
        )
        directory = run_dir / "evidence"
        directory.mkdir(exist_ok=True)
        # IDs are generated by the host, never supplied by a policy.
        if Path(evidence_id).name != evidence_id or evidence_id in {".", ".."}:
            raise ValueError("Invalid evidence ID")
        with (directory / f"{evidence_id}.json").open("x", encoding="utf-8") as stream:
            stream.write(evidence.model_dump_json(indent=2) + "\n")
        return evidence
