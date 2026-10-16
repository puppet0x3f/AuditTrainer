import json
import textwrap

import pytest
from pydantic import ValidationError

from audittrainer.artifacts import ArtifactStore
from audittrainer.sandbox import ExecutionLimits
from audittrainer.schema import Contract, ContractSpec, Dependency
from audittrainer.verifier import NO_APPLICABLE_SUITE, OracleCase, Suite, Verifier, digest


def prepare(runtime):
    runtime.step()
    runtime.step()
    return runtime


def check(runtime, suite_id="FX01_ROOT_VISIBLE", evidence_id="E_TEST"):
    return runtime.verifier.verify(runtime.run_dir, runtime.task.root_id, runtime.node.incoming_contract,
                                   runtime.artifact, suite_id, evidence_id)


@pytest.mark.parametrize("kind,reason", [("missing", "NO_APPLICABLE_ORACLE"), ("empty", "EMPTY_SUITE"),
                                        ("incomplete", "INCOMPLETE_OBLIGATION_COVERAGE")])
def test_missing_empty_and_incomplete_oracles_never_pass(make_f2_runtime, reviewed_suite, kind, reason):
    suites = [] if kind == "missing" else [reviewed_suite.model_copy(update={
        "cases": [] if kind == "empty" else reviewed_suite.cases,
        "covered_obligations": [] if kind == "incomplete" else reviewed_suite.covered_obligations,
    })]
    runtime = prepare(make_f2_runtime(suites=suites))
    evidence = check(runtime, NO_APPLICABLE_SUITE if kind == "missing" else reviewed_suite.suite_id)
    assert (evidence.status, evidence.details["reason"]) == ("UNKNOWN", reason)
    assert evidence.details["executed_cases"] == (len(reviewed_suite.cases) if kind == "incomplete" else 0)


def test_same_signature_different_semantic_contract_is_not_bound(make_f2_runtime, reviewed_suite):
    data = reviewed_suite.contract.model_dump()
    data["ensures"][0]["text"] = "返回输入最大值"
    wrong_suite = reviewed_suite.model_copy(update={"contract": ContractSpec.model_validate(data)})
    runtime = prepare(make_f2_runtime(suites=[wrong_suite]))
    assert runtime.build_observation()["available_suites"] == [NO_APPLICABLE_SUITE]
    evidence = check(runtime)
    assert (evidence.status, evidence.details["reason"]) == ("ERROR", "ORACLE_BINDING_MISMATCH")
    assert evidence.details["executed_cases"] == 0


def test_generic_local_callable_and_mutation_contract(tmp_path):
    contract = Contract(
        contract_id="C_LOCAL_1", version=1,
        signature={"name": "prefix_sums", "parameters": [{"name": "xs", "type": "list[int]"}], "returns": "list[int]"},
        requires=["xs contains integers"], ensures=[{"id": "O_PREFIX", "text": "zero followed by cumulative sums"}],
        owns=["prefix_sums"], uses=[], must_not=[],
        frame={"public_symbol": "prefix_sums", "mutate_inputs": False, "external_io": False},
    )
    spec = ContractSpec.model_validate(contract.model_dump(exclude={"contract_id", "version"}))
    suite = Suite(suite_id="PREFIX_LOCAL", root_id="prefix_fixture", contract=spec, check_kind="LOCAL", visibility="VISIBLE",
                  oracle_origin="Hand-reviewed prefix sums", covered_obligations=["O_PREFIX"],
                  cases=[OracleCase(case_id="signed", args=[[2, -1, 4]], expected=[0, 2, 1, 5])])
    source = "def prefix_sums(xs):\n    result = [0]\n    for x in xs: result.append(result[-1] + x)\n    return result\n"
    for name, candidate in [("valid", source),
                             ("mutates", source.replace("    return result", "    xs.clear()\n    return result"))]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        artifact = ArtifactStore(run_dir).save_initial("N0", contract, candidate)
        evidence = Verifier([suite]).verify(run_dir, "prefix_fixture", contract, artifact, suite.suite_id, "E1")
        assert evidence.check_kind == "LOCAL" and evidence.status == "PASS"
        assert evidence.responsibility_status == ("VIOLATION" if name == "mutates" else "UNKNOWN")


def test_actual_boolean_cannot_equal_numeric_oracle(make_f2_runtime, reviewed_suite):
    suite = reviewed_suite.model_copy(update={"cases": [OracleCase(case_id="type", args=[""], expected=1)]})
    runtime = prepare(make_f2_runtime(suites=[suite], source="def solve(text): return True"))
    evidence = check(runtime)
    assert (evidence.status, evidence.details["reason"]) == ("FAIL", "WRONG_OUTPUT")


def test_expected_answer_is_not_mounted_with_candidate(make_f2_runtime, reviewed_suite):
    secret = "ORACLE_SECRET_97591676"
    source = textwrap.dedent("""
        import re
        from pathlib import Path
        def solve(text):
            for p in Path('/work').iterdir():
                if p.is_file():
                    matches = re.findall(r'ORACLE_SECRET_[0-9]+', p.read_text())
                    if matches:
                        return matches[0]
            return 'not visible'
    """)
    suite = reviewed_suite.model_copy(update={"cases": [OracleCase(case_id="secret", args=[""], expected=secret)]})
    runtime = prepare(make_f2_runtime(suites=[suite], source=source))
    evidence = check(runtime)
    assert evidence.status == "FAIL"
    assert evidence.details["cases"][0]["execution"]["value"] == "not visible"


def test_evidence_fingerprint_covers_source_contract_suite_dependencies_and_execution(make_f2_runtime):
    runtime = prepare(make_f2_runtime())
    evidence = check(runtime)
    verifier, contract, artifact = runtime.verifier, runtime.node.incoming_contract, runtime.artifact

    def current(c=contract, a=artifact):
        return verifier.is_current(evidence, runtime.run_dir, runtime.task.root_id, c, a)

    assert evidence.status == "PASS" and current()
    assert evidence.fingerprint == digest(evidence.binding)
    assert json.loads((runtime.run_dir / "evidence/E_TEST.json").read_text())["fingerprint"] == evidence.fingerprint
    assert not current(c=contract.model_copy(update={"version": 2}))
    assert not current(c=contract.model_copy(update={"requires": ["different input domain"]}))
    assert not current(a=artifact.model_copy(update={"version": 2}))
    dependency = Dependency(child_id="N1", artifact_version=1, artifact_hash="a" * 64, alias="child")
    assert not current(a=artifact.model_copy(update={"dependency_bindings": [dependency]}))
    original_suite = verifier.suites[evidence.suite_id]
    verifier.suites[evidence.suite_id] = original_suite.model_copy(update={"cases": []})
    assert not current()
    verifier.suites[evidence.suite_id] = original_suite
    original_limits = verifier.sandbox.limits
    verifier.sandbox.limits = ExecutionLimits(per_case_timeout_seconds=3.0)
    assert not current()
    verifier.sandbox.limits = original_limits
    assert current()
    (runtime.run_dir / artifact.module_path).write_text("def solve(text): return 'stale'")
    assert not current()
    stale = check(runtime, evidence_id="E_STALE")
    assert (stale.status, stale.details["reason"]) == ("ERROR", "ARTIFACT_BINDING_MISMATCH")


def test_suite_rejects_invalid_host_oracle_binding(reviewed_suite):
    data = reviewed_suite.model_dump()
    data["cases"][0]["args"] = []
    with pytest.raises(ValidationError, match="arguments"):
        Suite.model_validate(data)
    data = reviewed_suite.model_dump()
    data["covered_obligations"] = ["invented"]
    with pytest.raises(ValidationError, match="coverage"):
        Suite.model_validate(data)
