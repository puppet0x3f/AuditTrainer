from copy import deepcopy
import json
from pathlib import Path

import pytest

from audittrainer.policy import ScriptedPolicy, ScriptStep
from audittrainer.runtime import AtomicRuntime, RecursiveRuntime, RepairRuntime, Limits
from audittrainer.schema import ContractSpec, RootTask
from audittrainer.verifier import Suite, Verifier


@pytest.fixture
def fixture_data():
    path = Path(__file__).resolve().parents[1] / "data/fixtures/fx01_integer_sum.json"
    return json.loads(path.read_text())


@pytest.fixture
def make_runtime(tmp_path, fixture_data):
    counter = 0

    def make(script=None, limits=None, policy=None):
        nonlocal counter
        counter += 1
        steps = deepcopy(fixture_data["script"] if script is None else script)
        if policy is None:
            policy = ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(s)) for s in steps])
        return AtomicRuntime(
            RootTask.model_validate(fixture_data["task"]),
            ContractSpec.model_validate(fixture_data["contract"]),
            policy, tmp_path / f"run_{counter}", limits=limits or Limits(),
        )

    return make


@pytest.fixture
def recursive_fixture():
    return json.loads((Path(__file__).resolve().parents[1] / "data/fixtures/fx02_range_sums.json").read_text())


@pytest.fixture
def make_f3_runtime(tmp_path, recursive_fixture):
    counter = 0

    def make(fixture=None, limits=None):
        nonlocal counter
        counter += 1
        data = deepcopy(recursive_fixture if fixture is None else fixture)
        root = ContractSpec.model_validate(data["contract"])
        suites = []
        for raw in data["suites"]:
            definition = dict(raw)
            key = definition.pop("contract_key", None)
            contract = ContractSpec.model_validate(data["local_contracts"][key]) if key else root
            suites.append(Suite(root_id=data["task"]["root_id"], contract=contract, **definition))
        steps = [ScriptStep.model_validate_json(json.dumps(step)) for step in data["script"]]
        return RecursiveRuntime(RootTask.model_validate(data["task"]), root, ScriptedPolicy(steps),
                                tmp_path / f"f3_{counter}", limits=limits, verifier=Verifier(suites))

    return make


@pytest.fixture
def repair_fixture():
    return json.loads((Path(__file__).resolve().parents[1] / "data/fixtures/fx03_bounded_repair.json").read_text())


@pytest.fixture
def make_f4_runtime(tmp_path, repair_fixture):
    counter = 0

    def make(fixture=None, limits=None):
        nonlocal counter
        counter += 1
        data = deepcopy(repair_fixture if fixture is None else fixture)
        contract = ContractSpec.model_validate(data["contract"])
        suites = []
        for raw in data["suites"]:
            definition = dict(raw)
            key = definition.pop("contract_key", None)
            reviewed = ContractSpec.model_validate(data["local_contracts"][key]) if key else contract
            suites.append(Suite(root_id=data["task"]["root_id"], contract=reviewed, **definition))
        return RepairRuntime(RootTask.model_validate(data["task"]), contract,
                             ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(s)) for s in data["script"]]),
                             tmp_path / f"f4_{counter}", limits=limits, verifier=Verifier(suites))

    return make


@pytest.fixture
def reviewed_suite(fixture_data):
    return Suite(contract=ContractSpec.model_validate(fixture_data["contract"]),
                 root_id=fixture_data["task"]["root_id"], **fixture_data["suites"][0])


@pytest.fixture
def make_f2_runtime(tmp_path, fixture_data, reviewed_suite):
    counter = 0

    def make(script=None, source=None, suites=None, sandbox=None, limits=None):
        nonlocal counter
        counter += 1
        steps = deepcopy(fixture_data["script"] + fixture_data["completion_script"] if script is None else script)
        if source is not None:
            steps[1]["output"]["args"]["source"] = source
        policy = ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(s)) for s in steps])
        return AtomicRuntime(
            RootTask.model_validate(fixture_data["task"]),
            ContractSpec.model_validate(fixture_data["contract"]), policy,
            tmp_path / f"f2_{counter}", limits=limits,
            verifier=Verifier([reviewed_suite] if suites is None else suites, sandbox),
        )

    return make
