"""Offline A2 function checks and small output interventions; no AST/PDG platform."""

import ast
from copy import deepcopy

from .artifacts import ArtifactStore, Bundle
from .evaluate import write_json


def audit_components(snapshot, definition, sandbox, measured_cost):
    """Named component checks are finite tests, not semantic responsibility proofs.

    No function is split or relabelled from reference code. Only the two public
    component interfaces, or the baseline's exact named private helpers, qualify.
    """
    if snapshot is None:
        return {"components": [], "checked": 0, "passed": 0, "output_insensitive": 0}
    from pathlib import Path
    from .schema import Artifact
    store = ArtifactStore(Path(snapshot["path"]))
    bundle = store.bundle(Artifact.model_validate(snapshot["artifact"]))
    root_source = bundle.sources[bundle.entry_node]
    suites = {s.get("contract_key"): s for s in definition["suites"] if s.get("contract_key") in {"a", "b"}}
    records = []
    for key, symbol in [("a", "transform"), ("b", "summarize")]:
        choices = [(n,symbol) for n,a in bundle.artifacts.items() if a.public_symbol == symbol]
        if not choices:
            choices = [(bundle.entry_node, "_"+symbol)] if any(isinstance(f,ast.FunctionDef) and f.name == "_"+symbol for f in ast.parse(root_source).body) else []
        if len(choices) != 1:
            records.append({"component": key, "status": "UNKNOWN", "reason": "NO_UNIQUE_DECLARED_INTERFACE"})
            continue
        node, name = choices[0]
        source = bundle.sources[node]
        probe_artifacts = dict(bundle.artifacts)
        probe_artifacts[node] = probe_artifacts[node].model_copy(update={"public_symbol": name})
        component_bundle = Bundle(node, probe_artifacts, bundle.sources)
        cases = []
        measured_cost["verify_calls"] += 1
        for case in suites[key]["cases"]:
            e = sandbox.run(source, name, case["args"], bundle=component_bundle)
            measured_cost["executed_cases"] += 1
            measured_cost["execution_seconds"] += e.elapsed_seconds
            cases.append({"case_id": case["case_id"], "pass": e.status == "RETURNED" and e.value == case["expected"] and not e.mutated_inputs,
                          "execution": e.model_dump()})
        # Override the component's output on a nonempty public input. Preserving
        # the root output here supports only finite output-insensitivity, not a
        # general claim that the parent reimplemented the child's responsibility.
        poison = [999999] if key == "a" else -999999
        tree = ast.parse(source)
        for f in tree.body:
            if isinstance(f, ast.FunctionDef) and f.name == name:
                f.body = [ast.Return(value=ast.parse(repr(poison), mode="eval").body)]
        changed = dict(bundle.sources)
        changed[node] = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        intervention = Bundle(bundle.entry_node, deepcopy(bundle.artifacts), changed)
        public = next(s for s in definition["suites"] if s["check_kind"] == "ROOT")["cases"][0]
        measured_cost["verify_calls"] += 2
        before = sandbox.run(root_source, bundle.artifacts[bundle.entry_node].public_symbol, public["args"], bundle=bundle)
        after = sandbox.run(changed[bundle.entry_node], bundle.artifacts[bundle.entry_node].public_symbol, public["args"], bundle=intervention)
        measured_cost["executed_cases"] += 2
        measured_cost["execution_seconds"] += before.elapsed_seconds + after.elapsed_seconds
        unchanged = before.status == "RETURNED" and before.value == public["expected"] and after.status == "RETURNED" and after.value == before.value
        records.append({"component": key, "node": node, "function": name, "status": "PASS" if all(c["pass"] for c in cases) else "FAIL",
                        "cases": cases, "intervention_source": changed[node], "output_insensitive_on_probe": unchanged,
                        "baseline": before.model_dump(), "intervention": after.model_dump()})
    result = {"components": records, "checked": sum(r["status"] != "UNKNOWN" for r in records),
              "passed": sum(r["status"] == "PASS" for r in records),
              "output_insensitive": sum(r.get("output_insensitive_on_probe", False) for r in records),
              "responsibility_redo": "UNKNOWN; output-insensitivity is a diagnostic, not a proof of recomputation"}
    write_json(Path(snapshot["path"]) / "component_audit.json", result)
    return result
