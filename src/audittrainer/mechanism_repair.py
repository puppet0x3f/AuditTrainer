"""P3: identical controlled failures, frozen repair model, visible-only selection."""

from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import time

from .artifacts import ArtifactStore
from .dataset import file_hash
from .evaluate import capture_program, load_task, score_program, write_json
from .mechanism import Session, atomic_args, cost, fixed_action, paired
from .runtime import Limits, RepairRuntime
from .schema import Artifact, Contract, Dependency, RepairTicket


ROOT_SOURCE = """import json
from nodes.N0_p1_a import transform as stage_a
from nodes.N0_p1_b import summarize as stage_b

def solve(text):
    items = json.loads(text)
    transformed = stage_a(items)
    result = stage_b(transformed)
    return json.dumps(result, separators=(',', ':'))
"""


class ReferencePolicy:
    """Authored setup only, explicitly excluded from model repair results."""
    def __init__(self, definition, reference, faulty=False):
        self.definition, self.reference, self.faulty = definition, reference, faulty

    def act(self, o):
        if o["phase"] == "NEW":
            args = atomic_args(o["incoming_contract"])
            if o["node_id"] == "N0":
                args.update(mode="SPLIT", children=[{"local_key": k, "contract": self.definition["local_contracts"][k]} for k in ("a", "b")],
                    obligation_map={"PARSE": ["self"], "FORMAT": ["self"], "TRANSFORM": ["child:a"], "SUMMARIZE": ["child:b"]},
                    dataflow=[{"from_child": "a", "from_output": "return", "to_child": "b", "to_parameter": "items"}])
            return fixed_action("plan", args)
        if o["phase"] == "IMPLEMENT":
            source = ROOT_SOURCE if o["node_id"] == "N0" else self.reference[o["node_id"][-1]]
            if self.faulty and o["node_id"] == self.reference["fault_node"]:
                if self.reference["fault_kind"] == "child_value":
                    symbol = o["incoming_contract"]["signature"]["name"]
                    value = "[999999]" if symbol == "transform" else "-999999"
                    source = f"def {symbol}(items):\n    return {value}\n"
                elif self.reference["fault_kind"] == "composition_bypass":
                    source = source.replace("stage_b(transformed)", "stage_b(items)")
                else:
                    source = source.replace("json.dumps(result, separators=(',', ':'))", "json.dumps(result, separators=(',', ':')) + '\\n'")
            return fixed_action("write_code", {"source": source, "contract_version": 1, "declared_dependencies": o["approved_dependencies"]})
        if o["phase"] == "RESOLVE_CHILD":
            child = o["returned_child"]["result"]
            artifact = child["artifact"]
            return fixed_action("consume", {"child_id": artifact["node_id"], "artifact_version": artifact["version"],
                "artifact_hash": artifact["source_hash"], "alias": "stage_"+artifact["node_id"][-1],
                "evidence_ids": [child["evidence"]["evidence_id"]], "accept_unverified": child["functional_status"] != "PASS"})
        if o["phase"] == "READY_VERIFY":
            return fixed_action("verify", {"target_ref": o["current_artifact"]["module_path"], "suite_id": o["available_suites"][0]})
        return fixed_action("finish", {"artifact_version": o["current_artifact"]["version"]})


def compact_evidence(evidence):
    return {"evidence_id": evidence.evidence_id, "target_ref": evidence.target_ref, "status": evidence.status,
            "visibility": "VISIBLE", "coverage": evidence.coverage, "reason": evidence.details["reason"],
            "cases": [{"case_id": c["case_id"], "args": c["args"], "expected": c["expected"],
                "status": c["status"], "reason": c["reason"], "actual": c["execution"]["value"],
                "stderr": c["execution"]["stderr"]} for c in evidence.details["cases"]]}


def failure_snapshot(row, tests, framework, config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    definition, task, contract, online, full = load_task(row, tests, framework)
    if file_hash(row["reference_path"]) != row["reference_hash"]:
        raise ValueError("Reference/fault specification differs from preregistered hash")
    reference = json.loads(Path(row["reference_path"]).read_text())
    setup = []
    for faulty in (False, True):
        name = "failure" if faulty else "reference"
        runtime = RepairRuntime(task, contract, ReferencePolicy(definition, reference, faulty), directory / name,
            limits=Limits(max_depth=1, max_nodes=3, max_children=2), verifier=online,
            repair_config=framework["repair"], implementation_stage="A2_CONTROLLED_SETUP")
        result = runtime.run()
        if result["functional_status"] != ("FAIL" if faulty else "PASS"):
            raise ValueError(f"Fixed fault construction did not produce its required visible outcome: {task.root_id}, {name}")
        snapshot = capture_program(runtime, directory / (name + "_snapshot"))
        setup.append({"kind": name, "usage": runtime.budget.model_dump(), "snapshot": snapshot})
    contexts = runtime.contexts
    bundle = runtime.store.bundle(contexts["N0"].artifact)
    contracts = {n: c.node.incoming_contract for n,c in contexts.items()}
    evidence = {n: c.node.last_evidence for n,c in contexts.items()}
    failed = [n for n,e in evidence.items() if e.status == "FAIL"]
    candidates = sorted(failed, key=lambda n: (n == "N0", n))[:3]
    public = {"root_id": task.root_id, "spec": task.spec,
              "sources": {f"nodes/{n}.py": s for n,s in bundle.sources.items()},
              "visible_evidence": {n: compact_evidence(e) for n,e in evidence.items()}}
    write_json(directory / "public_failure.json", public)
    write_json(directory / "setup.json", {"source": "controlled authored injection", "root_id": task.root_id,
        "fault_node": reference["fault_node"], "fault_kind": reference["fault_kind"], "runs": setup,
        "note": "Reference construction and fault labels never enter locator/repair prompts."})
    return {"public": public, "contracts": contracts, "bundle": bundle, "evidence": evidence, "online": online, "full": full,
            "fault_node": reference["fault_node"], "fault_kind": reference["fault_kind"], "snapshot": snapshot,
            "directory": directory, "candidates": candidates, "indexes": runtime.repair_manager.indexes()}


def make_candidate(failure, replacements, directory, *, ticket=None):
    """All arms share syntax/interface/import checks and immutable version storage."""
    sources = {f"nodes/{n}.py": s for n,s in failure["bundle"].sources.items()}
    if not isinstance(replacements, dict) or not replacements or not set(replacements) <= sources.keys():
        raise ValueError("INVALID_SOURCE_BOUNDARY")
    if any(not isinstance(v, str) for v in replacements.values()):
        raise ValueError("SOURCE_MUST_BE_TEXT")
    candidate_sources = {**sources, **replacements}
    if ticket:
        if not set(replacements) <= set(ticket.editable_files):
            raise ValueError("FROZEN_COMPLEMENT_VIOLATION")
        for path, expected in ticket.frozen_hashes.items():
            if hashlib.sha256(candidate_sources[path].encode()).hexdigest() != expected:
                raise ValueError("FROZEN_COMPLEMENT_VIOLATION")
    store = ArtifactStore(directory)
    artifacts = {}
    # The fixed three-node dependency graph is unchanged; all arms rebind its
    # original edges identically when materializing the independently proposed patch.
    for node in ("N0_p1_a", "N0_p1_b", "N0"):
        original = failure["bundle"].artifacts[node]
        deps = [Dependency(child_id=d.child_id, artifact_version=artifacts[d.child_id].version,
                artifact_hash=artifacts[d.child_id].source_hash, alias=d.alias) for d in original.dependency_bindings]
        artifacts[node] = store.save_initial(node, failure["contracts"][node], candidate_sources[f"nodes/{node}.py"], deps)
    root = artifacts["N0"]
    snapshot = {"path": str(directory), "root_id": failure["public"]["root_id"], "artifact": root.model_dump(),
                "contract": failure["contracts"]["N0"].model_dump(), "bundle": store.bundle(root).manifest()}
    write_json(directory / "snapshot.json", snapshot)
    return snapshot, artifacts, candidate_sources


def verify_all(failure, snapshot, artifacts, measured_cost, max_calls):
    evidence = {}
    for node, artifact in artifacts.items():
        contract = failure["contracts"][node]
        suite = failure["online"].available_suites(failure["public"]["root_id"], contract)[0]
        if measured_cost["verify_calls"] >= max_calls:
            raise ValueError("VERIFY_BUDGET")
        measured_cost["verify_calls"] += 1
        e = failure["online"].verify(Path(snapshot["path"]), failure["public"]["root_id"], contract, artifact, suite,
            f"P3_{measured_cost['verify_calls']:03d}")
        measured_cost["executed_cases"] += e.details["executed_cases"]
        measured_cost["execution_seconds"] += e.details["elapsed_seconds"]
        evidence[node] = e
    return evidence


def regression_count(baseline, evidence):
    passing = {(n,c["case_id"]) for n,e in baseline.items() for c in e.details["cases"] if c["status"] == "PASS"}
    return sum((n,c["case_id"]) in passing and c["status"] != "PASS" for n,e in evidence.items() for c in e.details["cases"])


def repair_arm(failure, config, model, arm, directory):
    session = Session(model, config, directory / "model_calls")
    record = {"root_id": failure["public"]["root_id"], "arm": arm, "status": "COMPLETED", "error": None,
              "fault_source": "CONTROLLED_INJECTION", "fault_kind": failure["fault_kind"], "fault_node": failure["fault_node"],
              "located": False, "candidate_hit": False, "abstained": False, "boundary_files": None, "boundary_lines": None,
              "authorized_files": None,
              "changed_files": 0, "changed_lines": 0, "outside_declared_boundary_changes": 0,
              "freeze_violations": 0, "regressions": 0, "attempts": [], "visible_pass": False, "private_pass": False,
              "online_cost": session.cost, "scoring_cost": cost(), "wall_seconds": 0.0,
              "failure_snapshot": str(failure["directory"] / "public_failure.json")}
    started = time.monotonic()
    selected, selected_sources = failure["snapshot"], failure["public"]["sources"]
    try:
        payload = deepcopy(failure["public"])
        nodes = list(failure["bundle"].artifacts)
        if arm != "R0":
            nodes = failure["candidates"]
            payload.update(responsibility_indexes=failure["indexes"], candidate_nodes=nodes)
        count = 3 if arm == "R2" else 1
        answer, error = session.ask(f'Return {{"candidates": ["node_id", ...], "abstain": false, "rationale": "short reason"}}. Propose at most {count} repair owner nodes from {nodes}. Use only the visible feedback. If evidence is insufficient, abstain. Do not write code yet.', payload)
        candidates = answer.get("candidates", []) if isinstance(answer, dict) else []
        if (not isinstance(candidates, list) or any(not isinstance(n,str) or n not in nodes for n in candidates)
                or len(candidates) > count or len(set(candidates)) != len(candidates)):
            candidates, error = [], "INVALID_CANDIDATES"
        record.update(localization_error=error, candidates=candidates,
            located=bool(candidates and candidates[0] == failure["fault_node"]), candidate_hit=failure["fault_node"] in candidates)
        if not candidates or answer.get("abstain") is not False:
            record["abstained"] = True
        else:
            preferred = f"nodes/{candidates[0]}.py"
            record["boundary_files"] = 1
            record["authorized_files"] = 1 if arm == "R2" else len(failure["public"]["sources"])
            record["boundary_lines"] = len(failure["public"]["sources"][preferred].splitlines())
            # R2 checks that an unchanged independent replay reproduces the same
            # visible evidence. It is paid from the same verification allowance.
            if arm == "R2":
                replay, artifacts, _ = make_candidate(failure, failure["public"]["sources"], directory / "replay")
                replayed = verify_all(failure, replay, artifacts, session.cost, config["budget"]["max_verify_calls"])
                if any([(c["case_id"], c["status"], c["execution"]["value"]) for c in replayed[n].details["cases"]] !=
                       [(c["case_id"], c["status"], c["execution"]["value"]) for c in e.details["cases"]] for n,e in failure["evidence"].items()):
                    raise ValueError("BASELINE_REPLAY_MISMATCH")
            feedback = None
            for attempt in range(config["budget"]["max_patch_attempts"]):
                if not session.available():
                    break
                target = candidates[min(attempt, len(candidates)-1)] if arm == "R2" else candidates[0]
                path = f"nodes/{target}.py"
                editable = [path] if arm == "R2" else list(failure["public"]["sources"])
                ticket = None
                if arm == "R2":
                    ticket = RepairTicket(ticket_id=f"A2_{attempt}", snapshot_id=record["failure_snapshot"],
                        target={"node_id": target}, boundary_kind="OWNER_MODULE", editable_files=editable, editable_plan_nodes=[],
                        rebind_edges=[], revalidate_nodes=list(failure["bundle"].artifacts),
                        frozen_hashes={p: hashlib.sha256(s.encode()).hexdigest() for p,s in failure["public"]["sources"].items() if p not in editable},
                        evidence_ids=[e.evidence_id for e in failure["evidence"].values()], remaining_attempts=config["budget"]["max_patch_attempts"]-attempt)
                    write_json(directory / f"ticket_{attempt}.json", ticket.model_dump())
                request = {**failure["public"], "preferred_target": path, "editable_files": editable,
                           "previous_attempt_feedback": feedback, "contract_signatures": {n: c.signature.model_dump() for n,c in failure["contracts"].items()}}
                patch, error = session.ask('Return {"replacements": {"nodes/ID.py": "complete Python module source"}, "abstain": false}. Repair the fixed failed program. Keep all signatures and imports. Use only editable_files. Each proposal starts from the supplied original snapshot. Prefer the declared target; other permitted files may change if necessary. Do not change contracts or tests.', request)
                item = {"attempt": attempt+1, "target": target, "status": "REJECTED", "reason": error, "regressions": 0}
                record["attempts"].append(item)
                if not isinstance(patch, dict) or patch.get("abstain") is not False:
                    record["abstained"] = True
                    break
                try:
                    candidate, artifacts, sources = make_candidate(failure, patch.get("replacements"), directory / f"candidate_{attempt}", ticket=ticket)
                    checked = verify_all(failure, candidate, artifacts, session.cost, config["budget"]["max_verify_calls"])
                    regressions = regression_count(failure["evidence"], checked)
                    item["regressions"] = regressions
                    record["regressions"] += regressions
                    feedback = {n: compact_evidence(e) for n,e in checked.items()}
                    item.update(reason="VISIBLE_ALL_PASS" if all(e.status == "PASS" for e in checked.values()) else "VISIBLE_FAILED", evidence=feedback)
                    if all(e.status == "PASS" for e in checked.values()) and not regressions:
                        selected, selected_sources = candidate, sources
                        record["visible_pass"] = True
                        item["status"] = "ACCEPTED"
                        break
                except Exception as error:
                    item["reason"] = str(error)
                    feedback = {"protocol_error": str(error)}
                    if str(error) == "FROZEN_COMPLEMENT_VIOLATION":
                        record["freeze_violations"] += 1
            for path, source in selected_sources.items():
                original = failure["public"]["sources"][path]
                if source != original:
                    record["changed_files"] += 1
                    record["outside_declared_boundary_changes"] += path != preferred
                    record["changed_lines"] += sum(line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                        for line in difflib.unified_diff(original.splitlines(), source.splitlines()))
        # The selected source is written before private tests can execute.
        selected, _, _ = make_candidate(failure, selected_sources, directory / "selected")
        write_json(directory / "selection.json", {"snapshot": selected, "sources": selected_sources,
            "rule": config["selection"], "private_feedback_used": False})
        record["scores"], _ = score_program(selected, failure["full"], record["root_id"], record["scoring_cost"])
        record["private_pass"] = record["scores"]["private"]["status"] == "PASS"
    except Exception as error:
        record.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    record["wall_seconds"] = time.monotonic() - started
    write_json(directory / "record.json", record)
    return record


def repair_summary(records, arms):
    result = {}
    for arm in arms:
        chosen = [r for r in records if r["arm"] == arm]
        successes = sum(r["private_pass"] for r in chosen)
        result[arm] = {"roots": len(chosen), **{k: sum(r[k] for r in chosen) for k in
            ("located", "candidate_hit", "abstained", "visible_pass", "private_pass", "changed_files", "changed_lines",
             "outside_declared_boundary_changes", "freeze_violations", "regressions")},
            "errors": sum(r["status"] == "ERROR" for r in chosen),
            "declared_boundary_files": [r["boundary_files"] for r in chosen],
            "authorized_files": [r["authorized_files"] for r in chosen],
            "declared_boundary_lines": [r["boundary_lines"] for r in chosen],
            "patch_attempts": sum(len(r["attempts"]) for r in chosen),
            "online_cost": {k: sum(r["online_cost"][k] for r in chosen) for k in cost()},
            "scoring_cost": {k: sum(r["scoring_cost"][k] for r in chosen) for k in cost()},
            "tokens_per_private_success": sum(r["online_cost"]["input_tokens"]+r["online_cost"]["output_tokens"] for r in chosen) / successes if successes else None}
    return {"stage": "A2", "scope": "DEV_MECHANISM", "fault_source": "CONTROLLED_INJECTION", "arms": result,
            "paired_private": paired(records, arms, lambda r: r["private_pass"])}
