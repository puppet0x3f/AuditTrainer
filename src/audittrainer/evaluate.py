"""One evaluator for scripted/base/SFT runs, with frozen first/final programs."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import yaml

from .artifacts import ArtifactStore
from .dataset import file_hash, load_root_manifest, read_jsonl
from .policy import LLMPolicy, ScriptedPolicy, ScriptStep
from .runtime import Budget, Limits, RepairRuntime
from .schema import Artifact, Contract, ContractSpec, RootTask
from .verifier import Verifier


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def capture_program(runtime, path):
    """Copy the exact transitive closure before later child versions can replace it."""
    context = runtime.contexts["N0"]
    if context.artifact is None:
        return None
    bundle = runtime.store.bundle(context.artifact)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for node, artifact in bundle.artifacts.items():
        source = path / artifact.module_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(bundle.sources[node])
        source.with_suffix(".json").write_text(artifact.model_dump_json(indent=2) + "\n")
    bundle.materialize(path / "program")
    snapshot = {"path": str(path), "root_id": runtime.task.root_id,
                "artifact": context.artifact.model_dump(mode="json"),
                "contract": context.node.incoming_contract.model_dump(mode="json"),
                "bundle": bundle.manifest(), "captured_usage": runtime.budget.model_dump(),
                "capture_event": runtime.trace.count}
    write_json(path / "snapshot.json", snapshot)
    return snapshot


def score_program(snapshot, verifier, root_id, cost=None):
    scores = {}
    if cost is None:
        cost = {"verify_calls": 0, "executed_cases": 0, "execution_seconds": 0.0}
    for visibility in ("VISIBLE", "PRIVATE"):
        group = [s for s in verifier.suites.values() if s.visibility == visibility and s.check_kind == "ROOT"]
        status = "NO_PROGRAM" if snapshot is None else "UNKNOWN"
        reason = "NO_PROGRAM" if snapshot is None else "NO_APPLICABLE_SUITE"
        evidence = []
        if snapshot is not None:
            contract = Contract.model_validate(snapshot["contract"])
            artifact = Artifact.model_validate(snapshot["artifact"])
            group = [s for s in group if s.applies(root_id, contract)]
            for suite in group:
                cost["verify_calls"] += 1
                result = verifier.verify(Path(snapshot["path"]), root_id, contract, artifact,
                    suite.suite_id, f"SCORE_{visibility}_{cost['verify_calls']}", allow_private=True)
                evidence.append({"suite_id": suite.suite_id, "status": result.status,
                    "reason": result.details["reason"], "responsibility_status": result.responsibility_status,
                    "evidence_path": str(Path(snapshot["path"]) / "evidence" / f"{result.evidence_id}.json")})
                cost["executed_cases"] += result.details["executed_cases"]
                cost["execution_seconds"] += result.details["elapsed_seconds"]
            if evidence:
                status = next((v for v in ("ERROR", "FAIL", "UNKNOWN") if any(e["status"] == v for e in evidence)), "PASS")
                reason = "ALL_SUITES_PASSED" if status == "PASS" else next(e["reason"] for e in evidence if e["status"] == status)
        scores[visibility.lower()] = {"status": status, "reason": reason, "suites": evidence}
    return scores, cost


def load_task(row, tests, framework):
    from .cli import fixture_verifier
    if row.get("io_type") not in {"stdin", "solve_text"}:
        raise ValueError("UNSUPPORTED_IO_TYPE")
    if file_hash(row["fixture_path"]) != row["source_hash"]:
        raise ValueError("Task source hash differs from the root manifest")
    definition = json.loads(Path(row["fixture_path"]).read_text())
    task = RootTask.model_validate(definition["task"])
    if (task.root_id, task.family_id, task.split) != (row["root_id"], row["family_id"], row["split"]):
        raise ValueError("Task differs from its registered root/family/split")
    if extra := tests.get(task.root_id):
        if file_hash(extra["path"]) != extra["sha256"]:
            raise ValueError("Private test manifest hash mismatch")
        private = json.loads(Path(extra["path"]).read_text())
        if any(s["visibility"] != "PRIVATE" for s in private["suites"]):
            raise ValueError("Additional scoring tests must be PRIVATE")
        definition["suites"] = definition.get("suites", []) + private["suites"]
    full = fixture_verifier(definition, framework)
    # Private suites are physically absent from the verifier reachable online.
    online = Verifier([s for s in full.suites.values() if s.visibility == "VISIBLE"], full.sandbox)
    return definition, task, ContractSpec.model_validate(definition["contract"]), online, full


def empty_record(row, policy):
    score, _ = score_program(None, Verifier([]), row["root_id"])
    return {"root_id": row["root_id"], "family_id": row["family_id"], "split": row["split"],
        "source": row.get("source"), "policy": policy, "status": "ERROR", "stop_reason": None,
        "error": None, "model_identity": None, "online_functional_status": None,
        "responsibility_status": "UNKNOWN", "node_count": 0, "first_program": None, "final_program": None,
        "first_scores": deepcopy(score), "final_scores": deepcopy(score),
        "online_cost": Budget(nodes_created=0).model_dump(), "scoring_cost": {"verify_calls": 0, "executed_cases": 0, "execution_seconds": 0.0},
        "wall_seconds": 0.0, "run_dir": None}


def aggregate(records):
    count = len(records)
    online = {key: sum(r["online_cost"][key] for r in records) for key in Budget.model_fields}
    scoring = {key: sum(r["scoring_cost"][key] for r in records) for key in ("verify_calls", "executed_cases", "execution_seconds")}
    first = sum(r["first_scores"]["private"]["status"] == "PASS" for r in records)
    final = sum(r["final_scores"]["private"]["status"] == "PASS" for r in records)
    tokens = online["input_tokens"] + online["output_tokens"]
    return {"roots": count, "first_program_successes": first, "budgeted_successes": final,
        "FirstProgramSuccess": first / count if count else None, "BudgetedSuccess": final / count if count else None,
        "first_private_status": dict(Counter(r["first_scores"]["private"]["status"] for r in records)),
        "final_private_status": dict(Counter(r["final_scores"]["private"]["status"] for r in records)),
        "final_visible_status": dict(Counter(r["final_scores"]["visible"]["status"] for r in records)),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
        "root_status": dict(Counter(r["status"] for r in records)),
        "responsibility_status": dict(Counter(r["responsibility_status"] for r in records)),
        "conservative_joint_successes": sum(r["responsibility_status"] == "SUPPORTED" and
            r["final_scores"]["private"]["status"] == "PASS" for r in records),
        "online_cost": online, "scoring_cost": scoring,
        "cost_per_budgeted_success": {"tokens": tokens / final, "policy_calls": online["policy_calls"] / final,
            "verify_calls": (online["verify_calls"] + scoring["verify_calls"]) / final,
            "executed_cases": (online["executed_cases"] + scoring["executed_cases"]) / final,
            "execution_seconds": (online["execution_seconds"] + scoring["execution_seconds"]) / final} if final else None}


def evaluate(config, *, policy_name="base", adapter=None, device="cuda:0", load_in_4bit=False, policy_factory=None):
    if (policy_name == "sft") != (adapter is not None) or policy_name not in {"script", "base", "sft"}:
        raise ValueError("Select script/base without adapter or sft with adapter")
    if config["selection"] != {"feedback": "visible_only", "private_tests": "score_selected_candidate_only"}:
        raise ValueError("Private feedback cannot select or repair candidates")
    if config["apps"]["supported_type"] != "stdin" or any(config["reporting"].get(k) is not True for k in (
            "first_program_snapshot", "budgeted_final_snapshot", "retain_all_roots",
            "include_unknown_and_errors", "include_all_calls_tokens_verifications_and_replays")):
        raise ValueError("F9 requires stdin scope, frozen candidates and complete denominators")
    framework = yaml.safe_load(Path(config["framework_config"]).read_text())
    framework["inference"]["seed"] = config["seed"]
    if load_in_4bit:
        framework["model"]["load_in_4bit"] = True
    if config.get("max_policy_calls"):
        framework["runtime"]["max_policy_calls"] = config["max_policy_calls"]
    roots = list(load_root_manifest(config["root_manifest"]).values())
    rows = [r for r in roots if r["split"] == config["split"]]
    if config.get("limit") is not None:
        if config["limit"] < 1:
            raise ValueError("Positive limit required")
        rows = rows[:config["limit"]]
    if not rows:
        raise ValueError("No registered roots in the selected split")
    if policy_name == "script" and any(row["split"] != "SYN_FIXTURE" for row in rows):
        raise ValueError("Scripted policy is restricted to SYN_FIXTURE")
    tests = {}
    if config.get("test_manifest"):
        for row in read_jsonl(config["test_manifest"]):
            if row["root_id"] in tests:
                raise ValueError("Duplicate root in test manifest")
            tests[row["root_id"]] = row
    directory = Path(config["output_dir"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = {"stage": config.get("stage", "F9"), "scope": config.get("scope", "SMOKE_ONLY"), "policy": policy_name, "adapter": str(adapter) if adapter else None,
        "config": config, "framework": framework, "roots": rows, "tests": [tests[r["root_id"]] for r in rows if r["root_id"] in tests],
        "first_selection": "first accepted complete root write, before root verification; child-local repair allowed and costed",
        "final_selection": "active root artifact at termination, including budget/abstention; never selected using private scores"}
    write_json(directory / "request.json", request)
    limits = Limits(**{key: framework["runtime"][key] for key in Limits.model_fields})
    policy, load_error, probe = None, None, None
    load_started = time.monotonic()
    if policy_name != "script":
        try:
            policy = (policy_factory or LLMPolicy.from_pretrained)(framework, device=device, adapter_path=adapter)
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"
    load_seconds = time.monotonic() - load_started
    records = []
    for row in rows:
        record = empty_record(row, policy_name)
        task_started = time.monotonic()
        runtime = None
        try:
            if load_error:
                record["stop_reason"] = "MODEL_LOAD_ERROR"
                raise RuntimeError(load_error)
            definition, task, contract, online, full = load_task(row, tests, framework)
            if policy_name == "script":
                steps = definition["script"] + definition.get("completion_script", [])
                policy = ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(s)) for s in steps], name=task.root_id)
            record["model_identity"] = policy.identity if hasattr(policy, "identity") else {"backend": "scripted", "name": task.root_id}
            runtime = RepairRuntime(task, contract, policy, directory / task.root_id / "runtime", limits=limits,
                protocol_version=framework["protocol_version"], verifier=online, repair_config=framework["repair"], implementation_stage="F9")
            record["run_dir"] = str(runtime.run_dir)
            if policy_name == "sft" and probe is None:
                import torch
                probe, logits = policy.check_adapter_effect(runtime.build_observation())
                write_json(directory / "adapter_probe.json", probe)
                torch.save(logits, directory / "adapter_probe_logits.pt")
            if policy_name == "sft" and probe["status"] != "PASS":
                record["stop_reason"] = "ADAPTER_CHECK_ERROR"
                raise RuntimeError("Adapter effect check failed")
            while runtime.stop_reason is None:
                event = runtime.step()
                if event.node_id == "N0" and event.host_result.code == "SOURCE_SAVED" and record["first_program"] is None:
                    record["first_program"] = capture_program(runtime, directory / task.root_id / "first_program")
            result = runtime.run()  # Already terminal: saves the common runtime result, no further decisions.
            record.update(stop_reason=result["stop_reason"], online_functional_status=result["functional_status"],
                          responsibility_status=result["responsibility_status"], node_count=result["node_count"])
            record["final_program"] = capture_program(runtime, directory / task.root_id / "final_program")
            errors = {"POLICY_ERROR", "MODEL_ERROR", "VERIFIER_ERROR", "INFRASTRUCTURE_ERROR"}
            record["status"] = "ERROR" if any(n["termination_reason"] in errors for n in result["node_results"].values()) else "COMPLETED"
            # Both selections are now frozen. Scoring has no reference to the policy/runtime.
            for kind in ("first", "final"):
                scores, _ = score_program(record[kind + "_program"], full, task.root_id, record["scoring_cost"])
                record[kind + "_scores"] = scores
                if any(s["status"] == "ERROR" for s in scores.values()):
                    record["status"] = "ERROR"
        except Exception as error:
            record["status"] = "ERROR"
            record["stop_reason"] = record["stop_reason"] or "EVALUATION_ERROR"
            record["error"] = f"{type(error).__name__}: {error}"
            for kind in ("first", "final"):
                if record[kind + "_program"] is not None:
                    for score in record[kind + "_scores"].values():
                        if score["status"] == "NO_PROGRAM":
                            score.update(status="ERROR", reason="SCORING_INTERRUPTED")
        finally:
            if runtime is not None:
                record["online_cost"] = runtime.budget.model_dump()
            record["wall_seconds"] = time.monotonic() - task_started
            records.append(record)
            with (directory / "roots.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            logging.info("root=%s stop=%s first_private=%s final_private=%s", row["root_id"], record["stop_reason"],
                         record["first_scores"]["private"]["status"], record["final_scores"]["private"]["status"])
    summary = {"stage": config.get("stage", "F9"), "scope": config.get("scope", "SMOKE_ONLY"), "policy": policy_name,
        "status": "ERROR" if any(r["status"] == "ERROR" for r in records) else "COMPLETED",
        "aggregate": aggregate(records), "model_load_seconds": load_seconds, "wall_seconds": time.monotonic() - started,
        "adapter_probe_forward_calls": probe["forward_calls"] if probe else 0,
        "root_results": str(directory / "roots.jsonl"), "run_dir": str(directory)}
    write_json(directory / "result.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--policy", choices=["script", "base", "sft"], default="base")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-policy-calls", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    for key in ("split", "limit", "max_policy_calls", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = evaluate(config, policy_name=args.policy, adapter=args.adapter, device=args.device, load_in_4bit=args.load_in_4bit)
    print(json.dumps(result, indent=2))
    return int(result["status"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
