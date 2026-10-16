"""PR0: same-state candidates, full root continuations, and zero-update report."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

from transformers import set_seed

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, load_task, score_program, write_json
from .refinement_policy import (RefinementRouter, adapter_fingerprint, load_policies,
                                two_level_advantages)
from .runtime import Budget, Limits, RecursiveRuntime
from .train_refinement import load_pr0_config
from .verifier import digest


def root_rows(config):
    roots = sorted((r for r in load_root_manifest(config["root_manifest"]).values()
                    if r["split"] == "TRAIN"), key=lambda r: r["root_id"])
    if len(roots) != config["root_limit"]:
        raise ValueError("PR0 requires the entire configured TRAIN root set")
    return roots


def framework_limits(framework):
    return Limits(**{k: v for k, v in framework["runtime"].items() if k in Limits.model_fields})


def task_parts(row, tests, framework):
    _, task, contract, online, full = load_task(row, tests, framework)
    if task.split != "TRAIN" and not (framework.get("allow_validation") and task.split in {"VALIDATION", "EXTERNAL_SEALED"}):
        raise ValueError("Only TRAIN rewards or explicitly enabled evaluation are allowed")
    return task, contract, online, full


def runtime_for(row, tests, framework, policy, directory):
    task, contract, online, full = task_parts(row, tests, framework)
    runtime = RecursiveRuntime(task, contract, policy, directory, verifier=online,
        limits=framework_limits(framework), protocol_version=framework["protocol_version"],
        implementation_stage="PR0", require_obligation_coverage=True)
    return runtime, full


def costs(planner, executor):
    return Counter(planner.cost)+Counter(executor.cost)


def split_candidate_count(config, root_id):
    sampling = config['candidate_sampling']
    return sampling.get('split_bodies_by_root', {}).get(root_id, sampling['split_bodies'])


def reward_record(scores, budget, limits, config):
    y = int(scores["private"]["status"] == "PASS")
    # Only current final-program scoring evidence, not runtime rejection flags.
    violations = [s["evidence_path"] for block in scores.values() for s in block["suites"]
                  if s["responsibility_status"] == "VIOLATION"]
    v = int(bool(violations))
    c = (0.7*min((budget["input_tokens"]+budget["output_tokens"])/limits.max_total_tokens, 1)
         +0.2*min(budget["policy_calls"]/limits.max_policy_calls, 1)
         +0.1*min(budget["verify_calls"]/limits.max_verify_calls, 1))
    return {"y": y, "v": v, "cost": c, "violation_evidence": violations,
            "responsibility_status": "VIOLATION" if v else "UNKNOWN",
            "reward": (1-v)*y-config["reward"]["cost_weight"]*c
                      -config["reward"]["confirmed_violation_weight"]*v}


def run_case(row, tests, framework, config, planner, executor, directory, seed, initial=None,
             reset_seed_after_root_plan=False, reset_seed_after_plan_node=None):
    directory = Path(directory)
    if (directory/"case_result.json").exists():
        return json.loads((directory/"case_result.json").read_text())
    if directory.exists():
        raise ValueError(f"Unfinished case exists; inspect it before restarting: {directory}")
    directory.mkdir(parents=True)
    router = RefinementRouter(planner, executor, initial=initial)
    set_seed(seed)
    started, before_cost = time.monotonic(), costs(planner, executor)
    runtime, full = runtime_for(row, tests, framework, router, directory/"runtime")
    if reset_seed_after_root_plan:
        event = runtime.step()
        if event.node_id != 'N0' or event.host_result.code != 'PLAN_REGISTERED':
            raise ValueError('Fixed root plan must register before continuation seeding')
        set_seed(seed)
    first = None
    internal_seed_resets = []
    while runtime.stop_reason is None:
        event = runtime.step()
        if event.node_id == reset_seed_after_plan_node and event.host_result.code == 'PLAN_REGISTERED':
            set_seed(seed)
            internal_seed_resets.append({'node_id': event.node_id, 'seed': seed})
        if event.node_id == "N0" and event.host_result.code == "SOURCE_SAVED" and first is None:
            first = capture_program(runtime, directory/"first_program")
    result = runtime.run()
    final = capture_program(runtime, directory/"final_program")
    scoring = {"verify_calls": 0, "executed_cases": 0, "execution_seconds": 0.0}
    first_scores, _ = score_program(first, full, row["root_id"], scoring)
    final_scores, _ = score_program(final, full, row["root_id"], scoring)
    events = read_jsonl(runtime.trace.path)
    infra = {"POLICY_ERROR", "MODEL_ERROR", "VERIFIER_ERROR", "INFRASTRUCTURE_ERROR"}
    errors = [n["termination_reason"] for n in result["node_results"].values() if n["termination_reason"] in infra]
    if any(v["status"] == "ERROR" for scores_ in (first_scores, final_scores) for v in scores_.values()):
        errors.append("SCORING_ERROR")
    plans = [e for e in events if e.get("parsed_action") and e["parsed_action"]["action"] == "plan"]
    accepted = [e for e in plans if e["host_result"]["code"] == "PLAN_REGISTERED"]
    rejections = Counter(e["host_result"]["code"] for e in events if e["host_result"]["status"] == "REJECTED")
    registered = next((e["host_result"]["code"] == "PLAN_REGISTERED" for e in events
                       if e["event_kind"] == "POLICY" and e["node_id"] == "N0"), False)
    physical = costs(planner, executor)-before_cost
    reward = reward_record(final_scores, result["usage"], runtime.limits, config) if not errors and row["split"] == "TRAIN" else None
    record = {"root_id": row["root_id"], "family_id": row["family_id"], "split": row["split"],
              "status": "ERROR" if errors else "COMPLETE", "errors": errors, "seed": seed,
              "model_identity": router.identity, "protocol_version": framework["protocol_version"],
              "first_program": first, "final_program": final, "first_scores": first_scores,
              "final_scores": final_scores, "reward": reward, "usage": result["usage"],
              "scoring_cost": scoring, "actual_model_cost": dict(physical),
              "role_calls": dict(router.calls), "stop_reason": result["stop_reason"],
              "root_first_plan_registered": registered,
              "registered_plan_modes": dict(Counter(e["parsed_action"]["args"]["mode"] for e in accepted)),
              "protocol_rejections": dict(rejections), "node_count": result["node_count"],
              "wall_seconds": time.monotonic()-started, "run_dir": str(directory)}
    if reset_seed_after_plan_node is not None:
        record['continuation_seed_resets_after_plan'] = internal_seed_resets
    if reset_seed_after_root_plan:
        record['continuation_seed_reset_after_root_plan'] = seed
    write_json(directory/"case_result.json", record)
    print(json.dumps({"root": row["root_id"], "case": directory.name, "status": record["status"],
                      "reward": reward, "stop": result["stop_reason"],
                      "seconds": round(record["wall_seconds"], 2)}, ensure_ascii=False), flush=True)
    return record


def baseline(rows, tests, framework, config, planner, executor, label, shard_index, shards):
    selected, seen = [], set()
    for row in rows:
        if row["family_id"] not in seen:
            selected.append(row)
            seen.add(row["family_id"])
    for index, row in enumerate(selected):
        if index % shards == shard_index:
            run_case(row, tests, framework, config, planner, executor,
                Path(config["output_dir"])/"baseline"/label/row["root_id"],
                config["seed"]+10000+index)


def collect(rows, tests, framework, config, planner, executor, shard_index, shards):
    directory = Path(config["output_dir"])/"groups"
    for index, row in enumerate(rows):
        if index % shards != shard_index:
            continue
        root_dir = directory/row["root_id"]
        sampling = config['candidate_sampling']
        count = split_candidate_count(config, row['root_id'])
        repetitions = sampling['continuations_per_candidate']
        completed = root_dir/"group_result.json"
        if completed.exists():
            continue
        gate_path = root_dir/"gate.json"
        sampling_before = costs(planner, executor)
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            candidates = [json.loads((root_dir/f"candidate_{i}.json").read_text()) for i in range(count+1)]
            generation_cost = json.loads((root_dir/"generation_cost.json").read_text())
        else:
            root_dir.mkdir(parents=True, exist_ok=True)
            template, _ = runtime_for(row, tests, framework, None, root_dir/"decision")
            observation = template.build_observation()
            set_seed(config["seed"]+100000+index)
            gate = planner.score_modes(observation)
            candidates = [planner.atomic(gate)]
            for candidate_index in range(count):
                set_seed(config["seed"]+200000+index*sampling.get('candidate_seed_stride', 3)+candidate_index)
                candidates.append(planner.sample_split(gate))
            generation_cost = dict(costs(planner, executor)-sampling_before)
            write_json(gate_path, gate)
            for i, candidate in enumerate(candidates):
                write_json(root_dir/f"candidate_{i}.json", candidate)
            write_json(root_dir/"generation_cost.json", generation_cost)
        values, outcomes = [], []
        for candidate_index, candidate in enumerate(candidates):
            cases = []
            for repetition in range(repetitions):
                cases.append(run_case(row, tests, framework, config, planner, executor,
                    root_dir/f"candidate_{candidate_index}_repeat_{repetition}",
                    config["seed"]+sampling.get('continuation_seed_offset', 300000)
                    +index*sampling.get('continuation_seed_stride', 2)+repetition, initial=(gate, candidate)))
            outcomes.append(cases)
            values.append(sum(c["reward"]["reward"] for c in cases)/repetitions if all(c["reward"] is not None for c in cases) else None)
        complete = all(v is not None for v in values)
        advantages = two_level_advantages(gate["mode_probabilities"]["SPLIT"], values[0], values[1:]) if complete else None
        result = {"root_id": row["root_id"], "family_id": row["family_id"], "split": "TRAIN",
                  "status": "COMPLETE" if complete else "ERROR", "rl_optimizer_steps": 0,
                  "candidate_count": len(candidates), "continuations": len(candidates)*repetitions,
                  "gate_path": str(gate_path), "observation_sha256": gate["observation_sha256"],
                  "q_values": values, "advantages": advantages, "mode_probabilities": gate["mode_probabilities"],
                  "candidate_parse_valid": [c["parse_error"] is None for c in candidates],
                  "candidate_generation_cost": generation_cost,
                  "case_paths": [[c["run_dir"] for c in group] for group in outcomes],
                  "continuation_reward_variance": [
                      sum((c["reward"]["reward"]-v)**2 for c in group)/repetitions if v is not None else None
                      for group, v in zip(outcomes, values)]}
        write_json(completed, result)
        print(json.dumps({"group_complete": row["root_id"], "q_values": values,
                          "p_split": gate["mode_probabilities"]["SPLIT"]}), flush=True)


def case_summary(cases):
    def summed(key):
        total = Counter()
        for c in cases:
            total.update(c.get(key, {}))
        return dict(total)
    return {"runs": len(cases), "statuses": dict(Counter(c["status"] for c in cases)),
            "final_programs": sum(c["final_program"] is not None for c in cases),
            "mean_reward": sum(c["reward"]["reward"] for c in cases)/len(cases) if cases and all(c["reward"] is not None for c in cases) else None,
            "first_successes": sum(c["first_scores"]["private"]["status"] == "PASS" for c in cases),
            "final_successes": sum(c["final_scores"]["private"]["status"] == "PASS" for c in cases),
            "root_first_plan_registered": sum(c["root_first_plan_registered"] for c in cases),
            "stop_reasons": dict(Counter(c["stop_reason"] for c in cases)),
            "registered_plan_modes": summed("registered_plan_modes"),
            "protocol_rejections": summed("protocol_rejections"), "logical_usage": summed("usage"),
            "scoring_cost": summed("scoring_cost"), "actual_model_cost": summed("actual_model_cost"),
            "role_calls": summed("role_calls"),
            "confirmed_violations": sum(c["reward"]["v"] for c in cases if c["reward"]),
            "unknown_responsibility": sum(c["reward"]["responsibility_status"] == "UNKNOWN" for c in cases if c["reward"]),
            "summed_wall_seconds": sum(c["wall_seconds"] for c in cases)}


def report(config):
    output = Path(config["output_dir"])
    rows = root_rows(config)
    groups, cases = [], []
    for row in rows:
        group = json.loads((output/"groups"/row["root_id"]/"group_result.json").read_text())
        if group["status"] != "COMPLETE" or group["continuations"] != 8:
            raise ValueError(f"Incomplete root group: {row['root_id']}")
        groups.append(group)
        for paths in group["case_paths"]:
            if len(paths) != 2:
                raise ValueError("Missing repetitions")
            for path in paths:
                record = json.loads((Path(path)/"case_result.json").read_text())
                if record["root_id"] != row["root_id"] or record["status"] != "COMPLETE":
                    raise ValueError("Bad case binding or unfinished infrastructure error")
                cases.append(record)
    if len(cases) != 400:
        raise ValueError("PR0 requires 400 completed continuations")
    adaptation = json.loads((Path(config["adaptation"]["output_dir"])/"training_report.json").read_text())
    if adaptation["status"] != "COMPLETE" or adaptation["rl_optimizer_steps"] != 0:
        raise ValueError("Missing completed representation SFT")
    frozen_workers = []
    for job in ("baseline_before", "collect_after"):
        requests = sorted(output.glob(f"{job}_shard*_request.json"))
        if not requests:
            raise ValueError(f"Missing fixed-weight worker evidence: {job}")
        job_shards, expected_shards = set(), None
        for request_path in requests:
            request = json.loads(request_path.read_text())
            completion_path = request_path.with_name(request_path.name.replace("_request.json", "_complete.json"))
            completion = json.loads(completion_path.read_text())
            expected_adapter = adaptation["adapter_sha256"] if job == "collect_after" else adaptation["initial_adapter_sha256"]
            if (request["config"] != config or request["adapter_sha256"] != expected_adapter
                    or completion["status"] != "COMPLETE" or completion["rl_optimizer_steps"] != 0
                    or completion["fingerprints_before"] != completion["fingerprints_after"]
                    or request["fingerprints_before"] != completion["fingerprints_before"]):
                raise ValueError(f"Fixed-weight worker mismatch: {request_path}")
            if expected_shards is not None and expected_shards != request["shards"]:
                raise ValueError("Mixed worker partitions")
            expected_shards = request["shards"]
            job_shards.add(request["shard"])
            frozen_workers.append({"request": str(request_path), "completion": str(completion_path),
                                   "weights_unchanged": True})
        if job_shards != set(range(expected_shards)):
            raise ValueError(f"Unfinished worker partition: {job}")
    base = {}
    for label in ("before", "after"):
        records = [json.loads(p.read_text()) for p in sorted((output/"baseline"/label).glob("*/case_result.json"))]
        if len(records) != len({r["family_id"] for r in rows}) or any(r["status"] != "COMPLETE" for r in records):
            raise ValueError(f"Incomplete same-environment {label} baseline")
        base[label] = case_summary(records)
    generation = Counter()
    for g in groups:
        generation.update(g["candidate_generation_cost"])
    deltas = [g["advantages"]["q_split"]-g["advantages"]["q_atomic"] for g in groups]
    split_cases = [c for c in cases if Path(c["run_dir"]).name.split("_")[1] != "0"]
    atomic_cases = [c for c in cases if Path(c["run_dir"]).name.split("_")[1] == "0"]
    by_path = {c["run_dir"]: c for c in cases}
    registered_sets = [[i for i in range(1, 4) if all(by_path[p]["root_first_plan_registered"]
                       for p in g["case_paths"][i])] for g in groups]
    # These conditional diagnostics locate grammar versus execution differences.
    # Q values and advantages above still include every original candidate.
    registered_diagnostics = {
        "initially_registered_split_candidates": sum(map(len, registered_sets)),
        "initially_registered_split_successes": sum(c["root_first_plan_registered"] and c["reward"]["y"] for c in split_cases),
        "successful_split_continuations_after_initial_rejection": sum(not c["root_first_plan_registered"] and c["reward"]["y"] for c in split_cases),
        "states_with_multiple_registered_splits": sum(len(ids) >= 2 for ids in registered_sets),
        "states_with_registered_split_value_difference": sum(
            len(ids) >= 2 and max(g["q_values"][i] for i in ids)-min(g["q_values"][i] for i in ids)>1e-8
            for g, ids in zip(groups, registered_sets)),
    }
    rejections_by_role = defaultdict(Counter)
    for case in cases:
        node_modes = {}
        for event in read_jsonl(Path(case["run_dir"])/"runtime"/"trace.jsonl"):
            code = event["host_result"]["code"]
            if code == "PLAN_REGISTERED":
                node_modes[event["node_id"]] = event["parsed_action"]["args"]["mode"]
            if event["host_result"]["status"] == "REJECTED":
                phase = event["observation_before"]["phase"]
                role = {"NEW": "planner", "RESOLVE_CHILD": "child_resolution"}.get(phase, "host_control")
                if phase == "IMPLEMENT":
                    role = "composition" if node_modes.get(event["node_id"]) == "SPLIT" else "implementation"
                rejections_by_role[role][code] += 1
    result = {"stage": "PR0", "status": "COMPLETE", "scope": "TRAIN_DIAGNOSTIC",
              "rl_optimizer_steps": 0, "roots": len(groups), "candidates": 200, "continuations": len(cases),
              "adaptation": adaptation, "baseline": base, "frozen_workers": frozen_workers,
              "all_cases": case_summary(cases),
              "atomic": case_summary(atomic_cases), "split": case_summary(split_cases),
              "candidate_generation_cost": dict(generation),
              "split_parse_valid": sum(sum(g["candidate_parse_valid"][1:]) for g in groups),
              "registered_split_diagnostics": registered_diagnostics,
              "protocol_rejections_by_role": {role: dict(counts) for role, counts in rejections_by_role.items()},
              "sampled_split_candidates": 150,
              "states_with_reward_difference": sum(max(g["q_values"])-min(g["q_values"])>1e-8 for g in groups),
              "states_with_split_body_difference": sum(max(g["q_values"][1:])-min(g["q_values"][1:])>1e-8 for g in groups),
              "states_split_better": sum(d>1e-8 for d in deltas),
              "states_atomic_better": sum(d < -1e-8 for d in deltas),
              "states_mode_tied": sum(abs(d)<=1e-8 for d in deltas),
              "mean_split_probability": sum(g["mode_probabilities"]["SPLIT"] for g in groups)/len(groups),
              "saturated_gate_states": sum(not .01 < g["mode_probabilities"]["SPLIT"] < .99 for g in groups),
              "candidates_with_continuation_variance": sum(v>0 for g in groups for v in g["continuation_reward_variance"]),
              "groups": groups}
    write_json(output/"result.json", result)
    print(json.dumps({k: v for k,v in result.items() if k not in {"groups", "adaptation", "baseline", "all_cases", "atomic", "split"}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "collect", "report"])
    parser.add_argument("--config", default="configs/refinement.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--label", default="after", choices=["before", "after"])
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    config, framework = load_pr0_config(args.config)
    if args.command == "report":
        report(config)
        return
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid worker shard")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter or str(Path(config["adaptation"]["output_dir"])/"final_adapter")
    planner, executor = load_policies(framework, adapter, config["executor_adapter"])
    rows = root_rows(config)
    tests = {r["root_id"]: r for r in read_jsonl(config["test_manifest"])}
    before = {name: adapter_fingerprint(planner.model, name) for name in ("default", "executor")}
    request = {"config": config, "adapter": adapter, "adapter_sha256": file_hash(Path(adapter)/"adapter_model.safetensors"),
               "executor_sha256": file_hash(Path(config["executor_adapter"])/"adapter_model.safetensors"),
               "model_identity": planner.identity, "framework": framework,
               "shard": args.shard_index, "shards": args.shards, "fingerprints_before": before}
    request_path = output/f"{args.command}_{args.label}_shard{args.shard_index}_request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text())
        if any(previous[k] != request[k] for k in ("config", "adapter_sha256", "executor_sha256", "framework", "shard", "shards")):
            raise ValueError("Existing collection has a different fixed experiment")
    else:
        write_json(request_path, request)
    if args.command == "baseline" or args.with_baseline:
        baseline(rows, tests, framework, config, planner, executor, args.label, args.shard_index, args.shards)
    if args.command == "collect":
        collect(rows, tests, framework, config, planner, executor, args.shard_index, args.shards)
    after = {name: adapter_fingerprint(planner.model, name) for name in before}
    if before != after:
        raise RuntimeError("Weights changed during the zero-update collection")
    write_json(output/f"{args.command}_{args.label}_shard{args.shard_index}_complete.json",
               {"status": "COMPLETE", "fingerprints_before": before, "fingerprints_after": after,
                "rl_optimizer_steps": 0, "actual_model_cost": dict(costs(planner, executor))})


if __name__ == "__main__":
    main()
