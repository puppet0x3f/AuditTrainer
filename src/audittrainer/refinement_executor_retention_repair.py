"""Small executor retention repair starting from the completed executor pilot.

The gate, BODY, planner decisions, and evaluation set stay frozen.  Training
replays the pilot's successful new-contract targets while replacing its random
retention slice with reviewed stock-status and stock-event responsibility
traces.  The completed pilot AFTER arm is reused as the paired baseline, so
only the repair adapter and its three evaluation shards need to run.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path("runs/refinement/executor_retention_repair")
DATA = Path("data/refinement/executor_retention_repair")
PILOT_ROOT = Path("runs/refinement/executor_pilot")
PILOT_DATA = Path("data/refinement/executor_pilot")
SOURCE_TRAINING = Path("data/refinement/contract_correction/training.jsonl")
BASELINE_ADAPTER = PILOT_ROOT / "training" / "final_adapter"
GATE = "runs/refinement/gate_refresh/training/final_adapter"
BODY = "runs/refinement/body_transfer/MULTI_BODY/final_adapter"
TARGETED_ROOTS = {
    "train_contract_course_stock_status",
    "train_contract_course_stock_events",
}

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {"train", "evaluate"}:
    from .gpu_resource import acquire

    _reservation = acquire(ROOT / "gpu_resources.json", "train" if sys.argv[1] == "train" else "collect")

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_executor_pilot import _checks, _planner
from .refinement_executor_sft import train as executor_train
from .refinement_policy import load_policies
from .refinement_responsibility_course import allocation
from .refinement_rollout import run_case
from .verifier import canonical


def _load(path):
    return json.loads(Path(path).read_text())


def _write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(row) + "\n" for row in rows))


def _selected_keys(results):
    return {(item["selection_index"], item["case"]["seed"]) for item in results}


def prepare():
    if (ROOT / "request.json").exists():
        return
    pilot_request = _load(PILOT_ROOT / "request.json")
    pilot_result = _load(PILOT_ROOT / "result.json")
    assert pilot_result["status"] == "COMPLETE"
    baseline = pilot_result["evaluation"]["AFTER"]
    assert len(baseline) == pilot_request["evaluation_cases"] == 32
    assert pilot_result["proceed_to_value_refresh"] is False

    pilot_training = read_jsonl(PILOT_DATA / "training.jsonl")
    anchors = [row for row in pilot_training if row["kind"] == "target"]
    pilot_retention = [row for row in pilot_training if row["kind"] == "retention"]
    assert len(anchors) == 20 and len(pilot_retention) == 40

    reviewed = read_jsonl(SOURCE_TRAINING)
    targeted = [
        {**row, "kind": "retention", "repair_source": "contract_correction_reviewed",
         "repair_kind": "targeted_stock_contract"}
        for row in reviewed if row["root_id"] in TARGETED_ROOTS
    ]
    assert len(targeted) == 18
    targeted_ids = {row["sample_id"] for row in targeted}
    assert len(targeted_ids) == len(targeted)

    broad_candidates = [
        row for row in pilot_retention
        if row["sample_id"] not in targeted_ids and row["root_id"] not in TARGETED_ROOTS
    ]
    broad_candidates.sort(key=lambda row: hashlib.sha256(
        f"retention_repair:{row['sample_id']}".encode()).hexdigest())
    broad = [
        {**row, "repair_source": "executor_pilot_retention",
         "repair_kind": "broad_retention"}
        for row in broad_candidates[:22]
    ]
    assert len(broad) == 22

    training = [
        {**row, "repair_source": "executor_pilot_target", "repair_kind": "new_contract_anchor"}
        for row in anchors
    ] + targeted + broad
    assert len(training) == 60
    assert sum(row["kind"] == "target" for row in training) == 20
    assert sum(row["kind"] == "retention" for row in training) == 40
    assert len({row["sample_id"] for row in training}) == len(training)

    ROOT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATA / "training.jsonl", training)

    config = deepcopy(pilot_request["config"])
    config["root_manifest"] = str(PILOT_DATA / "roots.jsonl")
    config["test_manifest"] = str(PILOT_DATA / "tests.jsonl")
    config["output_dir"] = str(ROOT)
    config["executor_adapter"] = str(BASELINE_ADAPTER)
    request = {
        "framework": pilot_request["framework"],
        "config": config,
        "selection": pilot_request["selection"],
        "baseline_result": str(PILOT_ROOT / "result.json"),
        "baseline_adapter": str(BASELINE_ADAPTER),
        "adapters": {"baseline": str(BASELINE_ADAPTER), "gate": GATE, "body": BODY},
        "source_hashes": {
            "pilot_request": file_hash(PILOT_ROOT / "request.json"),
            "pilot_training": file_hash(PILOT_DATA / "training.jsonl"),
            "reviewed_training": file_hash(SOURCE_TRAINING),
            "pilot_result": file_hash(PILOT_ROOT / "result.json"),
            "baseline_adapter": file_hash(BASELINE_ADAPTER / "adapter_model.safetensors"),
            "gate": file_hash(Path(GATE) / "adapter_model.safetensors"),
            "body": file_hash(Path(BODY) / "adapter_model.safetensors"),
        },
        "training_sha256": file_hash(DATA / "training.jsonl"),
        "steps": 24,
        "updates": 24,
        "learning_rate": 2e-6,
        "seed": 705000713,
        "eval_seed": pilot_request["eval_seed"],
        "target_examples": 20,
        "retention_examples": 40,
        "target_exposure": 48,
        "retention_exposure": 48,
        "repetitions": pilot_request["repetitions"],
        "evaluation_cases": pilot_request["evaluation_cases"],
        "known_failure_roots": sorted(TARGETED_ROOTS),
        "design": (
            "Start from the completed executor-pilot adapter. Freeze root gate, MULTI_BODY, "
            "planner decisions, tests, and the paired evaluation set. Replay all 20 pilot "
            "new-contract targets as anchors, replace the pilot random retention slice with "
            "18 reviewed stock-status/stock-events traces plus 22 broad retention traces, "
            "and run a 24-step low-learning-rate executor-only repair."
        ),
        "decision": (
            "Advance to ATOMIC/SPLIT value refresh only if retention is non-regressing root by "
            "root and recovers at least one baseline failure while fit/new capability does not "
            "regress. This is a retention repair test, not planner or gate training."
        ),
    }
    write_json(ROOT / "request.json", request)
    write_json(ROOT / "gpu_resources.json", {
        "devices": ["0", "1", "2"],
        "min_free_mib": {"train": 22000, "collect": 16000},
    })
    print(json.dumps({"targets": 20, "targeted_retention": 18, "broad_retention": 22,
                      "steps": 24, "evaluation_cases": 32}))


def check():
    request = _load(ROOT / "request.json")
    assert request["steps"] == request["updates"] == 24
    assert request["target_examples"] == 20 and request["retention_examples"] == 40
    assert request["evaluation_cases"] == 32 and request["repetitions"] == 2
    assert file_hash(DATA / "training.jsonl") == request["training_sha256"]
    for key, path in request["adapters"].items():
        expected = request["source_hashes"]["baseline_adapter" if key == "baseline" else key]
        assert file_hash(Path(path) / "adapter_model.safetensors") == expected
    assert file_hash(PILOT_ROOT / "request.json") == request["source_hashes"]["pilot_request"]
    assert file_hash(PILOT_DATA / "training.jsonl") == request["source_hashes"]["pilot_training"]
    assert file_hash(SOURCE_TRAINING) == request["source_hashes"]["reviewed_training"]
    assert file_hash(PILOT_ROOT / "result.json") == request["source_hashes"]["pilot_result"]

    rows = read_jsonl(DATA / "training.jsonl")
    assert len(rows) == 60 and len({row["sample_id"] for row in rows}) == 60
    assert sum(row["kind"] == "target" for row in rows) == 20
    assert sum(row["kind"] == "retention" for row in rows) == 40
    assert sum(row.get("repair_kind") == "targeted_stock_contract" for row in rows) == 18
    assert sum(row.get("repair_kind") == "broad_retention" for row in rows) == 22
    assert all(row["split"] == "TRAIN" for row in rows)

    pilot_request = _load(PILOT_ROOT / "request.json")
    pilot_result = _load(PILOT_ROOT / "result.json")
    assert pilot_result["status"] == "COMPLETE"
    baseline = pilot_result["evaluation"]["AFTER"]
    assert len(baseline) == pilot_request["evaluation_cases"]
    assert _selected_keys(baseline) == {
        (index, pilot_request["eval_seed"] + index * 100 + rep)
        for index in range(len(pilot_request["selection"]))
        for rep in range(pilot_request["repetitions"])
    }
    return request


def train():
    request = check()
    executor_train(ROOT, DATA, str(BASELINE_ADAPTER), progress=True)


def _context(request):
    adapter = str(ROOT / "training" / "final_adapter")
    base, executor = load_policies(request["framework"], GATE, adapter)
    base.model.load_adapter(BODY, adapter_name="body", is_trainable=False, local_files_only=True)
    base.model.set_adapter("default")
    base.model.requires_grad_(False)
    rows = load_root_manifest(request["config"]["root_manifest"])
    tests = {row["root_id"]: row for row in read_jsonl(request["config"]["test_manifest"])}
    return executor, rows, tests


def evaluate(shard):
    from tqdm import tqdm

    request = check()
    executor, rows, tests = _context(request)
    jobs = [
        (index, selected, rep)
        for index, selected in enumerate(request["selection"])
        if index % 3 == shard
        for rep in range(request["repetitions"])
    ]
    results = []
    for index, selected, rep in tqdm(jobs, desc=f"REPAIR 执行器分片{shard}",
                                     unit="条", dynamic_ncols=True):
        rid = selected["root_id"]
        definition = json.loads(Path(rows[rid]["fixture_path"]).read_text())
        planner = _planner(definition, "pipeline")
        directory = ROOT / "evaluation" / "REPAIR" / selected["group"] / rid / f"repeat_{rep:02d}"
        case = run_case(rows[rid], tests, request["framework"], request["config"], planner,
                        executor, directory, request["eval_seed"] + index * 100 + rep)
        assert case["status"] == "COMPLETE" or case["errors"] == ["SCORING_ERROR"]
        checks = _checks(case, rows[rid], tests, request["framework"])
        results.append({"selection_index": index, "group": selected["group"],
                        "case": case, "checks": checks})
        print(json.dumps({"method": "REPAIR", "group": selected["group"], "root": rid,
                          "rep": rep,
                          "success": int(case["final_scores"]["private"]["status"] == "PASS")}),
              flush=True)
    write_json(ROOT / "evaluation" / "REPAIR" / f"shard_{shard}.json",
               {"status": "COMPLETE", "results": results})


def _summary(results):
    summary = {}
    for group in ("fit", "new", "retention"):
        values = [item for item in results if item["group"] == group]
        summary[group] = {
            "n": len(values),
            "root_success": sum(item["case"]["final_scores"]["private"]["status"] == "PASS"
                                 for item in values),
            "local_private_pass": sum(any(v["visibility"] == "PRIVATE" and v["status"] == "PASS"
                                           for v in item["checks"].get("local", []))
                                       for item in values),
            "prepare_private_pass": sum(any(v["visibility"] == "PRIVATE" and v["status"] == "PASS"
                                             for v in item["checks"].get("prepare_local", []))
                                         for item in values),
            "parent_called": sum(item["checks"].get("parent_called", False) for item in values),
            "mean_tokens": sum(item["case"]["usage"]["input_tokens"] +
                                item["case"]["usage"]["output_tokens"] for item in values) / len(values),
            "by_root": {
                rid: sum(item["case"]["final_scores"]["private"]["status"] == "PASS"
                         for item in values if item["case"]["root_id"] == rid)
                for rid in sorted({item["case"]["root_id"] for item in values})
            },
        }
    return summary


def report():
    request = check()
    pilot_result = _load(PILOT_ROOT / "result.json")
    baseline = pilot_result["evaluation"]["AFTER"]
    pieces = [_load(ROOT / "evaluation" / "REPAIR" / f"shard_{i}.json") for i in range(3)]
    repair = [item for piece in pieces for item in piece["results"]]
    assert len(repair) == request["evaluation_cases"]
    assert _selected_keys(repair) == _selected_keys(baseline)

    errors = []
    for item in repair:
        if item["case"]["errors"] or "error" in item["checks"]:
            errors.append({"group": item["group"], "root_id": item["case"]["root_id"],
                           "seed": item["case"]["seed"], "errors": item["case"]["errors"],
                           "checks": item["checks"]})
    baseline_summary = _summary(baseline)
    repair_summary = _summary(repair)
    keyed = {
        "BASELINE": {(item["selection_index"], item["case"]["seed"]): item for item in baseline},
        "REPAIR": {(item["selection_index"], item["case"]["seed"]): item for item in repair},
    }
    paired = Counter()
    for key, before in keyed["BASELINE"].items():
        after = keyed["REPAIR"][key]
        a = before["case"]["final_scores"]["private"]["status"] == "PASS"
        b = after["case"]["final_scores"]["private"]["status"] == "PASS"
        paired["wins" if b > a else "losses" if b < a else "ties"] += 1

    retention_nonregressing = all(
        repair_summary["retention"]["by_root"][rid] >= value
        for rid, value in baseline_summary["retention"]["by_root"].items()
    )
    retention_recovered = (
        repair_summary["retention"]["root_success"] > baseline_summary["retention"]["root_success"]
    )
    new_nonregressing = (
        repair_summary["new"]["root_success"] >= baseline_summary["new"]["root_success"] and
        repair_summary["new"]["local_private_pass"] >= baseline_summary["new"]["local_private_pass"]
    )
    fit_nonregressing = repair_summary["fit"]["root_success"] >= baseline_summary["fit"]["root_success"]
    proceed = (not errors and retention_nonregressing and retention_recovered and
               new_nonregressing and fit_nonregressing)
    result = {
        "status": "COMPLETE",
        "baseline": {"adapter": str(BASELINE_ADAPTER), "summary": baseline_summary},
        "repair": {"summary": repair_summary},
        "paired": dict(paired),
        "errors": errors,
        "retention_nonregressing": retention_nonregressing,
        "retention_recovered": retention_recovered,
        "new_nonregressing": new_nonregressing,
        "fit_nonregressing": fit_nonregressing,
        "proceed_to_value_refresh": proceed,
        "known_failure_roots": request["known_failure_roots"],
    }
    write_json(ROOT / "result.json", result)
    write_json(ROOT / "validation.json", {
        "baseline_keys": sorted([list(key) for key in keyed["BASELINE"]]),
        "repair_keys": sorted([list(key) for key in keyed["REPAIR"]]),
        "errors": errors,
        "retention_nonregressing": retention_nonregressing,
    })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "check", "train", "evaluate", "report"])
    parser.add_argument("--shard", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "check":
        request = check()
        print(json.dumps({"status": "READY", "steps": request["steps"],
                          "targets": request["target_examples"],
                          "targeted_retention": 18,
                          "evaluation_cases": request["evaluation_cases"],
                          "model_loaded": False}))
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate(args.shard)
    else:
        report()
