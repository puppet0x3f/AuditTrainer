"""Independent responsibility-constrained executor pilot.

The planner and BODY stay frozen.  The pilot teaches a small set of new
prepare/aggregate contracts from independently checked authored executions,
keeps a fixed retention slice from the current executor, and evaluates the
new adapter on disjoint inputs before any value or gate refresh.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path("runs/refinement/executor_pilot")
DATA = Path("data/refinement/executor_pilot")
INITIAL = "runs/refinement/contract_correction/training/final_adapter"
GATE = "runs/refinement/gate_refresh/training/final_adapter"
BODY = "runs/refinement/body_transfer/MULTI_BODY/final_adapter"
TRAIN_VARIANTS = ("stock_delta", "coverage_histogram", "polynomial_pair", "routes_histogram")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {"train", "evaluate"}:
    from .gpu_resource import acquire

    _reservation = acquire(ROOT / "gpu_resources.json",
                           "train" if sys.argv[1] == "train" else "collect")

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import capture_program, score_program, write_json
from .refinement_body_transfer import reviewed_plans
from .refinement_complex_executor import prepare_checks
from .refinement_complex_tasks import TASKS, pack, oracle as base_oracle
from .refinement_executor_sft import train as executor_train
from .refinement_responsibility_course import CourseReference, allocation
from .refinement_rollout import run_case, runtime_for
from .refinement_separate_gate import fingerprints
from .refinement_compute_values import local_checks
from .refinement_interface_gate import FixedPlanner
from .verifier import canonical


PILOT_SPECS = {
    "stock_delta": (
        "stock_events",
        "Return {'delta_by_sku': [[SKU, stock-target], ...] sorted by SKU, "
        "'max_abs': maximum absolute delta or 0, 'rejected': unchanged rejected indices}.",
        """def aggregate(prepared):
    targets = dict(prepared['targets'])
    rows = [[sku, stock-targets.get(sku, 0)] for sku, stock in prepared['stocks']]
    return {'delta_by_sku': rows,
            'max_abs': max((abs(delta) for sku, delta in rows), default=0),
            'rejected': prepared['rejected']}
""",
    ),
    "coverage_histogram": (
        "coverage_queries",
        "For every half-open query return a histogram [[coverage_count, length], ...] "
        "including uncovered length at count zero, followed by the query in original order. "
        "Return {'histograms': [...]}.",
        """def aggregate(prepared):
    histograms = []
    for left, right in prepared['queries']:
        histogram = {0: max(0, right-left)}
        for a, b, count in prepared['segments']:
            length = max(0, min(right, b)-max(left, a))
            if length:
                histogram[0] -= length
                histogram[count] = histogram.get(count, 0) + length
        histograms.append([[count, histogram[count]] for count in sorted(histogram)
                           if histogram[count]])
    return {'histograms': histograms}
""",
    ),
    "polynomial_pair": (
        "polynomial_chain",
        "At each requested point return the original product value and its first derivative "
        "as [value, slope]. Return {'value_and_slope': [...], 'degree': degree of the "
        "original product or -1 for zero}; the requested derivative order remains part of "
        "the input but is not used for the slope.",
        """def aggregate(prepared):
    terms = dict(prepared['terms'])
    slope = {d-1: d*c for d, c in terms.items() if d}
    pairs = []
    for x in prepared['points']:
        value = sum(c*x**d for d, c in terms.items())
        derivative = sum(c*x**d for d, c in slope.items())
        pairs.append([value, derivative])
    return {'value_and_slope': pairs, 'degree': max(terms, default=-1)}
""",
    ),
    "routes_histogram": (
        "weighted_routes",
        "For every limit return the reachable distance histogram [[distance, count], ...] "
        "and the lexicographically sorted reachable node list. Return "
        "{'limits': [{'limit': limit, 'distance_histogram': ..., 'nodes': ...}, ...]}.",
        """def aggregate(prepared):
    distances = dict(prepared['distances'])
    rows = []
    for limit in prepared['limits']:
        chosen = [(node, distance) for node, distance in distances.items()
                  if distance <= limit]
        hist = {}
        for node, distance in chosen:
            hist[distance] = hist.get(distance, 0) + 1
        rows.append({'limit': limit,
                     'distance_histogram': [[d, hist[d]] for d in sorted(hist)],
                     'nodes': sorted(node for node, distance in chosen)})
    return {'limits': rows}
""",
    ),
}


TRAIN_INPUTS = {
    "stock_events": [
        {"initial": {"pa": 4, "pb": 1},
         "events": [[3, "pa", -2], [0, "pb", -3], [1, "pa", 5], [3, "pc", 2]],
         "targets": {"pa": 6, "pb": 0, "pc": 1, "pd": 0}},
        {"initial": {"qa": 1},
         "events": [[-2, "qa", 4], [2, "qb", -1], [2, "qa", -3], [1, "qc", 5]],
         "targets": {"qa": 2, "qb": 0, "qc": 8}},
        {"initial": {"ra": 7, "rb": 0},
         "events": [[0, "ra", -9], [0, "ra", 2], [4, "rb", 3], [2, "rc", -2]],
         "targets": {"ra": 1, "rb": 4, "rc": 0}},
        {"initial": {"sa": 2},
         "events": [[5, "sa", -1], [1, "sb", 2], [1, "sb", -4], [5, "sc", 6]],
         "targets": {"sa": 0, "sb": 1, "sc": 9}},
        {"initial": {"ta": 6, "tb": 2},
         "events": [[-3, "ta", -7], [-3, "ta", 4], [1, "tb", 5], [2, "tc", -3]],
         "targets": {"ta": 1, "tb": 5, "tc": 0}},
        {"initial": {"ua": 0},
         "events": [[0, "ua", 3], [0, "ua", -5], [4, "ub", 2], [3, "uc", 9]],
         "targets": {"ua": 2, "ub": 0, "uc": 10}},
    ],
    "coverage_queries": [
        {"intervals": [[-11, -5], [-8, 1], [-8, 1], [4, 10]],
         "queries": [[-12, 12], [-9, -2], [2, 7]]},
        {"intervals": [[-7, -1], [-3, 5], [0, 0], [3, 11]],
         "queries": [[-10, 0], [-2, 9], [7, 12]]},
        {"intervals": [[-12, -9], [-2, 2], [-2, 2], [-2, 2]],
         "queries": [[-12, 12], [-4, 4], [2, 2]]},
        {"intervals": [[-6, 6], [-6, -2], [1, 9], [1, 9]],
         "queries": [[-8, 10], [-5, 0], [6, 10]]},
        {"intervals": [[-12, -10], [-4, 3], [-4, 3], [2, 12]],
         "queries": [[-12, 12], [-6, 4], [3, 11]]},
        {"intervals": [[-9, -1], [-8, -2], [4, 5], [4, 10]],
         "queries": [[-10, 0], [-3, 6], [10, 12]]},
    ],
    "polynomial_chain": [
        {"polynomials": [[[2, 2], [0, 1]], [[1, -1], [0, 2]]],
         "derivative": 0, "points": [-3, 1, 4]},
        {"polynomials": [[[3, 1], [1, 2]], [[2, -2], [0, 1]]],
         "derivative": 2, "points": [-2, 0, 3]},
        {"polynomials": [[[4, 1], [2, -1], [0, 2]], [[1, 2], [0, -1]]],
         "derivative": 1, "points": [-1, 2, 3]},
        {"polynomials": [[[1, 2], [0, -2]], [[1, -1], [0, 3]], [[0, 1]]],
         "derivative": 4, "points": [-3, 0, 2]},
        {"polynomials": [[[4, -1], [0, 2]], [[2, 1], [1, 1], [0, -1]]],
         "derivative": 3, "points": [-2, 1, 4]},
        {"polynomials": [[[3, 2], [2, -1], [0, 1]], [[1, -2], [0, -1]]],
         "derivative": 1, "points": [-1, 0, 3]},
    ],
    "weighted_routes": [
        {"edges": [["aa", "bb", 2], ["bb", "cc", 3], ["aa", "cc", 8], ["cc", "dd", 1]],
         "source": "aa", "limits": [0, 2, 5, 9]},
        {"edges": [["ba", "bc", 0], ["ba", "bb", 4], ["bc", "bd", 2], ["bb", "bd", 1]],
         "source": "ba", "limits": [-1, 0, 3, 5]},
        {"edges": [["ca", "cb", 7], ["ca", "cc", 2], ["cc", "cb", 2], ["cb", "cd", 0]],
         "source": "ca", "limits": [1, 2, 4, 9]},
        {"edges": [["da", "db", 5], ["db", "dc", 5], ["da", "dc", 9], ["dc", "dd", 1]],
         "source": "da", "limits": [0, 5, 9, 10]},
        {"edges": [["ia", "ib", 6], ["ia", "ic", 1], ["ic", "ib", 2], ["ib", "id", 4]],
         "source": "ia", "limits": [0, 1, 3, 7]},
        {"edges": [["ja", "jb", 2], ["jb", "jc", 0], ["ja", "jc", 5], ["jc", "jd", 2]],
         "source": "ja", "limits": [-1, 2, 4, 6]},
    ],
}

VALIDATION_INPUTS = {
    "stock_events": [
        {"initial": {"va": 5}, "events": [[1, "va", -8], [1, "va", 4], [0, "vb", 3], [2, "vc", -1]],
         "targets": {"va": 0, "vb": 5, "vc": 0}},
        {"initial": {"wa": 0, "wb": 4}, "events": [[2, "wa", 6], [-1, "wb", -5], [2, "wb", 1], [4, "wc", 2]],
         "targets": {"wa": 7, "wb": 2, "wc": 4}},
        {"initial": {"xa": 3}, "events": [[0, "xa", -1], [0, "xa", -3], [3, "xb", 4], [2, "xa", 2]],
         "targets": {"xa": 3, "xb": 1}},
        {"initial": {"ya": 1}, "events": [[5, "ya", -2], [4, "yb", 7], [4, "yb", -2], [6, "yc", 1]],
         "targets": {"ya": 0, "yb": 3, "yc": 0}},
        {"initial": {"za": 8}, "events": [[-2, "za", -9], [-2, "za", 3], [1, "zb", 5], [2, "zc", -1]],
         "targets": {"za": 2, "zb": 9, "zc": 0}},
        {"initial": {"ka": 2, "kb": 0}, "events": [[3, "ka", 3], [1, "kb", -1], [1, "kb", 4], [5, "kc", 6]],
         "targets": {"ka": 4, "kb": 2, "kc": 1}},
    ],
    "coverage_queries": [
        {"intervals": [[-10, -3], [-5, 2], [1, 8], [1, 8]], "queries": [[-12, 10], [-4, 5], [8, 9]]},
        {"intervals": [[-9, -4], [-4, 4], [-1, 3], [6, 12]], "queries": [[-10, 12], [-6, 0], [3, 7]]},
        {"intervals": [[-2, 9], [-2, 9], [0, 2], [5, 5]], "queries": [[-3, 10], [-1, 1], [9, 12]]},
        {"intervals": [[-12, -8], [-6, -1], [2, 6], [2, 6]], "queries": [[-12, 8], [-7, 3], [6, 8]]},
        {"intervals": [[-11, -6], [-5, 0], [-5, 0], [7, 12]], "queries": [[-12, 12], [-6, 8], [0, 10]]},
        {"intervals": [[-3, 2], [-3, 2], [1, 9], [9, 12]], "queries": [[-4, 12], [-2, 3], [8, 12]]},
    ],
    "polynomial_chain": [
        {"polynomials": [[[2, -1], [0, 3]], [[1, 1], [0, 1]]], "derivative": 3, "points": [-4, 0, 2]},
        {"polynomials": [[[4, 1], [1, -2], [0, 1]], [[2, 1], [0, -1]]], "derivative": 0, "points": [-2, 1, 4]},
        {"polynomials": [[[3, -1], [0, 2]], [[1, 2], [0, 2]]], "derivative": 4, "points": [-3, 0, 3]},
        {"polynomials": [[[2, 1], [1, 1], [0, -1]], [[2, -1], [0, 2]]], "derivative": 2, "points": [-1, 2, 5]},
        {"polynomials": [[[4, 1], [2, 2], [0, -1]], [[1, -1], [0, 2]]], "derivative": 1, "points": [-2, 0, 3]},
        {"polynomials": [[[3, -2], [1, 1], [0, 1]], [[2, 1], [0, 1]]], "derivative": 3, "points": [-1, 1, 4]},
    ],
    "weighted_routes": [
        {"edges": [["ea", "eb", 1], ["eb", "ec", 4], ["ea", "ed", 6], ["ed", "ec", 1]], "source": "ea", "limits": [-1, 1, 5, 7]},
        {"edges": [["fa", "fb", 3], ["fa", "fc", 3], ["fb", "fd", 2], ["fc", "fd", 1]], "source": "fa", "limits": [0, 3, 5, 8]},
        {"edges": [["ga", "gb", 9], ["ga", "gc", 1], ["gc", "gb", 5], ["gb", "gd", 2]], "source": "ga", "limits": [0, 1, 6, 9]},
        {"edges": [["ha", "hb", 2], ["hb", "hc", 2], ["ha", "hc", 9], ["hc", "hd", 3]], "source": "ha", "limits": [2, 4, 7, 10]},
        {"edges": [["la", "lb", 4], ["la", "lc", 2], ["lc", "lb", 1], ["lb", "ld", 3]], "source": "la", "limits": [0, 2, 3, 7]},
        {"edges": [["ma", "mb", 8], ["ma", "mc", 4], ["mc", "md", 4], ["mb", "md", 1]], "source": "ma", "limits": [1, 4, 5, 9]},
    ],
}


def pilot_oracle(name, items):
    family = PILOT_SPECS[name][0]
    prepared, _ = base_oracle(family, deepcopy(items))
    if family == "stock_events":
        targets = dict(prepared["targets"])
        rows = [[sku, stock-targets.get(sku, 0)] for sku, stock in prepared["stocks"]]
        answer = {"delta_by_sku": rows, "max_abs": max((abs(v) for _, v in rows), default=0),
                  "rejected": prepared["rejected"]}
    elif family == "coverage_queries":
        histograms = []
        for left, right in items["queries"]:
            hist = {0: max(0, right-left)}
            for a, b, count in prepared["segments"]:
                length = max(0, min(right, b)-max(left, a))
                if length:
                    hist[0] -= length
                    hist[count] = hist.get(count, 0) + length
            histograms.append([[c, hist[c]] for c in sorted(hist) if hist[c]])
        answer = {"histograms": histograms}
    elif family == "polynomial_chain":
        terms = dict(prepared["terms"])
        slope = {d-1: d*c for d, c in terms.items() if d}
        answer = {"value_and_slope": [[sum(c*x**d for d, c in terms.items()),
                                         sum(c*x**d for d, c in slope.items())]
                                        for x in prepared["points"]],
                  "degree": max(terms, default=-1)}
    else:
        distances = dict(prepared["distances"]); rows = []
        for limit in prepared["limits"]:
            chosen = [(node, distance) for node, distance in distances.items() if distance <= limit]
            hist = Counter(distance for _, distance in chosen)
            rows.append({"limit": limit,
                         "distance_histogram": [[d, hist[d]] for d in sorted(hist)],
                         "nodes": sorted(node for node, _ in chosen)})
        answer = {"limits": rows}
    return prepared, answer


def _source_rows():
    source = json.loads(Path("runs/refinement/contract_correction/request.json").read_text())
    rows = load_root_manifest(source["config"]["root_manifest"])
    tests = {x["root_id"]: x for x in read_jsonl(source["config"]["test_manifest"])}
    return source, rows, tests


def _save_package(rid, definition, private, reference, source, rows, tests, split):
    folder = DATA / rid
    folder.mkdir(parents=True, exist_ok=True)
    for label, value in [("task", definition), ("private_tests", private), ("reference", reference)]:
        write_json(folder / f"{label}.json", value)
    rows[rid] = {**{k: definition["task"][k] for k in ["root_id", "family_id", "split"]},
                 "source": source, "original_split": split, "io_type": "solve_text",
                 "fixture_path": str(folder / "task.json"),
                 "source_hash": file_hash(folder / "task.json"),
                 "reference_path": str(folder / "reference.json"),
                 "reference_hash": file_hash(folder / "reference.json")}
    tests[rid] = {"root_id": rid, "path": str(folder / "private_tests.json"),
                  "sha256": file_hash(folder / "private_tests.json")}


def _planner(definition, structure="pipeline"):
    plans = {"N0": allocation(definition["contract"], definition, "fused"),
             "N0_p1_compute": allocation(definition["local_contracts"]["compute"], definition, structure)}
    planner = FixedPlanner(SimpleNamespace(cost={}, identity={}), plans)
    planner.identity = {"backend": "authored_fixed_executor_pilot", "structure": structure}
    return planner


class _Teacher:
    def __init__(self, planner, reference, definition):
        self.planner = planner
        self.reference = CourseReference(definition, reference, "recursive")
        self.identity = {"backend": "authored_executor_pilot_teacher"}

    def act(self, observation):
        return self.planner.act(observation) if observation["phase"] == "NEW" else self.reference.act(observation)


def _collect_teacher(definition, private, reference, row, tests, framework, rid, variant):
    planner = FixedPlanner(SimpleNamespace(cost={}, identity={}),
                           {"N0": allocation(definition["contract"], definition,
                                              "atomic" if variant == "atomic" else "fused"),
                            **({} if variant == "atomic" else {
                                "N0_p1_compute": allocation(definition["local_contracts"]["compute"], definition, "pipeline")})})
    teacher = _Teacher(planner, reference, definition)
    out = ROOT / "teacher" / rid / variant
    runtime, full = runtime_for(row, tests, framework, teacher, out / "runtime")
    result = runtime.run()
    scores, _ = score_program(capture_program(runtime, out / "program"), full, rid)
    assert all(score["status"] == "PASS" for score in scores.values())
    write_json(out / "result.json", {"scores": scores, "runtime": result})
    targets = []
    for event in read_jsonl(runtime.trace.path):
        action = event.get("parsed_action") or {}
        if action.get("action") == "write_code":
            assert event["host_result"]["code"] == "SOURCE_SAVED"
            targets.append({"sample_id": f"{rid}_{variant}_{event['event_id']}",
                            "root_id": rid, "split": "TRAIN", "kind": "target",
                            "observation": event["observation_before"],
                            "target_action": action})
    return targets


def prepare():
    if (ROOT / "request.json").exists():
        return
    source, rows, tests = _source_rows()
    ROOT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    targets = []
    fit = []
    new = []
    for name in TRAIN_VARIANTS:
        family, aggregate, code = PILOT_SPECS[name]
        train_id = f"train_executor_pilot_{name}"
        spec = {**TASKS[family], "aggregate": aggregate, "aggregate_code": code}
        definition, private, reference = pack(name, task_spec=spec,
                                               cases=TRAIN_INPUTS[family],
                                               oracle_fn=pilot_oracle, root_id=train_id,
                                               split="TRAIN")
        _save_package(train_id, definition, private, reference, "independent_executor_pilot_train",
                      rows, tests, "TRAIN")
        fit.append({"root_id": train_id, "variant": "pipeline"})
        for variant in ("atomic", "pipeline"):
            targets.extend(_collect_teacher(definition, private, reference, rows[train_id],
                                            tests, source["framework"], train_id, variant))
        valid_id = f"validation_executor_pilot_{name}"
        vdef, vprivate, vreference = pack(name, task_spec=spec,
                                          cases=VALIDATION_INPUTS[family],
                                          oracle_fn=pilot_oracle, root_id=valid_id,
                                          split="VALIDATION")
        vdef["task"]["family_id"] = f"internal_complex_{name}_holdout_v1"
        _save_package(valid_id, vdef, vprivate, vreference,
                      "independent_executor_pilot_validation", rows, tests, "VALIDATION")
        new.append({"root_id": valid_id, "variant": "pipeline"})

    old_training = list(read_jsonl("data/refinement/contract_correction/training.jsonl"))
    retention_candidates = sorted(old_training,
                                  key=lambda row: hashlib.sha256(
                                      f"executor_pilot:{row['sample_id']}".encode()).hexdigest())
    retention = []
    for row in retention_candidates[:40]:
        retention.append({**row, "kind": "retention", "retention_source": "contract_correction"})
    old_selection = source["selection"]
    old_rows = load_root_manifest(source["config"]["root_manifest"])
    old_tests = {x["root_id"]: x for x in read_jsonl(source["config"]["test_manifest"])}
    retention_roots = []
    for selected in old_selection:
        if selected.get("group") not in {"retention", "taught_retention"}:
            continue
        if selected["root_id"] not in retention_roots:
            retention_roots.append(selected["root_id"])
        if len(retention_roots) == 8:
            break
    for rid in retention_roots:
        rows[rid] = old_rows[rid]
        tests[rid] = old_tests[rid]
    assert len(targets) == 20, f"expected 20 target writes, got {len(targets)}"
    assert len(retention) == 40 and len(fit) == len(new) == 4 and len(retention_roots) == 8
    all_training = targets + retention
    (DATA / "training.jsonl").write_text("".join(canonical(row) + "\n" for row in all_training))
    (DATA / "roots.jsonl").write_text("".join(canonical(row) + "\n" for row in rows.values()))
    (DATA / "tests.jsonl").write_text("".join(canonical(row) + "\n" for row in tests.values()))
    selection = ([{"group": "fit", **item} for item in fit] +
                 [{"group": "new", **item} for item in new] +
                 [{"group": "retention", "root_id": rid, "variant": "pipeline"}
                  for rid in retention_roots])
    request = {
        "framework": source["framework"],
        "config": {**source["config"], "root_manifest": str(DATA / "roots.jsonl"),
                   "test_manifest": str(DATA / "tests.jsonl"), "output_dir": str(ROOT)},
        "selection": selection,
        "adapters": {"initial": INITIAL, "gate": GATE, "body": BODY},
        "source_hashes": {"initial": file_hash(Path(INITIAL) / "adapter_model.safetensors"),
                          "gate": file_hash(Path(GATE) / "adapter_model.safetensors"),
                          "body": file_hash(Path(BODY) / "adapter_model.safetensors")},
        "training_sha256": file_hash(DATA / "training.jsonl"),
        "steps": len(targets) * 4,
        "learning_rate": 1e-5,
        "seed": 703000713,
        "eval_seed": 704000713,
        "target_examples": len(targets),
        "retention_examples": len(retention),
        "target_exposure": 8,
        "retention_exposure": 4,
        "repetitions": 2,
        "evaluation_cases": len(selection) * 2,
        "updates": len(targets) * 4,
        "design": (
            "Freeze root gate, MULTI_BODY, and all planner decisions. Train only a copy of the "
            "current contract_correction executor. Four new output contracts use four independent "
            "TRAIN inputs; atomic and fused-root/pipeline teachers contribute exactly 20 reviewed "
            "write_code targets. Add 40 hashed retention targets from the current executor corpus. "
            "One final fixed checkpoint; no outcome filtering or private-test labels in prompts."
        ),
        "scope": (
            "Fit group is new executor-pilot training data; new group is disjoint input validation "
            "with the same four output contracts; retention is an eight-root current-capability "
            "control. This is executor capability evidence, not planner/gate generalization."
        ),
        "decision": (
            "Advance only if new-input root success and local responsibility checks improve over "
            "BEFORE while retention does not regress root by root. Only then refresh ATOMIC/SPLIT "
            "values and consider conditional planner data."
        ),
    }
    write_json(ROOT / "request.json", request)
    write_json(ROOT / "gpu_resources.json", {"devices": ["0", "1", "2"],
                                               "min_free_mib": {"train": 22000, "collect": 16000}})
    print(json.dumps({"targets": len(targets), "retention": len(retention),
                      "steps": request["steps"], "evaluation_cases": request["evaluation_cases"]}))


def check():
    request = json.loads((ROOT / "request.json").read_text())
    assert request["steps"] == request["updates"] == 80
    assert request["target_examples"] == 20 and request["retention_examples"] == 40
    assert request["evaluation_cases"] == 32 and request["repetitions"] == 2
    assert file_hash(DATA / "training.jsonl") == request["training_sha256"]
    for key, path in request["adapters"].items():
        if key == "initial" or key == "gate" or key == "body":
            assert file_hash(Path(path) / "adapter_model.safetensors") == request["source_hashes"][key]
    rows = load_root_manifest(request["config"]["root_manifest"])
    tests = {row["root_id"]: row for row in read_jsonl(request["config"]["test_manifest"])}
    for selected in request["selection"]:
        row = rows[selected["root_id"]]
        assert file_hash(row["fixture_path"]) == row["source_hash"]
        assert file_hash(tests[selected["root_id"]]["path"]) == tests[selected["root_id"]]["sha256"]
    training = read_jsonl(DATA / "training.jsonl")
    assert len(training) == 60
    assert sum(row["kind"] == "target" for row in training) == 20
    assert sum(row["kind"] == "retention" for row in training) == 40
    assert len({row["sample_id"] for row in training}) == 60
    return request


def _context(request, method):
    from .refinement_policy import load_policies

    adapter = INITIAL if method == "BEFORE" else str(ROOT / "training" / "final_adapter")
    base, executor = load_policies(request["framework"], GATE, adapter)
    base.model.load_adapter(BODY, adapter_name="body", is_trainable=False, local_files_only=True)
    base.model.set_adapter("default")
    base.model.requires_grad_(False)
    rows = load_root_manifest(request["config"]["root_manifest"])
    tests = {row["root_id"]: row for row in read_jsonl(request["config"]["test_manifest"])}
    return executor, rows, tests


def _checks(case, row, tests, framework):
    if case["errors"]:
        return {"local": [], "prepare_local": [], "parent_called": False, "error": case["errors"]}
    try:
        local = local_checks(case, row, tests, framework)
        local["prepare_local"] = prepare_checks(case, row, tests, framework)
        return local
    except Exception as error:
        return {"local": [], "prepare_local": [], "parent_called": False,
                "error": f"CHECK_ERROR:{type(error).__name__}:{error}"}


def evaluate(method, shard):
    from tqdm import tqdm

    request = check()
    executor, rows, tests = _context(request, method)
    jobs = [(index, selected, rep) for index, selected in enumerate(request["selection"])
            if index % 3 == shard for rep in range(request["repetitions"])]
    results = []
    for index, selected, rep in tqdm(jobs, desc=f"{method} 执行器分片{shard}",
                                     unit="条", dynamic_ncols=True):
        rid = selected["root_id"]
        definition = json.loads(Path(rows[rid]["fixture_path"]).read_text())
        planner = _planner(definition, "pipeline")
        directory = ROOT / "evaluation" / method / selected["group"] / rid / f"repeat_{rep:02d}"
        case = run_case(rows[rid], tests, request["framework"], request["config"], planner,
                        executor, directory, request["eval_seed"] + index * 100 + rep)
        assert case["status"] == "COMPLETE" or case["errors"] == ["SCORING_ERROR"]
        checks = _checks(case, rows[rid], tests, request["framework"])
        result = {"selection_index": index, "group": selected["group"],
                  "case": case, "checks": checks}
        results.append(result)
        print(json.dumps({"method": method, "group": selected["group"], "root": rid,
                          "rep": rep,
                          "success": int(case["final_scores"]["private"]["status"] == "PASS")}),
              flush=True)
    write_json(ROOT / "evaluation" / method / f"shard_{shard}.json",
               {"status": "COMPLETE", "results": results})


def report():
    request = check()
    arms = {}
    summary = {}
    errors = []
    groups = ["fit", "new", "retention"]
    for method in ["BEFORE", "AFTER"]:
        pieces = [json.loads((ROOT / "evaluation" / method / f"shard_{i}.json").read_text())
                  for i in range(3)]
        results = [item for piece in pieces for item in piece["results"]]
        # ``evaluation_cases`` is the count for each method; BEFORE and AFTER
        # are evaluated independently with the same number of cases.
        assert len(results) == request["evaluation_cases"]
        arms[method] = results
        summary[method] = {}
        for group in groups:
            values = [item for item in results if item["group"] == group]
            summary[method][group] = {
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
                "by_root": {rid: sum(item["case"]["final_scores"]["private"]["status"] == "PASS"
                                      for item in values if item["case"]["root_id"] == rid)
                            for rid in sorted({item["case"]["root_id"] for item in values})},
            }
            for item in values:
                if item["case"]["errors"] or "error" in item["checks"]:
                    errors.append({"method": method, "group": group,
                                   "root_id": item["case"]["root_id"],
                                   "seed": item["case"]["seed"],
                                   "errors": item["case"]["errors"],
                                   "checks": item["checks"]})
    keyed = {method: {(item["selection_index"], item["case"]["seed"]): item
                      for item in values} for method, values in arms.items()}
    assert keyed["BEFORE"].keys() == keyed["AFTER"].keys()
    paired = Counter()
    pairs = []
    for key, before in keyed["BEFORE"].items():
        after = keyed["AFTER"][key]
        a = before["case"]["final_scores"]["private"]["status"] == "PASS"
        b = after["case"]["final_scores"]["private"]["status"] == "PASS"
        paired["wins" if b > a else "losses" if b < a else "ties"] += 1
        pairs.append({"selection_index": key[0], "seed": key[1], "same_seed": True})
    retention_ok = all(summary["AFTER"]["retention"]["by_root"][rid] >= n
                       for rid, n in summary["BEFORE"]["retention"]["by_root"].items())
    proceed = (not errors and
               summary["AFTER"]["new"]["root_success"] > summary["BEFORE"]["new"]["root_success"] and
               summary["AFTER"]["new"]["local_private_pass"] > summary["BEFORE"]["new"]["local_private_pass"] and
               retention_ok)
    result = {"status": "COMPLETE", "updates": request["updates"],
              "summary": summary, "paired": dict(paired), "errors": errors,
              "retention_nonregressing": retention_ok,
              "proceed_to_value_refresh": proceed, "evaluation": arms}
    write_json(ROOT / "result.json", result)
    write_json(ROOT / "validation.json", {"pairs": pairs, "errors": errors,
                                           "retention_nonregressing": retention_ok})
    print(json.dumps({"summary": summary, "paired": dict(paired),
                      "proceed_to_value_refresh": proceed}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "check", "train", "evaluate", "report"])
    parser.add_argument("--method", choices=["BEFORE", "AFTER"], default="BEFORE")
    parser.add_argument("--shard", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "check":
        request = check()
        print(json.dumps({"status": "READY", "steps": request["steps"],
                          "targets": request["target_examples"],
                          "evaluation_cases": request["evaluation_cases"],
                          "model_loaded": False}))
    elif args.command == "train":
        request = check()
        executor_train(ROOT, DATA, INITIAL, progress=True)
    elif args.command == "evaluate":
        evaluate(args.method, args.shard)
    else:
        report()
