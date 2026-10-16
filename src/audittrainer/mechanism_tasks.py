"""Small, frozen development tasks for A2; never a training or sealed benchmark."""

import json
from pathlib import Path

from .dataset import file_hash, load_root_manifest
from .evaluate import write_json


# Trusted, authored oracle expressions. They are evaluated only by this offline
# builder; model-produced source is always executed by the existing Sandbox.
# Each task has a different operation, but all share one family/split.
CATALOG = [
    ("absolute_gaps", "Sorted distinct absolute values", "sorted(set(abs(x) for x in items))",
     "Largest gap between consecutive values; 0 with fewer than two values", "max([b-a for a,b in zip(items,items[1:])] or [0])"),
    ("nonnegative_squares", "Squares of nonnegative values, preserving order", "[x*x for x in items if x>=0]",
     "Sum all values", "sum(items)"),
    ("even_position_product", "Values at zero-based even positions", "items[::2]",
     "Product of all values; 1 for empty input", "__import__('math').prod(items)"),
    ("prefix_span", "Inclusive prefix sums, preserving order", "[sum(items[:i+1]) for i in range(len(items))]",
     "Maximum minus minimum; 0 for empty input", "max(items)-min(items) if items else 0"),
    ("run_lengths", "Lengths of consecutive equal-value runs", "[len(list(g)) for _,g in __import__('itertools').groupby(items)]",
     "Number of runs of length at least two", "sum(x>=2 for x in items)"),
    ("frequency_top_two", "Occurrence counts of distinct values, sorted descending", "sorted(__import__('collections').Counter(items).values(), reverse=True)",
     "Sum the first two values, or all available values if fewer than two", "sum(items[:2])"),
    ("odd_cubes", "Cubes of odd values, preserving order", "[x**3 for x in items if x%2]",
     "Alternating sum starting with plus at position zero", "sum(x if i%2==0 else -x for i,x in enumerate(items))"),
    ("adjacent_differences", "Signed consecutive differences (next minus previous)", "[b-a for a,b in zip(items,items[1:])]",
     "Count strictly positive values", "sum(x>0 for x in items)"),
    ("running_maxima", "Inclusive running maxima", "[max(items[:i+1]) for i in range(len(items))]",
     "Count distinct values", "len(set(items))"),
    ("sorted_median", "All values sorted ascending, retaining duplicates", "sorted(items)",
     "Lower median (index (n-1)//2); 0 for empty input", "items[(len(items)-1)//2] if items else 0"),
    ("positive_rotation", "Keep strictly positive values then rotate left once", "(lambda v: v[1:]+v[:1])([x for x in items if x>0])",
     "Sum (position+1)*value over zero-based positions", "sum((i+1)*x for i,x in enumerate(items))"),
    ("modulo_histogram", "Python nonnegative residues modulo 3, preserving order", "[x%3 for x in items]",
     "List counts of 0, 1, 2 in that order", "[items.count(i) for i in range(3)]"),
    ("digit_sums", "Sum decimal digits of the absolute value of each integer", "[sum(map(int,str(abs(x)))) for x in items]",
     "Maximum value; 0 for empty input", "max(items or [0])"),
    ("pair_sums", "Sum disjoint adjacent pairs; keep the last value if length is odd", "[sum(items[i:i+2]) for i in range(0,len(items),2)]",
     "List the values strictly above zero", "[x for x in items if x>0]"),
    ("zero_segments", "Sums of nonempty segments separated by zeros; ignore empty segments", "[sum(g) for k,g in __import__('itertools').groupby(items, key=lambda x:x!=0) if k]",
     "Maximum absolute value; 0 for empty input", "max([abs(x) for x in items] or [0])"),
    ("sorted_squares", "Squares of every value sorted ascending, retaining duplicates", "sorted(x*x for x in items)",
     "Sum the smallest three values, or all values if fewer than three", "sum(items[:3])"),
    ("parity_runs", "Parity (0 even, 1 odd) of each value", "[x%2 for x in items]",
     "Length of longest consecutive equal-value run; 0 for empty input", "max([len(list(g)) for _,g in __import__('itertools').groupby(items)] or [0])"),
    ("suffix_sums", "Inclusive suffix sum starting at each original position", "[sum(items[i:]) for i in range(len(items))]",
     "Count negative values", "sum(x<0 for x in items)"),
    ("reverse_negate", "Reverse the sequence and negate each value", "[-x for x in reversed(items)]",
     "List original zero-based indices of strictly negative values", "[i for i,x in enumerate(items) if x<0]"),
    ("absolute_ranks", "Sorted distinct values (signed, ascending)", "sorted(set(items))",
     "Sum position*absolute(value), zero-based positions", "sum(i*abs(x) for i,x in enumerate(items))"),
]

VISIBLE_INPUTS = [[3, -2, 3, 0, 5, -2], []]
PRIVATE_INPUTS = [[-7], [0, 0, 0], [2, 2, -1, -1, 0, 4, 4, 4],
                  [10, -30, 20, 5, -2, 0, 11], [-4, -3, -2, -1]]


def contract(name, parameter, argument_type, returns, goal, *, full=True):
    return {"signature": {"name": name, "parameters": [{"name": parameter, "type": argument_type}], "returns": returns},
            "requires": ["Input is a JSON-compatible list of integers." ] if parameter == "items" else ["text is a JSON list of integers."],
            "ensures": [{"id": "RESULT", "text": goal}],
            "owns": [goal] if full else [], "uses": [], "must_not": ["Do not mutate inputs or perform external I/O."] if full else [],
            "frame": {"public_symbol": name, "private_helper_prefix": "_", "mutate_inputs": False, "external_io": False}}


def definitions():
    for index, (name, goal_a, expr_a, goal_b, expr_b) in enumerate(CATALOG):
        root_id = f"mech{index+1:02d}_{name}"
        # This closed authored catalog, never a model string, defines the oracle.
        a = lambda items: eval(expr_a, {"items": items})
        b = lambda items: eval(expr_b, {"items": items})
        child_a = contract("transform", "items", "list[int]", "list[int]", goal_a)
        child_b = contract("summarize", "items", "list[int]", "int | list[int]", goal_b)
        root = contract("solve", "text", "str", "str", "Return JSON encoding of the required result.")
        root["ensures"] = [{"id": "PARSE", "text": "Parse the input JSON integer list."},
                           {"id": "TRANSFORM", "text": goal_a}, {"id": "SUMMARIZE", "text": goal_b},
                           {"id": "FORMAT", "text": "Return json.dumps(result, separators=(',', ':')), without a newline."}]
        root["owns"] = [x["text"] for x in root["ensures"]]
        local = {"a": child_a, "b": child_b,
                 "a_function": contract("transform", "items", "list[int]", "list[int]", goal_a, full=False),
                 "b_function": contract("summarize", "items", "list[int]", "int | list[int]", goal_b, full=False)}
        suites, private = [], []
        for visibility, inputs in [("VISIBLE", VISIBLE_INPUTS), ("PRIVATE", PRIVATE_INPUTS)]:
            cases = [{"case_id": f"{visibility}_{i}", "args": [json.dumps(xs)],
                      "expected": json.dumps(b(a(xs)), separators=(",", ":"))} for i, xs in enumerate(inputs)]
            suite = {"suite_id": f"{root_id}_{visibility.lower()}", "check_kind": "ROOT", "visibility": visibility,
                     "oracle_origin": "A2 authored closed catalog v1; development only", "covered_obligations": [x["id"] for x in root["ensures"]], "cases": cases}
            (suites if visibility == "VISIBLE" else private).append(suite)
        for key in local:
            cases = [{"case_id": f"{key}_{i}", "args": [xs if key.startswith('a') else a(xs)],
                      "expected": a(xs) if key.startswith('a') else b(a(xs))} for i, xs in enumerate(VISIBLE_INPUTS)]
            suites.append({"suite_id": f"{root_id}_{key}", "contract_key": key, "check_kind": "LOCAL", "visibility": "VISIBLE",
                           "oracle_origin": "A2 closed component contract; exact binding only", "covered_obligations": ["RESULT"], "cases": cases})
        task = {"root_id": root_id, "family_id": "a2_integer_pipeline_v1", "split": "DEV_MECHANISM",
                "spec": f"Input is a JSON list of integers. First: {goal_a}. Then, on that transformed list: {goal_b}. Return compact JSON with no newline. Empty input is valid. Implement solve(text: str) -> str.",
                "entrypoint": "solve", "visible_suite_ref": f"{root_id}_visible", "private_suite_ref": f"{root_id}_private"}
        yield {"task": task, "contract": root, "local_contracts": local, "suites": suites}, {"suites": private}, {
            "a": f"def transform(items):\n    return {expr_a}\n", "b": f"def summarize(items):\n    return {expr_b}\n",
            "fault_node": ["N0_p1_a", "N0_p1_b", "N0", "N0"][index % 4],
            "fault_kind": ["child_value", "child_value", "composition_bypass", "formatting"][index % 4]}


def build(directory=Path("data/mechanism"), manifest=Path("data/manifests/roots.jsonl"),
          test_manifest=Path("data/manifests/tests.jsonl")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    existing = load_root_manifest(manifest)
    rows, tests = [], []
    for definition, private, reference in definitions():
        root_id = definition["task"]["root_id"]
        if root_id in existing or any(r["family_id"] == definition["task"]["family_id"] for r in existing.values()):
            raise ValueError("A2 roots/family are already registered")
        target = directory / root_id
        target.mkdir()
        write_json(target / "task.json", definition)
        write_json(target / "private_tests.json", private)
        write_json(target / "reference.json", reference)
        rows.append({**{k: definition["task"][k] for k in ("root_id", "family_id", "split")},
                     "source": "authored_mechanism_development", "source_task_id": root_id,
                     "original_split": "DEV_MECHANISM", "io_type": "solve_text", "fixture_path": str(target / "task.json"),
                     "source_hash": file_hash(target / "task.json"), "reference_path": str(target / "reference.json"),
                     "reference_hash": file_hash(target / "reference.json")})
        tests.append({"root_id": root_id, "path": str(target / "private_tests.json"), "sha256": file_hash(target / "private_tests.json")})
    for path, values in [(manifest, rows), (test_manifest, tests)]:
        with Path(path).open("a") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(directory / "selection.json", {"scope": "DEV_MECHANISM", "roots": rows,
               "selection": "all 20 authored tasks, fixed before any model run", "independent_families": 1,
               "private_selection": "never; private cases score frozen candidates only"})
    return rows
