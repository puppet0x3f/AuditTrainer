"""Import APPS stdin records; preserve raw cases and the legacy text comparator."""

import argparse
import json
from pathlib import Path

from .dataset import file_hash, load_root_manifest, read_jsonl
from .verifier import canonical, digest


def compare_stdio(actual, expected):
    """APPS testing_util.py stdin string branches (including its permissive sets).

    Reference: https://github.com/hendrycks/apps/blob/main/eval/testing_util.py
    Input/output are strings; the importer joins original lists of lines first.
    This deliberately preserves the upstream unordered/rounded fallback, and is
    named separately from exact JSON equality. It is not a strict-order oracle.
    """
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    output = actual.splitlines()
    if ("\n".join(output).strip() == expected.strip()
            or "\n".join(line.strip() for line in output).strip() == expected.strip()):
        return True
    ground = [line.strip() for line in expected.split("\n") if line]
    if output == ground:
        return True
    output = [line for line in output if line]
    if output == ground:
        return True
    try:
        left, right = list(map(float, output)), list(map(float, ground))
        if len(left) == len(right) and all(a == b or abs(a - b) <= 1e-8 + 1e-5 * abs(b) for a, b in zip(left, right)):
            return True
    except ValueError:
        pass
    left = [set(line.split()) for line in output if line.split()]
    right = [set(line.split()) for line in ground]
    if {frozenset(line) for line in left} == {frozenset(line) for line in right}:
        return True
    try:
        return ({frozenset(round(float(token), 3) for token in line) for line in left}
                == {frozenset(round(float(token), 3) for token in line) for line in right})
    except ValueError:
        return False


def stdin_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "\n".join(value)
    raise ValueError("UNSUPPORTED_STDIN_ENCODING")


def convert_record(record, *, original_split, split, visible_cases=1):
    if original_split not in {"train", "test"}:
        raise ValueError("UNKNOWN_ORIGINAL_SPLIT")
    if (original_split == "test" and split != "APPS_SEALED") or (original_split == "train" and split not in {
            "DEV_ENGINEERING", "DEV_MECHANISM", "TRAIN", "VALIDATION", "MECHANISM_HOLDOUT"}):
        raise ValueError("FORBIDDEN_SPLIT_ASSIGNMENT")
    io = record["input_output"]
    io = json.loads(io) if isinstance(io, str) else io
    if io.get("fn_name") is not None:
        raise ValueError("UNSUPPORTED_CALL_BASED")
    inputs, outputs = io.get("inputs", []), io.get("outputs", [])
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("MISSING_OR_MISMATCHED_CASES")
    if visible_cases is None:
        visible_cases = int(len(inputs) > 1)
    if visible_cases < 0 or len(inputs) <= visible_cases:
        raise ValueError("NEED_DISJOINT_VISIBLE_AND_PRIVATE_CASES")
    problem_id = str(record.get("problem_id", record.get("id")))
    if not problem_id.isdigit() or not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError("INVALID_PROBLEM_ID_OR_QUESTION")
    root = f"apps_{original_split}_{int(problem_id):04d}"
    # A shared origin URL keeps reindexed copies of the same problem in one family.
    family = "apps_origin_" + digest(record.get("url") or root)[:16]
    visible_id, private_id = root + "_VISIBLE", root + "_PRIVATE"
    contract = {
        "signature": {"name": "solve", "parameters": [{"name": "text", "type": "str"}], "returns": "str"},
        "requires": ["text contains one valid standard-input instance described by the problem"],
        "ensures": [{"id": "O_PROBLEM", "text": "Return the standard output required by the problem for the given input"}],
        "owns": ["parse_input_solve_and_format"], "uses": [],
        "must_not": ["Read host stdin", "Access external files or network"],
        "frame": {"public_symbol": "solve", "private_helper_prefix": "_", "mutate_inputs": False, "external_io": False},
    }
    cases = [{"case_id": f"case_{i}", "args": [stdin_text(a)], "expected": stdin_text(b)}
             for i, (a, b) in enumerate(zip(inputs, outputs))]
    def suite(visibility, values):
        return {"suite_id": visible_id if visibility == "VISIBLE" else private_id, "contract_version": 1,
                "check_kind": "ROOT", "visibility": visibility, "covered_obligations": ["O_PROBLEM"],
                "oracle_origin": f"APPS {original_split}/{problem_id} original input_output; fixed index partition",
                "cases": values, "comparison": "apps_stdio_legacy"}
    public = {
        "task": {"root_id": root, "family_id": family, "split": split,
            "spec": record["question"] + "\n\nImplement solve(text: str) -> str: text supplies the full standard input; return the full standard output. Do not read stdin or print.",
            "entrypoint": "solve(text: str) -> str", "visible_suite_ref": visible_id if visible_cases else None, "private_suite_ref": private_id},
        "contract": contract, "suites": [suite("VISIBLE", cases[:visible_cases])] if visible_cases else [],
    }
    return public, {"suites": [suite("PRIVATE", cases[visible_cases:])]}


def import_apps(input_path, output_dir, root_manifest, test_manifest, *, original_split, split,
                source, visible_cases=1, limit=None):
    output_dir, root_manifest, test_manifest = map(Path, (output_dir, root_manifest, test_manifest))
    roots = load_root_manifest(root_manifest)
    known_tests = {r["root_id"] for r in read_jsonl(test_manifest)} if test_manifest.exists() else set()
    families = {r["family_id"]: r["split"] for r in roots.values()}
    converted, rejected = [], []
    for index, record in enumerate(read_jsonl(input_path)):
        try:
            public, private = convert_record(record, original_split=original_split, split=split, visible_cases=visible_cases)
            task = public["task"]
            if task["root_id"] in roots or task["root_id"] in known_tests:
                raise ValueError("DUPLICATE_ROOT")
            if task["family_id"] in families and families[task["family_id"]] != split:
                raise ValueError("FAMILY_CROSSES_SPLITS")
            if limit is not None and len(converted) >= limit:
                raise ValueError("IMPORT_LIMIT")
            converted.append((record, public, private))
            roots[task["root_id"]] = task
            families[task["family_id"]] = split
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            rejected.append({"record_index": index, "problem_id": record.get("problem_id", record.get("id")), "reason": str(error)})
    output_dir.mkdir(parents=True, exist_ok=False)
    root_rows, test_rows = [], []
    for record, public, private in converted:
        task = public["task"]
        directory = output_dir / task["root_id"]
        directory.mkdir()
        for name, obj in (("raw.json", record), ("task.json", public), ("private_tests.json", private)):
            (directory / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        public_visible = sum(len(s["cases"]) for s in public["suites"])
        root_rows.append({"root_id": task["root_id"], "family_id": task["family_id"], "split": split,
            "source": "APPS", "source_task_id": str(record.get("problem_id", record.get("id"))), "original_split": original_split,
            "source_location": source, "source_record_sha256": digest(record), "raw_path": str(directory / "raw.json"),
            "fixture_path": str(directory / "task.json"), "source_hash": file_hash(directory / "task.json"),
            "io_type": "stdin", "comparison": "apps_stdio_legacy", "visible_case_indices": list(range(public_visible))})
        test_rows.append({"root_id": task["root_id"], "path": str(directory / "private_tests.json"),
                          "sha256": file_hash(directory / "private_tests.json")})
    for path, rows in ((root_manifest, root_rows), (test_manifest, test_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write("".join(canonical(row) + "\n" for row in rows))
    report = {"source": source, "input_sha256": file_hash(input_path), "original_split": original_split,
              "split": split, "supported_type": "stdin", "comparison": "apps_stdio_legacy",
              "imported": root_rows, "rejected": rejected, "counts": {"imported": len(root_rows), "rejected": len(rejected)}}
    (output_dir / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root-manifest", type=Path, default=Path("data/manifests/roots.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/tests.jsonl"))
    parser.add_argument("--original-split", required=True, choices=["train", "test"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--visible-cases", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    report = import_apps(args.input, args.output_dir, args.root_manifest, args.test_manifest,
                         original_split=args.original_split, split=args.split, source=args.source,
                         visible_cases=args.visible_cases, limit=args.limit)
    print(json.dumps(report["counts"]))
    return 0 if report["imported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
