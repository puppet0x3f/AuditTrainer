"""Scripted smokes, real-model rollouts and evidence-scoped SFT export."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import yaml

from .policy import LLMPolicy, ScriptedPolicy, ScriptStep
from .runtime import AtomicRuntime, RecursiveRuntime, RepairRuntime, Limits
from .sandbox import ExecutionLimits, Sandbox
from .schema import ContractSpec, RootTask
from .verifier import Suite, Verifier


def fixture_verifier(fixture: dict, config: dict) -> Verifier:
    suites = []
    for definition in fixture.get("suites", []):
        data = dict(definition)
        key = data.pop("contract_key", None)
        reviewed = ContractSpec.model_validate(fixture["local_contracts"][key] if key else fixture["contract"])
        suites.append(Suite(contract=reviewed, root_id=fixture["task"]["root_id"], **data))
    return Verifier(suites, Sandbox(ExecutionLimits(**config["execution"])))


def rollout(args, parser) -> int:
    config = yaml.safe_load(args.config.read_text())
    if (args.policy == "sft") != (args.adapter is not None):
        parser.error("--policy sft requires --adapter; --policy base uses the public base alone")
    if args.load_in_4bit:
        config["model"]["load_in_4bit"] = True
    stage = "F8" if args.policy == "sft" else "F5"
    if args.limit < 1 or (args.max_policy_calls is not None and args.max_policy_calls < 1):
        parser.error("Rollout limits must be positive")
    if args.max_policy_calls is not None:
        config["runtime"]["max_policy_calls"] = args.max_policy_calls
    if config["runtime"]["max_protocol_retries"] > 1:
        parser.error("F5 permits at most one structured error retry")
    from .dataset import file_hash, load_root_manifest
    rows = list(load_root_manifest(args.manifest).values())
    rows = [row for row in rows if row["split"] == args.split][:args.limit]
    if not rows:
        parser.error(f"No registered {args.split} roots; current fixtures use SYN_FIXTURE")
    tasks = []
    for row in rows:
        if file_hash(row["fixture_path"]) != row["source_hash"]:
            parser.error("Task file differs from its registered source hash")
        fixture = json.loads(Path(row["fixture_path"]).read_text())
        task = RootTask.model_validate(fixture["task"])
        if (task.root_id, task.family_id, task.split) != (row["root_id"], row["family_id"], row["split"]):
            parser.error("Fixture does not match its root manifest")
        # Script/teacher fields never leave this loader. Only host suites and the
        # public task/contract are needed by the actual-model runtime.
        tasks.append((task, ContractSpec.model_validate(fixture["contract"]), fixture_verifier(fixture, config)))
    run_dir = (args.run_dir or Path("runs") / datetime.now(timezone.utc).strftime(f"{stage.lower()}_%Y%m%dT%H%M%S_%fZ")).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    request = {"stage": stage, "scope": "SMOKE_ONLY", "trajectory_source": "natural_model_rollout",
               "policy": args.policy, "split": args.split, "roots": rows, "config": config,
               "device": args.device, "adapter": str(args.adapter) if args.adapter else None}
    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        policy = LLMPolicy.from_pretrained(config, device=args.device, adapter_path=args.adapter)
    except Exception as error:
        failure = {"stage": stage, "status": "MODEL_LOAD_ERROR", "error": f"{type(error).__name__}: {error}",
                   "run_dir": str(run_dir)}
        (run_dir / "result.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    (run_dir / "model_identity.json").write_text(json.dumps(policy.identity, indent=2) + "\n")
    limits = Limits(**{key: config["runtime"][key] for key in Limits.model_fields})
    results = []
    infrastructure_failed = False
    adapter_check = None
    for task, contract, verifier in tasks:
        runtime = RepairRuntime(task, contract, policy, run_dir / task.root_id, limits=limits,
                                protocol_version=config["protocol_version"], verifier=verifier,
                                repair_config=config["repair"], implementation_stage=stage)
        if args.policy == "sft" and adapter_check is None:
            try:
                import torch
                adapter_check, logits = policy.check_adapter_effect(runtime.build_observation())
                torch.save(logits, run_dir / "adapter_probe_logits.pt")
                (run_dir / "adapter_probe.json").write_text(json.dumps(adapter_check, ensure_ascii=False, indent=2) + "\n")
                if adapter_check["status"] != "PASS":
                    raise RuntimeError("Adapter forward effect or restoration check failed")
            except Exception as error:
                failure = {"stage": stage, "status": "ADAPTER_CHECK_ERROR", "model_identity": policy.identity,
                           "error": f"{type(error).__name__}: {error}", "run_dir": str(run_dir)}
                (run_dir / "result.json").write_text(json.dumps(failure, indent=2) + "\n")
                print(json.dumps(failure, indent=2))
                return 1
        result = runtime.run()
        infrastructure_failed |= any(node["termination_reason"] in {
            "POLICY_ERROR", "MODEL_ERROR", "VERIFIER_ERROR", "INFRASTRUCTURE_ERROR",
        } for node in result["node_results"].values())
        results.append({key: result[key] for key in ("root_id", "phase", "stop_reason", "functional_status",
                        "responsibility_status", "node_count", "usage", "run_dir")})
        logging.info("root=%s stop=%s functional=%s", task.root_id, result["stop_reason"], result["functional_status"])
    summary = {"stage": stage, "scope": "SMOKE_ONLY", "trajectory_source": "natural_model_rollout",
               "status": "ERROR" if infrastructure_failed else "COMPLETED", "split": args.split,
               "model_identity": policy.identity, "generation_calls": policy.generation_calls,
               "adapter_check": ({key: adapter_check[key] for key in
                    ("status", "forward_calls", "enabled_disabled_max_abs", "restored_max_abs")}
                    if adapter_check else None), "results": results, "run_dir": str(run_dir)}
    (run_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Capability failure is an observed outcome; infrastructure failure is not a
    # completed F5 wiring check. Scripted smoke retains its separate PASS gate.
    return 1 if infrastructure_failed or policy.generation_calls == 0 else 0


def main(argv=None) -> int:
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    data_commands = {'import-roots', 'build-demos', 'collect', 'make-contrasts', 'validate-labels', 'make-schedule'}
    if arguments and arguments[0] in data_commands:
        from .training_data import main as data_main
        return data_main(arguments)
    if arguments and arguments[0] == "mechanism":
        from .mechanism import main as mechanism_main
        return mechanism_main(arguments[1:])
    if arguments and arguments[0] == "evaluate":
        from .evaluate import main as evaluate_main
        return evaluate_main(arguments[1:])
    if arguments and arguments[0] == "import-apps":
        from .apps import main as apps_main
        return apps_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="audittrainer")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in sorted(data_commands):
        commands.add_parser(command, help="A4 TRAIN data pipeline; use the command's --help for options")
    commands.add_parser("evaluate", help="Unified script/base/SFT evaluator")
    commands.add_parser("import-apps", help="Import APPS stdin records with fixed data splits")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--stage", choices=["f1", "f2", "f3", "f4", "f6"], required=True)
    smoke.add_argument("--fixture", type=Path)
    smoke.add_argument("--config", type=Path, default=Path("configs/framework.yaml"))
    smoke.add_argument("--run-dir", type=Path)
    smoke.add_argument("--reviews", type=Path, default=Path("data/fixtures/f6_reviews.json"))
    base = commands.add_parser("rollout")
    base.add_argument("--policy", choices=["base", "sft"], required=True)
    base.add_argument("--split", choices=["SYN_FIXTURE", "DEV_ENGINEERING"], required=True)
    base.add_argument("--limit", type=int, default=2)
    base.add_argument("--manifest", type=Path, default=Path("data/manifests/roots.jsonl"))
    base.add_argument("--config", type=Path, default=Path("configs/framework.yaml"))
    base.add_argument("--run-dir", type=Path)
    base.add_argument("--device", default="cuda:0")
    base.add_argument("--adapter", type=Path)
    base.add_argument("--load-in-4bit", action="store_true", help="NF4 double quantization with BF16 computation")
    base.add_argument("--max-policy-calls", type=int)
    export = commands.add_parser("export-sft")
    export.add_argument("--trace-dir", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--manifest", type=Path, default=Path("data/manifests/roots.jsonl"))
    export.add_argument("--config", type=Path, default=Path("configs/framework.yaml"))
    export.add_argument("--split", choices=["SYN_FIXTURE", "TRAIN"], default="SYN_FIXTURE")
    export.add_argument("--max-length", type=int, default=4096)
    export.add_argument("--limit", type=int)
    export.add_argument("--allow-function-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "rollout":
        return rollout(args, parser)
    if args.command == "export-sft":
        from .dataset import export_sft
        result = export_sft(args.trace_dir, args.out, root_manifest=args.manifest,
            config=yaml.safe_load(args.config.read_text()), max_length=args.max_length, limit=args.limit,
            allowed_splits=(args.split,), allow_function_only=args.allow_function_only)
        print(json.dumps({key: result[key] for key in ("scope", "samples", "rejected", "by_mode", "by_bucket", "by_scope", "rejection_reasons")}, indent=2))
        return 0 if result["samples"] else 1
    if args.stage == "f6":
        from .dataset import collect_fixture_suite
        if args.fixture:
            parser.error("F6 uses the registered fixture suite selected by --reviews")
        directory = args.run_dir or Path("runs") / datetime.now(timezone.utc).strftime("f6_%Y%m%dT%H%M%S_%fZ")
        result = collect_fixture_suite(directory, config=yaml.safe_load(args.config.read_text()),
            root_manifest=Path("data/manifests/roots.jsonl"), review_file=args.reviews)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if all(r["stop_reason"] == "FINISHED" for r in result["results"]) else 1
    default_fixtures = {"f3": "fx02_range_sums.json", "f4": "fx03_bounded_repair.json"}
    fixture_path = args.fixture or Path("data/fixtures") / default_fixtures.get(args.stage, "fx01_integer_sum.json")
    fixture = json.loads(fixture_path.read_text())
    task = RootTask.model_validate(fixture["task"])
    if task.split != "SYN_FIXTURE":
        parser.error("Scripted smoke accepts SYN_FIXTURE tasks only")
    contract = ContractSpec.model_validate(fixture["contract"])
    script = fixture["script"] + (fixture.get("completion_script", []) if args.stage == "f2" else [])
    steps = [ScriptStep.model_validate_json(json.dumps(step)) for step in script]
    config = yaml.safe_load(args.config.read_text())
    limits = Limits(**{key: config["runtime"][key] for key in Limits.model_fields})
    run_dir = args.run_dir or Path("runs") / datetime.now(timezone.utc).strftime(f"{args.stage}_%Y%m%dT%H%M%S_%fZ")
    verifier = None
    if args.stage != "f1":
        verifier = fixture_verifier(fixture, config)
    runtime_class = {"f3": RecursiveRuntime, "f4": RepairRuntime}.get(args.stage, AtomicRuntime)
    extra = {"repair_config": config["repair"]} if args.stage == "f4" else {}
    runtime = runtime_class(
        task, contract, ScriptedPolicy(steps, name=task.root_id), run_dir,
        limits=limits, protocol_version=config["protocol_version"], verifier=verifier, **extra,
    )
    result = runtime.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    success = result["stop_reason"] == "F1_READY_VERIFY" if args.stage == "f1" else (
        result["stop_reason"] == "FINISHED" and result["functional_status"] == "PASS"
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
