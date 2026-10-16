"""Evidence-scoped, one-action SFT export. Raw trajectories remain immutable."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from .prompts import PromptOverflow, prepare_prompt, render_messages
from .schema import Artifact, Evidence, Record, RootTask, TraceEvent, parse_action
from .trace import renderer_identity
from .verifier import canonical, digest


class Rejected(ValueError):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason


class EventReview(Record):
    event_id: str
    root_id: str
    node_id: str
    action_hash: str
    contract_hash: str
    verdict: Literal["SUPPORTED", "VIOLATION"]
    label_origin: Literal["scripted_fixture", "reviewed_reference", "controlled_transform"]
    rationale: str
    meta: dict = Field(default_factory=dict)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_root_manifest(path):
    roots, families = {}, {}
    splits = {"SYN_FIXTURE", "DEV_ENGINEERING", "DEV_MECHANISM", "TRAIN", "VALIDATION",
              "MECHANISM_HOLDOUT", "APPS_SEALED", "EXTERNAL_SEALED"}
    for row in read_jsonl(path):
        root, family, split = row["root_id"], row["family_id"], row["split"]
        if root in roots or split not in splits:
            raise ValueError("Duplicate root or unknown split in root manifest")
        if family in families and families[family] != split:
            raise ValueError(f"Family crosses splits: {family}")
        roots[root], families[family] = row, split
    for row in roots.values():
        if parent := row.get("parent_root_id"):
            if parent not in roots or (roots[parent]["family_id"], roots[parent]["split"]) != (row["family_id"], row["split"]):
                raise ValueError("Derived roots must inherit the parent's family and split")
    return roots


def _contained(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Trace reference escapes its run directory")
    return path


def _evidence_objects(value):
    if isinstance(value, dict):
        if {"evidence_id", "fingerprint", "binding", "coverage", "visibility"} <= value.keys():
            yield Evidence.model_validate_json(json.dumps(value))
        else:
            for item in value.values():
                yield from _evidence_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _evidence_objects(item)


def _matches(evidence, artifact, contract, root_id):
    b = evidence.binding
    return (b.get("root_id") == root_id and b.get("contract_id") == contract["contract_id"]
            and b.get("contract_version") == contract["version"] and b.get("contract_hash") == digest(contract)
            and b.get("artifact") == artifact and b.get("source_hash") == artifact["source_hash"]
            and evidence.target_ref == artifact["module_path"] and evidence.fingerprint == digest(b)
            and evidence.suite_hash == b.get("suite_hash") and not b.get("bundle_error"))


class RunData:
    def __init__(self, trace_path, roots):
        self.directory = trace_path.parent
        self.events = [TraceEvent.model_validate_json(json.dumps(row)) for row in read_jsonl(trace_path)]
        initial = json.loads((self.directory / "states/S0000.json").read_text())
        self.task = RootTask.model_validate(initial["root_task"])
        self.root = roots.get(self.task.root_id)
        if self.root is None or (self.task.family_id, self.task.split) != (self.root["family_id"], self.root["split"]):
            raise ValueError("Run root/family/split differs from the registered root")
        path = self.directory / "trace_manifest.json"
        self.manifest = json.loads(path.read_text()) if path.exists() else None
        if self.manifest and self.manifest["task"] != self.task.model_dump(mode="json"):
            raise ValueError("Trace manifest differs from its initial task")
        self.trace_hash = file_hash(trace_path)
        # F5 already saved exact model inputs; its pre-F6 runs have no run UUID.
        self.run_id = self.manifest["run_id"] if self.manifest else self.trace_hash
        result_path = self.directory / "result.json"
        self.result = json.loads(result_path.read_text()) if result_path.exists() else None
        self.evidence = {}
        for path in sorted((self.directory / "evidence").glob("*.json")):
            e = Evidence.model_validate_json(path.read_text())
            if e.visibility == "PRIVATE" and self.task.split != "TRAIN":
                continue  # SEALED/private evaluation cannot select training targets.
            self.evidence[e.evidence_id] = e
        self.reviews = {}
        path = self.directory / "supervision.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                review = EventReview.model_validate(row)
                if review.event_id in self.reviews:
                    raise ValueError("Duplicate event review")
                self.reviews[review.event_id] = review
        self.published_before = {}
        published = {}
        for event in self.events:
            if event.event_id in self.published_before:
                raise ValueError("Duplicate event identity within a trace")
            self.published_before[event.event_id] = dict(published)
            state = json.loads(_contained(self.directory, event.state_after_ref).read_text())
            for e in _evidence_objects([event.host_result.details, state.get("nodes", {})]):
                if e.visibility == "VISIBLE":
                    published[e.evidence_id] = e.model_dump(mode="json")

    def review(self, event, action):
        review = self.reviews.get(event.event_id)
        if review and (review.root_id != event.root_id or review.node_id != event.node_id
                       or review.action_hash != digest(action)
                       or review.contract_hash != digest(event.observation_before["incoming_contract"])):
            raise Rejected("REVIEW_BINDING_MISMATCH")
        return review

    def passing(self, artifact, contract):
        matched = [e for e in self.evidence.values() if _matches(e, artifact, contract, self.task.root_id)]
        if not matched:
            raise Rejected("NO_MATCHING_EXECUTION_EVIDENCE")
        # A later repaired version never rescues this version; a conflicting
        # matching suite cannot be hidden by selecting only a passing subset.
        if any(e.status != "PASS" for e in matched):
            raise Rejected("FUNCTION_NOT_PASS", ",".join(e.evidence_id + ":" + e.status for e in matched))
        if any(e.responsibility_status == "VIOLATION" for e in matched):
            raise Rejected("RESPONSIBILITY_VIOLATION")
        if any(e.details.get("executed_cases", 0) < 1 or not {o["id"] for o in contract["ensures"]} <= set(e.coverage) for e in matched):
            raise Rejected("INSUFFICIENT_EXECUTION_COVERAGE")
        return matched


def _artifact_for_write(event, action):
    artifact = Artifact.model_validate(event.host_result.details["artifact"]).model_dump()
    obs = event.observation_before
    if (artifact["node_id"] != event.node_id or artifact["contract_version"] != obs["incoming_contract"]["version"]
            or artifact["source_hash"] != hashlib.sha256(action["args"]["source"].encode()).hexdigest()
            or artifact["dependency_bindings"] != action["args"]["declared_dependencies"]):
        raise Rejected("ARTIFACT_BINDING_MISMATCH")
    return artifact


def _visible_match(obs, evidence_id, artifact, contract, root_id):
    matches = [Evidence.model_validate_json(json.dumps(e)) for e in obs["visible_evidence"] if e["evidence_id"] == evidence_id]
    if len(matches) != 1 or not _matches(matches[0], artifact, contract, root_id):
        raise Rejected("VISIBLE_EVIDENCE_BINDING_MISMATCH")
    return matches[0]


def classify(run, event, action, allow_function_only):
    obs, args, name = event.observation_before, action["args"], action["action"]
    contract = obs["incoming_contract"]
    review = run.review(event, action)
    if review and review.verdict == "VIOLATION":
        raise Rejected("REVIEWED_RESPONSIBILITY_VIOLATION", review.rationale)
    origin = review.label_origin if review else "verified_model_candidate" if event.model_identity.get("backend") == "transformers" else "independent_oracle"
    refs, artifact = [], None
    scope = "PROTOCOL_ONLY"
    if name in {"localize", "repair"} or (name == "plan" and args["base_plan_version"] > 0):
        raise Rejected("REPAIR_OUTSIDE_CORE_SFT")
    if name == "write_code":
        bucket = "D_COMPOSE" if obs["mode"] == "COMPOSE" else "D_IMPL"
        artifact = _artifact_for_write(event, action)
        evidence = run.passing(artifact, contract)
        refs = [e.evidence_id for e in evidence]
        if artifact["dependency_bindings"]:
            selected = {(d["child_id"], d["artifact_version"], d["artifact_hash"]) for d in artifact["dependency_bindings"]}
            for e in evidence:
                usage = e.details.get("dependency_usage", [])
                used = {(u["child_id"], u["artifact_version"], u["artifact_hash"]) for u in usage
                        if u.get("selected") and u.get("linked") and u.get("called") is True}
                if selected != used:
                    raise Rejected("DEPENDENCY_NOT_USED")
        scope = "RESPONSIBILITY_SUPPORTED" if review else "FUNCTION_ONLY"
        if not review and not allow_function_only:
            raise Rejected("FUNCTION_ONLY", "Passing tests do not establish responsibility boundaries")
    elif name == "plan":
        bucket = "D_PLAN"
        children = {c["local_key"] for c in args["children"]}
        for obligation in contract["ensures"]:
            owners = args["obligation_map"].get(obligation["id"], [])
            if not owners or any(not (v.lower() == "self" or v.startswith("SELF:") or v.startswith("self:")
                                     or (v.startswith("child:") and v[6:] in children)) for v in owners):
                raise Rejected("PLAN_OBLIGATION_GAP")
        if event.host_result.details.get("missing_obligation_ids"):
            raise Rejected("PLAN_OBLIGATION_GAP")
        if not review:
            raise Rejected("PLAN_PROTOCOL_ONLY", "Semantic plan quality has no independent review")
        writes = [e for e in run.events if e.node_id == event.node_id and e.host_result.code == "SOURCE_SAVED"
                  and e.observation_before.get("current_plan", {}).get("version") == args["base_plan_version"] + 1
                  and e.observation_before["incoming_contract"] == contract]
        if not writes:
            raise Rejected("PLAN_WITHOUT_EXECUTION")
        artifact = _artifact_for_write(writes[0], parse_action(writes[0].raw_output).model_dump(mode="json"))
        refs = [e.evidence_id for e in run.passing(artifact, contract)]
        scope = "RESPONSIBILITY_SUPPORTED"
    elif name in {"consume", "reject", "request_child_repair"}:
        bucket = "D_COMPOSE"
        returned = obs.get("returned_child")
        if not returned or returned["result"]["node_id"] != args["child_id"]:
            raise Rejected("CHILD_BINDING_MISMATCH")
        artifact = returned["result"]["artifact"]
        if not artifact or artifact["version"] != args["artifact_version"]:
            raise Rejected("CHILD_BINDING_MISMATCH")
        refs = list(args["evidence_ids"])
        evidence = [_visible_match(obs, ref, artifact, returned["contract"], event.root_id) for ref in refs]
        if name == "consume":
            if not evidence or any(e.status != "PASS" for e in evidence) or args["artifact_hash"] != artifact["source_hash"]:
                raise Rejected("UNVERIFIED_CHILD_CONSUMPTION")
            if not review:
                raise Rejected("CONSUMPTION_PROTOCOL_ONLY")
            scope = "RESPONSIBILITY_SUPPORTED"
        elif not evidence or not any(e.status in {"FAIL", "UNKNOWN"} for e in evidence):
            raise Rejected("UNSUPPORTED_CHILD_REJECTION")
    elif name in {"verify", "finish", "abstain"}:
        bucket = "D_CONTROL"
        artifact = obs.get("current_artifact")
        if name == "verify":
            e = Evidence.model_validate_json(json.dumps(event.host_result.details["evidence"]))
            if (not artifact or not _matches(e, artifact, contract, event.root_id) or e.visibility != "VISIBLE"
                    or e.status == "ERROR" or args["suite_id"] not in obs["available_suites"]
                    or args["target_ref"] != artifact["module_path"] or e.suite_id != args["suite_id"]):
                raise Rejected("INVALID_VERIFICATION_BINDING")
            refs = [e.evidence_id]
        elif name == "finish":
            if not artifact or args["artifact_version"] != artifact["version"]:
                raise Rejected("FINISH_VERSION_MISMATCH")
            e = _visible_match(obs, event.host_result.details["evidence_id"], artifact, contract, event.root_id)
            if e.status == "ERROR":
                raise Rejected("INFRASTRUCTURE_EVIDENCE")
            refs = [e.evidence_id]
        else:
            refs = list(args["evidence_ids"])
            known = {e["evidence_id"]: e for e in obs["visible_evidence"]}
            if not set(refs) <= known.keys():
                raise Rejected("UNKNOWN_EVIDENCE")
            justified = any(known[r]["status"] in {"FAIL", "UNKNOWN"} for r in refs)
            justified |= any(v == 0 for k, v in obs["budget_remaining"].items() if k in {"tokens", "policy_calls", "patch_attempts"})
            justified |= obs.get("last_action_result", {}).get("status") == "REJECTED" if obs.get("last_action_result") else False
            if not justified and not review:
                raise Rejected("ABSTENTION_NOT_JUSTIFIED")
    else:
        raise Rejected("UNSUPPORTED_SUPERVISION_ACTION")
    return {"bucket": bucket, "supervision_scope": scope, "label_origin": origin,
            "evidence_refs": refs, "artifact": artifact, "review": review.model_dump() if review else None}


def _render_input(run, event, tokenizer, config):
    obs = event.observation_before
    for e in _evidence_objects(obs):
        if e.visibility != "VISIBLE" or run.published_before[event.event_id].get(e.evidence_id) != e.model_dump(mode="json"):
            raise Rejected("FUTURE_OR_PRIVATE_EVIDENCE_IN_INPUT")
    if event.model_input is not None:
        saved = event.model_input
        messages = render_messages(obs, compact=saved["compression"] == "evidence_bindings_and_duplicates")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if messages != saved["messages"] or prompt != saved["prompt"]:
            raise Rejected("MODEL_INPUT_RENDERER_MISMATCH")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids != saved["input_token_ids"]:
            raise Rejected("MODEL_INPUT_TOKENIZER_MISMATCH")
        return messages, ids, saved["compression"]
    if not run.manifest or run.manifest["renderer"] != renderer_identity():
        raise Rejected("SCRIPTED_RENDERER_MISMATCH")
    prepared = prepare_prompt(tokenizer, obs, context_tokens=config["runtime"]["context_tokens"],
                              generation_tokens=config["runtime"]["generation_tokens"])
    return prepared.record["messages"], prepared.token_ids, prepared.record["compression"]


def export_sft(trace_dir, out, *, root_manifest, config, max_length=4096, limit=None,
               allowed_splits=("SYN_FIXTURE",), allow_function_only=False, tokenizer=None):
    allowed_splits = tuple(allowed_splits)
    if max_length < 1 or (limit is not None and limit < 1):
        raise ValueError("Positive sequence/sample limits are required")
    if not set(allowed_splits) <= {"SYN_FIXTURE", "TRAIN"}:
        raise ValueError("SFT export accepts SYN_FIXTURE smoke or TRAIN; development/holdout/SEALED are excluded")
    if len(set(allowed_splits)) != 1:
        raise ValueError("Keep fixture smoke and formal TRAIN exports separate")
    out = Path(out)
    paths = [out, out.with_suffix(".rejected.jsonl"), out.with_suffix(".manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose new SFT output paths; existing exports are immutable")
    roots = load_root_manifest(root_manifest)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["local_path"], local_files_only=True)
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    tokenizer_files = {name: file_hash(Path(config["model"]["local_path"]) / name) for name in tokenizer_names}
    tokenizer_hash = digest(tokenizer_files)
    model_manifest = json.loads(Path(config["model"]["identity_manifest"]).read_text())
    if (tokenizer_hash != model_manifest["tokenizer_sha256"]
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != model_manifest["chat_template_sha256"]):
        raise ValueError("SFT tokenizer/template differs from the pinned model identity")
    trace_paths = sorted(Path(trace_dir).rglob("trace.jsonl"))
    if not trace_paths:
        raise ValueError("No raw traces found")
    samples, rejections, sources, seen = [], [], [], set()
    for trace_path in trace_paths:
        run = RunData(trace_path, roots)
        sources.append({"run_id": run.run_id, "trace": str(trace_path.resolve()), "trace_sha256": run.trace_hash,
                        "root": run.root, "raw_events": len(run.events),
                        "policy_events": sum(e.event_kind == "POLICY" for e in run.events),
                        "result": str((run.directory / "result.json").resolve()) if run.result else None})
        for event in run.events:
            identity = (run.run_id, event.event_id)
            sample_id = ":".join(identity)
            rejected = {"sample_id": sample_id, "run_id": run.run_id, "event_id": event.event_id,
                        "root_id": event.root_id, "family_id": run.task.family_id, "split": run.task.split,
                        "trace": str(trace_path.resolve())}
            try:
                if identity in seen:
                    raise Rejected("DUPLICATE_EVENT")
                seen.add(identity)
                if event.event_kind != "POLICY":
                    raise Rejected("HOST_EVENT")
                if run.task.split not in allowed_splits:
                    raise Rejected("SPLIT_NOT_ALLOWED")
                if not run.result or run.result["phase"] != "TERMINAL":
                    raise Rejected("INCOMPLETE_RUN")
                obs = event.observation_before
                if (config.get("data_check_protocol_version")
                        and obs["protocol_version"] != config["protocol_version"]):
                    raise Rejected("RUNTIME_PROTOCOL_MISMATCH")
                if event.root_id != run.task.root_id or (obs["root_id"], obs["node_id"]) != (event.root_id, event.node_id):
                    raise Rejected("EVENT_IDENTITY_MISMATCH")
                if event.generation_truncated:
                    raise Rejected("TRUNCATED_OUTPUT")
                parsed = parse_action(event.raw_output or "")
                if parsed != event.parsed_action or parsed.action not in obs["allowed_actions"]:
                    raise Rejected("ACTION_MISMATCH_OR_NOT_ALLOWED")
                if event.host_result.status != "APPLIED":
                    raise Rejected("HOST_REJECTED_OR_ERROR", event.host_result.code)
                action = parsed.model_dump(mode="json")
                label = classify(run, event, action, allow_function_only)
                messages, prefix_ids, compression = _render_input(run, event, tokenizer, config)
                target = canonical(action)
                target_tokens = len(tokenizer.encode(target, add_special_tokens=False)) + 1  # EOS, mask belongs to F7.
                if len(prefix_ids) + target_tokens > max_length:
                    raise Rejected("SEQUENCE_TOO_LONG", f"{len(prefix_ids)} + {target_tokens} > {max_length}")
                if limit is not None and len(samples) >= limit:
                    raise Rejected("SAMPLE_LIMIT")
                samples.append({
                    "sample_id": sample_id, "root_id": event.root_id, "family_id": run.task.family_id,
                    "split": run.task.split, "mode": obs["mode"], "bucket": label["bucket"],
                    "messages": messages, "target_action": action, "target": target,
                    "evidence_refs": label["evidence_refs"], "supervision_scope": label["supervision_scope"],
                    "label_origin": label["label_origin"],
                    "meta": {"run_id": run.run_id, "event_id": event.event_id, "node_id": event.node_id,
                             "trace": str(trace_path.resolve()), "state_before_ref": obs["state_id"],
                             "protocol_version": obs["protocol_version"], "source": run.root,
                             "data_check_protocol_version": config.get("data_check_protocol_version"),
                             "model_identity": event.model_identity, "contract_id": obs["incoming_contract"]["contract_id"],
                             "contract_version": obs["incoming_contract"]["version"],
                             "plan_version": action["args"]["base_plan_version"] + 1 if parsed.action == "plan" else
                                             (obs["current_plan"]["version"] if obs.get("current_plan") else None),
                             "artifact": label["artifact"], "review": label["review"],
                             "root_functional_status": run.result["functional_status"],
                             "prefix_tokens": len(prefix_ids), "target_tokens": target_tokens, "compression": compression},
                })
            except (Rejected, PromptOverflow) as error:
                rejections.append({**rejected, "reason": error.reason if isinstance(error, Rejected) else error.code,
                                   "detail": str(error)})
            except (ValidationError, KeyError, TypeError, ValueError) as error:
                rejections.append({**rejected, "reason": "INVALID_EVENT", "detail": str(error)})
    manifest = {"dataset_version": "f6-one-action-v1", "scope": "SMOKE_ONLY" if allowed_splits == ("SYN_FIXTURE",) else "TRAIN",
                "protocol_version": config["protocol_version"],
                "data_check_protocol_version": config.get("data_check_protocol_version"),
                "allowed_splits": list(allowed_splits), "allow_function_only": allow_function_only,
                "max_length": max_length, "limit": limit, "raw_sources": sources,
                "root_manifest": str(Path(root_manifest).resolve()), "root_manifest_sha256": file_hash(root_manifest),
                "renderer": renderer_identity(), "base_model": config["model"],
                "tokenizer_sha256": tokenizer_hash, "exporter_sha256": file_hash(__file__),
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "eos_token_id": tokenizer.eos_token_id,
                "samples": len(samples), "rejected": len(rejections),
                "by_mode": dict(Counter(s["mode"] for s in samples)), "by_bucket": dict(Counter(s["bucket"] for s in samples)),
                "by_scope": dict(Counter(s["supervision_scope"] for s in samples)),
                "rejection_reasons": dict(Counter(r["reason"] for r in rejections))}
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths[:2], (samples, rejections)):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    manifest["output_sha256"] = file_hash(out)
    with paths[2].open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def collect_fixture_suite(run_dir, *, config, root_manifest, review_file):
    """Execute the reviewed fixtures first, then bind offline reviews to real events."""
    from .cli import fixture_verifier
    from .policy import ScriptedPolicy, ScriptStep
    from .runtime import Limits, RepairRuntime

    roots = load_root_manifest(root_manifest)
    definitions = json.loads(Path(review_file).read_text())
    prepared = []
    for definition in definitions["fixtures"]:
        path = Path(definition["fixture_path"])
        fixture = json.loads(path.read_text())
        task = RootTask.model_validate(fixture["task"])
        root = roots[task.root_id]
        if task.split != "SYN_FIXTURE" or (task.family_id, task.split) != (root["family_id"], root["split"]):
            raise ValueError("Fixture suite is SYN_FIXTURE only; derived roots must match the root manifest")
        if file_hash(path) != definition["fixture_hash"] or file_hash(path) != root["source_hash"]:
            raise ValueError("Fixture changed after its pre-execution review")
        steps = [ScriptStep.model_validate_json(json.dumps(s)) for s in fixture["script"] + fixture.get("completion_script", [])]
        for review in definition["reviews"]:
            step = steps[review["script_step"]]
            raw = step.output if isinstance(step.output, str) else json.dumps(step.output)
            if step.node_id != review["node_id"] or digest(parse_action(raw).model_dump(mode="json")) != review["action_hash"]:
                raise ValueError("Reviewed action differs from the fixture script")
        prepared.append((definition, fixture, task, steps))
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "fixture_reviews.json").write_text(json.dumps(definitions, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "request.json").write_text(json.dumps({"stage": "F6", "scope": "SMOKE_ONLY", "config": config,
        "root_manifest": str(Path(root_manifest).resolve()), "review_file": str(Path(review_file).resolve())}, indent=2) + "\n")
    results = []
    limits = Limits(**{key: config["runtime"][key] for key in Limits.model_fields})
    for definition, fixture, task, steps in prepared:
        from .schema import ContractSpec
        runtime = RepairRuntime(task, ContractSpec.model_validate(fixture["contract"]),
            ScriptedPolicy(steps, name=task.root_id), run_dir / task.root_id, limits=limits,
            verifier=fixture_verifier(fixture, config), protocol_version=config["protocol_version"],
            repair_config=config["repair"], implementation_stage="F6")
        result = runtime.run()
        events = [TraceEvent.model_validate_json(json.dumps(row)) for row in read_jsonl(runtime.trace.path)]
        decisions = [event for event in events if event.event_kind == "POLICY"]
        with (runtime.run_dir / "supervision.jsonl").open("x", encoding="utf-8") as stream:
            for review in definition["reviews"]:
                if review["script_step"] >= len(decisions):
                    continue  # No fabricated event if a fixture terminates before this decision.
                event = decisions[review["script_step"]]
                if event.parsed_action is None or digest(event.parsed_action.model_dump(mode="json")) != review["action_hash"]:
                    raise ValueError("Executed action differs from its pre-execution review")
                contract = event.observation_before["incoming_contract"]
                spec = {key: value for key, value in contract.items() if key not in {"contract_id", "version"}}
                if digest(spec) != review["contract_spec_hash"]:
                    raise ValueError("Executed contract differs from its pre-execution review")
                bound = EventReview(event_id=event.event_id, root_id=task.root_id, node_id=event.node_id,
                    action_hash=review["action_hash"], contract_hash=digest(contract), verdict=review["verdict"],
                    label_origin=review["label_origin"], rationale=review["rationale"],
                    meta={"fixture_hash": definition["fixture_hash"], "script_step": review["script_step"]})
                stream.write(bound.model_dump_json() + "\n")
        results.append({key: result[key] for key in ("root_id", "stop_reason", "functional_status", "usage", "run_dir")})
    summary = {"stage": "F6", "scope": "SMOKE_ONLY", "results": results, "run_dir": str(run_dir)}
    (run_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
