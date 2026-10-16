"""Evidence-bounded repair transactions over immutable source snapshots."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

from .artifacts import ArtifactStore, SourceError
from .runtime import ProtocolError
from .schema import Consumption, Dependency, HostResult, Phase, RepairTicket
from .verifier import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RepairManager:
    def __init__(self, runtime, candidate_limit=3, parent_expansions=1,
                 automatic_commit_rule="visible_all_pass_no_regression",
                 candidate_strategy="traceback"):
        if not 1 <= candidate_limit <= 3 or parent_expansions not in (0, 1):
            raise ValueError("F4 supports up to three candidates and one parent expansion")
        if automatic_commit_rule != "visible_all_pass_no_regression":
            raise ValueError("Unsupported repair acceptance rule")
        if candidate_strategy not in {"traceback", "responsibility_filtered"}:
            raise ValueError("Unsupported responsibility candidate strategy")
        self.candidate_strategy = candidate_strategy
        self.runtime, self.base_dir = runtime, runtime.run_dir
        self.candidate_limit, self.parent_limit = candidate_limit, parent_expansions
        self.ticket = self.snapshot = self.revision = None
        self.serial = self.expansions = 0
        self.plan_counts, self.failed_at, self.history = {}, {}, []

    def visible_evidence(self):
        r = self.runtime
        ids = {r.active_node_id, *r.context.children}
        if r.node.parent_id:
            ids.add(r.node.parent_id)
        evidence = [r.contexts[n].node.last_evidence for n in ids if n in r.contexts]
        return sorted((e for e in evidence if e and e.visibility == "VISIBLE"), key=lambda e: e.evidence_id)

    def indexes(self):
        r = self.runtime
        owners, artifacts, evidence = {}, {}, {}
        for node_id, context in r.contexts.items():
            if context.artifact:
                a = context.artifact
                owners[f"nodes/{node_id}.py:{a.public_symbol}"] = node_id
                consumers = [f"{owner}<-{node_id}" for owner, ctx in r.contexts.items() if ctx.artifact
                             and any(d.child_id == node_id for d in ctx.artifact.dependency_bindings)]
                artifacts[f"{node_id}@v{a.version}"] = {"contract_id": context.node.incoming_contract.contract_id,
                    "contract_version": a.contract_version, "source_hash": a.source_hash, "consumer_edges": consumers}
            e = context.node.last_evidence
            if e and e.visibility == "VISIBLE":
                evidence[e.evidence_id] = {"node_id": node_id, "target_ref": e.target_ref, "status": e.status,
                                           "coverage": e.coverage, "fingerprint": e.fingerprint}
        return {"module_owner": owners, "artifact_contract_consumers": artifacts, "evidence_target": evidence}

    def candidates(self):
        r = self.runtime
        traceback_nodes = []
        for e in self.visible_evidence():
            if e.status == "FAIL":
                traceback_nodes.extend(re.findall(r"/work/nodes/([A-Za-z_][A-Za-z0-9_]*)\.py", json.dumps(e.details)))
        if self.candidate_strategy == "responsibility_filtered":
            # R1 screens by current, contract-bound failures before the policy
            # chooses an owner/handoff. This performs no additional execution.
            failed = []
            for evidence in self.visible_evidence():
                node_id = evidence.binding["artifact"]["node_id"]
                ctx = r.contexts[node_id]
                if (evidence.status == "FAIL" and ctx.artifact is not None
                        and r.verifier.is_current(evidence, r.run_dir, r.task.root_id,
                                                  ctx.node.incoming_contract, ctx.artifact)):
                    failed.append(node_id)
            ids = sorted(set(failed), key=lambda n: (n == r.active_node_id, n))
            # Missing local oracles do not establish a failed child. Keep the
            # current owner available for an explicit omission/plan decision.
            if not ids:
                ids = [r.active_node_id]
        else:
            ids = list(reversed(traceback_nodes)) + [r.active_node_id] + r.context.children
        if self.failed_at.get(r.active_node_id) and self.expansions < self.parent_limit and r.node.parent_id:
            ids.append(r.node.parent_id)
        unique = list(dict.fromkeys(n for n in ids if n in r.contexts))[:self.candidate_limit]
        return [{"node_id": n, "parent_id": r.contexts[n].node.parent_id,
                 "handoff": f"{r.contexts[n].node.parent_id}<-{n}" if r.contexts[n].node.parent_id else None,
                 "status": r.contexts[n].node.last_evidence.status if r.contexts[n].node.last_evidence else None}
                for n in unique]

    def plan_authorized(self):
        r = self.runtime
        return bool(self.ticket and self.ticket.boundary_kind == "PLAN_SUBTREE"
                    and r.active_node_id == self.ticket.target["node_id"] and r.node.phase == Phase.REPAIR_AUTHORIZED)

    def _descendants(self, node_id):
        result = {node_id}
        for child in self.runtime.contexts[node_id].children:
            result.update(self._descendants(child))
        return result

    def _affected(self, node_id):
        r, affected = self.runtime, {node_id}
        while True:
            consumers = {n for n, ctx in r.contexts.items() if ctx.artifact
                         and any(d.child_id in affected for d in ctx.artifact.dependency_bindings)}
            if consumers <= affected:
                return affected
            affected |= consumers

    def _suite_identity(self):
        return digest({key: suite.model_dump(mode="json") for key, suite in self.runtime.verifier.suites.items()
                       if suite.visibility == "VISIBLE"})

    def _copy_program(self, source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("artifacts", "plans", "workspaces"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name)
        if (source / "consumptions.jsonl").exists():
            shutil.copyfile(source / "consumptions.jsonl", destination / "consumptions.jsonl")

    def _file_hashes(self, directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for name in ("artifacts", "plans") if (directory / name).exists()
                for p in (directory / name).rglob("*") if p.is_file()}

    def archive_evidence(self, evidence):
        r = self.runtime
        if r.run_dir != self.base_dir:
            source = r.run_dir / "evidence" / f"{evidence.evidence_id}.json"
            target = self.base_dir / "evidence" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)

    def _check(self, node_id, suite_id, *, update=True, reuse=False):
        r, ctx = self.runtime, self.runtime.contexts[node_id]
        previous = ctx.node.last_evidence
        if reuse and previous and previous.suite_id == suite_id and r.verifier.is_current(
                previous, r.run_dir, r.task.root_id, ctx.node.incoming_contract, ctx.artifact):
            return previous
        if r.budget.verify_calls >= r.limits.max_verify_calls:
            raise ProtocolError("VERIFY_BUDGET_EXHAUSTED", "Automatic replay shares the root verification budget")
        r.budget = r.budget.model_copy(update={"verify_calls": r.budget.verify_calls + 1})
        e = r.verifier.verify(r.run_dir, r.task.root_id, ctx.node.incoming_contract, ctx.artifact,
                              suite_id, f"E{r.budget.verify_calls:04d}")
        r.budget = r.budget.model_copy(update={
            "executed_cases": r.budget.executed_cases + e.details["executed_cases"],
            "execution_seconds": r.budget.execution_seconds + e.details["elapsed_seconds"],
        })
        self.archive_evidence(e)
        if update:
            ctx.node = ctx.node.model_copy(update={"last_evidence": e})
        return e

    @staticmethod
    def _outcome(evidence):
        return (evidence.status, evidence.details["reason"],
                [(c["case_id"], c["status"], c["reason"], c["execution"]["value"])
                 for c in evidence.details["cases"]])

    def localize(self, args):
        r = self.runtime
        if self.revision:
            raise ProtocolError("REVISION_IN_PROGRESS", "Complete or abandon the current plan reconstruction before another ticket")
        if r.budget.patch_attempts >= r.limits.max_patch_attempts:
            raise ProtocolError("PATCH_BUDGET_EXHAUSTED", "No root repair attempts remain")
        visible = {e.evidence_id: e for e in self.visible_evidence()}
        if not args.evidence_ids or not set(args.evidence_ids) <= visible.keys():
            raise ProtocolError("UNKNOWN_EVIDENCE", "Localization requires existing visible evidence")
        selected = [visible[key] for key in args.evidence_ids]
        if any(e.status == "ERROR" for e in selected):
            raise ProtocolError("INFRASTRUCTURE_EVIDENCE", "Runner errors do not authorize model code repair")
        target = args.target_id
        if args.target_type == "HANDOFF":
            parts = target.split("<-")
            if len(parts) != 2 or parts[0] not in r.contexts or parts[1] not in r.contexts[parts[0]].children:
                raise ProtocolError("INVALID_HANDOFF", "Handoff must be an existing direct parent-child edge")
            target = parts[0] if args.boundary_kind == "CONSUMER_MODULE" else parts[1]
        elif args.boundary_kind == "CONSUMER_MODULE":
            raise ProtocolError("INVALID_BOUNDARY", "Consumer-module boundaries require a handoff")
        if target not in r.contexts:
            raise ProtocolError("UNKNOWN_TARGET", "Unknown responsibility target")
        parent = r.node.parent_id
        expanding = target == parent
        if expanding and (not self.failed_at.get(r.active_node_id) or self.expansions >= self.parent_limit):
            raise ProtocolError("UNSUPPORTED_EXPANSION", "Parent expansion requires a failed local attempt and is allowed once")
        allowed = {c["node_id"] for c in self.candidates()}
        if target not in allowed and not expanding:
            raise ProtocolError("UNSUPPORTED_BOUNDARY", "Target is outside the evidence-derived candidate set")
        ctx = r.contexts[target]
        subtree = self._descendants(target) if args.boundary_kind == "PLAN_SUBTREE" else {target}
        if args.boundary_kind == "PLAN_SUBTREE":
            if args.target_type not in {"CONTRACT", "OMISSION"} or ctx.plan is None:
                raise ProtocolError("INVALID_PLAN_BOUNDARY", "Plan revisions require a contract/omission candidate with an existing plan")
            if self.plan_counts.get(target, 0) >= r.limits.max_plan_revisions:
                raise ProtocolError("PLAN_REVISION_BUDGET_EXHAUSTED", "Per-node plan revision budget exhausted")
            if self._affected(target) - subtree:
                raise ProtocolError("CONSUMER_PLAN_REQUIRED", "An already-built consumer must be included in the plan boundary")
        elif ctx.artifact is None:
            raise ProtocolError("NO_ARTIFACT_BOUNDARY", "Code repair requires an existing candidate module")
        anchor = next((e for e in selected if e.status == "FAIL"), selected[0])
        owner = anchor.binding["artifact"]["node_id"]
        if not r.verifier.is_current(anchor, r.run_dir, r.task.root_id, r.contexts[owner].node.incoming_contract, r.contexts[owner].artifact):
            raise ProtocolError("STALE_EVIDENCE", "Localization must reference the current candidate fingerprint")
        replay = self._check(owner, anchor.suite_id, update=False)
        if replay.status == "ERROR":
            return HostResult(status="ERROR", code="REPLAY_ERROR", details={"evidence": replay.model_dump()})
        if self._outcome(replay) != self._outcome(anchor):
            return HostResult(status="APPLIED", code="UNSTABLE_REPLAY", details={"original": anchor.model_dump(), "replay": replay.model_dump()})
        self.serial += 1
        ticket_id = f"T{self.serial}"
        directory = self.base_dir / "repairs" / ticket_id
        affected = self._affected(target)
        if r.contexts["N0"].artifact:
            affected.add("N0")
        edges = [f"{n}<-{d.child_id}" for n, c in r.contexts.items() if c.artifact
                 for d in c.artifact.dependency_bindings if n in affected and (n == target or d.child_id in affected)]
        frozen = {f"nodes/{n}.py": c.artifact.source_hash for n, c in r.contexts.items() if c.artifact and n not in subtree}
        self.ticket = RepairTicket(ticket_id=ticket_id, snapshot_id=r.state_ref,
            target={"type": args.target_type, "id": args.target_id, "node_id": target}, boundary_kind=args.boundary_kind,
            editable_files=[f"nodes/{target}.py"] if args.boundary_kind != "PLAN_SUBTREE" else [],
            editable_plan_nodes=[target] if args.boundary_kind == "PLAN_SUBTREE" else [],
            rebind_edges=edges, revalidate_nodes=sorted(affected), frozen_hashes=frozen,
            evidence_ids=args.evidence_ids, remaining_attempts=r.limits.max_patch_attempts-r.budget.patch_attempts)
        self.snapshot = {"contexts": deepcopy(r.contexts), "stack": list(r.stack), "active": r.active_node_id,
                         "stop_reason": r.stop_reason, "directory": directory, "target": target, "subtree": subtree,
                         "suite_hash": self._suite_identity(), "runner": r.verifier.sandbox.identity(),
                         "files": self._file_hashes(self.base_dir)}
        self._copy_program(self.base_dir, directory / "snapshot")
        write_json(directory / "snapshot/state.json", r._state())
        write_json(directory / "snapshot/visible_suites.json", {k:s.model_dump(mode="json") for k,s in r.verifier.suites.items() if s.visibility == "VISIBLE"})
        write_json(directory / "snapshot/runner.json", self.snapshot["runner"])
        write_json(directory / "indexes.json", self.indexes())
        write_json(directory / "ticket.json", self.ticket.model_dump())
        write_json(directory / "baseline.json", replay.model_dump())
        if expanding:
            self.expansions += 1
            r.context.termination_reason = "ESCALATED_TO_PARENT"
            r.node = r.node.model_copy(update={"phase": Phase.TERMINAL})
            r.active_node_id = target
            r.stack = r.stack[:r.stack.index(target)+1]
        elif args.boundary_kind == "PLAN_SUBTREE" and target != r.active_node_id:
            chain, cursor = [], target
            while cursor is not None:
                chain.append(cursor)
                cursor = r.contexts[cursor].node.parent_id
            r.active_node_id, r.stack = target, list(reversed(chain))
        r.node = r.node.model_copy(update={"phase": Phase.REPAIR_AUTHORIZED})
        return HostResult(status="APPLIED", code="REPAIR_AUTHORIZED", details={"ticket": self.ticket.model_dump(), "baseline_evidence": replay.evidence_id})

    def _charge(self):
        r = self.runtime
        if r.budget.patch_attempts >= r.limits.max_patch_attempts:
            raise ProtocolError("PATCH_BUDGET_EXHAUSTED", "All repair attempts share the root budget")
        r.budget = r.budget.model_copy(update={"patch_attempts": r.budget.patch_attempts + 1})
        if self.ticket:
            self.ticket = self.ticket.model_copy(update={"remaining_attempts": r.limits.max_patch_attempts-r.budget.patch_attempts})

    def _validate_ticket(self, ticket_id, kind):
        if not self.ticket or self.ticket.ticket_id != ticket_id or (self.ticket.boundary_kind == "PLAN_SUBTREE") != (kind == "plan"):
            raise ProtocolError("INVALID_TICKET", "Action does not match the active repair ticket")
        if self._suite_identity() != self.snapshot["suite_hash"] or self.runtime.verifier.sandbox.identity() != self.snapshot["runner"]:
            raise ProtocolError("STALE_TICKET", "Oracle or execution configuration changed since the snapshot")
        if self._file_hashes(self.base_dir) != self.snapshot["files"]:
            raise ProtocolError("STALE_TICKET", "Active program files changed since the snapshot")

    def _begin(self, action):
        r = self.runtime
        stage = self.snapshot["directory"] / f"attempt_{r.budget.patch_attempts}"
        self._copy_program(self.snapshot["directory"] / "snapshot", stage)
        write_json(stage / "action.json", action.model_dump(mode="json"))
        r.contexts = deepcopy(r.contexts)
        r.run_dir, r.store = stage, ArtifactStore(stage)
        return stage

    def _frozen_check(self):
        r = self.runtime
        if self._suite_identity() != self.snapshot["suite_hash"] or r.verifier.sandbox.identity() != self.snapshot["runner"]:
            raise ProtocolError("STALE_TICKET", "Fixed visible suites and runner configuration must remain unchanged")
        for path, expected in self.snapshot["files"].items():
            if hashlib.sha256((r.run_dir / path).read_bytes()).hexdigest() != expected:
                raise ProtocolError("FROZEN_FILE_CHANGED", f"Historical file changed: {path}")
        for node_id, old in self.snapshot["contexts"].items():
            if node_id in r.contexts and r.contexts[node_id].node.incoming_contract != old.node.incoming_contract:
                raise ProtocolError("INCOMING_CONTRACT_CHANGED", "Existing incoming contracts cannot be revised")
        for path, expected in self.ticket.frozen_hashes.items():
            node_id = Path(path).stem
            artifact = r.contexts[node_id].artifact
            if artifact.source_hash != expected or hashlib.sha256((r.run_dir / artifact.module_path).read_bytes()).hexdigest() != expected:
                raise ProtocolError("FROZEN_FILE_CHANGED", f"Frozen module changed: {path}")
        if self._file_hashes(self.base_dir) != self.snapshot["files"]:
            raise ProtocolError("FROZEN_FILE_CHANGED", "Live program changed while validating a candidate")

    def _checks(self, node_ids, *, reuse=False):
        r, evidence = self.runtime, []
        for node_id in sorted(node_ids):
            ctx = r.contexts[node_id]
            if ctx.artifact is None:
                continue
            for suite_id in r.verifier.available_suites(r.task.root_id, ctx.node.incoming_contract):
                evidence.append(self._check(node_id, suite_id, reuse=reuse))
        return evidence

    def _restore(self, stage, reason, evidence=()):
        r, snapshot = self.runtime, self.snapshot
        write_json(stage / "final_state.json", r._state())
        summary = {"ticket_id": self.ticket.ticket_id, "status": "ROLLED_BACK", "reason": reason,
                   "candidate_dir": str(stage), "evidence": [e.model_dump(mode="json") for e in evidence]}
        write_json(stage / "result.json", summary)
        self.history.append(summary)
        target = snapshot["target"]
        self.failed_at[target] = self.failed_at.get(target, 0) + 1
        r.contexts, r.stack = deepcopy(snapshot["contexts"]), list(snapshot["stack"])
        r.active_node_id, r.stop_reason = snapshot["active"], snapshot["stop_reason"]
        r.run_dir, r.store = self.base_dir, ArtifactStore(self.base_dir)
        self.ticket = self.snapshot = self.revision = None
        return HostResult(status="APPLIED", code="REPAIR_ROLLED_BACK", details=summary)

    def _commit(self, stage, evidence, status="CODE_REPAIR_ACCEPTED"):
        r = self.runtime
        self._frozen_check()
        # Only new immutable files may enter the active store; old versions stay untouched.
        additions = []
        for name in ("artifacts", "plans", "workspaces"):
            if not (stage / name).exists():
                continue
            for source in (stage / name).rglob("*"):
                if source.is_file():
                    dest = self.base_dir / source.relative_to(stage)
                    if dest.exists():
                        if source.read_bytes() != dest.read_bytes():
                            raise ProtocolError("IMMUTABLE_FILE_CHANGED", f"Cannot replace {dest.relative_to(self.base_dir)}")
                    else:
                        additions.append((source, dest))
        copied = []
        try:
            for source, dest in additions:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                copied.append(dest)
            if (stage / "consumptions.jsonl").exists():
                shutil.copyfile(stage / "consumptions.jsonl", self.base_dir / "consumptions.jsonl")
        except OSError:
            for path in reversed(copied):
                path.unlink(missing_ok=True)
            raise
        summary = {"ticket_id": self.ticket.ticket_id, "status": status, "candidate_dir": str(stage),
                   "evidence": [e.model_dump(mode="json") for e in evidence],
                   "root_checked": any(e.binding["artifact"]["node_id"] == "N0" for e in evidence)}
        write_json(stage / "result.json", summary)
        self.history.append(summary)
        r.run_dir, r.store = self.base_dir, ArtifactStore(self.base_dir)
        self.ticket = self.snapshot = self.revision = None
        return HostResult(status="APPLIED", code=status, details=summary)

    def repair(self, args):
        r = self.runtime
        self._charge()
        self._validate_ticket(args.ticket_id, "code")
        replacements = {p.file: p for p in args.replacements}
        if len(replacements) != len(args.replacements) or not set(replacements) <= set(self.ticket.editable_files):
            raise ProtocolError("OUTSIDE_REPAIR_BOUNDARY", "Patch may replace only ticket.editable_files")
        for path, patch in replacements.items():
            artifact = r.contexts[Path(path).stem].artifact
            if patch.base_hash != artifact.source_hash:
                raise ProtocolError("BASE_HASH_MISMATCH", "Patch must reference the snapshotted source hash")
        updates = {(u.consumer_node, u.child_id): u for u in args.dependency_updates}
        if len(updates) != len(args.dependency_updates) or any(f"{a}<-{b}" not in self.ticket.rebind_edges for a,b in updates):
            raise ProtocolError("UNAUTHORIZED_REBIND", "Dependency updates require an explicitly authorized edge")
        changed = {Path(path).stem for path in replacements} | {a for a,b in updates}
        if not changed:
            raise ProtocolError("EMPTY_REPAIR", "Repair requires source or binding changes")
        if any(u.artifact_ref.kind == "CANDIDATE" and (u.artifact_ref.node_id != u.child_id or u.child_id not in changed) for u in updates.values()):
            raise ProtocolError("INVALID_CANDIDATE_REF", "CANDIDATE must select a node changed by this transaction")
        stage = self._begin(args)
        evidence = []
        try:
            saved, visiting = {}, set()

            def save(node_id):
                if node_id in saved:
                    return saved[node_id]
                if node_id in visiting:
                    raise ProtocolError("REBIND_CYCLE", "Candidate updates must be acyclic")
                visiting.add(node_id)
                ctx = r.contexts[node_id]
                old = ctx.artifact
                dependencies = []
                for dep in old.dependency_bindings:
                    update = updates.get((node_id, dep.child_id))
                    if update:
                        child = save(update.child_id) if update.artifact_ref.kind == "CANDIDATE" else r.store.load(update.child_id, update.artifact_ref.version)
                        if update.artifact_ref.kind == "EXISTING" and update.artifact_ref.hash != child.source_hash:
                            raise ProtocolError("ARTIFACT_HASH_MISMATCH", "Existing dependency hash does not match the selected version")
                        dep = Dependency(child_id=child.node_id, artifact_version=child.version, artifact_hash=child.source_hash, alias=update.alias)
                        prior = ctx.consumptions[child.node_id]
                        decision = Consumption(**{**prior.model_dump(), "artifact_version": child.version,
                                                "artifact_hash": child.source_hash, "alias": update.alias,
                                                "evidence_refs": [], "accept_unverified": True,
                                                "known_failed_consumed": prior.known_failed_consumed
                                                    if (prior.artifact_version, prior.artifact_hash) == (child.version, child.source_hash) else False})
                        ctx.consumptions[child.node_id] = decision
                        with (stage / "consumptions.jsonl").open("a") as stream:
                            stream.write(decision.model_dump_json() + "\n")
                    dependencies.append(dep)
                source = replacements[f"nodes/{node_id}.py"].source if f"nodes/{node_id}.py" in replacements else (stage / old.module_path).read_text()
                artifact = r.store.save_next(old, ctx.node.incoming_contract, source, dependencies)
                ctx.artifact = saved[node_id] = artifact
                ctx.node = ctx.node.model_copy(update={"artifact_ref": artifact.module_path, "last_evidence": None})
                r.store.bundle(artifact).materialize(stage / "workspaces" / node_id / f"v{artifact.version}")
                visiting.remove(node_id)
                return artifact

            for node_id in sorted(changed):
                save(node_id)
            self._frozen_check()
            evidence = self._checks(set(self.ticket.revalidate_nodes) | changed)
            if not evidence or any(e.status != "PASS" for e in evidence):
                return self._restore(stage, "VISIBLE_CHECK_NOT_PASS", evidence)
            r.node = r.node.model_copy(update={"phase": Phase.REVIEW})
            return self._commit(stage, evidence)
        except (ProtocolError, SourceError, OSError, ValueError) as error:
            return self._restore(stage, f"{getattr(error, 'code', type(error).__name__)}: {error}", evidence)

    def revise_plan(self, args):
        r = self.runtime
        self._charge()
        self._validate_ticket(args.ticket_id, "plan")
        target = self.ticket.target["node_id"]
        if args.base_plan_version != r.contexts[target].plan.version:
            raise ProtocolError("PLAN_VERSION_MISMATCH", "Revision must reference the current internal plan")
        if self.plan_counts.get(target, 0) >= r.limits.max_plan_revisions:
            raise ProtocolError("PLAN_REVISION_BUDGET_EXHAUSTED", "Per-node plan revision budget exhausted")
        self.plan_counts[target] = self.plan_counts.get(target, 0) + 1
        r.budget = r.budget.model_copy(update={"plan_revisions": r.budget.plan_revisions + 1})
        version = args.base_plan_version + 1
        stage = self._begin(args)
        self.revision = {"target": target, "stage": stage}
        try:
            for node_id in self.snapshot["subtree"] - {target}:
                del r.contexts[node_id]
            r.active_node_id = target
            if target in r.stack:
                r.stack = r.stack[:r.stack.index(target)+1]
            ctx = r.context
            ctx.artifact = None
            ctx.children, ctx.consumptions, ctx.repair_requests = [], {}, {}
            ctx.returned_child = ctx.termination_reason = None
            ctx.node = ctx.node.model_copy(update={"phase": Phase.REPAIR_AUTHORIZED, "artifact_ref": None, "last_evidence": None, "pending_edges": []})
            result = r._register_plan(args, version=version)
            # The authorized plan was consumed; normal recursive actions now build its candidate subtree.
            return HostResult(status="APPLIED", code="PLAN_RECONSTRUCTION_STARTED", details={"plan": result.details, "ticket_id": self.ticket.ticket_id})
        except (ProtocolError, SourceError, OSError, ValueError) as error:
            return self._restore(stage, f"PLAN_REJECTED: {error}")

    def finish_revision(self):
        r, revision = self.runtime, self.revision
        evidence = []
        try:
            self._frozen_check()
            nodes = self._descendants(revision["target"])
            if r.contexts["N0"].artifact:
                nodes.add("N0")
            evidence = self._checks(nodes, reuse=True)
            if not evidence or any(e.status != "PASS" for e in evidence):
                return self._restore(revision["stage"], "PLAN_VISIBLE_CHECK_NOT_PASS", evidence)
            self._commit(revision["stage"], evidence, "PLAN_REVISION_ACCEPTED")
            return None
        except (ProtocolError, SourceError, OSError, ValueError) as error:
            return self._restore(revision["stage"], f"PLAN_VALIDATION_ERROR: {error}", evidence)

    def abort_revision(self, reason):
        if self.revision:
            return self._restore(self.revision["stage"], reason)
