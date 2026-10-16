"""Shared atomic and depth-first recursive execution with one root budget."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from pydantic import ValidationError

from .artifacts import ArtifactStore, SourceError
from .policy import Policy, PolicyError
from .schema import (
    Action, Artifact, Consumption, Contract, ContractSpec, Count, Dependency, HostResult, NodeState, Phase,
    NodeResult, Plan, PolicyOutput, Record, RootTask, TokenUsage, parse_action,
)
from .trace import TraceWriter
from .verifier import Verifier


class Limits(Record):
    max_policy_calls: Count = 64
    max_total_tokens: Count = 196608
    max_protocol_retries: Count = 1
    max_verify_calls: Count = 32
    max_depth: Count = 2
    max_nodes: Count = 7
    max_children: Count = 3
    max_child_repair_requests: Count = 1
    max_patch_attempts: Count = 3
    max_plan_revisions: Count = 1


class Budget(Record):
    policy_calls: Count = 0
    input_tokens: Count = 0
    output_tokens: Count = 0
    verify_calls: Count = 0
    executed_cases: Count = 0
    execution_seconds: float = 0.0
    patch_attempts: Count = 0
    plan_revisions: Count = 0
    nodes_created: Count = 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class NodeExecution:
    node: NodeState
    plan: Plan | None = None
    artifact: Artifact | None = None
    children: list[str] = field(default_factory=list)
    returned_child: str | None = None
    consumptions: dict[str, Consumption] = field(default_factory=dict)
    repair_requests: dict[str, int] = field(default_factory=dict)
    repair_request: dict | None = None
    termination_reason: str | None = None
    last_action_result: HostResult | None = None
    consecutive_errors: int = 0


class AtomicRuntime:
    def __init__(
        self, task: RootTask, contract: ContractSpec, policy: Policy, run_dir: Path,
        limits: Limits | None = None, protocol_version: str = "at-v3r1",
        verifier: Verifier | None = None,
        recursive: bool = False,
        enable_repair: bool = False, repair_config: dict | None = None,
        implementation_stage: str | None = None,
        require_obligation_coverage: bool = False,
    ):
        self.task = task.model_copy(deep=True)
        self.policy = policy
        self.limits = limits or Limits()
        self.protocol_version = protocol_version
        self.require_obligation_coverage = require_obligation_coverage
        self.verifier = verifier
        if recursive and verifier is None:
            raise ValueError("Recursive execution requires a real verifier")
        self.recursive = recursive
        self.stage = "F3" if recursive else ("F2" if verifier is not None else "F1")
        if enable_repair:
            self.stage = "F4"
        if implementation_stage is not None:
            self.stage = implementation_stage
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        incoming = Contract(**contract.model_dump(), contract_id="C_N0_1", version=1)
        self.active_node_id = "N0"
        self.contexts = {"N0": NodeExecution(NodeState(node_id="N0", parent_id=None, incoming_contract=incoming))}
        self.stack = ["N0"]
        self.budget = Budget()
        self.consecutive_errors = 0
        self.stop_reason: str | None = None
        self.last_action_result: HostResult | None = None
        self.store = ArtifactStore(self.run_dir)
        self.trace = TraceWriter(self.run_dir)
        self.trace.initialize(self.task, self.protocol_version)
        self.repair_manager = None
        if enable_repair:
            from .repair import RepairManager
            self.repair_manager = RepairManager(self, **(repair_config or {}))
        self.state_ref = self.trace.snapshot(self._state())

    @property
    def context(self):
        return self.contexts[self.active_node_id]

    @property
    def node(self):
        return self.context.node

    @node.setter
    def node(self, value):
        self.context.node = value

    @property
    def plan(self):
        return self.context.plan

    @plan.setter
    def plan(self, value):
        self.context.plan = value

    @property
    def artifact(self):
        return self.context.artifact

    @artifact.setter
    def artifact(self, value):
        self.context.artifact = value

    @property
    def last_action_result(self):
        return self.context.last_action_result

    @last_action_result.setter
    def last_action_result(self, value):
        self.context.last_action_result = value

    @property
    def consecutive_errors(self):
        return self.context.consecutive_errors

    @consecutive_errors.setter
    def consecutive_errors(self, value):
        self.context.consecutive_errors = value

    def _state(self) -> dict:
        return {
            "root_task": self.task.model_dump(mode="json"),
            "node": self.node.model_dump(mode="json"),
            "budget": self.budget.model_dump(),
            "limits": self.limits.model_dump(),
            "consecutive_protocol_errors": self.consecutive_errors,
            "last_action_result": self.last_action_result.model_dump() if self.last_action_result else None,
            "stop_reason": self.stop_reason,
            "execution_root": str(self.run_dir),
            "active_node_id": self.active_node_id, "stack": list(self.stack),
            "nodes": {node_id: {
                "node": ctx.node.model_dump(mode="json"), "children": ctx.children,
                "returned_child": ctx.returned_child,
                "consumptions": {key: value.model_dump() for key, value in ctx.consumptions.items()},
                "repair_requests": ctx.repair_requests, "repair_request": ctx.repair_request,
                "termination_reason": ctx.termination_reason,
                "consecutive_protocol_errors": ctx.consecutive_errors,
                "last_action_result": ctx.last_action_result.model_dump() if ctx.last_action_result else None,
            } for node_id, ctx in self.contexts.items()},
        }

    def _plan_modes(self):
        revising = self.repair_manager and self.repair_manager.plan_authorized()
        if self.node.phase != Phase.NEW and not revising:
            return []
        modes = ["ATOMIC"]
        if (self.recursive and len(self.stack) - 1 < self.limits.max_depth
                and self.budget.nodes_created < self.limits.max_nodes and self.limits.max_children > 0):
            modes.append("SPLIT")
        return modes

    def _unfulfilled(self, context):
        rejected = [c for c, decision in context.consumptions.items() if decision.decision == "REJECT"]
        obligations = []
        if context.plan:
            for obligation, owners in context.plan.obligation_map.items():
                for child in context.plan.children:
                    child_id = f"{context.node.node_id}_p{context.plan.version}_{child.local_key}"
                    if child_id in rejected and f"child:{child.local_key}" in owners:
                        obligations.append(obligation)
                        break
        return rejected, obligations

    def build_observation(self) -> dict:
        actions = {Phase.NEW: ["plan"], Phase.IMPLEMENT: ["write_code"]}
        if self.verifier is not None:
            actions = {
                Phase.NEW: ["plan", "abstain"], Phase.IMPLEMENT: ["write_code", "abstain"],
                Phase.READY_VERIFY: ["verify", "abstain"], Phase.REVIEW: ["verify", "finish", "abstain"],
            }
        if self.recursive:
            actions[Phase.RESOLVE_CHILD] = ["consume", "reject", "request_child_repair", "abstain"]
        if self.repair_manager:
            for phase in (Phase.REVIEW, Phase.RESOLVE_CHILD, Phase.IMPLEMENT):
                actions[phase] = [*actions[phase], "localize"]
            actions[Phase.REPAIR_AUTHORIZED] = ["plan" if self.repair_manager.plan_authorized() else "repair", "abstain"]
        evidence = self.node.last_evidence
        _, unfulfilled = self._unfulfilled(self.context)
        returned = self.contexts.get(self.context.returned_child)
        approved = [Dependency(child_id=c.child_node, artifact_version=c.artifact_version,
                               artifact_hash=c.artifact_hash, alias=c.alias)
                    for c in self.context.consumptions.values() if c.decision == "CONSUME"]
        visible = [evidence.model_dump(mode="json")] if evidence and evidence.visibility == "VISIBLE" else []
        if returned and returned.node.last_evidence and returned.node.last_evidence.visibility == "VISIBLE":
            visible.append(returned.node.last_evidence.model_dump(mode="json"))
        if self.repair_manager:
            visible = [e.model_dump(mode="json") for e in self.repair_manager.visible_evidence()]
        return {
            "protocol_version": self.protocol_version,
            "root_id": self.task.root_id,
            "node_id": self.node.node_id,
            "state_id": self.state_ref,
            "phase": self.node.phase.value,
            "mode": {Phase.NEW: "PLAN", Phase.RESOLVE_CHILD: "COMPOSE",
                     Phase.REPAIR_AUTHORIZED: "PLAN" if self.repair_manager and self.repair_manager.plan_authorized() else "REPAIR",
                     Phase.REVIEW: "LOCATE" if self.repair_manager and evidence and evidence.status == "FAIL" else "CONTROL",
                     Phase.IMPLEMENT: "COMPOSE" if self.context.children else "IMPLEMENT"}.get(self.node.phase, "CONTROL"),
            "root_spec": self.task.spec,
            "incoming_contract": self.node.incoming_contract.model_dump(mode="json"),
            "current_plan": self.plan.model_dump(mode="json") if self.plan else None,
            "own_code": (self.run_dir / self.artifact.module_path).read_text() if self.artifact else None,
            "current_artifact": self.artifact.model_dump() if self.artifact else None,
            "available_suites": self.verifier.available_suites(self.task.root_id, self.node.incoming_contract) if self.verifier else [],
            "returned_child": ({"result": self._node_result(returned).model_dump(mode="json"),
                                "contract": returned.node.incoming_contract.model_dump(),
                                "source": (self.run_dir / returned.artifact.module_path).read_text() if returned.artifact else None,
                                "import_module": f"nodes.{returned.node.node_id}"} if returned else None),
            "approved_dependencies": [d.model_dump() for d in approved],
            "child_interfaces": [{"node_id": child_id, "contract": self.contexts[child_id].node.incoming_contract.model_dump()}
                                 for child_id in self.context.children],
            "sibling_interfaces": [{"node_id": c, "signature": self.contexts[c].node.incoming_contract.signature.model_dump()}
                                   for c in self.contexts[self.node.parent_id].children if c != self.node.node_id] if self.node.parent_id else [],
            "visible_evidence": visible,
            "pending_obligations": [o.id for o in self.node.incoming_contract.ensures
                                    if o.id in unfulfilled or not evidence or evidence.status != "PASS" or o.id not in evidence.coverage],
            "unfulfilled_child_ids": self._unfulfilled(self.context)[0],
            "repair_ticket": self.repair_manager.ticket.model_dump() if self.repair_manager and self.repair_manager.ticket else None,
            "responsibility_candidates": self.repair_manager.candidates() if self.repair_manager else [],
            "child_repair_request": self.context.repair_request,
            "last_action_result": self.last_action_result.model_dump() if self.last_action_result else None,
            "budget_remaining": {
                "policy_calls": max(0, self.limits.max_policy_calls - self.budget.policy_calls),
                "tokens": max(0, self.limits.max_total_tokens - self.budget.total_tokens),
                "verify_calls": max(0, self.limits.max_verify_calls - self.budget.verify_calls),
                "nodes": max(0, self.limits.max_nodes - self.budget.nodes_created),
                "child_slots": (max(0, min(self.limits.max_children, self.limits.max_nodes - self.budget.nodes_created))
                                if "SPLIT" in self._plan_modes() else 0),
                "patch_attempts": max(0, self.limits.max_patch_attempts - self.budget.patch_attempts),
                "depth": max(0, self.limits.max_depth - (len(self.stack) - 1)),
            },
            "allowed_actions": actions.get(self.node.phase, []) if self.stop_reason is None else [],
            "allowed_plan_modes": self._plan_modes(),
            "structure_budget_limited": self.recursive and self.node.phase == Phase.NEW and "SPLIT" not in self._plan_modes(),
            "implementation_stage": self.stage,
        }

    def _record(self, observation, result, output=None, parsed=None, kind="POLICY"):
        if observation["node_id"] in self.contexts:
            self.contexts[observation["node_id"]].last_action_result = result
        self.trace.count += 1
        self.state_ref = self.trace.snapshot(self._state())
        return self.trace.append(
            event_kind=kind, root_id=self.task.root_id, node_id=observation["node_id"],
            observation_before=observation,
            raw_output=output.raw_text if output else None,
            parsed_action=parsed, host_result=result, state_after_ref=self.state_ref,
            usage=output.token_usage if output else TokenUsage(source="unavailable" if kind == "POLICY" else "scripted"),
            model_identity=output.model_identity if output else {},
            generation_truncated=output.generation_truncated if output else False,
            model_input=output.model_input if output else None,
            output_token_ids=output.output_token_ids if output else [],
        )

    def _stop(self, reason: str):
        observation = self.build_observation()
        if reason == "F1_READY_VERIFY":
            self.stop_reason = reason
        elif reason == "BUDGET_EXHAUSTED":
            if self.repair_manager and self.repair_manager.revision:
                self.repair_manager.abort_revision(reason)
            self.stop_reason = reason
            for context in self.contexts.values():
                if context.node.phase != Phase.TERMINAL:
                    context.node = context.node.model_copy(update={"phase": Phase.TERMINAL})
                    context.termination_reason = reason
        else:
            self._terminate_node(reason)
        return self._record(observation, HostResult(status="STOPPED", code=reason), kind="HOST")

    def _terminate_node(self, reason):
        if (self.repair_manager and self.repair_manager.revision
                and self.active_node_id == self.repair_manager.revision["target"] and reason != "FINISHED"):
            target = self.active_node_id
            self.repair_manager.abort_revision(reason)
            self.active_node_id = target
            chain, cursor = [], target
            while cursor is not None:
                chain.append(cursor)
                cursor = self.contexts[cursor].node.parent_id
            self.stack = list(reversed(chain))
        self.context.termination_reason = reason
        self.node = self.node.model_copy(update={"phase": Phase.TERMINAL})
        if self.node.parent_id is None:
            self.stop_reason = reason

    def _schedule(self):
        observation = self.build_observation()
        if self.node.phase == Phase.WAIT_CHILD:
            child_id = self.node.pending_edges[0]
            self.stack.append(child_id)
            self.active_node_id = child_id
            return self._record(observation, HostResult(status="APPLIED", code="CHILD_ENTERED", details={"child_id": child_id}), kind="HOST")
        child = self.context
        self.stack.pop()
        self.active_node_id = self.stack[-1]
        self.context.returned_child = child.node.node_id
        self.node = self.node.model_copy(update={"phase": Phase.RESOLVE_CHILD})
        result = HostResult(status="APPLIED", code="CHILD_RETURNED", details={"result": self._node_result(child).model_dump(mode="json")})
        self.last_action_result = result
        return self._record(observation, result, kind="HOST")

    def _exhausted(self) -> bool:
        return (
            self.budget.policy_calls >= self.limits.max_policy_calls
            or self.budget.total_tokens >= self.limits.max_total_tokens
        )

    def _dispatch(self, action: Action) -> HostResult:
        if action.action not in self.build_observation()["allowed_actions"]:
            raise ProtocolError("ACTION_NOT_ALLOWED", f"{action.action} is unavailable in {self.node.phase.value} during {self.stage}")
        if action.action == "plan":
            if self.repair_manager and action.args.base_plan_version > 0:
                return self.repair_manager.revise_plan(action.args)
            return self._register_plan(action.args)
        if action.action == "localize":
            return self.repair_manager.localize(action.args)
        if action.action == "repair":
            return self.repair_manager.repair(action.args)

        return self._dispatch_implementation(action)

    def _register_plan(self, args, version=1):
        # Both initial plans and authorized revisions use the same registration path.
        if version == 1 and args.base_plan_version != 0:
            raise ProtocolError("PLAN_VERSION_MISMATCH", "Initial plan requires base_plan_version=0")
        if version == 1 and args.ticket_id is not None:
            raise ProtocolError("UNAUTHORIZED_TICKET", "Atomic execution has no authorized repair ticket")
        if args.mode != "ATOMIC" and not self.recursive:
            raise ProtocolError("F1_UNSUPPORTED_MODE", "Recursive scheduling is implemented in F3")
        if args.mode not in self._plan_modes():
            raise ProtocolError("STRUCTURE_BUDGET_EXHAUSTED", "Choose ATOMIC or abstain at the depth/node limit")
        if (len(args.children) > self.limits.max_children
                or self.budget.nodes_created + len(args.children) > self.limits.max_nodes):
            raise ProtocolError("STRUCTURE_BUDGET_EXHAUSTED",
                                f"Plan requests {len(args.children)} children; per-node limit is {self.limits.max_children}, "
                                f"remaining node slots are {self.limits.max_nodes - self.budget.nodes_created}")
        plan = Plan(**args.model_dump(), node_id=self.node.node_id, version=version)
        if self.require_obligation_coverage:
            owners = {"self", *(f"child:{c.local_key}" for c in plan.children)}
            if any(not plan.obligation_map.get(o.id) for o in self.node.incoming_contract.ensures):
                raise ProtocolError("MISSING_OBLIGATION_OWNER", "Every incoming obligation needs an explicit owner")
            if any(owner not in owners for values in plan.obligation_map.values() for owner in values):
                raise ProtocolError("UNKNOWN_OBLIGATION_OWNER", "Owners must be self or a declared child")
        child_ids = [f"{self.node.node_id}_p{plan.version}_{child.local_key}" for child in plan.children]
        if any(c in self.contexts for c in child_ids):
            raise ProtocolError("CHILD_ID_CONFLICT", "Plan would reuse an existing node identity")
        child_specs = {c.local_key: c.contract for c in plan.children}
        if any(edge.from_child not in child_specs or edge.to_child not in child_specs
               or edge.from_output != "return"
               or edge.to_parameter not in [p.name for p in child_specs[edge.to_child].signature.parameters]
               for edge in plan.dataflow):
            raise ProtocolError("INVALID_DATAFLOW", "Dataflow must reference declared child outputs and parameters")
        relative = f"plans/{self.node.node_id}.v{version}.json"
        path = self.run_dir / relative
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(plan.model_dump_json(indent=2) + "\n")
        self.plan = plan
        for child_id, child in zip(child_ids, plan.children):
            incoming = Contract(**child.contract.model_dump(), contract_id=f"C_{child_id}_1", version=1)
            self.contexts[child_id] = NodeExecution(NodeState(node_id=child_id, parent_id=self.node.node_id, incoming_contract=incoming))
        self.context.children = child_ids
        self.budget = self.budget.model_copy(update={"nodes_created": self.budget.nodes_created + len(child_ids)})
        self.node = self.node.model_copy(update={"phase": Phase.WAIT_CHILD if child_ids else Phase.IMPLEMENT,
                                                 "plan_ref": relative, "pending_edges": child_ids})
        missing = [o.id for o in self.node.incoming_contract.ensures if not args.obligation_map.get(o.id)]
        return HostResult(status="APPLIED", code="PLAN_REGISTERED", details={
            "missing_obligation_ids": missing, "semantic_coverage": "UNKNOWN", "created_children": child_ids,
        })

    def _dispatch_implementation(self, action):
        if action.action in {"consume", "reject", "request_child_repair"}:
            return self._resolve_child(action)
        if action.action == "verify":
            return self._verify(action.args)
        if action.action == "finish":
            if action.args.artifact_version != self.artifact.version:
                raise ProtocolError("ARTIFACT_VERSION_MISMATCH", "finish must reference the current artifact version")
            if self.repair_manager and self.repair_manager.revision and self.active_node_id == self.repair_manager.revision["target"]:
                outcome = self.repair_manager.finish_revision()
                if outcome is not None:
                    return outcome
            evidence = self.node.last_evidence
            if (evidence is None or evidence.visibility != "VISIBLE" or not self.verifier.is_current(
                    evidence, self.run_dir, self.task.root_id, self.node.incoming_contract, self.artifact)):
                raise ProtocolError("STALE_OR_MISSING_EVIDENCE", "finish requires verification of the current source, contract, suite and runner")
            self._terminate_node("FINISHED")
            return HostResult(status="APPLIED", code="FINISHED", details={"functional_status": evidence.status, "evidence_id": evidence.evidence_id})
        if action.action == "abstain":
            evidence = self.node.last_evidence
            known = {e["evidence_id"] for e in self.build_observation()["visible_evidence"]}
            if not set(action.args.evidence_ids) <= known:
                raise ProtocolError("UNKNOWN_EVIDENCE", "abstain can reference only current visible evidence")
            self._terminate_node("ABSTAINED")
            return HostResult(status="APPLIED", code="ABSTAINED", details=action.args.model_dump())

        args = action.args
        if args.contract_version != self.node.incoming_contract.version:
            raise ProtocolError("CONTRACT_VERSION_MISMATCH", "write_code must reference the incoming contract version")
        approved = self.build_observation()["approved_dependencies"]
        if sorted([d.model_dump() for d in args.declared_dependencies], key=lambda d: d["child_id"]) != sorted(approved, key=lambda d: d["child_id"]):
            raise ProtocolError("UNAPPROVED_DEPENDENCY", "Declared dependencies must exactly match explicit consumption decisions")
        if self.artifact is not None:
            raise ProtocolError("ARTIFACT_EXISTS", "Existing source changes require the repair path")
        previous_version = self.store.latest_version(self.node.node_id)
        if self.repair_manager and self.repair_manager.revision and previous_version:
            artifact = self.store.save_next(self.store.load(self.node.node_id, previous_version), self.node.incoming_contract, args.source, args.declared_dependencies)
        else:
            artifact = self.store.save_initial(self.node.node_id, self.node.incoming_contract, args.source, args.declared_dependencies)
        if self.recursive:
            self.store.bundle(artifact).materialize(self.run_dir / "workspaces" / self.node.node_id / f"v{artifact.version}")
        self.artifact = artifact
        self.node = self.node.model_copy(update={"phase": Phase.READY_VERIFY, "artifact_ref": artifact.module_path})
        return HostResult(status="APPLIED", code="SOURCE_SAVED", details={"artifact": artifact.model_dump()})

    def _resolve_child(self, action):
        args = action.args
        if args.child_id != self.context.returned_child:
            raise ProtocolError("CHILD_NOT_PENDING", "Only the currently returned direct child can be resolved")
        child = self.contexts[args.child_id]
        artifact = child.artifact
        if artifact is None or args.artifact_version != artifact.version or self.store.latest_version(args.child_id) != artifact.version:
            raise ProtocolError("ARTIFACT_VERSION_MISMATCH", "Select the current returned child version")
        visible = {e["evidence_id"] for e in self.build_observation()["visible_evidence"]}
        if not set(args.evidence_ids) <= visible:
            raise ProtocolError("UNKNOWN_EVIDENCE", "Child decisions can reference visible evidence only")
        if action.action == "request_child_repair":
            attempts = self.context.repair_requests.get(args.child_id, 0)
            if attempts >= self.limits.max_child_repair_requests:
                raise ProtocolError("CHILD_REPAIR_BUDGET_EXHAUSTED", "Per-edge repair request budget exhausted")
            self.context.repair_requests[args.child_id] = attempts + 1
            child.repair_request = {"parent_id": self.node.node_id, **args.model_dump()}
            child.node = child.node.model_copy(update={"phase": Phase.REVIEW})
            child.termination_reason, child.consecutive_errors = None, 0
            self.context.returned_child = None
            self.node = self.node.model_copy(update={"phase": Phase.WAIT_CHILD})
            return HostResult(status="APPLIED", code="CHILD_REPAIR_REQUESTED", details={"child_id": args.child_id, "request_number": attempts + 1})
        if action.action == "consume":
            if args.artifact_hash != artifact.source_hash:
                raise ProtocolError("ARTIFACT_HASH_MISMATCH", "Consume must select the returned artifact hash")
            self.store.bundle(artifact) # Check the complete selected dependency closure before accepting it.
            if any(c.alias == args.alias for c in self.context.consumptions.values() if c.decision == "CONSUME"):
                raise ProtocolError("ALIAS_CONFLICT", "Dependency aliases must be unique within the consumer")
            result = self._node_result(child)
            if result.functional_status == "ERROR":
                raise ProtocolError("CHILD_INFRASTRUCTURE_ERROR", "An infrastructure error cannot be accepted as an unverified candidate result")
            if result.functional_status != "PASS" and not args.accept_unverified:
                raise ProtocolError("UNVERIFIED_CONSUMPTION", "FAIL/UNKNOWN/unchecked artifacts require accept_unverified=true")
            if result.functional_status == "PASS" and child.node.last_evidence.evidence_id not in args.evidence_ids:
                raise ProtocolError("MISSING_CHILD_EVIDENCE", "Verified consumption must reference the current child evidence")
            decision = Consumption(consumer_node=self.node.node_id, child_node=args.child_id,
                artifact_version=artifact.version, artifact_hash=artifact.source_hash, alias=args.alias,
                decision="CONSUME", evidence_refs=args.evidence_ids, accept_unverified=args.accept_unverified,
                known_failed_consumed=result.functional_status == "FAIL")
        else:
            decision = Consumption(consumer_node=self.node.node_id, child_node=args.child_id,
                artifact_version=artifact.version, artifact_hash=artifact.source_hash, alias=None,
                decision="REJECT", evidence_refs=args.evidence_ids)
        self.context.consumptions[args.child_id] = decision
        with (self.run_dir / "consumptions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(decision.model_dump_json() + "\n")
        remaining = self.node.pending_edges[1:]
        self.context.returned_child = None
        self.node = self.node.model_copy(update={"pending_edges": remaining,
            "phase": Phase.WAIT_CHILD if remaining else Phase.IMPLEMENT})
        return HostResult(status="APPLIED", code="CHILD_CONSUMED" if decision.decision == "CONSUME" else "CHILD_REJECTED",
                          details={"consumption": decision.model_dump(), "unfulfilled_child_ids": self._unfulfilled(self.context)[0]})

    def _verify(self, args) -> HostResult:
        if args.target_ref != self.node.artifact_ref:
            raise ProtocolError("TARGET_MISMATCH", "verify must reference the current node artifact")
        if args.suite_id not in self.verifier.available_suites(self.task.root_id, self.node.incoming_contract):
            raise ProtocolError("SUITE_UNAVAILABLE", "verify must select an advertised visible suite")
        if self.budget.verify_calls >= self.limits.max_verify_calls:
            raise ProtocolError("VERIFY_BUDGET_EXHAUSTED", "Root verification budget exhausted; finish or abstain remains available")
        self.budget = self.budget.model_copy(update={"verify_calls": self.budget.verify_calls + 1})
        evidence = self.verifier.verify(
            self.run_dir, self.task.root_id, self.node.incoming_contract, self.artifact,
            args.suite_id, f"E{self.budget.verify_calls:04d}",
        )
        self.budget = self.budget.model_copy(update={
            "executed_cases": self.budget.executed_cases + evidence.details["executed_cases"],
            "execution_seconds": self.budget.execution_seconds + evidence.details["elapsed_seconds"],
        })
        self.node = self.node.model_copy(update={"phase": Phase.REVIEW, "last_evidence": evidence})
        if self.repair_manager:
            self.repair_manager.archive_evidence(evidence)
        if evidence.status == "ERROR":
            self._terminate_node("VERIFIER_ERROR")
        return HostResult(
            status="ERROR" if evidence.status == "ERROR" else "APPLIED", code="VERIFIED",
            details={"evidence": evidence.model_dump(mode="json")},
        )

    def step(self):
        if self.stop_reason is not None:
            raise RuntimeError("Cannot advance a stopped run")
        if self.recursive and self.node.phase == Phase.TERMINAL and self.node.parent_id is not None:
            return self._schedule()
        if self._exhausted():
            return self._stop("BUDGET_EXHAUSTED")
        if self.consecutive_errors > self.limits.max_protocol_retries:
            return self._stop("PROTOCOL_RETRIES_EXHAUSTED")
        if self.recursive and self.node.phase == Phase.WAIT_CHILD:
            return self._schedule()
        observation = self.build_observation()
        self.budget = self.budget.model_copy(update={"policy_calls": self.budget.policy_calls + 1})
        try:
            output = self.policy.act(deepcopy(observation))
            if not isinstance(output, PolicyOutput):
                raise TypeError("Policy.act must return PolicyOutput")
        except PolicyError as error:
            self._terminate_node(error.code)
            return self._record(observation, HostResult(
                status="ERROR", code=error.code, message=str(error),
            ), output=error.output)
        except Exception as error:
            self._terminate_node("POLICY_ERROR")
            return self._record(observation, HostResult(
                status="ERROR", code="POLICY_ERROR", message=f"{type(error).__name__}: {error}",
            ))
        self.budget = self.budget.model_copy(update={
            "input_tokens": self.budget.input_tokens + output.token_usage.input_tokens,
            "output_tokens": self.budget.output_tokens + output.token_usage.output_tokens,
        })
        parsed = None
        try:
            if output.generation_truncated:
                raise ProtocolError("TRUNCATED_OUTPUT", "Truncated output is not an executable action")
            # Always validate raw output; a policy-provided parsed action is not authoritative.
            if self.require_obligation_coverage:
                def unique_keys(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ProtocolError("DUPLICATE_JSON_KEY", f"Repeated action field: {key}")
                        result[key] = value
                    return result
                try:
                    json.loads(output.raw_text, object_pairs_hook=unique_keys)
                except json.JSONDecodeError:
                    pass  # The normal action parser reports malformed JSON below.
            parsed = parse_action(output.raw_text)
            if self.budget.total_tokens > self.limits.max_total_tokens:
                raise ProtocolError("TOKEN_BUDGET_EXCEEDED", "Actual input and output tokens exceed the root budget")
            result = self._dispatch(parsed)
        except ValidationError as error:
            result = HostResult(status="REJECTED", code="INVALID_ACTION_JSON", message=str(error))
        except (ProtocolError, SourceError) as error:
            result = HostResult(status="REJECTED", code=error.code, message=str(error))
        except Exception as error:
            self._terminate_node("INFRASTRUCTURE_ERROR")
            result = HostResult(status="ERROR", code="INFRASTRUCTURE_ERROR", message=f"{type(error).__name__}: {error}")
        self.consecutive_errors = self.consecutive_errors + 1 if result.status == "REJECTED" else 0
        return self._record(observation, result, output, parsed)

    def _node_result(self, context):
        evidence, artifact = context.node.last_evidence, context.artifact
        current = bool(evidence and self.verifier.is_current(
            evidence, self.run_dir, self.task.root_id, context.node.incoming_contract, artifact))
        rejected, obligations = self._unfulfilled(context)
        responsibility = evidence.responsibility_status if current else "UNKNOWN"
        if rejected:
            responsibility = "VIOLATION"
        return NodeResult(
            node_id=context.node.node_id, artifact=artifact,
            functional_status=(evidence.status if current or evidence.status == "ERROR" else "UNKNOWN") if evidence else None,
            evidence=evidence, termination_reason=context.termination_reason or
                (self.stop_reason if context.node.parent_id is None else None) or "NOT_FINISHED",
            responsibility_status=responsibility,
            responsibility_details={"unfulfilled_child_ids": rejected, "unfulfilled_obligation_ids": obligations,
                                    "consumptions": [c.model_dump() for c in context.consumptions.values()]},
        )

    def run(self) -> dict:
        while self.stop_reason is None:
            if self.verifier is None and self.node.phase == Phase.READY_VERIFY:
                self._stop("F1_READY_VERIFY")
            else:
                self.step()
        self.active_node_id = "N0"
        evidence = self.node.last_evidence
        evidence_current = (self.verifier.is_current(
            evidence, self.run_dir, self.task.root_id, self.node.incoming_contract, self.artifact,
        ) if evidence else None)
        node_result = self._node_result(self.context)
        result = {
            "scope": "SMOKE_ONLY", "stage": self.stage, "root_id": self.task.root_id,
            "split": self.task.split, "phase": self.node.phase.value,
            "stop_reason": self.stop_reason, "node_count": len(self.contexts),
            "verification_performed": evidence is not None, "functional_status": node_result.functional_status,
            "evidence_current": evidence_current,
            "responsibility_status": node_result.responsibility_status,
            "node_result": node_result.model_dump(mode="json"),
            "node_results": {node_id: self._node_result(ctx).model_dump(mode="json") for node_id, ctx in self.contexts.items()},
            "artifact": self.artifact.model_dump() if self.artifact else None,
            "usage": self.budget.model_dump(), "trace_events": self.trace.count,
            "run_dir": str(self.run_dir),
            "repair_history": self.repair_manager.history if self.repair_manager else [],
        }
        (self.run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result


class RecursiveRuntime(AtomicRuntime):
    """F3 uses the same policy, verifier, trace and root budget at every depth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, recursive=True)


class RepairRuntime(AtomicRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, recursive=True, enable_repair=True)
