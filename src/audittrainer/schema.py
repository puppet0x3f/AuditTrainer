"""Version-one action payloads and atomic runtime/evidence records."""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

Text = Annotated[str, StringConstraints(min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Version = Annotated[int, Field(ge=1)]
Count = Annotated[int, Field(ge=0)]
Reason = Literal[
    "CONTRACT_INSUFFICIENT", "LOCAL_CHECK_FAILED", "INSUFFICIENT_EVIDENCE",
    "UNFULFILLED_OBLIGATION", "BUDGET_LIMIT", "OTHER",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Parameter(Record):
    name: Identifier
    type: Text


class Signature(Record):
    name: Identifier
    parameters: list[Parameter]
    returns: Text


class Obligation(Record):
    id: Text
    text: Text


class Frame(Record):
    public_symbol: Identifier
    private_helper_prefix: Literal["_"] = "_"
    mutate_inputs: bool
    external_io: bool


class ContractSpec(Record):
    signature: Signature
    requires: list[Text]
    ensures: list[Obligation]
    owns: list[Text]
    uses: list[Text]
    must_not: list[Text]
    frame: Frame

    @model_validator(mode="after")
    def check_interface(self):
        if self.signature.name != self.frame.public_symbol:
            raise ValueError("signature and public_symbol must agree")
        for values in ([p.name for p in self.signature.parameters], [o.id for o in self.ensures]):
            if len(values) != len(set(values)):
                raise ValueError("parameter names and obligation IDs must be unique")
        return self


class Contract(ContractSpec):
    contract_id: Text
    version: Version


class ChildSpec(Record):
    local_key: Identifier
    contract: ContractSpec


class Dataflow(Record):
    from_child: Text
    from_output: Text
    to_child: Text
    to_parameter: Text


class PlanArgs(Record):
    base_plan_version: Count
    mode: Literal["ATOMIC", "SPLIT"]
    children: list[ChildSpec]
    obligation_map: dict[str, list[Text]]
    retained_responsibilities: list[Text]
    dataflow: list[Dataflow]
    ticket_id: Text | None = None

    @model_validator(mode="after")
    def check_structure(self):
        if self.mode == "ATOMIC" and (self.children or self.dataflow):
            raise ValueError("ATOMIC has no child or inter-child dataflow")
        if self.mode == "SPLIT" and not self.children:
            raise ValueError("SPLIT requires children")
        keys = [child.local_key for child in self.children]
        if len(keys) != len(set(keys)):
            raise ValueError("child local_key must be unique")
        if self.base_plan_version > 0 and self.ticket_id is None:
            raise ValueError("plan revision requires a ticket")
        return self


class Dependency(Record):
    child_id: Text
    artifact_version: Version
    artifact_hash: Digest
    alias: Identifier


class WriteCodeArgs(Record):
    contract_version: Version
    source: Text
    declared_dependencies: list[Dependency]


class VerifyArgs(Record):
    target_ref: Text
    suite_id: Text


class ConsumeArgs(Dependency):
    evidence_ids: list[Text]
    accept_unverified: bool = False


class RejectArgs(Record):
    child_id: Text
    artifact_version: Version
    reason_code: Reason
    evidence_ids: list[Text]


class ChildRepairArgs(Record):
    child_id: Text
    artifact_version: Version
    evidence_ids: list[Text]
    instruction: Text


class LocalizeArgs(Record):
    target_type: Literal["NODE", "HANDOFF", "OMISSION", "CONTRACT"]
    target_id: Text
    evidence_ids: list[Text]
    boundary_kind: Literal["OWNER_MODULE", "CONSUMER_MODULE", "PLAN_SUBTREE"]
    rationale: Text


class Replacement(Record):
    file: Text
    base_hash: Digest
    source: Text


class ExistingRef(Record):
    kind: Literal["EXISTING"]
    version: Version
    hash: Digest


class CandidateRef(Record):
    kind: Literal["CANDIDATE"]
    node_id: Text


class DependencyUpdate(Record):
    consumer_node: Text
    child_id: Text
    artifact_ref: Annotated[ExistingRef | CandidateRef, Field(discriminator="kind")]
    alias: Identifier


class RepairArgs(Record):
    ticket_id: Text
    replacements: list[Replacement]
    dependency_updates: list[DependencyUpdate]


class FinishArgs(Record):
    artifact_version: Version


class AbstainArgs(Record):
    reason_code: Reason
    evidence_ids: list[Text]
    suggested_change: str | None = None


class PlanAction(Record):
    action: Literal["plan"]
    args: PlanArgs


class WriteCodeAction(Record):
    action: Literal["write_code"]
    args: WriteCodeArgs


class VerifyAction(Record):
    action: Literal["verify"]
    args: VerifyArgs


class ConsumeAction(Record):
    action: Literal["consume"]
    args: ConsumeArgs


class RejectAction(Record):
    action: Literal["reject"]
    args: RejectArgs


class ChildRepairAction(Record):
    action: Literal["request_child_repair"]
    args: ChildRepairArgs


class LocalizeAction(Record):
    action: Literal["localize"]
    args: LocalizeArgs


class RepairAction(Record):
    action: Literal["repair"]
    args: RepairArgs


class FinishAction(Record):
    action: Literal["finish"]
    args: FinishArgs


class AbstainAction(Record):
    action: Literal["abstain"]
    args: AbstainArgs


Action = Annotated[
    PlanAction | WriteCodeAction | VerifyAction | ConsumeAction | RejectAction
    | ChildRepairAction | LocalizeAction | RepairAction | FinishAction | AbstainAction,
    Field(discriminator="action"),
]
ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(raw_text: str) -> Action:
    """Parse exactly one action JSON; no Markdown extraction or output repair."""
    return ACTION_ADAPTER.validate_json(raw_text)


class RootTask(Record):
    root_id: Text
    family_id: Text
    split: Literal[
        "SYN_FIXTURE", "DEV_ENGINEERING", "DEV_MECHANISM", "TRAIN", "VALIDATION",
        "MECHANISM_HOLDOUT", "APPS_SEALED", "EXTERNAL_SEALED",
    ]
    spec: Text
    entrypoint: Text
    visible_suite_ref: Text | None
    private_suite_ref: Text | None


class Plan(PlanArgs):
    node_id: Identifier
    version: Version


class Artifact(Record):
    node_id: Identifier
    version: Version
    module_path: Text
    public_symbol: Identifier
    source_hash: Digest
    contract_version: Version
    dependency_bindings: list[Dependency]


class Consumption(Record):
    consumer_node: Identifier
    child_node: Identifier
    artifact_version: Version
    artifact_hash: Digest
    alias: Identifier | None
    decision: Literal["CONSUME", "REJECT"]
    evidence_refs: list[Text]
    accept_unverified: bool = False
    known_failed_consumed: bool = False


class RepairTicket(Record):
    ticket_id: Identifier
    snapshot_id: Text
    target: dict[str, str]
    boundary_kind: Literal["OWNER_MODULE", "CONSUMER_MODULE", "PLAN_SUBTREE"]
    editable_files: list[Text]
    editable_plan_nodes: list[Identifier]
    rebind_edges: list[Text]
    revalidate_nodes: list[Identifier]
    frozen_hashes: dict[str, Digest]
    evidence_ids: list[Text]
    remaining_attempts: Count


class Phase(str, Enum):
    NEW = "NEW"
    WAIT_CHILD = "WAIT_CHILD"
    RESOLVE_CHILD = "RESOLVE_CHILD"
    IMPLEMENT = "IMPLEMENT"
    READY_VERIFY = "READY_VERIFY"
    REVIEW = "REVIEW"
    REPAIR_AUTHORIZED = "REPAIR_AUTHORIZED"
    TERMINAL = "TERMINAL"


Verdict = Literal["PASS", "FAIL", "UNKNOWN", "ERROR"]


class Evidence(Record):
    evidence_id: Text
    check_kind: Literal["ROOT", "LOCAL", "UNBOUND"]
    target_ref: Text
    suite_id: Text
    suite_hash: Digest
    status: Verdict
    coverage: list[str]
    visibility: Literal["VISIBLE", "PRIVATE"]
    fingerprint: Digest
    binding: dict[str, Any]
    responsibility_status: Literal["SUPPORTED", "VIOLATION", "UNKNOWN"] = "UNKNOWN"
    details: dict[str, Any]


class NodeResult(Record):
    node_id: Identifier
    artifact: Artifact | None
    functional_status: Verdict | None
    evidence: Evidence | None
    termination_reason: Text
    responsibility_status: Literal["SUPPORTED", "VIOLATION", "UNKNOWN"] = "UNKNOWN"
    responsibility_details: dict[str, Any] = Field(default_factory=dict)


class NodeState(Record):
    node_id: Identifier
    parent_id: Identifier | None
    phase: Phase = Phase.NEW
    incoming_contract: Contract
    plan_ref: Text | None = None
    artifact_ref: Text | None = None
    pending_edges: list[str] = Field(default_factory=list)
    last_evidence: Evidence | None = None


class TokenUsage(Record):
    input_tokens: Count = 0
    output_tokens: Count = 0
    source: Literal["scripted", "model", "unavailable"] = "scripted"


class PolicyOutput(Record):
    raw_text: str
    parsed_action: Action | None = None
    token_usage: TokenUsage
    model_identity: dict[str, str]
    generation_truncated: bool = False
    model_input: dict[str, Any] | None = None
    output_token_ids: list[int] = Field(default_factory=list)


class HostResult(Record):
    status: Literal["APPLIED", "REJECTED", "ERROR", "STOPPED"]
    code: Text
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(Record):
    event_id: Text
    event_kind: Literal["POLICY", "HOST"]
    root_id: Text
    node_id: Identifier
    observation_before: dict[str, Any]
    raw_output: str | None
    parsed_action: Action | None
    host_result: HostResult
    state_after_ref: Text
    usage: TokenUsage
    model_identity: dict[str, str]
    generation_truncated: bool = False
    model_input: dict[str, Any] | None = None
    output_token_ids: list[int] = Field(default_factory=list)
