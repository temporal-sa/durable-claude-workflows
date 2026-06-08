"""Shared data models (Pydantic) and the plan JSON schema.

The "dynamic workflow" Claude authors is a **plan**: a directed acyclic graph of
typed nodes (`agent` / `review` / `synthesize`) with dependencies. Temporal runs
the graph durably - each node becomes a child workflow that calls Claude.

These cross the workflow/activity/client boundary, so they're serialized by
Temporal's ``pydantic_data_converter``. Keep this module dependency-light - it's
imported into the workflow sandbox (through a pass-through import).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

NodeKind = Literal["agent", "review", "synthesize", "apply"]
Phase = Literal["planning", "awaiting_approval", "running", "done", "error", "cancelled"]
NodeStatus = Literal["pending", "running", "done", "failed"]


# --- The plan Claude authors (the "dynamic workflow" as a DAG) --------------
class PlanNode(BaseModel):
    id: str                                       # stable, unique (e.g. "a1", "review", "final")
    kind: NodeKind                                # agent | review | synthesize
    title: str                                    # short label
    instruction: str                              # what this node should do
    depends_on: list[str] = Field(default_factory=list)  # ids whose results feed this node
    use_web_search: bool = False
    use_filesystem: bool = False                  # true => read/write files + run bash (coding)


class WorkflowPlan(BaseModel):
    title: str = ""
    summary: str = ""                             # 1-2 sentences, shown at the approval prompt
    nodes: list[PlanNode] = Field(default_factory=list)
    output: str = ""                              # id of the node whose result is the final answer
    input_tokens: int = 0                         # full-price input tokens Claude used to author this plan
    cached_tokens: int = 0                        # cache-read input tokens (~10% price)
    output_tokens: int = 0


class NodeResult(BaseModel):
    id: str
    kind: NodeKind
    title: str = ""
    output: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    error: Optional[str] = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# --- Live progress, exposed via the workflow query --------------------------
class NodeProgress(BaseModel):
    id: str
    kind: NodeKind
    title: str
    instruction: str = ""
    depends_on: list[str] = Field(default_factory=list)
    use_filesystem: bool = False
    status: NodeStatus = "pending"
    workflow_id: Optional[str] = None             # the node's child workflow (visible in the Temporal UI)
    confidence: Optional[float] = None
    note: Optional[str] = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


class TurnProgress(BaseModel):
    index: int
    user_prompt: str
    phase: Phase = "planning"
    plan_title: Optional[str] = None
    plan_summary: Optional[str] = None
    plan_input_tokens: int = 0
    plan_cached_tokens: int = 0
    plan_output_tokens: int = 0
    nodes: list[NodeProgress] = Field(default_factory=list)
    report: Optional[str] = None
    error: Optional[str] = None


class AgentSnapshot(BaseModel):
    """Return type of the ``get_state`` query - drives the chat client UI."""

    session_id: str
    task_queue: str
    busy: bool = False
    turns_completed: int = 0
    transcript: list[Turn] = Field(default_factory=list)
    current: Optional[TurnProgress] = None
    last_turn: Optional[TurnProgress] = None      # the most recently finished turn (final tokens/status)
    mock: bool = False
    models: dict[str, str] = Field(default_factory=dict)


# --- Workflow input --------------------------------------------------------
class StartRequest(BaseModel):
    session_id: str
    initial_prompt: Optional[str] = None
    auto_approve: bool = False                    # one-shot mode approves the plan automatically
    max_nodes: int = 10
    mock: bool = False
    models: dict[str, str] = Field(default_factory=dict)
    # Carried across continue-as-new to bound Event History on long chats.
    carry_transcript: list[Turn] = Field(default_factory=list)
    turns_completed: int = 0


# --- Activity inputs -------------------------------------------------------
class PlanInput(BaseModel):
    goal: str
    history: list[Turn] = Field(default_factory=list)
    max_nodes: int = 10


class NodeRunInput(BaseModel):
    goal: str
    node: PlanNode
    upstream: list[NodeResult] = Field(default_factory=list)  # results of this node's depends_on


# --- JSON schema for Claude's structured plan output -----------------------
# Hand-written to satisfy the structured-output constraints (additionalProperties:false,
# no min/max keywords). Claude returns the graph; we validate + normalize it.
_NODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "description": "Stable unique id, e.g. a1, review, final"},
        "kind": {"type": "string", "enum": ["agent", "review", "synthesize", "apply"]},
        "title": {"type": "string", "description": "Short label, 2-6 words"},
        "instruction": {"type": "string", "description": "What this step should do"},
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ids of steps whose outputs feed this one; empty = runs immediately",
        },
        "use_web_search": {"type": "boolean", "description": "true if this step needs current/external info"},
        "use_filesystem": {"type": "boolean",
                           "description": "true if this step must read/write files or run shell commands (e.g. coding)"},
    },
    "required": ["id", "kind", "title", "instruction", "depends_on", "use_web_search", "use_filesystem"],
}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string", "description": "Short title for the workflow"},
        "summary": {"type": "string", "description": "1-2 sentence plain-English description of the plan"},
        "nodes": {"type": "array", "items": _NODE_SCHEMA},
        "output": {"type": "string", "description": "id of the node whose result is the final deliverable"},
    },
    "required": ["title", "summary", "nodes", "output"],
}
