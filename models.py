"""Shared data models (Pydantic) and JSON schemas.

These cross the workflow/activity/client boundary, so they're serialized by
Temporal's ``pydantic_data_converter``. Keep this module dependency-light — it's
imported into the workflow sandbox (through a pass-through import).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Phase = Literal[
    "planning", "researching", "reviewing", "refining", "synthesizing", "done", "error"
]
SubStatus = Literal["pending", "running", "done", "failed"]


# --- The plan Claude authors (the "dynamic workflow") ----------------------
class SubAgentTask(BaseModel):
    """One unit of the fan-out — researched by its own child workflow."""

    id: int = 0
    title: str
    question: str
    search_hint: str = ""


class Plan(BaseModel):
    """What the planner activity returns: the orchestration Temporal will run."""

    strategy: str
    review_focus: str
    tasks: list[SubAgentTask] = Field(default_factory=list)


class SubAgentResult(BaseModel):
    id: int
    title: str
    findings: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None


class ReviewResult(BaseModel):
    satisfied: bool
    critique: str = ""
    gaps: list[str] = Field(default_factory=list)
    followup_tasks: list[SubAgentTask] = Field(default_factory=list)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# --- Live progress, exposed via the workflow query -------------------------
class SubAgentProgress(BaseModel):
    id: int
    title: str
    status: SubStatus = "pending"
    workflow_id: Optional[str] = None  # the child workflow execution (visible in the Temporal UI)
    confidence: Optional[float] = None
    note: Optional[str] = None


class TurnProgress(BaseModel):
    index: int
    user_prompt: str
    phase: Phase = "planning"
    strategy: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 1
    subagents: list[SubAgentProgress] = Field(default_factory=list)
    report: Optional[str] = None
    error: Optional[str] = None


class AgentSnapshot(BaseModel):
    """Return type of the ``get_state`` query — drives the chat client UI."""

    session_id: str
    task_queue: str
    busy: bool = False
    turns_completed: int = 0
    transcript: list[Turn] = Field(default_factory=list)
    current: Optional[TurnProgress] = None
    mock: bool = False
    models: dict[str, str] = Field(default_factory=dict)


# --- Workflow input --------------------------------------------------------
class StartRequest(BaseModel):
    session_id: str
    initial_prompt: Optional[str] = None
    max_iterations: int = 2
    max_subagents: int = 5
    mock: bool = False
    models: dict[str, str] = Field(default_factory=dict)
    # Carried across continue-as-new to bound Event History on long chats.
    carry_transcript: list[Turn] = Field(default_factory=list)
    turns_completed: int = 0


# --- Activity inputs -------------------------------------------------------
class PlanInput(BaseModel):
    goal: str
    history: list[Turn] = Field(default_factory=list)
    max_subagents: int = 5


class ResearchInput(BaseModel):
    goal: str
    task: SubAgentTask


class ReviewInput(BaseModel):
    goal: str
    strategy: str
    findings: list[SubAgentResult] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 1


class SynthesisInput(BaseModel):
    goal: str
    history: list[Turn] = Field(default_factory=list)
    findings: list[SubAgentResult] = Field(default_factory=list)
    review: Optional[ReviewResult] = None


# --- JSON schemas for Claude structured outputs ----------------------------
# Hand-written so they satisfy the structured-output constraints
# (additionalProperties:false, no min/max/length keywords). Claude returns the
# JSON; we assign task ids ourselves afterwards.
_TASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string", "description": "Short label, 3-6 words"},
        "question": {
            "type": "string",
            "description": "The focused, independently-researchable sub-question",
        },
        "search_hint": {
            "type": "string",
            "description": "What to search for / which kinds of sources to prioritize",
        },
    },
    "required": ["title", "question", "search_hint"],
}

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "strategy": {
            "type": "string",
            "description": "1-2 sentences describing the overall research approach",
        },
        "review_focus": {
            "type": "string",
            "description": "What the adversarial reviewer should scrutinize most",
        },
        "tasks": {"type": "array", "items": _TASK_SCHEMA},
    },
    "required": ["strategy", "review_focus", "tasks"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "satisfied": {
            "type": "boolean",
            "description": "True if the evidence is sufficient to synthesize a verified answer",
        },
        "critique": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "followup_tasks": {"type": "array", "items": _TASK_SCHEMA},
    },
    "required": ["satisfied", "critique", "gaps", "followup_tasks"],
}
