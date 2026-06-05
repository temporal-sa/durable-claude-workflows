"""Temporal activities - the non-deterministic edge of the system.

Two activities:
  plan_workflow  -> Claude authors the workflow DAG (the "dynamic workflow")
  run_node       -> execute one node of the DAG (agent / review / synthesize)

Both have a MOCK path (deterministic, fast, no API key) so the whole Temporal
orchestration is runnable out of the box. Set ANTHROPIC_API_KEY (or unset
DURABLE_CLAUDE_MOCK) to switch to real Claude.
"""

from __future__ import annotations

import asyncio
import os

from temporalio import activity

import claude_llm
import config
from models import NodeResult, NodeRunInput, PlanInput, PlanNode, WorkflowPlan

_AGENT_MODEL = {"agent": config.AGENT_MODEL, "review": config.REVIEW_MODEL, "synthesize": config.SYNTH_MODEL}
_AGENT_EFFORT = {"agent": config.AGENT_EFFORT, "review": config.REVIEW_EFFORT, "synthesize": config.SYNTH_EFFORT}
_MOCK_INTERVAL = {
    "agent": config.AGENT_HEARTBEAT_INTERVAL,
    "review": config.REVIEW_HEARTBEAT_INTERVAL,
    "synthesize": config.SYNTH_HEARTBEAT_INTERVAL,
}


def _short(s: str, n: int = 120) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "..."


# ==========================================================================
# Activities
# ==========================================================================
@activity.defn
async def plan_workflow(inp: PlanInput) -> WorkflowPlan:
    """Claude designs the multi-agent DAG that Temporal will durably execute."""
    if config.mock_mode():
        return _mock_plan(inp)
    return await claude_llm.plan(
        inp.goal, inp.history, inp.max_nodes, model=config.PLANNER_MODEL, effort=config.PLANNER_EFFORT
    )


@activity.defn
async def run_node(inp: NodeRunInput) -> NodeResult:
    """Execute one node of the DAG. Invoked from the node's own child workflow."""
    if config.mock_mode():
        return await _mock_node(inp)
    return await claude_llm.run_node(
        inp.goal, inp.node, inp.upstream,
        model=_AGENT_MODEL[inp.node.kind], effort=_AGENT_EFFORT[inp.node.kind],
        web_search_enabled=config.ENABLE_WEB_SEARCH,
    )


# ==========================================================================
# Mock implementations (no API key required)
# ==========================================================================
def _mock_plan(inp: PlanInput) -> WorkflowPlan:
    g = _short(inp.goal, 70)
    nodes = [
        PlanNode(id="a1", kind="agent", title="Gather key facts",
                 instruction=f"Find the key facts and context for: {inp.goal}", use_web_search=True),
        PlanNode(id="a2", kind="agent", title="Evidence & data",
                 instruction=f"Find supporting evidence and data for: {inp.goal}", use_web_search=True),
        PlanNode(id="a3", kind="agent", title="Risks & counterpoints",
                 instruction=f"Find risks, limits, and counterarguments for: {inp.goal}", use_web_search=True),
        PlanNode(id="review", kind="review", title="Adversarial review",
                 instruction="Cross-check the findings for gaps and contradictions.",
                 depends_on=["a1", "a2", "a3"]),
        PlanNode(id="final", kind="synthesize", title="Synthesize answer",
                 instruction=f"Write the final answer to: {inp.goal}",
                 depends_on=["a1", "a2", "a3", "review"]),
    ]
    return WorkflowPlan(
        title=f"Workflow: {g}",
        summary=(f"Fan out 3 agents to investigate “{g}” in parallel, adversarially review the "
                 "findings, then synthesize a verified answer."),
        nodes=nodes,
        output="final",
    )


async def _mock_node(inp: NodeRunInput) -> NodeResult:
    node = inp.node
    async with claude_llm.heartbeater(_MOCK_INTERVAL[node.kind], f"{node.kind}:{node.id}"):
        await asyncio.sleep(config.MOCK_LATENCY)

    if node.kind == "agent":
        if node.use_filesystem and config.ENABLE_FILE_TOOLS:
            ws = config.workspace_dir()
            fname = f"{node.id}.md"
            with open(os.path.join(ws, fname), "w", encoding="utf-8") as f:
                f.write(f"# {node.title} (mock)\n\n{_short(node.instruction, 200)}\n")
            out = (f"- **{node.title}** (mock, coding): wrote `{fname}` to the workspace.\n"
                   f"- In live mode this step uses the bash + text-editor tools to build real files.\n\n"
                   f"Confidence: 0.8")
            return NodeResult(id=node.id, kind="agent", title=node.title, output=out,
                              sources=[], confidence=0.8)
        out = (
            f"- **{node.title}** (mock): {_short(node.instruction, 90)}\n"
            f"- Representative point A (illustrative).\n"
            f"- Representative point B with a sample figure (~42%).\n"
            f"- Caveat: generated in MOCK mode - no web search performed.\n\n"
            f"Sources:\n- https://example.com/{node.id}\n\nConfidence: 0.8"
        )
        return NodeResult(id=node.id, kind="agent", title=node.title, output=out,
                          sources=[f"https://example.com/{node.id}"], confidence=0.8)

    if node.kind == "review":
        return NodeResult(id=node.id, kind="review", title=node.title,
                          output="Adversarial review (mock): findings are coherent, sourced, and cover the "
                                 "main facets; no blocking contradictions. Safe to synthesize.")

    # synthesize
    lines = [f"# {_short(inp.goal, 80)}", "",
             f"> Synthesized by a durable Temporal workflow that ran {len(inp.upstream)} upstream steps (MOCK).", ""]
    for r in inp.upstream:
        if r.kind == "agent":
            lines.append(f"## {r.title}\n{r.output}\n")
    lines.append("---\n*MOCK mode - Claude plans the DAG, Temporal executes it. Set ANTHROPIC_API_KEY for real Claude.*")
    return NodeResult(id=node.id, kind="synthesize", title=node.title, output="\n".join(lines))
