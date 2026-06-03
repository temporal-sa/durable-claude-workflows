"""Temporal activities — the non-deterministic edge of the system.

Four activities mirror the stages of the dynamic workflow:
  plan_workflow      → Claude authors the plan (the "dynamic workflow")
  run_subagent       → one research subagent (run inside a child workflow)
  adversarial_review → Claude cross-checks the findings
  synthesize_report  → Claude writes the final verified report

Each has a MOCK path (deterministic, fast, no API key) so the entire Temporal
orchestration is runnable and demoable out of the box. Set ANTHROPIC_API_KEY
(or unset DURABLE_CLAUDE_MOCK) to switch to real Claude.
"""

from __future__ import annotations

import asyncio

from temporalio import activity

import claude_llm
import config
from models import (
    Plan,
    PlanInput,
    ResearchInput,
    ReviewInput,
    ReviewResult,
    SubAgentResult,
    SubAgentTask,
    SynthesisInput,
)


def _short(s: str, n: int = 120) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ==========================================================================
# Activities
# ==========================================================================
@activity.defn
async def plan_workflow(inp: PlanInput) -> Plan:
    """Claude designs the multi-agent plan that Temporal will durably execute."""
    if config.mock_mode():
        return _mock_plan(inp)
    return await claude_llm.plan(
        inp.goal, inp.history, inp.max_subagents,
        model=config.PLANNER_MODEL, effort=config.PLANNER_EFFORT,
    )


@activity.defn
async def run_subagent(inp: ResearchInput) -> SubAgentResult:
    """One research subagent. Invoked from its own child workflow."""
    if config.mock_mode():
        return await _mock_research(inp)
    return await claude_llm.research(
        inp.goal, inp.task,
        model=config.SUBAGENT_MODEL, effort=config.SUBAGENT_EFFORT,
        web_search=config.ENABLE_WEB_SEARCH,
    )


@activity.defn
async def adversarial_review(inp: ReviewInput) -> ReviewResult:
    """Claude cross-checks the findings and decides whether to iterate."""
    if config.mock_mode():
        return _mock_review(inp)
    return await claude_llm.review(
        inp.goal, inp.strategy, inp.findings, inp.iteration, inp.max_iterations,
        model=config.REVIEW_MODEL, effort=config.REVIEW_EFFORT,
    )


@activity.defn
async def synthesize_report(inp: SynthesisInput) -> str:
    """Claude writes the final, verified report."""
    if config.mock_mode():
        return await _mock_synthesize(inp)
    return await claude_llm.synthesize(
        inp.goal, inp.history, inp.findings, inp.review,
        model=config.SYNTH_MODEL, effort=config.SYNTH_EFFORT,
    )


# ==========================================================================
# Mock implementations (no API key required)
# ==========================================================================
_FACETS = [
    ("Background & definitions", "What is the essential background and key definitions for", "overview, primers, authoritative definitions"),
    ("Current evidence & data", "What does current evidence and data show about", "recent studies, statistics, primary sources"),
    ("Comparisons & alternatives", "What are the main comparisons, alternatives, or trade-offs for", "comparison articles, benchmarks"),
    ("Risks, limits & counterarguments", "What are the risks, limitations, and strongest counterarguments about", "critiques, skeptical analyses"),
    ("Recent developments", "What are the most recent (2025-2026) developments relevant to", "news, release notes, announcements"),
]


def _mock_plan(inp: PlanInput) -> Plan:
    n = max(2, min(inp.max_subagents, len(_FACETS)))
    tasks = [
        SubAgentTask(
            id=i + 1,
            title=title,
            question=f"{stem}: {inp.goal}?",
            search_hint=hint,
        )
        for i, (title, stem, hint) in enumerate(_FACETS[:n])
    ]
    return Plan(
        strategy=(
            f"Decompose “{_short(inp.goal, 80)}” into {len(tasks)} parallel research threads, "
            "fan them out as durable child workflows, adversarially review the findings, then "
            "synthesize a verified answer."
        ),
        review_focus="Unsupported claims, missing facets, and contradictions across subagents.",
        tasks=tasks,
    )


async def _mock_research(inp: ResearchInput) -> SubAgentResult:
    t = inp.task
    async with claude_llm.heartbeater(config.SUBAGENT_HEARTBEAT_INTERVAL, f"subagent:{t.id}"):
        await asyncio.sleep(config.MOCK_LATENCY)
    findings = (
        f"- **{t.title}** — simulated findings for: _{t.question}_\n"
        f"- Representative point A about {_short(inp.goal, 60)} (illustrative).\n"
        f"- Representative point B with a sample figure (~42%).\n"
        f"- Caveat: generated in MOCK mode — no web search was performed.\n\n"
        f"Sources:\n- https://example.com/{t.id}/a\n- https://example.com/{t.id}/b\n\n"
        f"Confidence: 0.78"
    )
    return SubAgentResult(
        id=t.id,
        title=t.title,
        findings=findings,
        sources=[f"https://example.com/{t.id}/a", f"https://example.com/{t.id}/b"],
        confidence=0.78,
    )


def _mock_review(inp: ReviewInput) -> ReviewResult:
    # Demonstrate the iterate-on-gaps loop on the first pass when iterations allow.
    if inp.iteration == 0 and inp.max_iterations > 1:
        return ReviewResult(
            satisfied=False,
            critique=(
                "Initial findings are reasonable but lack independent cross-verification of the "
                "headline claim and recent data. One more targeted pass is recommended."
            ),
            gaps=["No independent cross-check of the central claim", "Limited 2026 data"],
            followup_tasks=[
                SubAgentTask(
                    id=0,
                    title="Cross-verify central claim",
                    question=f"Independently verify the central claim about: {inp.goal}.",
                    search_hint="primary sources, original studies",
                )
            ],
        )
    return ReviewResult(
        satisfied=True,
        critique="Findings are coherent, sourced, and cover the key facets. Sufficient to synthesize.",
        gaps=[],
        followup_tasks=[],
    )


async def _mock_synthesize(inp: SynthesisInput) -> str:
    async with claude_llm.heartbeater(config.SYNTH_HEARTBEAT_INTERVAL, "synthesize"):
        await asyncio.sleep(config.MOCK_LATENCY)
    lines = [
        f"# {_short(inp.goal, 90)}",
        "",
        (
            "> **Executive summary (MOCK).** A durable Temporal workflow fanned out "
            f"{len(inp.findings)} research subagents as child workflows, adversarially reviewed "
            "their findings, and synthesized this report. Set `ANTHROPIC_API_KEY` "
            "(or unset `DURABLE_CLAUDE_MOCK`) to run it with real Claude."
        ),
        "",
        "## Findings by subagent",
        "",
    ]
    for f in inp.findings:
        lines.append(f"### {f.title}  ")
        lines.append(f"_confidence {f.confidence:.2f}_")
        lines.append("")
        lines.append(f.findings or f.error or "(no findings)")
        lines.append("")
    if inp.review is not None:
        lines.append("## Adversarial review")
        lines.append("")
        verdict = "satisfied ✓" if inp.review.satisfied else "needed another pass ↻"
        lines.append(f"- Verdict: **{verdict}**")
        lines.append(f"- {inp.review.critique}")
        lines.append("")
    lines.append("---")
    lines.append("*Generated in MOCK mode • Claude plans, Temporal executes.*")
    return "\n".join(lines)
