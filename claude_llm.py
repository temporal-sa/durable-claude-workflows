"""Anthropic API helpers — the ONLY place that calls Claude.

Every function here runs inside a Temporal activity, so:
  * the SDK's own retries are disabled (``max_retries=0``) — Temporal owns retries;
  * exceptions are mapped to ``ApplicationError`` with sensible retryable/
    non-retryable classification;
  * calls heartbeat on a fixed cadence via a background ticker, so the worker can
    detect stalls and deliver cancellation even while a call is blocked.

This is where "Claude plans" happens (``plan``) and where each subagent's
research, the adversarial review, and the final synthesis run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re

import anthropic
from anthropic import AsyncAnthropic
from temporalio import activity
from temporalio.exceptions import ApplicationError

import config
from models import (
    PLAN_SCHEMA,
    REVIEW_SCHEMA,
    Plan,
    ReviewResult,
    SubAgentResult,
    SubAgentTask,
    Turn,
)

# --------------------------------------------------------------------------
# Client + error mapping
# --------------------------------------------------------------------------
_client_obj: AsyncAnthropic | None = None


def _client() -> AsyncAnthropic:
    global _client_obj
    if _client_obj is None:
        # Disable SDK retries — Temporal's RetryPolicy handles them durably.
        # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ant profile.
        _client_obj = AsyncAnthropic(max_retries=0, timeout=600.0)
    return _client_obj


def _map_error(e: Exception) -> ApplicationError:
    if isinstance(e, ApplicationError):
        return e
    if isinstance(e, anthropic.AuthenticationError):
        return ApplicationError(
            f"Anthropic authentication failed (set ANTHROPIC_API_KEY): {e}",
            type="AuthError",
            non_retryable=True,
        )
    if isinstance(e, anthropic.PermissionDeniedError):
        return ApplicationError(f"Permission denied: {e}", type="PermissionError", non_retryable=True)
    if isinstance(e, anthropic.NotFoundError):
        return ApplicationError(
            f"Not found — check the model id: {e}", type="NotFound", non_retryable=True
        )
    if isinstance(e, anthropic.BadRequestError):
        return ApplicationError(f"Bad request: {e}", type="BadRequest", non_retryable=True)
    if isinstance(e, anthropic.RateLimitError):
        return ApplicationError(f"Rate limited: {e}", type="RateLimit")  # retryable
    if isinstance(e, anthropic.APIStatusError):
        status = getattr(e, "status_code", None)
        if status and status >= 500:  # includes 529 overloaded
            return ApplicationError(f"Anthropic server error {status}: {e}", type="ServerError")
        return ApplicationError(f"Anthropic API error {status}: {e}", type="APIError", non_retryable=True)
    if isinstance(e, anthropic.APIConnectionError):
        return ApplicationError(f"Connection error: {e}", type="ConnectionError")  # retryable
    return ApplicationError(f"Unexpected error calling Claude: {e}", type="Unknown")


# --------------------------------------------------------------------------
# Low-level call helpers
# --------------------------------------------------------------------------
def _system(text: str) -> list[dict]:
    # Stable system prompt → cache it (cheap reuse across the many subagent calls).
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _first_text(resp) -> str:
    return next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")


def _loads(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)  # tolerate fences / stray prose
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ApplicationError(
        "Claude did not return valid JSON for a structured step", type="ParseError"
    )


@contextlib.asynccontextmanager
async def heartbeater(interval: float, label: str):
    """Heartbeat the current activity every ``interval`` seconds until the block exits.

    Runs as a background task so the heartbeat fires even while the Claude call is
    blocked (adaptive thinking before the first token, between web searches, etc.).
    A task created with ``create_task`` inherits the activity's context, so
    ``activity.heartbeat`` targets the right activity.
    """
    async def _beat() -> None:
        while True:
            await asyncio.sleep(interval)
            activity.heartbeat(label)

    task = asyncio.create_task(_beat())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _structured(
    system: str, user: str, schema: dict, *, model: str, effort: str, interval: float
) -> dict:
    client = _client()
    try:
        async with heartbeater(interval, "structured"):
            resp = await client.messages.create(
                model=model,
                max_tokens=8000,
                system=_system(system),
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
    except Exception as e:  # noqa: BLE001 — mapped to ApplicationError
        raise _map_error(e)
    return _loads(_first_text(resp))


async def _stream_text(
    system: str,
    messages: list[dict],
    *,
    model: str,
    effort: str,
    interval: float,
    tools: list[dict] | None = None,
    max_tokens: int = 16000,
    hb: str = "claude",
    max_pause_continuations: int = 3,
) -> tuple[str, object]:
    """Stream a text response while a background ticker heartbeats on a fixed cadence.

    Handles server-tool ``pause_turn`` (e.g. web_search hitting its server-side
    iteration cap) by re-sending the accumulated assistant turn and continuing.
    """
    client = _client()
    msgs = list(messages)
    text_parts: list[str] = []
    final = None
    try:
        async with heartbeater(interval, hb):
            for _ in range(max_pause_continuations + 1):
                kwargs: dict = dict(
                    model=model,
                    max_tokens=max_tokens,
                    system=_system(system),
                    messages=msgs,
                    thinking={"type": "adaptive"},
                    output_config={"effort": effort},
                )
                if tools:
                    kwargs["tools"] = tools
                async with client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                            text_parts.append(event.delta.text)
                    final = await stream.get_final_message()
                if getattr(final, "stop_reason", None) != "pause_turn":
                    break
                # Resume the server-tool loop: re-send the assistant turn verbatim.
                msgs.append({"role": "assistant", "content": final.content})
    except Exception as e:  # noqa: BLE001
        raise _map_error(e)
    return "".join(text_parts).strip(), final


# --------------------------------------------------------------------------
# Source / confidence extraction (best effort)
# --------------------------------------------------------------------------
_URL_RE = re.compile(r'https?://[^\s<>")\]]+')
_CONF_RE = re.compile(r"confidence[:=]\s*([01](?:\.\d+)?)", re.I)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _extract_confidence(text: str) -> float:
    m = _CONF_RE.search(text or "")
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.6


def _extract_search_sources(final) -> list[str]:
    urls: list[str] = []
    try:
        for b in getattr(final, "content", []) or []:
            if "web_search" in getattr(b, "type", ""):
                for item in getattr(b, "content", None) or []:
                    u = getattr(item, "url", None)
                    if u:
                        urls.append(u)
    except Exception:  # noqa: BLE001 — purely best-effort
        pass
    return urls


# --------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------
def _history_block(history: list[Turn]) -> str:
    if not history:
        return ""
    lines = ["Conversation so far (most recent last):"]
    for t in history[-6:]:
        snippet = t.content if len(t.content) <= 600 else t.content[:600] + "…"
        lines.append(f"[{t.role}] {snippet}")
    return "\n".join(lines) + "\n\n"


def _findings_block(findings: list[SubAgentResult]) -> str:
    parts = []
    for f in findings:
        head = f"### [{f.id}] {f.title} (confidence {f.confidence:.2f})"
        if f.error:
            head += " — FAILED"
        parts.append(head + "\n" + (f.findings or f.error or "(no findings)"))
    return "\n\n".join(parts)


PLAN_SYSTEM = (
    "You are the PLANNER of a durable multi-agent research workflow running on Temporal. "
    "Given the user's research goal (and any prior conversation), design the plan that "
    "Temporal will execute: a set of focused, parallelizable research subagent tasks plus "
    "guidance for an adversarial reviewer.\n\n"
    "Guidelines:\n"
    "- Produce {max_subagents} tasks or fewer; each must be independent so they can run in parallel.\n"
    "- Cover distinct facets (background/definitions, evidence/data, comparisons, "
    "risks/limitations, recent developments) rather than overlapping.\n"
    "- Make each question specific and answerable via web research; keep titles to 3-6 words.\n"
    "- review_focus should name the riskiest things to double-check.\n"
    "Respond ONLY with the JSON object matching the provided schema."
)

RESEARCH_SYSTEM = (
    "You are a RESEARCH SUBAGENT in a larger multi-agent workflow. Investigate ONLY your "
    "assigned sub-question — do not try to answer the whole goal.\n"
    "- Use web search to find current, credible sources; prefer primary/authoritative ones.\n"
    "- Report concise markdown bullet findings with specific facts, figures, and dates, each "
    "tied to a source.\n"
    "- End with a 'Sources:' list of URLs, then a final line exactly like 'Confidence: 0.8' "
    "reflecting how well-supported your findings are."
)

RESEARCH_SYSTEM_NOSEARCH = (
    "You are a RESEARCH SUBAGENT in a larger multi-agent workflow. Investigate ONLY your "
    "assigned sub-question. Web search is unavailable, so answer from your own knowledge and "
    "clearly flag anything uncertain or potentially out of date. Report concise markdown "
    "bullet findings, then a final line exactly like 'Confidence: 0.5'."
)

REVIEW_SYSTEM = (
    "You are an ADVERSARIAL REVIEWER. Scrutinize the subagents' findings against the research "
    "goal. Identify contradictions, unsupported claims, missing facets, and weak sourcing, then "
    "decide whether the evidence is sufficient to write a high-quality, verified answer.\n"
    "- If it is sufficient, set satisfied=true with empty followup_tasks.\n"
    "- If not, set satisfied=false and propose a FEW targeted follow-up subagent tasks that would "
    "close the most important gaps.\n"
    "Be strict but fair. Respond ONLY with the JSON object matching the provided schema."
)

SYNTH_SYSTEM = (
    "You are the SYNTHESIZER. Write the final, verified report answering the user's goal. "
    "Integrate the subagents' findings, resolve conflicts, and explicitly account for the "
    "reviewer's critique. Lead with a direct answer / executive summary, use clear markdown "
    "headings, support claims with the gathered evidence, and include a 'Sources' section. "
    "Be accurate and concise; do not invent facts beyond the findings."
)


# --------------------------------------------------------------------------
# Public activity-facing functions
# --------------------------------------------------------------------------
async def plan(goal: str, history: list[Turn], max_subagents: int, *, model: str, effort: str) -> Plan:
    data = await _structured(
        PLAN_SYSTEM.format(max_subagents=max_subagents),
        _history_block(history) + f"Research goal:\n{goal}\n\nDesign the research plan.",
        PLAN_SCHEMA,
        model=model,
        effort=effort,
        interval=config.PLAN_HEARTBEAT_INTERVAL,
    )
    tasks = [
        SubAgentTask(id=i + 1, title=t["title"], question=t["question"], search_hint=t.get("search_hint", ""))
        for i, t in enumerate((data.get("tasks") or [])[:max_subagents])
    ]
    if not tasks:  # never hand Temporal an empty fan-out
        tasks = [SubAgentTask(id=1, title="Investigate goal", question=goal, search_hint="")]
    return Plan(
        strategy=data.get("strategy", ""),
        review_focus=data.get("review_focus", ""),
        tasks=tasks,
    )


async def research(goal: str, task: SubAgentTask, *, model: str, effort: str, web_search: bool) -> SubAgentResult:
    user = (
        f"Overall research goal: {goal}\n\n"
        f"Your assigned sub-question:\n"
        f"- Title: {task.title}\n- Question: {task.question}\n- Search hint: {task.search_hint}\n\n"
        "Investigate and report well-sourced findings."
    )
    tools = (
        [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}] if web_search else None
    )
    system = RESEARCH_SYSTEM if web_search else RESEARCH_SYSTEM_NOSEARCH
    try:
        text, final = await _stream_text(
            system, [{"role": "user", "content": user}],
            model=model, effort=effort, interval=config.SUBAGENT_HEARTBEAT_INTERVAL,
            tools=tools, max_tokens=8000, hb=f"subagent:{task.id}",
        )
    except ApplicationError as e:
        # Web search not enabled on this account → retry once without tools.
        if web_search and getattr(e, "type", None) == "BadRequest":
            text, final = await _stream_text(
                RESEARCH_SYSTEM_NOSEARCH, [{"role": "user", "content": user}],
                model=model, effort=effort, interval=config.SUBAGENT_HEARTBEAT_INTERVAL,
                tools=None, max_tokens=8000, hb=f"subagent:{task.id}",
            )
        else:
            raise
    sources = _dedupe(_extract_search_sources(final) + _URL_RE.findall(text))[:10]
    return SubAgentResult(
        id=task.id,
        title=task.title,
        findings=text or "(no findings produced)",
        sources=sources,
        confidence=_extract_confidence(text),
    )


async def review(
    goal: str, strategy: str, findings: list[SubAgentResult], iteration: int, max_iterations: int,
    *, model: str, effort: str,
) -> ReviewResult:
    user = (
        f"Research goal: {goal}\n\nPlanned strategy: {strategy}\n\n"
        f"This is review pass {iteration + 1} of at most {max_iterations}.\n\n"
        f"Subagent findings to scrutinize:\n\n{_findings_block(findings)}"
    )
    data = await _structured(
        REVIEW_SYSTEM, user, REVIEW_SCHEMA, model=model, effort=effort,
        interval=config.REVIEW_HEARTBEAT_INTERVAL,
    )
    followups = [
        SubAgentTask(id=0, title=t["title"], question=t["question"], search_hint=t.get("search_hint", ""))
        for t in (data.get("followup_tasks") or [])
    ]
    return ReviewResult(
        satisfied=bool(data.get("satisfied")),
        critique=data.get("critique", ""),
        gaps=list(data.get("gaps") or []),
        followup_tasks=followups,
    )


async def synthesize(
    goal: str, history: list[Turn], findings: list[SubAgentResult], review_result: ReviewResult | None,
    *, model: str, effort: str,
) -> str:
    critique = ""
    if review_result is not None:
        critique = (
            f"Reviewer verdict: {'satisfied' if review_result.satisfied else 'gaps remained'}\n"
            f"Reviewer critique: {review_result.critique}\n"
        )
        if review_result.gaps:
            critique += "Known gaps to acknowledge: " + "; ".join(review_result.gaps) + "\n"
    user = (
        _history_block(history)
        + f"Research goal: {goal}\n\n{critique}\n"
        + f"Subagent findings:\n\n{_findings_block(findings)}\n\n"
        + "Write the final verified report."
    )
    text, _ = await _stream_text(
        SYNTH_SYSTEM, [{"role": "user", "content": user}],
        model=model, effort=effort, interval=config.SYNTH_HEARTBEAT_INTERVAL,
        tools=None, max_tokens=20000, hb="synthesize",
    )
    return text or "(synthesis produced no text)"
