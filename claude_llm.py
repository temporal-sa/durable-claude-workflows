"""Anthropic API helpers - the ONLY place that calls Claude.

Every function here runs inside a Temporal activity, so:
  * the SDK's own retries are disabled (``max_retries=0``) - Temporal owns retries;
  * exceptions are mapped to ``ApplicationError`` with retryable / non-retryable
    classification;
  * calls heartbeat on a fixed cadence via a background ticker, so the worker can
    detect stalls and deliver cancellation even while a call is blocked.

``plan`` is where "Claude authors the dynamic workflow" happens: it returns a DAG
of typed nodes. ``run_node`` executes one node (agent / review / synthesize).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re

import anthropic
from anthropic import AsyncAnthropic
from temporalio import activity
from temporalio.exceptions import ApplicationError

import config
from models import (
    PLAN_SCHEMA,
    NodeResult,
    PlanNode,
    Turn,
    WorkflowPlan,
)

# --------------------------------------------------------------------------
# Client + error mapping
# --------------------------------------------------------------------------
_client_obj: AsyncAnthropic | None = None


def _client() -> AsyncAnthropic:
    global _client_obj
    if _client_obj is None:
        _client_obj = AsyncAnthropic(max_retries=0, timeout=600.0)
    return _client_obj


def _map_error(e: Exception) -> ApplicationError:
    if isinstance(e, ApplicationError):
        return e
    if isinstance(e, anthropic.AuthenticationError):
        return ApplicationError(f"Anthropic authentication failed (set ANTHROPIC_API_KEY): {e}",
                                type="AuthError", non_retryable=True)
    if isinstance(e, anthropic.PermissionDeniedError):
        return ApplicationError(f"Permission denied: {e}", type="PermissionError", non_retryable=True)
    if isinstance(e, anthropic.NotFoundError):
        return ApplicationError(f"Not found - check the model id: {e}", type="NotFound", non_retryable=True)
    if isinstance(e, anthropic.BadRequestError):
        return ApplicationError(f"Bad request: {e}", type="BadRequest", non_retryable=True)
    if isinstance(e, anthropic.RateLimitError):
        return ApplicationError(f"Rate limited: {e}", type="RateLimit")
    if isinstance(e, anthropic.APIStatusError):
        status = getattr(e, "status_code", None)
        if status and status >= 500:
            return ApplicationError(f"Anthropic server error {status}: {e}", type="ServerError")
        return ApplicationError(f"Anthropic API error {status}: {e}", type="APIError", non_retryable=True)
    if isinstance(e, anthropic.APIConnectionError):
        return ApplicationError(f"Connection error: {e}", type="ConnectionError")
    return ApplicationError(f"Unexpected error calling Claude: {e}", type="Unknown")


# --------------------------------------------------------------------------
# Low-level call helpers
# --------------------------------------------------------------------------
def _system(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _first_text(resp) -> str:
    return next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")


def _loads(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ApplicationError("Claude did not return valid JSON for the plan", type="ParseError")


@contextlib.asynccontextmanager
async def heartbeater(interval: float, label: str):
    """Heartbeat the current activity every ``interval`` seconds until the block exits.

    Runs as a background task so the heartbeat fires even while the Claude call is
    blocked. A task created with ``create_task`` inherits the activity's context,
    so ``activity.heartbeat`` targets the right activity.
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


async def _structured(system: str, user: str, schema: dict, *, model: str, effort: str, interval: float) -> dict:
    client = _client()
    try:
        async with heartbeater(interval, "plan"):
            resp = await client.messages.create(
                model=model,
                max_tokens=8000,
                system=_system(system),
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
    except Exception as e:  # noqa: BLE001
        raise _map_error(e)
    return _loads(_first_text(resp))


async def _stream_text(
    system: str,
    user: str,
    *,
    model: str,
    effort: str,
    interval: float,
    tools: list[dict] | None = None,
    max_tokens: int = 8000,
    hb: str = "claude",
    max_pause_continuations: int = 3,
) -> tuple[str, object]:
    """Stream a text response while a background ticker heartbeats on a fixed cadence.

    Handles server-tool ``pause_turn`` (e.g. web_search hitting its iteration cap)
    by re-sending the accumulated assistant turn and continuing.
    """
    client = _client()
    msgs: list[dict] = [{"role": "user", "content": user}]
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
    except Exception:  # noqa: BLE001
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
        snippet = t.content if len(t.content) <= 600 else t.content[:600] + "..."
        lines.append(f"[{t.role}] {snippet}")
    return "\n".join(lines) + "\n\n"


def _upstream_block(upstream: list[NodeResult]) -> str:
    if not upstream:
        return ""
    parts = ["Results from the steps this one depends on:\n"]
    for r in upstream:
        head = f"### [{r.id}] {r.title} ({r.kind})"
        parts.append(head + "\n" + (r.output or r.error or "(no output)"))
    return "\n\n".join(parts) + "\n\n"


PLAN_SYSTEM = (
    "You are the PLANNER of a durable multi-agent workflow that runs on Temporal. For ANY task "
    "the user gives, design the workflow as a directed acyclic graph (DAG) of steps that a runtime "
    "will execute, fanning independent steps out in parallel.\n\n"
    "Node kinds:\n"
    "- \"agent\": a worker that performs one focused piece of the task by reasoning (and optional "
    "web search). ONLY agent steps can read/write files and run shell commands (via use_filesystem). "
    "Use several independent agents (empty depends_on) for parts that can run in parallel.\n"
    "- \"review\": adversarially cross-checks / validates the outputs of the steps it depends on "
    "(finds gaps, errors, contradictions). Reads upstream text only; it cannot touch the filesystem.\n"
    "- \"synthesize\": writes the final SUMMARY/answer text from the steps it depends on. It also cannot "
    "touch the filesystem, so never put file-writing, build, or fix-applying steps here.\n\n"
    "Rules:\n"
    "- Use {max_nodes} nodes or fewer. Give each a short unique id (e.g. a1, a2, review, final), a "
    "short title, and a clear instruction.\n"
    "- depends_on lists the ids whose outputs this node needs; nodes with empty depends_on run first, in parallel.\n"
    "- Include at least one \"review\" step that depends on the agent steps, and exactly one final "
    "\"synthesize\" step (set it as `output`) that depends on the review.\n"
    "- Set use_web_search=true on steps that need current or external information; false otherwise.\n"
    "- For tasks that PRODUCE files (writing code/software, generating a project): EVERY step that creates, "
    "edits, or runs files MUST be an \"agent\" with use_filesystem=true. Those steps share ONE working "
    "directory, so later steps can read what earlier steps wrote; when file-writing steps run in parallel, "
    "give each its own subdirectory so they do not clobber shared files. The final assembly/build/test step "
    "that writes files must also be an \"agent\" with use_filesystem=true (NOT synthesize). Use the single "
    "synthesize step only for a short text summary of what was built and how to run it.\n"
    "- It MUST be a DAG (no cycles). Tailor the steps to the actual task - this is not limited to research.\n"
    "Respond ONLY with the JSON object matching the provided schema."
)

AGENT_SYSTEM = (
    "You are a worker agent inside a larger workflow. Do ONLY your assigned step - not the whole task. "
    "Be concrete and well-grounded. If web search is available, use it for current or external facts "
    "and cite sources. Return your result directly (markdown is fine). If you used sources, end with a "
    "'Sources:' list of URLs and a final line exactly like 'Confidence: 0.8'."
)

REVIEW_SYSTEM = (
    "You are an ADVERSARIAL REVIEWER inside a workflow. Scrutinize the upstream results against the "
    "overall goal and your instruction: surface errors, gaps, unsupported claims, and contradictions. "
    "Return a concise, specific critique that the synthesizer should account for."
)

SYNTH_SYSTEM = (
    "You are the SYNTHESIZER. Produce the final deliverable for the user's goal, following your "
    "instruction, integrating the upstream results and explicitly addressing the reviewer's critique. "
    "Lead with a direct answer, use clear markdown structure, and do not invent facts beyond the inputs."
)


# --------------------------------------------------------------------------
# Plan: Claude authors the DAG (then we validate + normalize it)
# --------------------------------------------------------------------------
def _normalize_plan(data: dict, goal: str, max_nodes: int) -> WorkflowPlan:
    raw = (data.get("nodes") or [])[:max_nodes]
    nodes: list[PlanNode] = []
    seen: set[str] = set()
    for i, n in enumerate(raw):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or f"n{i + 1}").strip() or f"n{i + 1}"
        while nid in seen:  # de-duplicate ids
            nid = f"{nid}_"
        seen.add(nid)
        kind = n.get("kind")
        if kind not in ("agent", "review", "synthesize"):
            kind = "agent"
        nodes.append(PlanNode(
            id=nid,
            kind=kind,
            title=str(n.get("title") or nid)[:80],
            instruction=str(n.get("instruction") or goal),
            depends_on=[str(d) for d in (n.get("depends_on") or [])],
            use_web_search=bool(n.get("use_web_search", False)),
            use_filesystem=bool(n.get("use_filesystem", False)) and kind == "agent",
        ))

    if not nodes:  # never hand Temporal an empty graph
        nodes = [PlanNode(id="a1", kind="agent", title="Work the task", instruction=goal, use_web_search=True),
                 PlanNode(id="final", kind="synthesize", title="Synthesize answer",
                          instruction=f"Answer: {goal}", depends_on=["a1"])]

    ids = {n.id for n in nodes}
    for n in nodes:
        n.depends_on = [d for d in n.depends_on if d in ids and d != n.id]  # drop dangling/self deps
    _break_cycles(nodes)

    output = str(data.get("output") or "")
    if output not in ids:
        synth = [n.id for n in nodes if n.kind == "synthesize"]
        output = synth[-1] if synth else nodes[-1].id

    return WorkflowPlan(
        title=str(data.get("title") or "Workflow")[:120],
        summary=str(data.get("summary") or "")[:600],
        nodes=nodes,
        output=output,
    )


def _break_cycles(nodes: list[PlanNode]) -> None:
    """Drop edges that would create a cycle (keep it a DAG), via a simple DFS."""
    by_id = {n.id: n for n in nodes}
    color: dict[str, int] = {}  # 0=unvisited, 1=in-stack, 2=done

    def visit(nid: str) -> None:
        color[nid] = 1
        node = by_id[nid]
        for d in list(node.depends_on):
            c = color.get(d, 0)
            if c == 1:  # back-edge -> cycle; drop it
                node.depends_on.remove(d)
            elif c == 0:
                visit(d)
        color[nid] = 2

    for n in nodes:
        if color.get(n.id, 0) == 0:
            visit(n.id)


async def plan(goal: str, history: list[Turn], max_nodes: int, *, model: str, effort: str) -> WorkflowPlan:
    data = await _structured(
        PLAN_SYSTEM.format(max_nodes=max_nodes),
        _history_block(history) + f"Task:\n{goal}\n\nDesign the workflow DAG.",
        PLAN_SCHEMA,
        model=model,
        effort=effort,
        interval=config.PLAN_HEARTBEAT_INTERVAL,
    )
    return _normalize_plan(data, goal, max_nodes)


# --------------------------------------------------------------------------
# Run one node
# --------------------------------------------------------------------------
_NODE_SYSTEM = {"agent": AGENT_SYSTEM, "review": REVIEW_SYSTEM, "synthesize": SYNTH_SYSTEM}
_NODE_INTERVAL = {
    "agent": config.AGENT_HEARTBEAT_INTERVAL,
    "review": config.REVIEW_HEARTBEAT_INTERVAL,
    "synthesize": config.SYNTH_HEARTBEAT_INTERVAL,
}

_WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
_BASH_TOOL = {"type": "bash_20250124", "name": "bash"}
_EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}

AGENT_SYSTEM_CODING = (
    "You are a worker agent in a workflow that CAN read, write, and edit files and run shell commands. "
    "Your working directory - the only place you may touch - is:\n  {workspace}\n"
    "Actually create and modify files there using the text editor tool (view / create / str_replace / "
    "insert) and run commands with bash. Do NOT just print code in your reply - code only counts if it is "
    "written to disk with the tools. Earlier steps may have already written files here, so view or `ls` the "
    "directory first and build on what exists rather than overwriting it. Use paths inside the working "
    "directory (relative paths are fine); if your instruction names an absolute path, treat THIS working "
    "directory as that location. When you finish, syntax-check or run what you wrote if practical, then end "
    "with a short summary listing the files you created or changed."
)


# --- Local tool executors (run inside the activity, on the worker's disk) ---
def _safe_path(workspace: str, path: str) -> str | None:
    p = path if os.path.isabs(path) else os.path.join(workspace, path)
    rp = os.path.realpath(p)
    root = os.path.realpath(workspace)
    return rp if rp == root or rp.startswith(root + os.sep) else None


async def _run_bash(command: str, workspace: str, timeout: float = 120.0) -> tuple[str, bool]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=workspace,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    except Exception as e:  # noqa: BLE001
        return f"bash error: {e}", True
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"(bash timed out after {timeout:.0f}s)", True
    text = (out or b"").decode("utf-8", "replace")
    if len(text) > 16000:
        text = text[:16000] + "\n...(truncated)"
    return text or f"(exit {proc.returncode}, no output)", proc.returncode != 0


def _text_editor(inp: dict, workspace: str) -> tuple[str, bool]:
    cmd = inp.get("command")
    rp = _safe_path(workspace, inp.get("path", ""))
    if rp is None:
        return f"Error: path is outside the working directory ({workspace}). Use paths within it.", True
    try:
        if cmd == "view":
            if os.path.isdir(rp):
                return "\n".join(sorted(os.listdir(rp))) or "(empty directory)", False
            lines = open(rp, encoding="utf-8", errors="replace").read().splitlines()
            vr = inp.get("view_range")
            start = 1
            if isinstance(vr, list) and len(vr) == 2:
                s, e = vr
                e = len(lines) if e == -1 else e
                lines, start = lines[max(0, s - 1):e], max(1, s)
            return "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(lines)) or "(empty file)", False
        if cmd == "create":
            os.makedirs(os.path.dirname(rp) or ".", exist_ok=True)
            with open(rp, "w", encoding="utf-8") as f:
                f.write(inp.get("file_text", ""))
            return f"Created {inp.get('path')}", False
        if cmd == "str_replace":
            content = open(rp, encoding="utf-8").read()
            old = inp.get("old_str", "")
            n = content.count(old)
            if n != 1:
                return f"Error: old_str must match exactly once (matched {n}).", True
            with open(rp, "w", encoding="utf-8") as f:
                f.write(content.replace(old, inp.get("new_str", ""), 1))
            return f"Edited {inp.get('path')}", False
        if cmd == "insert":
            lines = open(rp, encoding="utf-8").readlines()
            new = inp.get("new_str", "")
            if not new.endswith("\n"):
                new += "\n"
            lines.insert(min(int(inp.get("insert_line", 0)), len(lines)), new)
            with open(rp, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return f"Inserted into {inp.get('path')}", False
        return f"Error: unsupported text-editor command '{cmd}'", True
    except FileNotFoundError:
        return f"Error: file not found: {inp.get('path')}", True
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}", True


async def _agentic(system: str, user: str, *, model: str, effort: str, interval: float, hb: str,
                   tools: list[dict], workspace: str, max_iters: int = 30) -> str:
    """Run Claude with client-side bash / text-editor tools, executing each tool call on
    the worker's filesystem and feeding results back, until Claude is done."""
    client = _client()
    messages: list[dict] = [{"role": "user", "content": user}]
    text_parts: list[str] = []
    try:
        async with heartbeater(interval, hb):
            for _ in range(max_iters):
                resp = await client.messages.create(
                    model=model, max_tokens=16000, system=_system(system), messages=messages,
                    tools=tools, thinking={"type": "adaptive"}, output_config={"effort": effort})
                for b in resp.content:
                    if getattr(b, "type", None) == "text":
                        text_parts.append(b.text)
                if resp.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": resp.content})
                    results = []
                    for b in resp.content:
                        if getattr(b, "type", None) != "tool_use":
                            continue
                        activity.heartbeat(hb)
                        if b.name == "bash":
                            out, err = (("(shell restarted)", False) if b.input.get("restart")
                                        else await _run_bash(b.input.get("command", ""), workspace))
                        else:  # text editor
                            out, err = _text_editor(b.input, workspace)
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
                    messages.append({"role": "user", "content": results})
                    continue
                if resp.stop_reason == "pause_turn":  # server tool (web_search) mid-loop
                    messages.append({"role": "assistant", "content": resp.content})
                    continue
                break
    except Exception as e:  # noqa: BLE001
        raise _map_error(e)
    return "".join(text_parts).strip()


async def run_node(goal: str, node: PlanNode, upstream: list[NodeResult], *, model: str, effort: str,
                   web_search_enabled: bool) -> NodeResult:
    user = (
        f"Overall goal: {goal}\n\n"
        f"{_upstream_block(upstream)}"
        f"Your step ({node.kind}): {node.title}\n{node.instruction}"
    )
    interval = _NODE_INTERVAL[node.kind]
    want_search = node.use_web_search and web_search_enabled and node.kind in ("agent", "review")
    want_fs = node.use_filesystem and config.ENABLE_FILE_TOOLS and node.kind == "agent"

    # Coding path: bash + text editor (+ optional web search), executed on the worker's disk.
    if want_fs:
        ws = config.workspace_dir()
        tools = [_BASH_TOOL, _EDITOR_TOOL] + ([_WEB_SEARCH_TOOL] if want_search else [])
        text = await _agentic(AGENT_SYSTEM_CODING.format(workspace=ws), user, model=model, effort=effort,
                              interval=interval, hb=f"{node.kind}:{node.id}", tools=tools, workspace=ws)
        return NodeResult(id=node.id, kind=node.kind, title=node.title,
                          output=text or "(no output produced)",
                          sources=_dedupe(_URL_RE.findall(text))[:10], confidence=_extract_confidence(text))

    # Text / web-search path (no filesystem).
    tools = [_WEB_SEARCH_TOOL] if want_search else None
    max_tokens = 16000 if node.kind == "synthesize" else 8000
    try:
        text, final = await _stream_text(_NODE_SYSTEM[node.kind], user, model=model, effort=effort,
                                         interval=interval, tools=tools, max_tokens=max_tokens,
                                         hb=f"{node.kind}:{node.id}")
    except ApplicationError as e:
        if tools and getattr(e, "type", None) == "BadRequest":  # web search unavailable -> retry w/o tools
            text, final = await _stream_text(_NODE_SYSTEM[node.kind], user, model=model, effort=effort,
                                             interval=interval, tools=None, max_tokens=max_tokens,
                                             hb=f"{node.kind}:{node.id}")
        else:
            raise
    sources = _dedupe(_extract_search_sources(final) + _URL_RE.findall(text))[:10] if tools else []
    confidence = _extract_confidence(text) if node.kind == "agent" else None
    return NodeResult(id=node.id, kind=node.kind, title=node.title,
                      output=text or "(no output produced)", sources=sources, confidence=confidence)
