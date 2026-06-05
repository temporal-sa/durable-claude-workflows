"""Temporal-branded chat client - Claude-CLI look and feel.

On <enter> the client signals the durable Temporal workflow (the agent loop). The
workflow asks Claude to plan a workflow DAG, shows it to you for approval (y/N),
then runs it - rendering each node (its own child workflow) live as the DAG executes.

    uv run python client.py                          # interactive chat (approves each plan)
    uv run python client.py --once "your task"       # one-shot, auto-approves (great for demos/CI)

Ctrl-C terminates the current workflow (and, via ParentClosePolicy.TERMINATE, its nodes).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import threading
import uuid

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from temporalio.client import Client

import config
from models import StartRequest, TurnProgress
from workflows import DurableClaudeAgentWorkflow

ACCENT = "#7C5CFF"  # Temporal-ish purple
CLAUDE = "#D97757"  # Claude/Anthropic-ish terracotta

_KIND_TAG = {
    "agent": f"[{ACCENT}]agent[/]",
    "review": "[magenta]review[/]",
    "synthesize": "[green]synth[/]",
}
_STATUS_ICON = {
    "pending": "[dim]o[/]",
    "running": f"[{ACCENT}]*[/]",
    "done": "[green]+[/]",
    "failed": "[red]x[/]",
}
_PHASE_STEPS = [("planning", "plan"), ("running", "execute"), ("done", "done")]
_PHASE_RANK = {"planning": 0, "awaiting_approval": 0, "running": 1, "done": 2, "error": 2, "cancelled": 2}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _trunc(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "..."


def _waves(nodes) -> list[list]:
    """Group nodes into topological levels (= execution phases) by dependency depth."""
    by_id = {n.id: n for n in nodes}
    level: dict[str, int] = {}
    for _ in range(len(nodes) + 1):  # relax until stable
        for n in nodes:
            deps = [d for d in n.depends_on if d in by_id]
            level[n.id] = 0 if not deps else 1 + max(level.get(d, 0) for d in deps)
    waves: dict[int, list] = {}
    for n in nodes:
        waves.setdefault(level[n.id], []).append(n)
    return [waves[k] for k in sorted(waves)]


def _phase_line(phase: str) -> str:
    cur = _PHASE_RANK.get(phase, 0)
    parts = []
    for i, (_p, label) in enumerate(_PHASE_STEPS):
        if phase in ("done", "error", "cancelled") or i < cur:
            parts.append(f"[green]+ {label}[/]")
        elif i == cur:
            parts.append(f"[bold {ACCENT}]> {label}[/]")
        else:
            parts.append(f"[dim]{label}[/]")
    return "  ->  ".join(parts)


def render_plan(console: Console, cur: TurnProgress) -> None:
    body: list = [Text.from_markup(f"[bold]{escape(cur.plan_title or 'Workflow')}[/]")]
    if cur.plan_summary:
        body.append(Text.from_markup(f"[dim]{escape(cur.plan_summary)}[/]"))
    body.append(Text(""))
    for i, wave in enumerate(_waves(cur.nodes), 1):
        tag = " [dim](parallel)[/]" if len(wave) > 1 else ""
        body.append(Text.from_markup(f"[dim]phase {i}[/]{tag}"))
        for n in wave:
            deps = f"  [dim]<- {', '.join(n.depends_on)}[/]" if n.depends_on else ""
            files = " [green](files)[/]" if getattr(n, "use_filesystem", False) else ""
            body.append(Text.from_markup(f"  {_KIND_TAG.get(n.kind, n.kind)}  {escape(n.title)}{files}{deps}"))
            if n.instruction:
                body.append(Text.from_markup(f"       [dim]{escape(_trunc(n.instruction, 100))}[/]"))
    console.print(
        Panel(Group(*body), title=f"[bold {ACCENT}]proposed workflow[/]", title_align="left",
              border_style=ACCENT, padding=(1, 2))
    )


def render_dag_panel(cur: TurnProgress, mock: bool) -> Panel:
    body: list = [Text.from_markup(_phase_line(cur.phase))]
    if cur.plan_summary:
        body.append(Text.from_markup(f"[italic dim]{escape(cur.plan_summary)}[/]"))
    t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    t.add_column("", width=2, no_wrap=True)
    t.add_column("step", ratio=3)
    t.add_column("status", ratio=5, overflow="fold")
    t.add_column("execution", ratio=2, style="dim", overflow="fold")
    for n in cur.nodes:
        if n.status == "running":
            info = f"[{ACCENT}]running... (child workflow)[/]"
        elif n.status == "done":
            conf = f"conf {n.confidence:.2f} - " if n.confidence is not None else ""
            info = f"[dim]{conf}{escape(n.note or '')}[/]"
        elif n.status == "failed":
            info = f"[red]{escape(n.note or 'failed')}[/]"
        else:
            info = "[dim]queued[/]"
        wf = n.workflow_id.rsplit("-", 1)[-1] if n.workflow_id else ""
        t.add_row(_STATUS_ICON.get(n.status, "o"), f"{_KIND_TAG.get(n.kind, n.kind)} {escape(n.title)}", info, wf)
    body.append(t)
    title = f"[bold {ACCENT}]dynamic workflow[/] [dim]- turn {cur.index + 1}[/]"
    return Panel(Group(*body), title=title, title_align="left",
                 subtitle="[yellow]mock[/]" if mock else None, border_style=ACCENT, padding=(1, 2))


def render_report(console: Console, report: str) -> None:
    console.print()
    console.print(
        Panel(Markdown(report), title="[bold green]verified report[/]", title_align="left",
              border_style="green", padding=(1, 2))
    )


def print_banner(console: Console, mock: bool, models: dict, address: str) -> None:
    title = Text()
    title.append("Temporal", style=f"bold {ACCENT}")
    title.append("  x  ", style="dim")
    title.append("Claude", style=f"bold {CLAUDE}")
    subtitle = Text("Durable Dynamic Workflows - Claude plans the DAG, Temporal executes it", style="dim")
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right")
    info.add_column()
    mode = "[yellow]MOCK[/]  (set ANTHROPIC_API_KEY for live Claude)" if mock else "[green]LIVE[/]"
    info.add_row("mode", mode)
    info.add_row("models", f"planner={models['planner']} - agent={models['agent']} - "
                           f"reviewer={models['reviewer']} - synth={models['synthesizer']}")
    info.add_row("temporal", f"{address} - ns={config.TEMPORAL_NAMESPACE} - auth={config.temporal_auth_mode()}")
    console.print(Panel(Group(title, subtitle, Text(""), info), border_style=ACCENT, padding=(1, 3)))


def print_help(console: Console) -> None:
    console.print(
        Panel(
            "[bold]/help[/]   show this help\n"
            "[bold]/new[/]    start a fresh chat session (new workflow)\n"
            "[bold]/ui[/]     print the Temporal Web UI link for this session\n"
            "[bold]/exit[/]   end the session gracefully (also /quit, :q)\n"
            "[bold]Ctrl-C[/]  terminate the workflow + its nodes, then quit\n\n"
            "[dim]Anything else is a task. Claude plans a workflow; you approve it (y/N); Temporal runs it.[/]",
            border_style="dim", title="commands", title_align="left", padding=(1, 2),
        )
    )


def _ui_base(address: str) -> str | None:
    host = address.split(":")[0]
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        return f"http://localhost:8233/namespaces/{config.TEMPORAL_NAMESPACE}/workflows"
    return None


def _print_session(console: Console, wf_id: str, ui_base: str | None) -> None:
    line = f"[dim]session:[/] {wf_id}"
    if ui_base:
        line += f"    [dim]watch:[/] {ui_base}/{wf_id}"
    console.print(line)


# --------------------------------------------------------------------------
# Input + Ctrl-C handling
# --------------------------------------------------------------------------
async def _ainput(console: Console, prompt: str) -> str | None:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def _work() -> None:
        try:
            line = console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(None))
            return
        except BaseException as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_exception(exc))
            return
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(line))

    threading.Thread(target=_work, daemon=True).start()
    return await fut


async def _until_interrupt(coro, interrupted: asyncio.Event):
    """Run ``coro`` but bail out if ``interrupted`` (Ctrl-C) fires first."""
    task = asyncio.ensure_future(coro)
    stopper = asyncio.ensure_future(interrupted.wait())
    await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
    if interrupted.is_set():
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        return False, None
    stopper.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stopper
    return True, task.result()


# --------------------------------------------------------------------------
# Turn streaming: plan -> approve -> run -> report
# --------------------------------------------------------------------------
async def _query(handle):
    while True:
        try:
            return await handle.query(DurableClaudeAgentWorkflow.get_state)
        except Exception:  # noqa: BLE001 - transient right after start
            await asyncio.sleep(0.3)


async def stream_turn(console: Console, handle, mock: bool) -> str | None:
    base: int | None = None
    plan_rendered = False
    approved: bool | None = None

    # Phase A: wait for the plan, render it, and (if interactive) approve it.
    while True:
        snap = await _query(handle)
        if base is None:
            base = snap.turns_completed
        cur = snap.current
        if cur is not None and cur.plan_title is not None and not plan_rendered:
            render_plan(console, cur)
            plan_rendered = True
        if snap.turns_completed > base:  # finished already (very fast / auto)
            if snap.transcript and snap.transcript[-1].role == "assistant":
                render_report(console, snap.transcript[-1].content)
                return snap.transcript[-1].content
            return None
        if cur is not None and cur.phase == "awaiting_approval":
            ans = await _ainput(console, "[bold]Run this workflow?[/] [y/N] ")
            approved = (ans or "").strip().lower() in ("y", "yes")
            await handle.signal(DurableClaudeAgentWorkflow.approve_plan, approved)
            break
        if cur is not None and cur.phase in ("running", "done"):  # auto-approved
            approved = True
            break
        await asyncio.sleep(0.3)

    if approved is False:
        while True:  # let the (cancelled) turn close
            snap = await _query(handle)
            if snap.turns_completed > base:
                break
            await asyncio.sleep(0.2)
        console.print("[dim]declined - nothing ran.[/]")
        return None

    # Phase B: live DAG execution.
    final_report: str | None = None
    with Live(console=console, refresh_per_second=8, transient=False) as live:
        while True:
            snap = await _query(handle)
            cur = snap.current
            if cur is not None:
                live.update(render_dag_panel(cur, mock))
            if snap.turns_completed > base:
                if snap.transcript and snap.transcript[-1].role == "assistant":
                    final_report = snap.transcript[-1].content
                break
            await asyncio.sleep(0.35)
    if final_report:
        render_report(console, final_report)
    return final_report


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------
async def _start_session(client: Client, args, mock: bool, models: dict, prompt: str | None) -> object:
    prefix = "once" if prompt is not None else "chat"
    session_id = args.session_id or f"{prefix}-{uuid.uuid4().hex[:8]}"
    auto_approve = prompt is not None or bool(getattr(args, "yes", False))
    return await client.start_workflow(
        DurableClaudeAgentWorkflow.run,
        StartRequest(
            session_id=session_id,
            initial_prompt=prompt,
            auto_approve=auto_approve,
            max_nodes=args.max_nodes,
            mock=mock,
            models=models,
        ),
        id=session_id,
        task_queue=config.TASK_QUEUE,
    )


async def amain(args) -> int:
    console = Console()
    address = args.address or config.TEMPORAL_ADDRESS
    try:
        client = await config.connect_temporal_client(address)
    except Exception as e:  # noqa: BLE001
        console.print(
            f"[red]Could not connect to Temporal at {address}.[/]\n"
            "Start a local server:  [bold]temporal server start-dev[/]  (or set TEMPORAL_API_KEY for Temporal Cloud)\n"
            f"[dim]{e}[/]"
        )
        return 1

    mock = config.mock_mode()
    models = config.model_summary()
    ui_base = _ui_base(address)
    print_banner(console, mock, models, address)

    interrupted = asyncio.Event()
    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, interrupted.set)

    handle = None
    try:
        if args.once is not None:
            handle = await _start_session(client, args, mock, models, prompt=args.once)
            _print_session(console, handle.id, ui_base)
            console.print(f"\n[bold {ACCENT}]temporal >[/] {escape(args.once)}")
            await _until_interrupt(stream_turn(console, handle, mock), interrupted)
            if not interrupted.is_set():
                with contextlib.suppress(Exception):
                    await handle.signal(DurableClaudeAgentWorkflow.end_session)
            return 130 if interrupted.is_set() else 0

        handle = await _start_session(client, args, mock, models, prompt=None)
        _print_session(console, handle.id, ui_base)
        console.print("[dim]type a task and press enter - /help for commands - Ctrl-C terminates[/]")

        while not interrupted.is_set():
            done, line = await _until_interrupt(_ainput(console, f"\n[bold {ACCENT}]temporal >[/] "), interrupted)
            if not done or line is None:  # Ctrl-C, or EOF (Ctrl-D)
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/exit", "/quit", ":q"):
                break
            if line == "/help":
                print_help(console)
                continue
            if line == "/ui":
                console.print(f"{ui_base}/{handle.id}" if ui_base else "[dim](no local Web UI)[/]")
                continue
            if line == "/new":
                handle = await _start_session(client, args, mock, models, prompt=None)
                console.print()
                _print_session(console, handle.id, ui_base)
                continue
            await handle.signal(DurableClaudeAgentWorkflow.submit_prompt, line)
            done, _ = await _until_interrupt(stream_turn(console, handle, mock), interrupted)
            if not done:  # Ctrl-C mid-turn
                break

        if not interrupted.is_set():
            with contextlib.suppress(Exception):
                await handle.signal(DurableClaudeAgentWorkflow.end_session)
            console.print("\n[dim]session ended - the workflow completed in Temporal.[/]")
    finally:
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGINT)
        if interrupted.is_set() and handle is not None:
            console.print("\n[dim]Ctrl-C - terminating the workflow and its nodes...[/]")
            with contextlib.suppress(Exception):
                await asyncio.shield(handle.terminate(reason="ctrl-c from cli"))

    return 130 if interrupted.is_set() else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal-branded chat client for Durable Claude Workflows")
    p.add_argument("--once", metavar="TASK", help="run a single task non-interactively (auto-approves the plan) and exit")
    p.add_argument("--address", help=f"Temporal address (default {config.TEMPORAL_ADDRESS})")
    p.add_argument("--session-id", help="explicit workflow/session id (default: random)")
    p.add_argument("--max-nodes", type=int, default=10, help="max nodes Claude may put in the plan (default 10)")
    p.add_argument("--yes", action="store_true", help="auto-approve plans in interactive mode too")
    args = p.parse_args()
    try:
        code = asyncio.run(amain(args))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
