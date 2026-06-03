"""Temporal-branded chat client — Claude-CLI look and feel.

On <enter>, the client signals the durable Temporal workflow (the agent loop) and
then renders the dynamic workflow live: planning, the subagent fan-out (each a
child workflow), adversarial review, and the synthesized report.

    uv run python client.py                          # interactive chat
    uv run python client.py --once "your question"   # one-shot (great for demos/CI)

Ctrl-C terminates the current workflow (and, via ParentClosePolicy.TERMINATE, its
subagents) instead of leaving it running on the server.
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

_STEPS = [
    ("planning", "plan"),
    ("researching", "research"),
    ("reviewing", "review"),
    ("synthesizing", "synthesize"),
]
_RANK = {
    "planning": 0,
    "researching": 1,
    "reviewing": 2,
    "refining": 2,
    "synthesizing": 3,
    "done": 4,
    "error": 4,
}
_ICON = {
    "pending": "[dim]○[/]",
    "running": f"[{ACCENT}]◐[/]",
    "done": "[green]✓[/]",
    "failed": "[red]✗[/]",
}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _tracker(phase: str) -> str:
    cur = _RANK.get(phase, 0)
    parts = []
    for i, (p, label) in enumerate(_STEPS):
        if phase == "done" or i < cur:
            parts.append(f"[green]✓ {label}[/]")
        elif i == cur and phase != "done":
            parts.append(f"[bold {ACCENT}]▶ {label}[/]")
        else:
            parts.append(f"[dim]{label}[/]")
    s = "  →  ".join(parts)
    if phase == "refining":
        s += f"    [bold {ACCENT}](refine ↻)[/]"
    return s


def _subagent_table(cur: TurnProgress) -> Table:
    t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    t.add_column("", width=2, no_wrap=True)
    t.add_column("subagent", ratio=3)
    t.add_column("status", ratio=5, overflow="fold")
    t.add_column("execution", ratio=2, style="dim", overflow="fold")
    for s in cur.subagents:
        if s.status == "running":
            info = f"[{ACCENT}]researching… (child workflow running)[/]"
        elif s.status == "done":
            conf = f"conf {s.confidence:.2f} · " if s.confidence is not None else ""
            info = f"[dim]{conf}{escape(s.note or '')}[/]"
        elif s.status == "failed":
            info = f"[red]{escape(s.note or 'failed')}[/]"
        else:
            info = "[dim]queued[/]"
        wf = s.workflow_id.rsplit("-", 3)[-1] if s.workflow_id else ""
        t.add_row(_ICON.get(s.status, "○"), escape(s.title), info, wf)
    return t


def render_panel(cur: TurnProgress, mock: bool) -> Panel:
    body: list = [Text.from_markup(_tracker(cur.phase))]
    if cur.max_iterations > 1:
        body.append(Text.from_markup(f"[dim]iteration {cur.iteration + 1}/{cur.max_iterations}[/]"))
    if cur.strategy:
        body.append(Text.from_markup(f"[italic dim]{escape(cur.strategy)}[/]"))
    if cur.subagents:
        body.append(_subagent_table(cur))
    title = f"[bold {ACCENT}]◆ Dynamic workflow[/] [dim]· turn {cur.index + 1}[/]"
    subtitle = "[yellow]mock[/]" if mock else None
    return Panel(Group(*body), title=title, title_align="left", subtitle=subtitle,
                 border_style=ACCENT, padding=(1, 2))


def render_report(console: Console, report: str) -> None:
    console.print()
    console.print(
        Panel(Markdown(report), title=f"[bold green]✦ verified report[/]", title_align="left",
              border_style="green", padding=(1, 2))
    )


def print_banner(console: Console, mock: bool, models: dict, address: str) -> None:
    title = Text()
    title.append("◆ Temporal", style=f"bold {ACCENT}")
    title.append("  ×  ", style="dim")
    title.append("Claude ✱", style=f"bold {CLAUDE}")
    subtitle = Text("Durable Dynamic Workflows — Claude plans, Temporal executes", style="dim")
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim", justify="right")
    info.add_column()
    mode = "[yellow]MOCK[/]  (set ANTHROPIC_API_KEY for live Claude)" if mock else "[green]LIVE[/]"
    info.add_row("mode", mode)
    info.add_row(
        "models",
        f"planner={models['planner']} · subagent={models['subagent']} · "
        f"reviewer={models['reviewer']} · synth={models['synthesizer']}",
    )
    info.add_row("temporal", f"{address} · ns={config.TEMPORAL_NAMESPACE} · auth={config.temporal_auth_mode()}")
    console.print(Panel(Group(title, subtitle, Text(""), info), border_style=ACCENT, padding=(1, 3)))


def print_help(console: Console) -> None:
    console.print(
        Panel(
            "[bold]/help[/]   show this help\n"
            "[bold]/new[/]    start a fresh chat session (new workflow)\n"
            "[bold]/ui[/]     print the Temporal Web UI link for this session\n"
            "[bold]/exit[/]   end the session gracefully (also /quit, :q)\n"
            "[bold]Ctrl-C[/]  terminate the workflow + its subagents, then quit\n\n"
            "[dim]Anything else is sent to the durable agent as a research goal.[/]",
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
# Live turn streaming
# --------------------------------------------------------------------------
async def stream_turn(console: Console, handle, mock: bool) -> str | None:
    base: int | None = None
    final_report: str | None = None
    with Live(console=console, refresh_per_second=8, transient=False) as live:
        while True:
            try:
                snap = await handle.query(DurableClaudeAgentWorkflow.get_state)
            except Exception:
                await asyncio.sleep(0.3)
                continue
            if base is None:
                base = snap.turns_completed
            if snap.current is not None:
                live.update(render_panel(snap.current, mock))
                if snap.current.report:
                    final_report = snap.current.report
            if snap.turns_completed > base:
                if snap.transcript and snap.transcript[-1].role == "assistant":
                    final_report = snap.transcript[-1].content
                break
            await asyncio.sleep(0.35)
    if final_report:
        render_report(console, final_report)
    return final_report


# --------------------------------------------------------------------------
# Input + Ctrl-C handling
# --------------------------------------------------------------------------
async def _ainput(console: Console, prompt: str) -> str | None:
    """Read a line on a daemon thread so Ctrl-C never blocks process exit.

    Returns the line, or None on EOF (Ctrl-D).
    """
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
    """Run ``coro`` but bail out if ``interrupted`` (Ctrl-C) fires first.

    Returns ``(finished, result)``; ``finished`` is False if we aborted on interrupt.
    """
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
# Entrypoints
# --------------------------------------------------------------------------
async def _start_session(client: Client, args, mock: bool, models: dict, prompt: str | None) -> object:
    prefix = "once" if prompt is not None else "chat"
    session_id = args.session_id or f"{prefix}-{uuid.uuid4().hex[:8]}"
    return await client.start_workflow(
        DurableClaudeAgentWorkflow.run,
        StartRequest(
            session_id=session_id,
            initial_prompt=prompt,
            max_iterations=args.max_iterations,
            max_subagents=args.max_subagents,
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

    # Ctrl-C sets an event (instead of raising) so we can terminate the workflow —
    # which, via ParentClosePolicy.TERMINATE, also terminates its subagents.
    interrupted = asyncio.Event()
    loop = asyncio.get_running_loop()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, interrupted.set)

    handle = None
    try:
        if args.once is not None:
            handle = await _start_session(client, args, mock, models, prompt=args.once)
            _print_session(console, handle.id, ui_base)
            console.print(f"\n[bold {ACCENT}]temporal ❯[/] {escape(args.once)}")
            await _until_interrupt(stream_turn(console, handle, mock), interrupted)
            if not interrupted.is_set():
                with contextlib.suppress(Exception):
                    await handle.signal(DurableClaudeAgentWorkflow.end_session)
            return 130 if interrupted.is_set() else 0

        handle = await _start_session(client, args, mock, models, prompt=None)
        _print_session(console, handle.id, ui_base)
        console.print("[dim]type a research goal and press enter · /help for commands · Ctrl-C terminates[/]")

        while not interrupted.is_set():
            done, line = await _until_interrupt(
                _ainput(console, f"\n[bold {ACCENT}]temporal ❯[/] "), interrupted
            )
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

        # Graceful end for /exit, EOF, or a normal break (not Ctrl-C).
        if not interrupted.is_set():
            with contextlib.suppress(Exception):
                await handle.signal(DurableClaudeAgentWorkflow.end_session)
            console.print("\n[dim]session ended — the workflow completed in Temporal.[/]")
    finally:
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGINT)
        if interrupted.is_set() and handle is not None:
            console.print("\n[dim]Ctrl-C — terminating the workflow and its subagents…[/]")
            with contextlib.suppress(Exception):
                await asyncio.shield(handle.terminate(reason="ctrl-c from cli"))

    return 130 if interrupted.is_set() else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal-branded chat client for Durable Claude Workflows")
    p.add_argument("--once", metavar="PROMPT", help="run a single research goal non-interactively and exit")
    p.add_argument("--address", help=f"Temporal address (default {config.TEMPORAL_ADDRESS})")
    p.add_argument("--session-id", help="explicit workflow/session id (default: random)")
    p.add_argument("--max-iterations", type=int, default=2, help="max plan→review→refine rounds (default 2)")
    p.add_argument("--max-subagents", type=int, default=5, help="max parallel subagents per round (default 5)")
    args = p.parse_args()
    try:
        code = asyncio.run(amain(args))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
