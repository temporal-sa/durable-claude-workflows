"""The durable orchestrator: Claude plans a DAG, Temporal executes it.

``DurableClaudeAgentWorkflow`` is a long-lived, signal-driven chat session. Each
user prompt runs a turn:

    1. PLAN     - an activity asks Claude for a workflow DAG (typed nodes + deps).
    2. APPROVE  - the plan is surfaced to the user, who approves it (y/n), mirroring
                  Claude Code's "approve the plan before it runs". One-shot mode auto-approves.
    3. EXECUTE  - a deterministic DAG interpreter runs the graph: every node becomes
                  its own ``NodeWorkflow`` child execution (which calls Claude), with
                  independent nodes fanned out in parallel and dependencies respected.

The plan and every node result are recorded in Event History, so the whole
orchestration is replay-safe: a worker crash, deploy, or reboot resumes exactly
where it left off, without re-running completed nodes.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import plan_workflow, run_node
    from models import (
        AgentSnapshot,
        NodeProgress,
        NodeResult,
        NodeRunInput,
        PlanInput,
        StartRequest,
        Turn,
        TurnProgress,
        WorkflowPlan,
    )

# Unlimited retries (maximum_attempts=0). Transient errors retry forever with capped
# backoff; non-retryable errors (auth, bad request) still fail fast.
_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=0,
)

# Per-kind activity timeouts: (start_to_close, heartbeat_timeout). Heartbeat cadence
# is in config.py and stays under these.
_PLAN_STC, _PLAN_HB = timedelta(minutes=15), timedelta(seconds=60)
_NODE_TIMEOUTS = {
    "agent": (timedelta(minutes=60), timedelta(seconds=15)),
    "review": (timedelta(minutes=15), timedelta(seconds=60)),
    "synthesize": (timedelta(minutes=15), timedelta(seconds=60)),
}


def _first_line(text: str, n: int = 140) -> str:
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#-*> ").strip().replace("**", "").replace("`", "")
        if line:
            return line if len(line) <= n else line[: n - 1] + "..."
    return ""


def _safe(node_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in node_id)[:40] or "n"


@workflow.defn
class NodeWorkflow:
    """One node of the DAG, as its own durable execution.

    Modeling every node as a child workflow means each shows up independently in
    the Temporal UI, retries on its own, and is schedulable on any worker.
    """

    @workflow.run
    async def run(self, inp: NodeRunInput) -> NodeResult:
        stc, hb = _NODE_TIMEOUTS.get(inp.node.kind, (timedelta(minutes=15), timedelta(seconds=60)))
        return await workflow.execute_activity(
            run_node, inp, start_to_close_timeout=stc, heartbeat_timeout=hb, retry_policy=_RETRY
        )


@workflow.defn
class DurableClaudeAgentWorkflow:
    """A durable, multi-turn chat session that plans and runs Claude dynamic workflows."""

    @workflow.init
    def __init__(self, req: StartRequest) -> None:
        self._req = req
        self._session_id = req.session_id
        self._pending: list[str] = []
        self._transcript: list[Turn] = list(req.carry_transcript)
        self._turns_completed = req.turns_completed
        self._busy = False
        self._ended = False
        self._approval: bool | None = None
        self._current: TurnProgress | None = None
        if req.initial_prompt:
            self._pending.append(req.initial_prompt)

    # --- Inbound signals (chat client -> workflow) -------------------------
    @workflow.signal
    def submit_prompt(self, text: str) -> None:
        if text and text.strip():
            self._pending.append(text.strip())

    @workflow.signal
    def approve_plan(self, approved: bool) -> None:
        self._approval = bool(approved)

    @workflow.signal
    def end_session(self) -> None:
        self._ended = True

    # --- Query (workflow -> chat client UI) --------------------------------
    @workflow.query
    def get_state(self) -> AgentSnapshot:
        return AgentSnapshot(
            session_id=self._session_id,
            task_queue=workflow.info().task_queue,
            busy=self._busy,
            turns_completed=self._turns_completed,
            transcript=list(self._transcript),
            current=self._current,
            mock=self._req.mock,
            models=self._req.models,
        )

    # --- The agent loop ----------------------------------------------------
    @workflow.run
    async def run(self, req: StartRequest) -> str:
        while True:
            await workflow.wait_condition(lambda: bool(self._pending) or self._ended)
            if self._ended and not self._pending:
                return f"session {self._session_id} ended after {self._turns_completed} turns"

            prompt = self._pending.pop(0)
            self._busy = True
            self._approval = None
            self._transcript.append(Turn(role="user", content=prompt))
            try:
                report = await self._run_turn(prompt)
                self._transcript.append(Turn(role="assistant", content=report))
            except Exception as e:  # noqa: BLE001 - surface failure into the transcript
                workflow.logger.exception("turn failed")
                if self._current is not None:
                    self._current.phase = "error"
                    self._current.error = str(e)
                self._transcript.append(Turn(role="assistant", content=f"The durable workflow hit an error: {e}"))
            finally:
                self._turns_completed += 1
                self._busy = False
                self._current = None

            if (not self._pending and not self._ended
                    and workflow.info().is_continue_as_new_suggested()):
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(args=[StartRequest(
                    session_id=self._session_id,
                    auto_approve=self._req.auto_approve,
                    max_nodes=self._req.max_nodes,
                    mock=self._req.mock,
                    models=self._req.models,
                    carry_transcript=self._transcript[-8:],
                    turns_completed=self._turns_completed,
                )])

    # --- One turn = plan -> approve -> execute the DAG ---------------------
    async def _run_turn(self, prompt: str) -> str:
        history = self._transcript[:-1]
        cur = TurnProgress(index=self._turns_completed, user_prompt=prompt, phase="planning")
        self._current = cur

        # 1) PLAN - Claude authors the DAG (recorded in history).
        plan: WorkflowPlan = await workflow.execute_activity(
            plan_workflow,
            PlanInput(goal=prompt, history=history, max_nodes=self._req.max_nodes),
            start_to_close_timeout=_PLAN_STC,
            heartbeat_timeout=_PLAN_HB,
            retry_policy=_RETRY,
        )
        cur.plan_title = plan.title
        cur.plan_summary = plan.summary
        cur.nodes = [
            NodeProgress(id=n.id, kind=n.kind, title=n.title, instruction=n.instruction, depends_on=n.depends_on)
            for n in plan.nodes
        ]

        # 2) APPROVE - surface the plan and wait for y/n (one-shot auto-approves).
        if not self._req.auto_approve:
            cur.phase = "awaiting_approval"
            await workflow.wait_condition(lambda: self._approval is not None or self._ended)
            if self._approval is None:  # session ended before approving
                cur.phase = "cancelled"
                return "(session ended before the workflow was approved)"
            if self._approval is False:
                cur.phase = "cancelled"
                return "Workflow declined - nothing ran."

        # 3) EXECUTE - durable DAG interpreter.
        cur.phase = "running"
        results = await self._run_dag(prompt, plan)
        out = results.get(plan.output) or (results.get(plan.nodes[-1].id) if plan.nodes else None)
        report = out.output if out else "(workflow produced no output)"
        cur.phase = "done"
        cur.report = report
        return report

    async def _run_dag(self, goal: str, plan: WorkflowPlan) -> dict[str, NodeResult]:
        """Run the DAG in topological waves; nodes whose deps are met run in parallel."""
        cur = self._current
        results: dict[str, NodeResult] = {}
        remaining = list(plan.nodes)

        while remaining:
            ready = sorted(
                [n for n in remaining if all(d in results for d in n.depends_on)],
                key=lambda n: n.id,
            )
            if not ready:  # validated to be a DAG, so this shouldn't happen
                raise RuntimeError("plan is not a DAG (cycle or missing dependency)")

            async def _run_one(node):
                self._set_node(node.id, status="running")
                upstream = [results[d] for d in node.depends_on if d in results]
                child_id = f"{workflow.info().workflow_id}-t{cur.index}-{_safe(node.id)}"
                try:
                    handle = await workflow.start_child_workflow(
                        NodeWorkflow.run,
                        NodeRunInput(goal=goal, node=node, upstream=upstream),
                        id=child_id,
                        parent_close_policy=workflow.ParentClosePolicy.TERMINATE,
                    )
                    self._set_node(node.id, status="running", workflow_id=handle.id)
                    res: NodeResult = await handle
                    self._set_node(
                        node.id,
                        status="failed" if res.error else "done",
                        confidence=res.confidence,
                        note=res.error or _first_line(res.output),
                    )
                    return node.id, res
                except Exception as e:  # noqa: BLE001 - one node failing must not kill the turn
                    workflow.logger.warning(f"node {node.id} failed: {e}")
                    self._set_node(node.id, status="failed", note=str(e))
                    return node.id, NodeResult(id=node.id, kind=node.kind, title=node.title, output="", error=str(e))

            wave = await asyncio.gather(*[_run_one(n) for n in ready])
            for nid, res in wave:
                results[nid] = res
            remaining = [n for n in remaining if n.id not in results]

        return results

    def _set_node(self, node_id: str, **fields) -> None:
        if self._current is None:
            return
        for n in self._current.nodes:
            if n.id == node_id:
                for k, v in fields.items():
                    setattr(n, k, v)
                return
