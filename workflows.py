"""The durable orchestrator: Claude plans, Temporal executes.

``DurableClaudeAgentWorkflow`` is a long-lived, signal-driven chat session — the
"agent loop". Each user prompt runs a turn:

    1. PLAN        — an activity asks Claude for a structured plan (the dynamic
                     workflow Claude authors).
    2. FAN OUT     — Temporal runs each plan task as its own child workflow,
                     concurrently, across the worker fleet. Add workers => more
                     parallelism. There is no fixed concurrency cap.
    3. REVIEW      — an adversarial-review activity cross-checks the findings.
    4. REFINE      — if the review found gaps, the plan changes and we iterate
                     (genuinely dynamic control flow), bounded by max_iterations.
    5. SYNTHESIZE  — an activity asks Claude for the final verified report.

Because the plan and every result are recorded in Event History, the whole
orchestration is replay-safe: a worker crash, deploy, or machine reboot resumes
exactly where it left off — the thing an in-session dynamic workflow cannot do.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        adversarial_review,
        plan_workflow,
        run_subagent,
        synthesize_report,
    )
    from models import (
        AgentSnapshot,
        PlanInput,
        ResearchInput,
        ReviewInput,
        ReviewResult,
        StartRequest,
        SubAgentProgress,
        SubAgentResult,
        SynthesisInput,
        Turn,
        TurnProgress,
    )

# Unlimited retries (maximum_attempts=0). Transient errors (rate limit, 5xx,
# connection, timeouts) retry forever with capped backoff; non-retryable errors
# (auth, bad request) still fail fast because claude_llm marks them non_retryable.
_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=0,
)

# Per-activity timeouts: (start_to_close, heartbeat_timeout). Heartbeat *cadence*
# lives in config.py and is kept comfortably under these heartbeat timeouts.
_PLAN_STC, _PLAN_HB = timedelta(minutes=15), timedelta(seconds=60)
_SUBAGENT_STC, _SUBAGENT_HB = timedelta(minutes=60), timedelta(seconds=15)
_REVIEW_STC, _REVIEW_HB = timedelta(minutes=15), timedelta(seconds=60)
_SYNTH_STC, _SYNTH_HB = timedelta(minutes=15), timedelta(seconds=60)


def _first_line(text: str, n: int = 140) -> str:
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#-*> ").strip().replace("**", "").replace("`", "")
        if line:
            return line if len(line) <= n else line[: n - 1] + "…"
    return ""


@workflow.defn
class ResearchSubagentWorkflow:
    """A single research subagent, as its own durable execution.

    Modeling each subagent as a child workflow (rather than a bare activity)
    means every subagent shows up independently in the Temporal UI, retries on
    its own, and is schedulable on any worker in the fleet.
    """

    @workflow.run
    async def run(self, inp: ResearchInput) -> SubAgentResult:
        return await workflow.execute_activity(
            run_subagent,
            inp,
            start_to_close_timeout=_SUBAGENT_STC,
            heartbeat_timeout=_SUBAGENT_HB,
            retry_policy=_RETRY,
        )


@workflow.defn
class DurableClaudeAgentWorkflow:
    """A durable, multi-turn chat session that orchestrates Claude dynamic workflows."""

    @workflow.init
    def __init__(self, req: StartRequest) -> None:
        self._req = req
        self._session_id = req.session_id
        self._pending: list[str] = []
        self._transcript: list[Turn] = list(req.carry_transcript)
        self._turns_completed = req.turns_completed
        self._busy = False
        self._ended = False
        self._current: TurnProgress | None = None
        if req.initial_prompt:
            self._pending.append(req.initial_prompt)

    # --- Inbound signals (chat client → workflow) --------------------------
    @workflow.signal
    def submit_prompt(self, text: str) -> None:
        if text and text.strip():
            self._pending.append(text.strip())

    @workflow.signal
    def end_session(self) -> None:
        self._ended = True

    # --- Query (workflow → chat client UI) ---------------------------------
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
            self._transcript.append(Turn(role="user", content=prompt))
            try:
                report = await self._run_turn(prompt)
                self._transcript.append(Turn(role="assistant", content=report))
            except Exception as e:  # noqa: BLE001 — surface failure into the transcript
                workflow.logger.exception("turn failed")
                if self._current is not None:
                    self._current.phase = "error"
                    self._current.error = str(e)
                self._transcript.append(
                    Turn(role="assistant", content=f"⚠️ The durable workflow hit an error: {e}")
                )
            finally:
                self._turns_completed += 1
                self._busy = False
                self._current = None

            # Bound Event History on very long chats: hand off to a fresh run.
            if (
                not self._pending
                and not self._ended
                and workflow.info().is_continue_as_new_suggested()
            ):
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(
                    args=[
                        StartRequest(
                            session_id=self._session_id,
                            max_iterations=self._req.max_iterations,
                            max_subagents=self._req.max_subagents,
                            mock=self._req.mock,
                            models=self._req.models,
                            carry_transcript=self._transcript[-8:],
                            turns_completed=self._turns_completed,
                        )
                    ]
                )

    # --- One turn = one Claude dynamic workflow ----------------------------
    async def _run_turn(self, prompt: str) -> str:
        history = self._transcript[:-1]  # everything before the just-added user prompt
        cur = TurnProgress(
            index=self._turns_completed,
            user_prompt=prompt,
            phase="planning",
            max_iterations=self._req.max_iterations,
        )
        self._current = cur

        # 1) PLAN — Claude authors the dynamic workflow (recorded in history).
        plan = await workflow.execute_activity(
            plan_workflow,
            PlanInput(goal=prompt, history=history, max_subagents=self._req.max_subagents),
            start_to_close_timeout=_PLAN_STC,
            heartbeat_timeout=_PLAN_HB,
            retry_policy=_RETRY,
        )
        cur.strategy = plan.strategy
        cur.subagents = [SubAgentProgress(id=t.id, title=t.title) for t in plan.tasks]

        all_findings: list[SubAgentResult] = []
        review: ReviewResult | None = None
        tasks = plan.tasks
        next_id = max((t.id for t in tasks), default=0) + 1

        for iteration in range(self._req.max_iterations):
            cur.iteration = iteration

            # 2) FAN OUT — each subagent is its own durable child workflow.
            cur.phase = "researching"
            all_findings.extend(await self._fan_out(prompt, tasks, iteration))

            # 3) ADVERSARIAL REVIEW.
            cur.phase = "reviewing"
            review = await workflow.execute_activity(
                adversarial_review,
                ReviewInput(
                    goal=prompt,
                    strategy=plan.strategy,
                    findings=all_findings,
                    iteration=iteration,
                    max_iterations=self._req.max_iterations,
                ),
                start_to_close_timeout=_REVIEW_STC,
                heartbeat_timeout=_REVIEW_HB,
                retry_policy=_RETRY,
            )
            if review.satisfied or not review.followup_tasks or iteration == self._req.max_iterations - 1:
                break

            # 4) REFINE — the plan changes based on intermediate results.
            cur.phase = "refining"
            tasks = []
            for t in review.followup_tasks:
                t.id = next_id
                next_id += 1
                tasks.append(t)
            cur.subagents = cur.subagents + [SubAgentProgress(id=t.id, title=t.title) for t in tasks]

        # 5) SYNTHESIZE — the final verified report.
        cur.phase = "synthesizing"
        report = await workflow.execute_activity(
            synthesize_report,
            SynthesisInput(goal=prompt, history=history, findings=all_findings, review=review),
            start_to_close_timeout=_SYNTH_STC,
            heartbeat_timeout=_SYNTH_HB,
            retry_policy=_RETRY,
        )
        cur.phase = "done"
        cur.report = report
        return report

    async def _fan_out(self, goal: str, tasks, iteration: int) -> list[SubAgentResult]:
        # Start every subagent as a child workflow, then await them concurrently.
        started = []
        for t in tasks:
            child_id = f"{workflow.info().workflow_id}-t{self._current.index}-i{iteration}-s{t.id}"
            handle = await workflow.start_child_workflow(
                ResearchSubagentWorkflow.run,
                ResearchInput(goal=goal, task=t),
                id=child_id,
                # Default policy, set explicitly: if the parent closes (including a
                # terminate from the CLI), terminate the children too — no orphans.
                parent_close_policy=workflow.ParentClosePolicy.TERMINATE,
            )
            self._set_sub(t.id, status="running", workflow_id=handle.id)
            started.append((t, handle))

        async def _await_one(task, handle) -> SubAgentResult:
            try:
                res: SubAgentResult = await handle
                self._set_sub(
                    res.id,
                    status="failed" if res.error else "done",
                    confidence=res.confidence,
                    note=res.error or _first_line(res.findings),
                )
                return res
            except Exception as e:  # noqa: BLE001 — one subagent failing must not kill the turn
                workflow.logger.warning(f"subagent {task.id} failed: {e}")
                self._set_sub(task.id, status="failed", note=str(e))
                return SubAgentResult(id=task.id, title=task.title, findings="", error=str(e))

        return list(await asyncio.gather(*[_await_one(t, h) for t, h in started]))

    def _set_sub(self, sub_id: int, **fields) -> None:
        if self._current is None:
            return
        for s in self._current.subagents:
            if s.id == sub_id:
                for k, v in fields.items():
                    setattr(s, k, v)
                return
