"""Temporal worker - hosts the workflows and activities.

Run several of these (same task queue) to scale the node fan-out across machines.
All activities are async, so no ThreadPoolExecutor is needed.

    uv run python worker.py
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

import config
from activities import plan_workflow, run_node
from workflows import DurableClaudeAgentWorkflow, NodeWorkflow


def _banner() -> None:
    mode = "MOCK (simulated Claude - set ANTHROPIC_API_KEY for live)" if config.mock_mode() else "LIVE"
    m = config.model_summary()
    print("+- Durable Claude Workflows - worker ----------------------------")
    print(f"|  Temporal : {config.TEMPORAL_ADDRESS}  ns={config.TEMPORAL_NAMESPACE}  auth={config.temporal_auth_mode()}")
    print(f"|  Queue    : {config.TASK_QUEUE}")
    print(f"|  Mode     : {mode}")
    print(f"|  Models   : planner={m['planner']} agent={m['agent']} reviewer={m['reviewer']} synth={m['synthesizer']}")
    if not config.mock_mode():
        print(f"|  WebSearch: {'on' if config.ENABLE_WEB_SEARCH else 'off'}")
    print("+- waiting for work (Ctrl-C to stop) ----------------------------", flush=True)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = await config.connect_temporal_client()
    _banner()
    # Let Pydantic's compiled core load from the host instead of being re-imported
    # inside the workflow sandbox (avoids "imported after initial workflow load" warnings).
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules("pydantic", "pydantic_core")
    )
    # Worker.run() performs a one-time namespace-validation RPC at startup; a freshly
    # started dev server can answer it with a transient error. Retry briefly.
    for attempt in range(1, 11):
        worker = Worker(
            client,
            task_queue=config.TASK_QUEUE,
            workflows=[DurableClaudeAgentWorkflow, NodeWorkflow],
            activities=[plan_workflow, run_node],
            workflow_runner=runner,
            max_concurrent_activities=64,
        )
        try:
            await worker.run()
            return
        except RuntimeError as e:
            if "validation failed" in str(e).lower() and attempt < 10:
                logging.warning("worker startup transient (attempt %d/10), retrying in 2s...", attempt)
                await asyncio.sleep(2)
                continue
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nworker stopped")
