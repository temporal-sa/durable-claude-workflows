"""Central configuration — loaded from the environment (and a local ``.env``).

Imported by the worker, the client, and the activities. Workflow code must stay
deterministic, so workflows do NOT read this module for anything that can change
between runs — they receive their knobs as workflow arguments instead.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

# Load .env from the current working directory (no-op if absent). Safe to call
# from any entrypoint; values already set in the real environment win.
load_dotenv()

# --- Temporal connection ---------------------------------------------------
TASK_QUEUE: str = os.getenv("DURABLE_CLAUDE_TASK_QUEUE", "durable-claude")
TEMPORAL_ADDRESS: str = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE: str = os.getenv("TEMPORAL_NAMESPACE", "default")
# Temporal Cloud: set TEMPORAL_API_KEY (TLS is implied). Self-hosted behind TLS: TEMPORAL_TLS=true.
TEMPORAL_API_KEY: str | None = os.getenv("TEMPORAL_API_KEY") or None
TEMPORAL_TLS: bool = os.getenv("TEMPORAL_TLS", "").lower() in ("1", "true", "yes")


def temporal_auth_mode() -> str:
    return "api-key" if TEMPORAL_API_KEY else ("tls" if TEMPORAL_TLS else "none")


async def connect_temporal_client(address: str | None = None) -> Client:
    """Connect to Temporal, with the Pydantic data converter on every path.

    Three modes (same shape as the temporal-workflow-throttler sample):
      1. TEMPORAL_API_KEY set  -> Temporal Cloud API-key auth (TLS implied).
      2. TEMPORAL_TLS=true     -> system-trust TLS (self-hosted behind TLS, no API key).
      3. neither               -> plain TCP, for `temporal server start-dev`.
    """
    target = address or TEMPORAL_ADDRESS
    common = dict(namespace=TEMPORAL_NAMESPACE, data_converter=pydantic_data_converter)
    if TEMPORAL_API_KEY:
        return await Client.connect(target, api_key=TEMPORAL_API_KEY, tls=True, **common)
    if TEMPORAL_TLS:
        return await Client.connect(target, tls=True, **common)
    return await Client.connect(target, **common)

# --- Models (per role) -----------------------------------------------------
# Default everything to the most capable model. Override per role via env if you
# want to trade cost/latency on the high-volume subagent fan-out, e.g.
#   SUBAGENT_MODEL=claude-haiku-4-5
PLANNER_MODEL: str = os.getenv("PLANNER_MODEL", "claude-opus-4-8")
SUBAGENT_MODEL: str = os.getenv("SUBAGENT_MODEL", "claude-opus-4-8")
REVIEW_MODEL: str = os.getenv("REVIEW_MODEL", "claude-opus-4-8")
SYNTH_MODEL: str = os.getenv("SYNTH_MODEL", "claude-opus-4-8")

# Effort knob (low | medium | high | xhigh | max) for adaptive thinking.
PLANNER_EFFORT: str = os.getenv("PLANNER_EFFORT", "high")
SUBAGENT_EFFORT: str = os.getenv("SUBAGENT_EFFORT", "medium")
REVIEW_EFFORT: str = os.getenv("REVIEW_EFFORT", "high")
SYNTH_EFFORT: str = os.getenv("SYNTH_EFFORT", "high")

# Heartbeat cadence (seconds) per role — how often an activity pings Temporal
# while a Claude call is in flight. Each is kept well under that role's
# heartbeat_timeout (set in workflows.py) so a healthy call never trips it.
PLAN_HEARTBEAT_INTERVAL: int = 30
SUBAGENT_HEARTBEAT_INTERVAL: int = 5
REVIEW_HEARTBEAT_INTERVAL: int = 30
SYNTH_HEARTBEAT_INTERVAL: int = 30

# --- Behavior --------------------------------------------------------------
ENABLE_WEB_SEARCH: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Mock-mode latency per Claude call (seconds). Bump it up to open a window for the
# crash-recovery demo: kill the worker mid-run, restart, watch the workflow resume.
MOCK_LATENCY: float = float(os.getenv("DURABLE_CLAUDE_MOCK_LATENCY", "1.2"))


def _is_truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def mock_mode() -> bool:
    """Should activities simulate Claude instead of calling the API?

    True when explicitly requested (``DURABLE_CLAUDE_MOCK=1``), OR when no
    Anthropic credentials are present (so the sample runs out of the box with a
    loud MOCK banner). Set ``DURABLE_CLAUDE_NO_AUTOMOCK=1`` to disable the
    credential-based fallback (e.g. if you authenticate via an ``ant auth login``
    profile that isn't visible as an env var).
    """
    if _is_truthy("DURABLE_CLAUDE_MOCK"):
        return True
    if _is_truthy("DURABLE_CLAUDE_NO_AUTOMOCK"):
        return False
    has_creds = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return not has_creds


def model_summary() -> dict[str, str]:
    """Human-readable per-role model map, surfaced in the chat client header."""
    if mock_mode():
        return {r: "mock" for r in ("planner", "subagent", "reviewer", "synthesizer")}
    return {
        "planner": PLANNER_MODEL,
        "subagent": SUBAGENT_MODEL,
        "reviewer": REVIEW_MODEL,
        "synthesizer": SYNTH_MODEL,
    }
