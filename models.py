"""
models.py — Typed contracts for the SRE Incident Response OpenEnv environment.

All Action, Observation, State, and Reward types are defined here.
Both the server (environment.py / app.py) and the client (baseline.py) import from this file.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    P1 = "P1"  # Critical  — customer-facing, immediate response
    P2 = "P2"  # High      — significant impact, respond within 30 min
    P3 = "P3"  # Medium    — degraded experience, respond within 2 h
    P4 = "P4"  # Low       — minor issue, respond within 24 h


class ActionType(str, Enum):
    CLASSIFY = "classify"              # Classify an alert (severity, service, team)
    INVESTIGATE = "investigate"        # Pull logs/metrics for a specific service
    HYPOTHESIZE = "hypothesize"        # State a root-cause hypothesis
    MITIGATE = "mitigate"              # Apply a mitigation action
    ESCALATE = "escalate"              # Escalate to another team
    RESOLVE = "resolve"                # Declare the incident resolved
    WRITE_POSTMORTEM = "write_postmortem"  # Write a structured post-mortem


# ---------------------------------------------------------------------------
# Action — what the agent sends
# ---------------------------------------------------------------------------


class ClassifyPayload(BaseModel):
    severity: Severity
    affected_service: str = Field(..., description="Canonical service name, e.g. 'payment-service'")
    affected_team: str = Field(..., description="Owning team slug, e.g. 'payments-team'")
    summary: str = Field(..., description="One-sentence human-readable summary")


class InvestigatePayload(BaseModel):
    service_name: str = Field(..., description="Service to fetch logs/metrics for")
    log_type: Literal["application", "system", "network", "database"] = "application"


class HypothesizePayload(BaseModel):
    root_cause: str = Field(..., description="Plain-English root cause description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent confidence 0–1")
    evidence: List[str] = Field(default_factory=list, description="Log/metric IDs or excerpts supporting this hypothesis")


class MitigatePayload(BaseModel):
    action: str = Field(..., description="Mitigation command or procedure, e.g. 'restart redis-cluster'")
    expected_outcome: str = Field(..., description="What improvement is expected")
    risk_level: Literal["low", "medium", "high"] = "low"


class EscalatePayload(BaseModel):
    team: str = Field(..., description="Team to escalate to")
    reason: str
    priority: Severity


class ResolvePayload(BaseModel):
    resolution_summary: str
    root_cause: str
    time_to_resolve_minutes: int = Field(..., ge=0)


class PostmortemPayload(BaseModel):
    title: str
    severity: Severity
    duration_minutes: int = Field(..., ge=0)
    timeline: List[Dict[str, str]] = Field(..., description="List of {time, event} dicts")
    root_cause: str
    contributing_factors: List[str]
    impact: str = Field(..., description="Customer / business impact description")
    action_items: List[Dict[str, str]] = Field(..., description="List of {owner, action, due} dicts")


class Action(BaseModel):
    """The single top-level action type the agent submits on each step."""

    action_type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict, description="Type-specific payload fields")
    task_id: Optional[str] = Field(None, description="Must match the active task; validated server-side")


# ---------------------------------------------------------------------------
# Observation — what the environment returns
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    alert_id: str
    service: str
    message: str
    timestamp: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    labels: Dict[str, str] = Field(default_factory=dict)


class LogEntry(BaseModel):
    timestamp: str
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    service: str
    message: str
    trace_id: Optional[str] = None


class MetricPoint(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: str
    service: str


class Observation(BaseModel):
    """Everything the agent can observe at a given step."""

    task_id: str
    task_description: str
    step_count: int
    max_steps: int
    current_alerts: List[Alert] = Field(default_factory=list)
    available_services: List[str] = Field(default_factory=list)
    visible_logs: List[LogEntry] = Field(default_factory=list)
    visible_metrics: List[MetricPoint] = Field(default_factory=list)
    team_responses: List[str] = Field(default_factory=list)
    incident_timeline: List[Dict[str, str]] = Field(default_factory=list)
    investigations_done: List[str] = Field(default_factory=list, description="Services already investigated")
    feedback: str = Field("", description="Human-readable feedback about the last action")
    is_done: bool = False


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State — episode metadata
# ---------------------------------------------------------------------------


class State(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    total_reward: float
    is_done: bool
    investigations_performed: List[str] = Field(default_factory=list)
    hypotheses_submitted: List[str] = Field(default_factory=list)
    mitigations_applied: List[str] = Field(default_factory=list)
    resolved: bool = False
    postmortem_written: bool = False


# ---------------------------------------------------------------------------
# Reward (for documentation / validation)
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """Structured breakdown of a reward signal (returned in StepResult.info)."""

    raw: float = Field(..., description="Scalar reward for this step")
    components: Dict[str, float] = Field(default_factory=dict, description="Named sub-rewards")
    explanation: str = ""
