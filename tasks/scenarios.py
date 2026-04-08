"""
tasks/scenarios.py — Deterministic task definitions for the SRE Incident Response environment.

Each task is a frozen dataclass specifying:
  - The initial observation data (alerts, logs hidden behind investigation gates)
  - The ground-truth solution used by the grader
  - A grading function that scores an episode history 0.0–1.0

Task difficulty:
  EASY   (task_id="alert_triage")         — classify 3 independent alerts
  MEDIUM (task_id="cascading_failure")    — diagnose a multi-service cascading failure
  HARD   (task_id="full_postmortem")      — reconstruct + document a complex 4-hour incident
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    Alert,
    LogEntry,
    MetricPoint,
    Observation,
    Severity,
    StepResult,
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _fuzzy_match(candidate: str, target: str, threshold: float = 0.70) -> bool:
    """Case-insensitive fuzzy string match using SequenceMatcher."""
    if not candidate or not target:
        return False
    ratio = difflib.SequenceMatcher(
        None,
        candidate.lower().strip(),
        target.lower().strip(),
    ).ratio()
    return ratio >= threshold


def _keyword_score(text: str, required_keywords: List[str], bonus_keywords: List[str] = []) -> float:
    """Score text coverage of required + bonus keywords (0.0–1.0)."""
    if not text:
        return 0.0
    text_lower = text.lower()
    required_hit = sum(1 for kw in required_keywords if kw.lower() in text_lower)
    bonus_hit = sum(1 for kw in bonus_keywords if kw.lower() in text_lower)
    required_score = required_hit / len(required_keywords) if required_keywords else 1.0
    bonus_score = (bonus_hit / len(bonus_keywords) * 0.2) if bonus_keywords else 0.0
    return min(1.0, required_score * 0.8 + bonus_score)


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (never exactly 0.0 or 1.0)."""
    return max(0.01, min(0.99, score))


# ===========================================================================
# TASK 1 — EASY: Alert Triage
# ===========================================================================

ALERT_TRIAGE_ALERTS: List[Alert] = [
    Alert(
        alert_id="alert-001",
        service="api-gateway",
        message="HTTP 5xx error rate exceeded 5% threshold — currently at 18.3%. Upstream payment-service returning ECONNREFUSED.",
        timestamp="2024-03-15T14:32:01Z",
        metric_name="http_error_rate_5xx",
        metric_value=18.3,
        threshold=5.0,
        labels={"env": "production", "region": "us-east-1", "tier": "frontend"},
    ),
    Alert(
        alert_id="alert-002",
        service="redis-cluster",
        message="Redis memory usage at 94% capacity. Eviction policy maxmemory-policy=noeviction — writes are now failing with OOM errors.",
        timestamp="2024-03-15T14:30:47Z",
        metric_name="redis_memory_usage_percent",
        metric_value=94.0,
        threshold=85.0,
        labels={"env": "production", "cluster": "cache-prod-01", "tier": "cache"},
    ),
    Alert(
        alert_id="alert-003",
        service="postgres-primary",
        message="Replication lag on postgres-replica-01 has exceeded 30 seconds (current: 187 seconds). Replica is falling behind.",
        timestamp="2024-03-15T14:28:15Z",
        metric_name="replication_lag_seconds",
        metric_value=187.0,
        threshold=30.0,
        labels={"env": "production", "role": "replica", "tier": "database"},
    ),
]

ALERT_TRIAGE_GROUND_TRUTH: Dict[str, Dict] = {
    "alert-001": {
        "severity": Severity.P1,
        "affected_service": "api-gateway",
        "affected_team": "platform-team",
        "keywords": ["5xx", "error rate", "payment", "econnrefused", "upstream"],
    },
    "alert-002": {
        "severity": Severity.P1,
        "affected_service": "redis-cluster",
        "affected_team": "infrastructure-team",
        "keywords": ["redis", "memory", "oom", "eviction", "cache"],
    },
    "alert-003": {
        "severity": Severity.P2,
        "affected_service": "postgres-primary",
        "affected_team": "database-team",
        "keywords": ["replication", "lag", "replica", "postgres"],
    },
}

ALERT_TRIAGE_DESCRIPTION = """
# Task: Alert Triage (Easy)

You are an on-call SRE. Three monitoring alerts have just fired in production simultaneously.
Your job is to triage each alert by classifying its severity, identifying the affected service,
assigning it to the correct team, and writing a short summary.

**Available teams:** platform-team, infrastructure-team, database-team, payments-team, security-team

**Severity guide:**
- P1: Customer-facing outage or data loss risk — respond immediately
- P2: Significant degradation — respond within 30 minutes
- P3: Minor degradation — respond within 2 hours
- P4: Informational — respond within 24 hours

**Your goal:** Use the `classify` action for each alert. Each correct classification earns reward.
You have a maximum of 8 steps.

**Alerts currently firing:**
- alert-001: api-gateway HTTP 5xx spike (18.3% error rate)
- alert-002: redis-cluster OOM / memory at 94%
- alert-003: postgres-primary replication lag (187 seconds)
""".strip()


def grade_alert_triage(history: List[Tuple[Action, StepResult]]) -> Tuple[float, Dict[str, Any]]:
    """Grade an alert triage episode. Returns (score 0.0–1.0, details dict)."""
    classified: Dict[str, Dict] = {}
    details: Dict[str, Any] = {"per_alert": {}, "explanations": []}

    for action, result in history:
        if action.action_type != ActionType.CLASSIFY:
            continue
        payload = action.payload
        alert_id = payload.get("alert_id") or payload.get("summary", "")[:20]

        # Try to match which alert is being classified
        matched_id = None
        for aid in ALERT_TRIAGE_GROUND_TRUTH:
            if aid in str(payload) or _fuzzy_match(
                payload.get("affected_service", ""),
                ALERT_TRIAGE_GROUND_TRUTH[aid]["affected_service"],
            ):
                matched_id = aid
                break

        if matched_id is None:
            continue
        if matched_id in classified:
            continue  # only first classification counts per alert
        classified[matched_id] = payload

    total_score = 0.0
    for aid, gt in ALERT_TRIAGE_GROUND_TRUTH.items():
        if aid not in classified:
            details["per_alert"][aid] = {"score": 0.0, "reason": "not classified"}
            continue

        cl = classified[aid]
        alert_score = 0.0
        reasons = []

        # Severity: 40% of alert score
        sev_guess = cl.get("severity", "")
        if isinstance(sev_guess, str):
            sev_guess = sev_guess.upper()
        if sev_guess == gt["severity"].value:
            alert_score += 0.40
            reasons.append("✓ severity correct")
        else:
            # Partial credit for adjacent severity
            sev_order = ["P1", "P2", "P3", "P4"]
            gt_idx = sev_order.index(gt["severity"].value)
            cl_idx = sev_order.index(sev_guess) if sev_guess in sev_order else -1
            if cl_idx >= 0 and abs(gt_idx - cl_idx) == 1:
                alert_score += 0.15
                reasons.append("~ severity off by one")
            else:
                reasons.append("✗ severity wrong")

        # Team: 30% of alert score
        team_guess = cl.get("affected_team", "")
        if _fuzzy_match(team_guess, gt["affected_team"], threshold=0.6):
            alert_score += 0.30
            reasons.append("✓ team correct")
        else:
            reasons.append(f"✗ team wrong (got '{team_guess}', expected '{gt['affected_team']}')")

        # Summary quality: 30% of alert score
        summary = cl.get("summary", "")
        kw_score = _keyword_score(summary, gt["keywords"][:2], gt["keywords"][2:])
        alert_score += 0.30 * kw_score
        reasons.append(f"~ summary keyword coverage {kw_score:.0%}")

        details["per_alert"][aid] = {"score": round(alert_score, 3), "reasons": reasons}
        total_score += alert_score / len(ALERT_TRIAGE_GROUND_TRUTH)

    total_score = _clamp(total_score)
    details["final_score"] = round(total_score, 4)
    return total_score, details


# ===========================================================================
# TASK 2 — MEDIUM: Cascading Failure Diagnosis
# ===========================================================================

CASCADING_FAILURE_ALERTS: List[Alert] = [
    Alert(
        alert_id="cf-001",
        service="payment-service",
        message="Payment service p99 latency spiked to 12,400ms (SLO threshold: 500ms). Checkout flow is timing out.",
        timestamp="2024-06-10T09:15:22Z",
        metric_name="latency_p99_ms",
        metric_value=12400.0,
        threshold=500.0,
        labels={"env": "production", "tier": "application"},
    ),
    Alert(
        alert_id="cf-002",
        service="payment-service",
        message="Payment service error budget burned: 74% of monthly error budget consumed in 3 hours.",
        timestamp="2024-06-10T09:17:05Z",
        metric_name="error_budget_percent",
        metric_value=74.0,
        threshold=10.0,
        labels={"env": "production", "tier": "application"},
    ),
    Alert(
        alert_id="cf-003",
        service="order-service",
        message="Order creation failures at 31%. Downstream dependency on payment-service failing.",
        timestamp="2024-06-10T09:18:44Z",
        metric_name="order_creation_failure_rate",
        metric_value=31.0,
        threshold=1.0,
        labels={"env": "production", "tier": "application"},
    ),
]

# Hidden logs — revealed only when agent investigates the correct service
CASCADING_FAILURE_HIDDEN_LOGS: Dict[str, List[LogEntry]] = {
    "payment-service": [
        LogEntry(
            timestamp="2024-06-10T09:14:55Z",
            level="ERROR",
            service="payment-service",
            message="redis.exceptions.ConnectionError: Error 111 connecting to cache-prod-01:6379. Connection refused.",
            trace_id="trace-abc123",
        ),
        LogEntry(
            timestamp="2024-06-10T09:14:56Z",
            level="ERROR",
            service="payment-service",
            message="Cache miss fallback failed — unable to reach Redis. Falling back to database for session lookup.",
            trace_id="trace-abc123",
        ),
        LogEntry(
            timestamp="2024-06-10T09:15:01Z",
            level="FATAL",
            service="payment-service",
            message="Database connection pool exhausted (max_connections=100, active=100). New queries queuing.",
            trace_id="trace-def456",
        ),
    ],
    "redis-cluster": [
        LogEntry(
            timestamp="2024-06-10T09:10:03Z",
            level="WARN",
            service="redis-cluster",
            message="Redis sentinel failover triggered. Primary cache-prod-01 unreachable, promoting cache-prod-02.",
            trace_id=None,
        ),
        LogEntry(
            timestamp="2024-06-10T09:12:19Z",
            level="ERROR",
            service="redis-cluster",
            message="Failover incomplete — cache-prod-02 misconfigured: bind address 127.0.0.1 prevents external connections.",
            trace_id=None,
        ),
        LogEntry(
            timestamp="2024-06-10T09:13:00Z",
            level="ERROR",
            service="redis-cluster",
            message="All Redis nodes unreachable. Cluster in DOWN state. Clients will receive ECONNREFUSED.",
            trace_id=None,
        ),
    ],
    "order-service": [
        LogEntry(
            timestamp="2024-06-10T09:15:30Z",
            level="ERROR",
            service="order-service",
            message="POST /api/v1/orders — payment-service returned 504 Gateway Timeout after 10000ms.",
            trace_id="trace-ghi789",
        ),
        LogEntry(
            timestamp="2024-06-10T09:16:00Z",
            level="WARN",
            service="order-service",
            message="Circuit breaker for payment-service OPEN. Order creation requests fast-failing.",
            trace_id=None,
        ),
    ],
    "postgres-primary": [
        LogEntry(
            timestamp="2024-06-10T09:14:50Z",
            level="WARN",
            service="postgres-primary",
            message="Connection count approaching max: 89/100 active connections.",
            trace_id=None,
        ),
        LogEntry(
            timestamp="2024-06-10T09:15:05Z",
            level="ERROR",
            service="postgres-primary",
            message="FATAL: remaining connection slots reserved for replication. Application connections rejected.",
            trace_id=None,
        ),
    ],
    "api-gateway": [
        LogEntry(
            timestamp="2024-06-10T09:15:10Z",
            level="WARN",
            service="api-gateway",
            message="Upstream payment-service health check failed 3 consecutive times. Removing from load balancer pool.",
            trace_id=None,
        ),
    ],
}

CASCADING_FAILURE_METRICS: List[MetricPoint] = [
    MetricPoint(name="latency_p99_ms", value=12400.0, unit="ms", timestamp="2024-06-10T09:15:22Z", service="payment-service"),
    MetricPoint(name="cache_hit_rate", value=0.0, unit="percent", timestamp="2024-06-10T09:15:00Z", service="redis-cluster"),
    MetricPoint(name="db_connections_active", value=100.0, unit="count", timestamp="2024-06-10T09:15:05Z", service="postgres-primary"),
    MetricPoint(name="order_failure_rate", value=31.0, unit="percent", timestamp="2024-06-10T09:18:44Z", service="order-service"),
]

CASCADING_FAILURE_DESCRIPTION = """
# Task: Cascading Failure Diagnosis (Medium)

You are the on-call SRE for a payments platform. Alerts fired 4 minutes ago. The checkout flow is broken.

**Your goal:** Diagnose the root cause of this cascading failure, propose mitigation, and resolve the incident.

**Available services to investigate:** payment-service, redis-cluster, order-service, postgres-primary, api-gateway

**Recommended action sequence:**
1. Use `investigate` to examine logs from suspicious services
2. Use `hypothesize` to state your root cause theory
3. Use `mitigate` to propose an immediate fix
4. Use `resolve` to close the incident with a summary

**Scoring rewards:**
- Investigating redis-cluster: +reward (it's the root cause chain)
- Correct root cause hypothesis mentioning Redis/cache: +large reward
- Valid mitigation (restart/reconfigure Redis): +reward
- Clean resolve with accurate root cause: +reward

You have a maximum of 12 steps.
""".strip()

CASCADING_FAILURE_GROUND_TRUTH = {
    "root_cause_service": "redis-cluster",
    "root_cause_keywords": ["redis", "cache", "failover", "misconfigured", "bind", "econnrefused", "sentinel"],
    "mitigation_keywords": ["restart", "redis", "reconfigure", "bind", "failover", "cache", "flush", "sentinel"],
    "resolve_keywords": ["redis", "cache", "failover", "misconfigured"],
    "key_services_to_investigate": ["redis-cluster", "payment-service"],
}


def grade_cascading_failure(history: List[Tuple[Action, StepResult]]) -> Tuple[float, Dict[str, Any]]:
    """Grade a cascading failure episode. Returns (score 0.0–1.0, details dict)."""
    details: Dict[str, Any] = {
        "investigated": [],
        "hypotheses": [],
        "mitigations": [],
        "resolved": False,
        "components": {},
    }
    score = 0.0

    investigated_services = set()
    best_hypothesis_score = 0.0
    best_mitigation_score = 0.0
    resolved = False
    resolve_score = 0.0
    step_penalty = 0.0

    gt = CASCADING_FAILURE_GROUND_TRUTH

    for i, (action, result) in enumerate(history):
        if action.action_type == ActionType.INVESTIGATE:
            svc = action.payload.get("service_name", "")
            investigated_services.add(svc)
            details["investigated"].append(svc)

        elif action.action_type == ActionType.HYPOTHESIZE:
            rc = action.payload.get("root_cause", "")
            ev = " ".join(action.payload.get("evidence", []))
            combined = rc + " " + ev
            hyp_score = _keyword_score(combined, gt["root_cause_keywords"][:3], gt["root_cause_keywords"][3:])
            # Bonus for high confidence when correct
            if hyp_score > 0.6:
                confidence = float(action.payload.get("confidence", 0.5))
                hyp_score = hyp_score * (0.8 + 0.2 * confidence)
            best_hypothesis_score = max(best_hypothesis_score, hyp_score)
            details["hypotheses"].append({"text": rc[:80], "score": round(hyp_score, 3)})

        elif action.action_type == ActionType.MITIGATE:
            mit_text = action.payload.get("action", "") + " " + action.payload.get("expected_outcome", "")
            mit_score = _keyword_score(mit_text, gt["mitigation_keywords"][:3], gt["mitigation_keywords"][3:])
            best_mitigation_score = max(best_mitigation_score, mit_score)
            details["mitigations"].append({"text": action.payload.get("action", "")[:80], "score": round(mit_score, 3)})

        elif action.action_type == ActionType.RESOLVE:
            resolved = True
            details["resolved"] = True
            res_text = action.payload.get("root_cause", "") + " " + action.payload.get("resolution_summary", "")
            resolve_score = _keyword_score(res_text, gt["resolve_keywords"][:2], gt["resolve_keywords"][2:])

    # Investigation coverage: up to 0.15
    key_investigated = len([s for s in gt["key_services_to_investigate"] if s in investigated_services])
    invest_score = key_investigated / len(gt["key_services_to_investigate"]) * 0.15
    details["components"]["investigation"] = round(invest_score, 3)

    # Root cause accuracy: up to 0.40
    hyp_component = best_hypothesis_score * 0.40
    details["components"]["hypothesis"] = round(hyp_component, 3)

    # Mitigation quality: up to 0.25
    mit_component = best_mitigation_score * 0.25
    details["components"]["mitigation"] = round(mit_component, 3)

    # Resolution: up to 0.20
    if resolved:
        res_component = resolve_score * 0.20
    else:
        res_component = 0.0
    details["components"]["resolution"] = round(res_component, 3)

    # Step efficiency bonus: if solved in ≤8 steps, +0.05
    if resolved and len(history) <= 8:
        details["components"]["efficiency_bonus"] = 0.05
    else:
        details["components"]["efficiency_bonus"] = 0.0

    score = sum(details["components"].values())
    score = _clamp(round(score, 4))
    details["final_score"] = score
    return score, details


# ===========================================================================
# TASK 3 — HARD: Full Post-mortem Writing
# ===========================================================================

POSTMORTEM_INCIDENT_TIMELINE: List[Dict[str, str]] = [
    {"time": "2024-09-02T02:00:00Z", "event": "Scheduled maintenance window begins. DBA team runs index rebuild on payments_db."},
    {"time": "2024-09-02T02:23:00Z", "event": "Index rebuild completes. DBA runs VACUUM ANALYZE on payments_db.transactions table (87M rows)."},
    {"time": "2024-09-02T02:26:00Z", "event": "VACUUM ANALYZE holds AccessShareLock on transactions table."},
    {"time": "2024-09-02T02:27:00Z", "event": "payment-service write queries begin queuing behind VACUUM lock. Connection pool fills."},
    {"time": "2024-09-02T02:28:00Z", "event": "First alert fires: payment-service p99 latency > 2000ms."},
    {"time": "2024-09-02T02:29:00Z", "event": "On-call engineer (Priya) paged. Begins investigation."},
    {"time": "2024-09-02T02:31:00Z", "event": "Priya investigates payment-service — sees connection pool exhaustion but no obvious application error."},
    {"time": "2024-09-02T02:35:00Z", "event": "Priya escalates to DBA team. DBA team lead (Arjun) joins the call."},
    {"time": "2024-09-02T02:38:00Z", "event": "Arjun identifies VACUUM ANALYZE running with lock. Cancels the VACUUM process: SELECT pg_cancel_backend(<pid>)."},
    {"time": "2024-09-02T02:39:00Z", "event": "payment-service connection pool drains. Latency recovers to baseline."},
    {"time": "2024-09-02T02:41:00Z", "event": "All alerts resolve. Incident declared resolved."},
    {"time": "2024-09-02T06:15:00Z", "event": "Post-incident: payments team reports 1,247 failed payment transactions during the 13-minute window."},
]

POSTMORTEM_FULL_LOGS: List[LogEntry] = [
    LogEntry(timestamp="2024-09-02T02:27:01Z", level="ERROR", service="payment-service",
             message="could not obtain lock on relation 'transactions' — waiting for AccessShareLock held by pid 39211", trace_id="t-0001"),
    LogEntry(timestamp="2024-09-02T02:27:30Z", level="FATAL", service="payment-service",
             message="connection pool exhausted: 50/50 connections active. New requests failing with 503.", trace_id="t-0002"),
    LogEntry(timestamp="2024-09-02T02:28:05Z", level="INFO", service="postgres-primary",
             message="autovacuum: VACUUM ANALYZE payments_db.transactions — estimated rows: 87433210", trace_id=None),
    LogEntry(timestamp="2024-09-02T02:38:42Z", level="INFO", service="postgres-primary",
             message="pg_cancel_backend(39211) called — VACUUM process terminated.", trace_id=None),
    LogEntry(timestamp="2024-09-02T02:39:10Z", level="INFO", service="payment-service",
             message="Connection pool draining. Active: 12/50. Latency recovering.", trace_id=None),
]

POSTMORTEM_GROUND_TRUTH = {
    "severity": Severity.P1,
    "duration_minutes": 13,  # 02:28 to 02:41
    "root_cause_keywords": ["vacuum", "analyze", "lock", "transaction", "postgres", "dba", "maintenance", "autovacuum"],
    "contributing_keywords": ["maintenance window", "connection pool", "no staging test", "lock", "vacuum"],
    "impact_keywords": ["1247", "payment", "failed", "transaction", "customer"],
    "timeline_events_required": 5,  # at least 5 timeline entries
    "action_items_required": 2,     # at least 2 action items
    "action_item_keywords": ["never vacuum production during peak", "staging", "lock", "runbook", "alert"],
}

POSTMORTEM_DESCRIPTION = """
# Task: Full Post-mortem Writing (Hard)

A P1 production incident occurred on 2024-09-02 between 02:28Z and 02:41Z.
You have access to the full incident timeline, all logs, and the metrics from that window.

**Your goal:** Write a complete, structured post-mortem using the `write_postmortem` action.

**Required post-mortem fields:**
- title: Descriptive incident title
- severity: P1/P2/P3/P4
- duration_minutes: How long did the incident last?
- timeline: At least 5 key events with {time, event} entries
- root_cause: Clear, technical root cause description
- contributing_factors: List of factors that allowed this to happen
- impact: Customer/business impact (include numbers where possible)
- action_items: At least 2 specific, actionable items with {owner, action, due}

**Full incident data is immediately available in this task** (no investigation needed).

**Scoring criteria:**
- Correct severity (P1): required
- Root cause accuracy (mentions VACUUM/lock/postgres): 30% of score
- Impact accuracy (mentions failed payments/customer impact): 20%
- Timeline completeness (≥5 events, chronologically correct): 20%
- Action item quality (specific, preventive): 20%
- Duration accuracy (within 5 minutes of 13 min): 10%

You have a maximum of 4 steps (this is a writing task, not exploration).
""".strip()


def grade_postmortem(history: List[Tuple[Action, StepResult]]) -> Tuple[float, Dict[str, Any]]:
    """Grade a post-mortem writing episode. Returns (score 0.0–1.0, details dict)."""
    details: Dict[str, Any] = {"components": {}, "postmortems_found": 0}
    gt = POSTMORTEM_GROUND_TRUTH

    best_pm = None
    for action, result in history:
        if action.action_type == ActionType.WRITE_POSTMORTEM:
            details["postmortems_found"] += 1
            best_pm = action.payload  # take the last one written

    if best_pm is None:
        details["final_score"] = 0.01
        details["reason"] = "No write_postmortem action found"
        return 0.01, details

    score = 0.0

    # 1. Severity (10%)
    sev = str(best_pm.get("severity", "")).upper()
    if sev == gt["severity"].value:
        details["components"]["severity"] = 0.10
    else:
        details["components"]["severity"] = 0.0

    # 2. Root cause accuracy (30%)
    rc = str(best_pm.get("root_cause", ""))
    rc_score = _keyword_score(rc, gt["root_cause_keywords"][:4], gt["root_cause_keywords"][4:])
    details["components"]["root_cause"] = round(rc_score * 0.30, 4)

    # 3. Impact accuracy (20%)
    impact = str(best_pm.get("impact", ""))
    imp_score = _keyword_score(impact, gt["impact_keywords"][:3], gt["impact_keywords"][3:])
    details["components"]["impact"] = round(imp_score * 0.20, 4)

    # 4. Timeline completeness (20%)
    timeline = best_pm.get("timeline", [])
    if isinstance(timeline, list) and len(timeline) >= gt["timeline_events_required"]:
        # Check that events are roughly chronological
        times = []
        for ev in timeline:
            t = ev.get("time", "") if isinstance(ev, dict) else ""
            times.append(t)
        is_sorted = all(times[i] <= times[i + 1] for i in range(len(times) - 1) if times[i] and times[i + 1])
        timeline_score = min(1.0, len(timeline) / 8) * (1.0 if is_sorted else 0.7)
    else:
        timeline_score = len(timeline) / gt["timeline_events_required"] * 0.5 if timeline else 0.0
    details["components"]["timeline"] = round(timeline_score * 0.20, 4)

    # 5. Action items quality (20%)
    action_items = best_pm.get("action_items", [])
    if isinstance(action_items, list) and len(action_items) >= gt["action_items_required"]:
        combined_actions = " ".join(
            str(ai.get("action", "")) + " " + str(ai.get("owner", ""))
            for ai in action_items if isinstance(ai, dict)
        )
        ai_score = _keyword_score(combined_actions, gt["action_item_keywords"][:2], gt["action_item_keywords"][2:])
        # Check all items have owner + action + due
        completeness = sum(
            1 for ai in action_items
            if isinstance(ai, dict) and ai.get("owner") and ai.get("action") and ai.get("due")
        ) / max(len(action_items), 1)
        ai_score = ai_score * (0.7 + 0.3 * completeness)
    else:
        ai_score = len(action_items) / gt["action_items_required"] * 0.3 if action_items else 0.0
    details["components"]["action_items"] = round(ai_score * 0.20, 4)

    # 6. Duration accuracy (10%)
    duration = int(best_pm.get("duration_minutes", 0))
    if abs(duration - gt["duration_minutes"]) <= 5:
        details["components"]["duration"] = 0.10
    elif abs(duration - gt["duration_minutes"]) <= 15:
        details["components"]["duration"] = 0.05
    else:
        details["components"]["duration"] = 0.0

    # Contributing factors (bonus check, included in root_cause score)
    cf = " ".join(str(f) for f in best_pm.get("contributing_factors", []))
    cf_bonus = _keyword_score(cf, gt["contributing_keywords"][:2], gt["contributing_keywords"][2:]) * 0.05
    details["components"]["contributing_factors_bonus"] = round(cf_bonus, 4)

    score = sum(details["components"].values())
    score = _clamp(round(score, 4))
    details["final_score"] = score
    return score, details


# ===========================================================================
# Task registry
# ===========================================================================


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "alert_triage": {
        "task_id": "alert_triage",
        "difficulty": "easy",
        "description": ALERT_TRIAGE_DESCRIPTION,
        "alerts": ALERT_TRIAGE_ALERTS,
        "hidden_logs": {},
        "metrics": [],
        "available_services": ["api-gateway", "redis-cluster", "postgres-primary"],
        "incident_timeline": [],
        "max_steps": 8,
        "grader": grade_alert_triage,
        "action_schema": {
            "action_type": "classify",
            "payload": {
                "alert_id": "str — which alert (e.g. alert-001)",
                "severity": "P1|P2|P3|P4",
                "affected_service": "str — canonical service name",
                "affected_team": "str — owning team slug",
                "summary": "str — one-sentence description",
            },
        },
    },
    "cascading_failure": {
        "task_id": "cascading_failure",
        "difficulty": "medium",
        "description": CASCADING_FAILURE_DESCRIPTION,
        "alerts": CASCADING_FAILURE_ALERTS,
        "hidden_logs": CASCADING_FAILURE_HIDDEN_LOGS,
        "metrics": CASCADING_FAILURE_METRICS,
        "available_services": list(CASCADING_FAILURE_HIDDEN_LOGS.keys()),
        "incident_timeline": [],
        "max_steps": 12,
        "grader": grade_cascading_failure,
        "action_schema": {
            "investigate": {"service_name": "str", "log_type": "application|system|network|database"},
            "hypothesize": {"root_cause": "str", "confidence": "float 0–1", "evidence": "list[str]"},
            "mitigate": {"action": "str", "expected_outcome": "str", "risk_level": "low|medium|high"},
            "escalate": {"team": "str", "reason": "str", "priority": "P1|P2|P3|P4"},
            "resolve": {"resolution_summary": "str", "root_cause": "str", "time_to_resolve_minutes": "int"},
        },
    },
    "full_postmortem": {
        "task_id": "full_postmortem",
        "difficulty": "hard",
        "description": POSTMORTEM_DESCRIPTION,
        "alerts": [],
        "hidden_logs": {},
        "metrics": [],
        "available_services": ["payment-service", "postgres-primary"],
        "incident_timeline": POSTMORTEM_INCIDENT_TIMELINE,
        "visible_logs": POSTMORTEM_FULL_LOGS,
        "max_steps": 4,
        "grader": grade_postmortem,
        "action_schema": {
            "action_type": "write_postmortem",
            "payload": {
                "title": "str",
                "severity": "P1|P2|P3|P4",
                "duration_minutes": "int",
                "timeline": "list[{time: str, event: str}]",
                "root_cause": "str",
                "contributing_factors": "list[str]",
                "impact": "str",
                "action_items": "list[{owner: str, action: str, due: str}]",
            },
        },
    },
}
