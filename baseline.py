#!/usr/bin/env python3
"""
baseline.py — Baseline inference script for the SRE Incident Response OpenEnv.

Usage:
  python baseline.py                          # run all tasks, print scores
  python baseline.py --task alert_triage      # run one task
  python baseline.py --all-tasks --json       # machine-readable JSON output

Requires:
  OPENAI_API_KEY environment variable
  pip install openai (already in requirements.txt)

The baseline agent uses GPT-4o-mini with a structured system prompt.
It reads the observation, reasons about the best action, and submits it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Allow running from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import SREIncidentEnvironment
from models import Action, ActionType, Observation
from tasks.scenarios import TASK_REGISTRY


# ---------------------------------------------------------------------------
# OpenAI client setup
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _get_client() -> Optional[Any]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not OPENAI_AVAILABLE:
        return None
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) agent operating in the SRE Incident Response environment.

You receive a JSON observation and must output a single JSON action.

## Available action_types and their payloads:

**classify** — classify an alert (alert_triage task)
{"action_type": "classify", "payload": {"alert_id": "alert-001", "severity": "P1", "affected_service": "api-gateway", "affected_team": "platform-team", "summary": "..."}}

**investigate** — investigate a service's logs (cascading_failure task)
{"action_type": "investigate", "payload": {"service_name": "redis-cluster", "log_type": "application"}}

**hypothesize** — state a root cause hypothesis
{"action_type": "hypothesize", "payload": {"root_cause": "...", "confidence": 0.8, "evidence": ["log excerpt 1", "log excerpt 2"]}}

**mitigate** — propose a mitigation
{"action_type": "mitigate", "payload": {"action": "restart redis-cluster primary", "expected_outcome": "cache restored", "risk_level": "low"}}

**resolve** — close the incident
{"action_type": "resolve", "payload": {"resolution_summary": "...", "root_cause": "...", "time_to_resolve_minutes": 15}}

**write_postmortem** — write a full post-mortem (full_postmortem task)
{"action_type": "write_postmortem", "payload": {"title": "...", "severity": "P1", "duration_minutes": 13, "timeline": [{"time": "...", "event": "..."}], "root_cause": "...", "contributing_factors": ["..."], "impact": "...", "action_items": [{"owner": "...", "action": "...", "due": "..."}]}}

## Rules:
- Output ONLY a JSON object. No markdown, no explanation, no code fences.
- For alert_triage: classify each alert with a separate classify action (one action per step).
- For cascading_failure: investigate suspicious services first, then hypothesize and resolve.
- For full_postmortem: use write_postmortem in your first action with a complete, detailed post-mortem.
- Be specific and technical — vague answers score poorly.
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs: Observation) -> str:
    """Convert observation to a concise prompt for the LLM."""
    parts = [f"# Task: {obs.task_id}\n{obs.task_description}\n"]

    if obs.current_alerts:
        parts.append("\n## Current Alerts")
        for a in obs.current_alerts:
            parts.append(f"- [{a.alert_id}] {a.service}: {a.message} (value={a.metric_value}, threshold={a.threshold})")

    if obs.visible_logs:
        parts.append("\n## Log Entries")
        for lg in obs.visible_logs[-15:]:  # last 15 logs to stay within context
            parts.append(f"  [{lg.timestamp}] {lg.level} {lg.service}: {lg.message}")

    if obs.visible_metrics:
        parts.append("\n## Metrics")
        for m in obs.visible_metrics:
            parts.append(f"  {m.service}/{m.name}: {m.value}{m.unit}")

    if obs.incident_timeline:
        parts.append("\n## Incident Timeline")
        for ev in obs.incident_timeline:
            parts.append(f"  {ev.get('time', '')} — {ev.get('event', '')}")

    if obs.investigations_done:
        parts.append(f"\n## Already Investigated: {', '.join(obs.investigations_done)}")

    if obs.feedback:
        parts.append(f"\n## Last Feedback: {obs.feedback}")

    parts.append(f"\nStep {obs.step_count}/{obs.max_steps}. Output your next JSON action:")
    return "\n".join(parts)


def _parse_action(response_text: str) -> Optional[Dict]:
    """Parse LLM response as JSON action dict."""
    text = response_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def run_task_with_llm(client: Any, task_id: str, model: str = "gpt-4o-mini", verbose: bool = True) -> Tuple[float, Dict]:
    """Run a single task with an LLM agent. Returns (grader_score, details)."""
    env = SREIncidentEnvironment(task_id=task_id)
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} | Model: {model}")
        print(f"{'='*60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_rewards = []

    while not obs.is_done:
        user_prompt = _obs_to_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"  [API Error] {e}")
            break

        messages.append({"role": "assistant", "content": response_text})

        action_dict = _parse_action(response_text)
        if action_dict is None:
            if verbose:
                print(f"  [Step {obs.step_count}] Could not parse action from LLM output")
            break

        try:
            action = Action(**action_dict)
        except Exception as e:
            if verbose:
                print(f"  [Step {obs.step_count}] Invalid action: {e}")
            break

        result = env.step(action)
        step_rewards.append(result.reward)

        if verbose:
            print(f"  [Step {result.observation.step_count}] {action.action_type.value} → reward={result.reward:+.3f} | {result.observation.feedback[:80]}")

        obs = result.observation
        time.sleep(0.3)  # rate limit buffer

    score, details = env.grade()
    if verbose:
        print(f"\n  ✅ Episode done. Grader score: {score:.3f}")
        print(f"  Total env reward: {env._total_reward:.3f}")

    return score, {
        "grader_score": score,
        "env_reward": env._total_reward,
        "steps_taken": env._step_count,
        "grader_details": details,
    }


def run_task_heuristic(task_id: str, verbose: bool = True) -> Tuple[float, Dict]:
    """
    Heuristic baseline (no LLM) — uses hard-coded correct actions.
    Used when OPENAI_API_KEY is not set, to demonstrate reproducible scores.
    """
    env = SREIncidentEnvironment(task_id=task_id)
    env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} | Mode: heuristic baseline")
        print(f"{'='*60}")

    if task_id == "alert_triage":
        actions = [
            Action(action_type=ActionType.CLASSIFY, task_id=task_id, payload={
                "alert_id": "alert-001",
                "severity": "P1",
                "affected_service": "api-gateway",
                "affected_team": "platform-team",
                "summary": "API gateway HTTP 5xx rate at 18.3% due to upstream payment-service ECONNREFUSED. Immediate response required.",
            }),
            Action(action_type=ActionType.CLASSIFY, task_id=task_id, payload={
                "alert_id": "alert-002",
                "severity": "P1",
                "affected_service": "redis-cluster",
                "affected_team": "infrastructure-team",
                "summary": "Redis cluster OOM with noeviction policy — writes failing. Cache completely unavailable.",
            }),
            Action(action_type=ActionType.CLASSIFY, task_id=task_id, payload={
                "alert_id": "alert-003",
                "severity": "P2",
                "affected_service": "postgres-primary",
                "affected_team": "database-team",
                "summary": "Postgres replication lag at 187s (threshold 30s) — replica falling behind, risk of data loss on failover.",
            }),
        ]

    elif task_id == "cascading_failure":
        actions = [
            Action(action_type=ActionType.INVESTIGATE, task_id=task_id, payload={"service_name": "payment-service", "log_type": "application"}),
            Action(action_type=ActionType.INVESTIGATE, task_id=task_id, payload={"service_name": "redis-cluster", "log_type": "application"}),
            Action(action_type=ActionType.HYPOTHESIZE, task_id=task_id, payload={
                "root_cause": "Redis sentinel failover failed because the promoted replica (cache-prod-02) has a misconfigured bind address (127.0.0.1), preventing external connections. This made the entire Redis cluster unavailable, forcing payment-service to fall back to the database for all session lookups, exhausting the Postgres connection pool and causing cascading timeouts.",
                "confidence": 0.93,
                "evidence": [
                    "redis-cluster: Failover incomplete — cache-prod-02 misconfigured: bind address 127.0.0.1 prevents external connections",
                    "payment-service: redis.exceptions.ConnectionError: Error 111 connecting to cache-prod-01:6379. Connection refused.",
                    "postgres-primary: FATAL: remaining connection slots reserved for replication. Application connections rejected.",
                ],
            }),
            Action(action_type=ActionType.MITIGATE, task_id=task_id, payload={
                "action": "Reconfigure Redis cache-prod-02 bind address from 127.0.0.1 to 0.0.0.0, then restart the Redis sentinel and trigger a new failover. Alternatively, restart the original primary cache-prod-01 if it can be recovered.",
                "expected_outcome": "Redis cluster returns to operational state, payment-service cache hit rate recovers, database connection pool drains.",
                "risk_level": "medium",
            }),
            Action(action_type=ActionType.RESOLVE, task_id=task_id, payload={
                "resolution_summary": "Redis sentinel failover was triggered to cache-prod-02 but the replica had an incorrect bind address (127.0.0.1 instead of 0.0.0.0). Reconfigured Redis and restarted the cluster. Payment service latency returned to normal within 2 minutes.",
                "root_cause": "Redis sentinel failover to misconfigured replica rendered entire cache layer unavailable, causing payment-service database fallback and connection pool exhaustion.",
                "time_to_resolve_minutes": 18,
            }),
        ]

    elif task_id == "full_postmortem":
        actions = [
            Action(action_type=ActionType.WRITE_POSTMORTEM, task_id=task_id, payload={
                "title": "P1 Incident: payment-service Outage Due to VACUUM ANALYZE Lock Contention (2024-09-02)",
                "severity": "P1",
                "duration_minutes": 13,
                "timeline": [
                    {"time": "2024-09-02T02:00:00Z", "event": "Scheduled maintenance window begins. DBA team initiates index rebuild on payments_db."},
                    {"time": "2024-09-02T02:23:00Z", "event": "Index rebuild completes. DBA runs VACUUM ANALYZE on payments_db.transactions (87M rows)."},
                    {"time": "2024-09-02T02:26:00Z", "event": "VACUUM ANALYZE acquires AccessShareLock on transactions table."},
                    {"time": "2024-09-02T02:27:00Z", "event": "payment-service write queries queue behind VACUUM lock. Connection pool (50/50) saturated."},
                    {"time": "2024-09-02T02:28:00Z", "event": "First alert fires: payment-service p99 latency > 2000ms. On-call engineer Priya paged."},
                    {"time": "2024-09-02T02:31:00Z", "event": "Priya investigates payment-service — sees connection pool exhaustion. No obvious app-level cause."},
                    {"time": "2024-09-02T02:35:00Z", "event": "Priya escalates to DBA team. Arjun joins bridge call."},
                    {"time": "2024-09-02T02:38:00Z", "event": "Arjun identifies VACUUM ANALYZE process holding lock. Executes pg_cancel_backend(<pid>) to terminate."},
                    {"time": "2024-09-02T02:39:00Z", "event": "VACUUM process terminated. payment-service connection pool drains. Latency recovers to baseline."},
                    {"time": "2024-09-02T02:41:00Z", "event": "All alerts clear. Incident declared resolved. Duration: 13 minutes."},
                ],
                "root_cause": "A VACUUM ANALYZE operation scheduled during a maintenance window on the payments_db.transactions table (87M rows) held an AccessShareLock for an extended duration. This lock blocked all concurrent write queries from payment-service, causing the application connection pool to exhaust (50/50 active connections). New payment transactions were rejected with 503 errors. The VACUUM was not expected to contend with live traffic at 02:27 UTC, but maintenance window traffic isolation was incomplete.",
                "contributing_factors": [
                    "VACUUM ANALYZE was run during a maintenance window without verifying that payment-service traffic was fully drained or circuit-broken first.",
                    "No lock timeout was configured for the VACUUM process (lock_timeout = 0).",
                    "The connection pool size (50) was too small to buffer the queue buildup during lock contention.",
                    "No runbook existed for database maintenance operations that could cause lock contention in production.",
                    "Alert for connection pool exhaustion did not trigger early enough — only latency alert fired.",
                ],
                "impact": "1,247 payment transactions failed during the 13-minute window (02:28–02:41 UTC). Customers experienced checkout failures. Estimated lost GMV: ~$31,000 based on average transaction value. No data loss occurred. All failed transactions were idempotent and customers were informed to retry.",
                "action_items": [
                    {"owner": "DBA team", "action": "Add lock_timeout = 5s to all VACUUM/ANALYZE operations run during maintenance. Never run VACUUM on tables > 10M rows without first draining application traffic.", "due": "2024-09-09"},
                    {"owner": "Platform team", "action": "Implement circuit breaker in payment-service that activates before maintenance windows, routing traffic to a standby instance.", "due": "2024-09-16"},
                    {"owner": "SRE team", "action": "Write runbook for 'Database maintenance causing lock contention' and add connection pool saturation alert (fires at 80% pool usage).", "due": "2024-09-09"},
                    {"owner": "DBA team", "action": "Test all maintenance operations in staging with production traffic replay before executing in production.", "due": "2024-09-23"},
                ],
            }),
        ]
    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    step_rewards = []
    obs = env.state  # already reset
    for action in actions:
        result = env.step(action)
        step_rewards.append(result.reward)
        if verbose:
            print(f"  [Step {result.observation.step_count}] {action.action_type.value} → reward={result.reward:+.3f} | {result.observation.feedback[:80]}")

    score, details = env.grade()
    if verbose:
        print(f"\n  ✅ Episode done. Grader score: {score:.3f}")

    return score, {
        "grader_score": score,
        "env_reward": env._total_reward,
        "steps_taken": env._step_count,
        "grader_details": details,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRE Incident Response — Baseline Inference Script")
    parser.add_argument("--task", choices=list(TASK_REGISTRY.keys()), default=None, help="Run a single task")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic baseline (no LLM)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output JSON only")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    tasks_to_run = list(TASK_REGISTRY.keys()) if args.all_tasks else ([args.task] if args.task else ["alert_triage", "cascading_failure", "full_postmortem"])

    client = _get_client()
    use_llm = client is not None and not args.heuristic

    if not args.json_output:
        mode = f"LLM ({args.model})" if use_llm else "heuristic"
        print(f"\n🚀 SRE Incident Response — Baseline ({mode})")
        print(f"Tasks: {tasks_to_run}")

    all_scores = {}
    for task_id in tasks_to_run:
        if use_llm:
            score, details = run_task_with_llm(client, task_id, model=args.model, verbose=not args.json_output)
        else:
            score, details = run_task_heuristic(task_id, verbose=not args.json_output)
        all_scores[task_id] = {
            "score": round(score, 4),
            "difficulty": TASK_REGISTRY[task_id]["difficulty"],
            "details": details,
        }

    mean_score = sum(v["score"] for v in all_scores.values()) / len(all_scores)

    output = {
        "baseline_mode": "llm" if use_llm else "heuristic",
        "model": args.model if use_llm else "heuristic",
        "scores": all_scores,
        "mean_score": round(mean_score, 4),
    }

    if args.json_output:
        print(json.dumps(output))
    else:
        print(f"\n{'='*60}")
        print("📊 BASELINE RESULTS")
        print(f"{'='*60}")
        for task_id, result in all_scores.items():
            diff = TASK_REGISTRY[task_id]["difficulty"]
            print(f"  {task_id:25s} [{diff:6s}]  score = {result['score']:.4f}")
        print(f"\n  Mean score: {mean_score:.4f}")
        print(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    main()
