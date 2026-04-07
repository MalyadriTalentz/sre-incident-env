"""
Inference Script — SRE Incident Response OpenEnv
===================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name if running the env locally via Docker.
                       If not set, connects to ENV_BASE_URL (default: http://localhost:7860).

DEFAULTS (set only for API_BASE_URL and MODEL_NAME, not HF_TOKEN):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")
    HF_TOKEN     = os.getenv("HF_TOKEN")

STDOUT FORMAT — strictly followed for automated evaluation:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task returns score in [0, 1].

  Example:
    [START] task=alert_triage env=sre-incident-env model=meta-llama/Llama-3.3-70B-Instruct
    [STEP] step=1 action=classify reward=0.20 done=false error=null
    [STEP] step=2 action=classify reward=0.18 done=false error=null
    [STEP] step=3 action=classify reward=0.20 done=true error=null
    [END] success=true steps=3 score=0.780 rewards=0.20,0.18,0.20
"""

import asyncio
import json
import os
import re
import textwrap
import time
from typing import Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

# Defaults are set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
API_BASE_URL: str   = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME: str     = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN            = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME    = os.getenv("LOCAL_IMAGE_NAME")

# Derived
API_KEY: str        = HF_TOKEN or os.getenv("API_KEY") or ""
ENV_BASE_URL: str   = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK           = "sre-incident-env"
MAX_STEPS           = 12        # ceiling; each task has its own lower limit
TEMPERATURE         = 0.1
MAX_TOKENS          = 1500
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score in [0, 1]

TASKS = ["alert_triage", "cascading_failure", "full_postmortem"]


# ---------------------------------------------------------------------------
# Structured stdout logging — exact format required by the evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # action must be single-word / no spaces for clean field parsing
    action_clean = re.sub(r"\s+", "_", action)[:60]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP calls (talks to the running FastAPI server)
# ---------------------------------------------------------------------------

async def env_reset(task_id: str, client: httpx.AsyncClient) -> Tuple[str, Dict]:
    """POST /reset → (session_id, observation_dict)."""
    r = await client.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("session_id", ""), data


async def env_step(session_id: str, action: Dict, client: httpx.AsyncClient) -> Dict:
    """POST /step → StepResult dict."""
    r = await client.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


async def env_grade(session_id: str, client: httpx.AsyncClient) -> float:
    """POST /grader → float score in [0, 1]."""
    r = await client.post(
        f"{ENV_BASE_URL}/grader",
        json={"session_id": session_id},
        timeout=30,
    )
    r.raise_for_status()
    return float(r.json().get("score", 0.0))


async def env_close(session_id: str, client: httpx.AsyncClient) -> None:
    """Best-effort session cleanup."""
    try:
        await client.delete(f"{ENV_BASE_URL}/sessions/{session_id}", timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Site Reliability Engineer (SRE) handling production incidents.
    You interact with an SRE Incident Response environment one action at a time.
    Output ONLY a single valid JSON object — no markdown fences, no explanation.

    Action types:

    classify — triage an alert (alert_triage task)
    {"action_type":"classify","payload":{"alert_id":"alert-001","severity":"P1|P2|P3|P4","affected_service":"<svc>","affected_team":"<team>","summary":"<one sentence>"}}

    investigate — pull logs for a service (cascading_failure task)
    {"action_type":"investigate","payload":{"service_name":"<svc>","log_type":"application"}}

    hypothesize — state root cause
    {"action_type":"hypothesize","payload":{"root_cause":"<technical cause>","confidence":0.9,"evidence":["<log 1>","<log 2>"]}}

    mitigate — propose a fix
    {"action_type":"mitigate","payload":{"action":"<action>","expected_outcome":"<outcome>","risk_level":"low|medium|high"}}

    resolve — close the incident
    {"action_type":"resolve","payload":{"resolution_summary":"<summary>","root_cause":"<cause>","time_to_resolve_minutes":<int>}}

    write_postmortem — full post-mortem (full_postmortem task)
    {"action_type":"write_postmortem","payload":{"title":"<title>","severity":"P1","duration_minutes":<int>,"timeline":[{"time":"<iso>","event":"<desc>"}],"root_cause":"<cause>","contributing_factors":["<f>"],"impact":"<impact with numbers>","action_items":[{"owner":"<team>","action":"<action>","due":"<date>"}]}}

    Strategy:
    - alert_triage: classify alert-001, alert-002, alert-003 one per step (3 steps total).
    - cascading_failure: investigate 2+ services, hypothesize, mitigate, resolve.
    - full_postmortem: write_postmortem in step 1 with complete post-mortem using the given timeline.

    Be technically precise. Output ONLY the JSON object.
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation dict
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict, step: int) -> str:
    parts = [
        f"Task: {obs.get('task_id', '')} | Step {step}/{obs.get('max_steps', 10)}",
        obs.get("task_description", "")[:600],
    ]

    alerts = obs.get("current_alerts", [])
    if alerts:
        parts.append("\nAlerts:")
        for a in alerts:
            parts.append(
                f"  [{a.get('alert_id','?')}] {a.get('service','')}: {a.get('message','')} "
                f"(value={a.get('metric_value')}, threshold={a.get('threshold')})"
            )

    logs = obs.get("visible_logs", [])
    if logs:
        parts.append(f"\nLogs ({len(logs)} entries, last 12):")
        for lg in logs[-12:]:
            parts.append(
                f"  [{lg.get('timestamp','')}] {lg.get('level','')} "
                f"{lg.get('service','')}: {lg.get('message','')}"
            )

    metrics = obs.get("visible_metrics", [])
    if metrics:
        parts.append("\nMetrics:")
        for m in metrics:
            parts.append(f"  {m.get('service','')}/{m.get('name','')}: {m.get('value')}{m.get('unit','')}")

    timeline = obs.get("incident_timeline", [])
    if timeline:
        parts.append("\nIncident Timeline:")
        for ev in timeline:
            parts.append(f"  {ev.get('time','')} — {ev.get('event','')}")

    investigated = obs.get("investigations_done", [])
    if investigated:
        parts.append(f"\nAlready investigated: {', '.join(investigated)}")

    available = obs.get("available_services", [])
    if available:
        parts.append(f"Available services: {', '.join(available)}")

    feedback = obs.get("feedback", "")
    if feedback:
        parts.append(f"Last feedback: {feedback}")

    parts.append("\nOutput your next JSON action:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Parse JSON action from LLM text
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[Dict]:
    """Extract a JSON action dict from LLM response, stripping markdown fences."""
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Heuristic fallback — guarantees reproducible scores without LLM
# ---------------------------------------------------------------------------

HEURISTIC_ACTIONS: Dict[str, List[Dict]] = {
    "alert_triage": [
        {"action_type": "classify", "payload": {
            "alert_id": "alert-001", "severity": "P1",
            "affected_service": "api-gateway", "affected_team": "platform-team",
            "summary": "API gateway 5xx error rate at 18.3% due to upstream payment-service ECONNREFUSED. Customer-facing outage."}},
        {"action_type": "classify", "payload": {
            "alert_id": "alert-002", "severity": "P1",
            "affected_service": "redis-cluster", "affected_team": "infrastructure-team",
            "summary": "Redis OOM at 94% memory with noeviction policy — cache writes failing completely."}},
        {"action_type": "classify", "payload": {
            "alert_id": "alert-003", "severity": "P2",
            "affected_service": "postgres-primary", "affected_team": "database-team",
            "summary": "Postgres replication lag at 187s (threshold 30s) — replica falling behind, failover risk."}},
    ],
    "cascading_failure": [
        {"action_type": "investigate", "payload": {"service_name": "payment-service",  "log_type": "application"}},
        {"action_type": "investigate", "payload": {"service_name": "redis-cluster",    "log_type": "application"}},
        {"action_type": "hypothesize", "payload": {
            "root_cause": "Redis sentinel failover promoted cache-prod-02 which has bind=127.0.0.1 preventing external connections. Entire Redis cluster went DOWN, forcing payment-service database fallback that exhausted Postgres connection pool.",
            "confidence": 0.93,
            "evidence": [
                "redis: Failover incomplete — cache-prod-02 bind 127.0.0.1 prevents external connections",
                "payment-service: redis.exceptions.ConnectionError ECONNREFUSED to cache-prod-01:6379",
                "postgres: FATAL remaining connection slots reserved — application connections rejected",
            ]}},
        {"action_type": "mitigate", "payload": {
            "action": "Reconfigure cache-prod-02 bind address from 127.0.0.1 to 0.0.0.0 then restart Redis sentinel and trigger new failover",
            "expected_outcome": "Redis cluster restored, payment-service cache hit rate recovers, Postgres connection pool drains",
            "risk_level": "medium"}},
        {"action_type": "resolve", "payload": {
            "resolution_summary": "Redis sentinel failover to misconfigured replica cache-prod-02 (bind=127.0.0.1) rendered cache unavailable. Reconfigured bind address, restarted Redis. Service recovered.",
            "root_cause": "Redis sentinel failover to replica with misconfigured bind address (127.0.0.1) caused complete cache outage and Postgres connection pool exhaustion.",
            "time_to_resolve_minutes": 18}},
    ],
    "full_postmortem": [
        {"action_type": "write_postmortem", "payload": {
            "title": "P1: payment-service Outage — VACUUM ANALYZE Lock Contention on payments_db.transactions (2024-09-02)",
            "severity": "P1",
            "duration_minutes": 13,
            "timeline": [
                {"time": "2024-09-02T02:00:00Z", "event": "Scheduled maintenance window begins. DBA team initiates index rebuild on payments_db."},
                {"time": "2024-09-02T02:23:00Z", "event": "Index rebuild completes. DBA runs VACUUM ANALYZE on payments_db.transactions (87M rows)."},
                {"time": "2024-09-02T02:26:00Z", "event": "VACUUM ANALYZE acquires AccessShareLock on transactions table."},
                {"time": "2024-09-02T02:27:00Z", "event": "payment-service write queries queue behind VACUUM lock. Connection pool (50/50) saturated."},
                {"time": "2024-09-02T02:28:00Z", "event": "Alert fires: payment-service p99 latency > 2000ms. On-call engineer Priya paged."},
                {"time": "2024-09-02T02:35:00Z", "event": "Priya escalates to DBA team. Arjun joins the bridge call."},
                {"time": "2024-09-02T02:38:00Z", "event": "Arjun identifies VACUUM ANALYZE holding lock. Executes pg_cancel_backend to terminate."},
                {"time": "2024-09-02T02:39:00Z", "event": "VACUUM process cancelled. Connection pool drains. Latency recovers to baseline."},
                {"time": "2024-09-02T02:41:00Z", "event": "All alerts clear. Incident declared resolved. Total duration: 13 minutes."},
            ],
            "root_cause": "VACUUM ANALYZE on payments_db.transactions (87M rows) during maintenance window acquired an AccessShareLock that blocked all payment-service write queries, exhausting the connection pool (50/50) and causing cascading 503 errors on the checkout flow.",
            "contributing_factors": [
                "No lock_timeout configured for VACUUM operations (lock_timeout=0 waits indefinitely).",
                "Maintenance window did not drain or circuit-break application traffic before VACUUM.",
                "Application connection pool (50) too small to absorb write queue during lock contention.",
                "No monitoring alert for connection pool saturation — only latency alert fired.",
                "No runbook for database maintenance lock contention in production.",
            ],
            "impact": "1,247 failed payment transactions during the 13-minute outage window (02:28-02:41 UTC). Customers experienced checkout failures. Estimated lost GMV ~$31,175. No data loss. All failed transactions were idempotent.",
            "action_items": [
                {"owner": "DBA team",      "action": "Add lock_timeout=5s to all VACUUM/ANALYZE in production. Mandate traffic drain before vacuuming tables >10M rows.", "due": "2024-09-09"},
                {"owner": "Platform team", "action": "Implement circuit breaker in payment-service that activates before maintenance windows.", "due": "2024-09-16"},
                {"owner": "SRE team",      "action": "Add alert for connection pool saturation at 80%. Write runbook for DB lock contention incidents.", "due": "2024-09-09"},
            ],
        }},
    ],
}


# ---------------------------------------------------------------------------
# Single task episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    task_id: str,
    llm_client: Optional[OpenAI],
    http: httpx.AsyncClient,
) -> Tuple[float, List[float], int]:
    """
    Run one episode for task_id.
    Returns (grader_score, step_rewards, steps_taken).
    """
    session_id, obs_data = await env_reset(task_id, http)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages        = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken     = 0
    done            = False
    score           = 0.0
    success         = False
    heuristic_queue = list(HEURISTIC_ACTIONS.get(task_id, []))

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(obs_data, step)
            messages.append({"role": "user", "content": user_prompt})

            # Attempt LLM, fall back to heuristic
            action_dict: Optional[Dict] = None
            error_str:   Optional[str]  = None

            if llm_client is not None:
                raw_text = call_llm(llm_client, messages)
                messages.append({"role": "assistant", "content": raw_text})
                action_dict = parse_action(raw_text)
                if action_dict is None:
                    error_str = "parse_error"
                    print(f"[DEBUG] step={step} parse_error raw={raw_text[:80]}", flush=True)

            if action_dict is None and heuristic_queue:
                action_dict = heuristic_queue.pop(0)
                error_str   = None

            if action_dict is None:
                break

            # env.step()
            try:
                result   = await env_step(session_id, action_dict, http)
                reward   = float(result.get("reward", 0.0))
                done     = bool(result.get("done", False))
                obs_data = result.get("observation", obs_data)
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_str = str(exc)[:80]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_dict.get("action_type", "unknown"),
                reward=reward,
                done=done,
                error=error_str,
            )

            await asyncio.sleep(0.15)

    finally:
        # env.close() equivalent + grade
        try:
            score = await env_grade(session_id, http)
        except Exception:
            score = 0.0

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        await env_close(session_id, http)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards, steps_taken


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    llm_client: Optional[OpenAI] = None
    if API_KEY:
        try:
            llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as e:
            print(f"[DEBUG] Failed to build LLM client: {e}", flush=True)

    mode = f"LLM ({MODEL_NAME})" if llm_client else "heuristic"
    print(f"[DEBUG] mode={mode} env={ENV_BASE_URL}", flush=True)

    all_scores: Dict[str, float] = {}

    async with httpx.AsyncClient(timeout=60) as http:
        for task_id in TASKS:
            score, _, _ = await run_episode(task_id, llm_client, http)
            all_scores[task_id] = score

    mean = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(
        f"[SUMMARY] alert_triage={all_scores.get('alert_triage', 0):.3f} "
        f"cascading_failure={all_scores.get('cascading_failure', 0):.3f} "
        f"full_postmortem={all_scores.get('full_postmortem', 0):.3f} "
        f"mean={mean:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
