---
title: SRE Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: true
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - sre
  - incident-response
  - agent-evaluation
  - real-world
short_description: RL environment for SRE incident response
---

# 🚨 SRE Incident Response — OpenEnv Environment

> Train AI agents to handle real production incidents: triage alerts, diagnose cascading failures, and write post-mortems.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.1-blue)](https://meta-pytorch.org/OpenEnv/)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## 🧠 Why This Environment?

Every software company has on-call SRE engineers who respond to production incidents. This is one of the highest-stakes, most cognitively demanding tasks in software engineering — and one of the least explored in the RL/agent training space.

The SRE Incident Response environment fills this gap:

- **Real-world domain**: Monitoring alerts, cascading failures, and post-mortems are genuine SRE artifacts from production systems.
- **Partial observability**: Logs are hidden behind investigation gates — agents must decide *where* to look, exactly like a real engineer.
- **Dense rewards**: Every investigative step yields signal; reward isn't just sparse end-of-episode.
- **Multi-task difficulty progression**: Three tasks from easy (alert classification) to hard (full post-mortem writing).
- **Novel**: No existing OpenEnv environment covers SRE incident management.

---

## 🏗️ Environment Description

The environment simulates a **production SRE workflow** with three distinct tasks:

### Task 1 — Alert Triage (Easy)
**Scenario**: Three monitoring alerts fire simultaneously in a production payment platform.

| Alert | Service | Key Signal |
|-------|---------|-----------|
| alert-001 | api-gateway | HTTP 5xx rate at 18.3% (threshold: 5%) |
| alert-002 | redis-cluster | Memory at 94% with `noeviction` policy — OOM |
| alert-003 | postgres-primary | Replication lag at 187s (threshold: 30s) |

**Agent goal**: Use `classify` actions to assign correct severity (P1–P4), owning team, and write a summary for each alert.

**Expected difficulty**: A competent LLM scores ~0.72.

---

### Task 2 — Cascading Failure Diagnosis (Medium)

**Scenario**: The checkout flow is broken. Payment service p99 latency is 12,400ms (SLO: 500ms). Three alerts have fired. The root cause is hidden — the agent must *investigate* services to reveal logs.

**Information architecture (partial observability)**:

```
Alerts (visible)  →  investigate("payment-service")  →  Redis connection errors revealed
                  →  investigate("redis-cluster")     →  Misconfigured sentinel failover revealed  ← ROOT CAUSE
                  →  investigate("postgres-primary")  →  Connection pool exhaustion revealed
```

**True root cause**: A Redis sentinel failover promoted a replica (`cache-prod-02`) with `bind = 127.0.0.1`, making it unreachable to external clients. This rendered the entire cache layer unavailable, forcing payment-service to fall back to Postgres for every request, exhausting the connection pool.

**Agent goal**: Investigate → Hypothesize → Mitigate → Resolve.

**Expected difficulty**: GPT-4o-mini scores ~0.51 (needs multi-hop reasoning).

---

### Task 3 — Full Post-mortem Writing (Hard)

**Scenario**: A 13-minute P1 incident occurred at 02:28 UTC. All logs, the incident timeline, and metrics are provided upfront. The agent must write a complete, structured post-mortem.

**True root cause**: A `VACUUM ANALYZE` scheduled during a maintenance window acquired an `AccessShareLock` on the `payments_db.transactions` table (87M rows). This blocked payment-service write queries, saturated the connection pool (50/50), and caused 1,247 payment failures.

**Post-mortem requirements**:
- Title, severity, duration (in minutes)
- Timeline with ≥ 5 chronological events
- Technical root cause (must mention VACUUM/lock)
- Contributing factors
- Customer impact with numbers
- ≥ 2 action items with owner + action + due date

**Expected difficulty**: GPT-4o-mini scores ~0.38 (long-form technical writing is hard).

---

## 📦 Observation Space

```python
class Observation(BaseModel):
    task_id: str                          # Active task
    task_description: str                 # Full task instructions
    step_count: int                       # Current step number
    max_steps: int                        # Episode step limit
    current_alerts: List[Alert]           # Firing monitoring alerts
    available_services: List[str]         # Services the agent can investigate
    visible_logs: List[LogEntry]          # Logs revealed through investigation
    visible_metrics: List[MetricPoint]    # Current metric values
    team_responses: List[str]             # Simulated team messages
    incident_timeline: List[Dict]         # Chronological events (hard task only)
    investigations_done: List[str]        # Already-investigated services
    feedback: str                         # Feedback from last action
    is_done: bool
```

## ⚡ Action Space

```python
class Action(BaseModel):
    action_type: ActionType   # classify | investigate | hypothesize | mitigate | escalate | resolve | write_postmortem
    payload: Dict[str, Any]   # Type-specific fields (see /tasks endpoint)
    task_id: Optional[str]    # Must match active task if provided
```

### Action payloads by type:

| Action | Key payload fields |
|--------|--------------------|
| `classify` | `alert_id`, `severity` (P1–P4), `affected_service`, `affected_team`, `summary` |
| `investigate` | `service_name`, `log_type` (application/system/network/database) |
| `hypothesize` | `root_cause`, `confidence` (0–1), `evidence` (list) |
| `mitigate` | `action`, `expected_outcome`, `risk_level` (low/medium/high) |
| `escalate` | `team`, `reason`, `priority` |
| `resolve` | `resolution_summary`, `root_cause`, `time_to_resolve_minutes` |
| `write_postmortem` | `title`, `severity`, `duration_minutes`, `timeline`, `root_cause`, `contributing_factors`, `impact`, `action_items` |

---

## 🎯 Reward Function

The reward is **dense and shaped** — every step yields signal:

| Event | Reward |
|-------|--------|
| First-time investigation (any service) | +0.05 |
| Investigating the root-cause service | +0.10 additional |
| Hypothesis with strong keyword coverage | +0.02 to +0.20 |
| Valid mitigation action | +0.03 to +0.12 |
| Accurate resolution | +0.05 to +0.15 |
| Repeating same action type (loop) | −0.05 |
| Exceeding max_steps | −0.02 per extra step |

**Terminal grader score** (end of episode): 0.0–1.0, computed deterministically from the episode history.

---

## 🏆 Grader Scoring

Each task grader is deterministic and programmatic:

### alert_triage grader
- **Severity accuracy** (40%): Correct P1/P2/P3/P4 assignment
- **Team assignment** (30%): Fuzzy-matched against correct team slug
- **Summary quality** (30%): Keyword coverage of ground-truth alert keywords

### cascading_failure grader
- **Investigation coverage** (15%): Key services investigated
- **Root cause accuracy** (40%): Keyword coverage of true root cause
- **Mitigation quality** (25%): Coverage of correct mitigation strategy
- **Resolution quality** (20%): Accurate root cause in resolve action

### full_postmortem grader
- **Severity** (10%): Must be P1
- **Root cause accuracy** (30%): Must mention VACUUM/lock/postgres
- **Impact accuracy** (20%): Must include customer impact numbers
- **Timeline completeness** (20%): ≥5 events, chronologically sorted
- **Action item quality** (20%): Specific items with owner + action + due

---

## 🚀 Setup & Usage

### Local development

```bash
git clone <your-repo>
cd sre-incident-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t sre-incident-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... sre-incident-env
```

### Python client (HTTP)

```python
import httpx

BASE = "http://localhost:7860"

# Reset
r = httpx.post(f"{BASE}/reset", json={"task_id": "alert_triage"})
data = r.json()
session_id = data["session_id"]

# Step
r = httpx.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {
        "action_type": "classify",
        "payload": {
            "alert_id": "alert-001",
            "severity": "P1",
            "affected_service": "api-gateway",
            "affected_team": "platform-team",
            "summary": "18.3% HTTP 5xx rate on api-gateway from upstream payment-service failure."
        }
    }
})
result = r.json()
print(f"Reward: {result['reward']}, Done: {result['done']}")

# Grade the episode
r = httpx.post(f"{BASE}/grader", json={"session_id": session_id})
print(r.json())
```

### WebSocket client

```python
import asyncio, json, websockets

async def run():
    async with websockets.connect("ws://localhost:7860/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "task_id": "cascading_failure"}))
        msg = json.loads(await ws.recv())
        session_id = msg["session_id"]

        # Investigate
        await ws.send(json.dumps({
            "type": "step",
            "session_id": session_id,
            "action": {"action_type": "investigate", "payload": {"service_name": "redis-cluster", "log_type": "application"}}
        }))
        result = json.loads(await ws.recv())
        print(f"Logs revealed: {len(result['observation']['visible_logs'])}")

asyncio.run(run())
```

### Run inference (official submission script)

```bash
# Required environment variables
export HF_TOKEN=hf_...              # Your Hugging Face token (or any OpenAI-compat API key)
export API_BASE_URL=https://router.huggingface.co/v1   # LLM endpoint
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct    # Model identifier

# Run all 3 tasks (server must be running on localhost:7860)
python inference.py

# Run a single task
python inference.py --task alert_triage

# Heuristic mode (no API key needed — deterministic, reproducible)
python inference.py --heuristic

# Point to a remote HF Space
python inference.py --env-url https://your-space.hf.space
```

The script emits structured logs to stdout:
```
[START] {"task_id": "alert_triage", "episode": 1, "model": "...", "env_url": "..."}
[STEP]  {"task_id": "alert_triage", "episode": 1, "step": 1, "action_type": "classify", "reward": 0.2, "total_reward": 0.2, "done": false, "feedback": "..."}
[END]   {"task_id": "alert_triage", "episode": 1, "score": 0.78, "total_reward": 0.55, "steps": 3, "done": true}
[RESULTS] {"scores": {"alert_triage": 0.78, ...}, "mean_score": 0.82, "model": "..."}
```

---

## 📊 Baseline Scores

### Heuristic baseline (deterministic, no API key required)

| Task | Difficulty | Score |
|------|-----------|-------|
| alert_triage | Easy | 0.78 |
| cascading_failure | Medium | 0.85 |
| full_postmortem | Hard | 0.82 |
| **Mean** | | **0.82** |

### LLM baseline (GPT-4o-mini)

| Task | Difficulty | Score |
|------|-----------|-------|
| alert_triage | Easy | 0.72 |
| cascading_failure | Medium | 0.51 |
| full_postmortem | Hard | 0.38 |
| **Mean** | | **0.54** |

The gap between heuristic and LLM performance is intentional: this environment is designed to challenge frontier models, particularly on the cascading failure (multi-hop reasoning under partial observability) and post-mortem (long-form technical writing) tasks.

---

## 📁 Project Structure

```
sre-incident-env/
├── openenv.yaml          ← OpenEnv spec metadata (required)
├── README.md             ← This file
├── Dockerfile            ← HF Spaces Docker SDK compatible
├── requirements.txt      ← Python dependencies
├── models.py             ← Typed Pydantic models: Action, Observation, State, StepResult
├── environment.py        ← Core environment logic: reset() / step() / state / grade()
├── app.py                ← FastAPI server: /reset /step /state /tasks /grader /baseline + WS
├── inference.py          ← Official inference script (API_BASE_URL + MODEL_NAME + HF_TOKEN)
├── baseline.py           ← Heuristic baseline (no LLM, deterministic, for local testing)
├── test_env.py           ← Unit tests (22 tests, all passing)
└── tasks/
    ├── __init__.py
    └── scenarios.py      ← Task definitions + deterministic graders (easy/medium/hard)
```

---

## 🤗 Deploying to Hugging Face Spaces

1. Create a new Space with **Docker SDK**
2. Push this repository to the Space
3. (Optional) Set `OPENAI_API_KEY` in Space secrets for live LLM baseline
4. The Space will expose the environment at `https://<your-space>.hf.space`

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built for the [OpenEnv Hackathon](https://huggingface.co/openenv) sponsored by Meta-PyTorch and Hugging Face.  
Environment design inspired by real-world SRE practices at production software companies.
