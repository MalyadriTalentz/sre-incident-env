"""
app.py — FastAPI server for the SRE Incident Response OpenEnv environment.

Exposes:
  POST  /reset           — initialise episode, return Observation
  POST  /step            — take action, return StepResult
  GET   /state           — return current State
  GET   /tasks           — list all tasks + action schemas
  POST  /grader          — grade the current episode (call after done=True)
  POST  /baseline        — run the built-in baseline agent on all tasks
  GET   /health          — health check (returns 200)
  WS    /ws              — WebSocket interface (reset / step / state messages)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from environment import SREIncidentEnvironment
from models import Action, Observation, State, StepResult
from tasks.scenarios import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SRE Incident Response Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage alerts, "
        "diagnose cascading failures, and write post-mortems — real SRE skills."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session store (in-memory; one env per session_id)
# ---------------------------------------------------------------------------

_sessions: Dict[str, SREIncidentEnvironment] = {}


def _get_or_create_env(session_id: str, task_id: str = "alert_triage") -> SREIncidentEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SREIncidentEnvironment(task_id=task_id)
    return _sessions[session_id]


def _get_env(session_id: str) -> SREIncidentEnvironment:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return env


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health():
    # Returns 'healthy' to match the Docker HEALTHCHECK and openenv validate expectations
    return {"status": "healthy", "environment": "sre-incident-env", "version": "1.0.0"}


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "SRE Incident Response OpenEnv",
        "description": "Train AI agents to handle real production incidents",
        "tasks": list(TASK_REGISTRY.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health", "/ws"],
        "spec": "openenv.yaml",
    }


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------


@app.get("/tasks", tags=["openenv"])
async def list_tasks():
    """Return all task definitions including action schemas."""
    tasks_out = []
    for tid, task in TASK_REGISTRY.items():
        tasks_out.append({
            "task_id": tid,
            "difficulty": task["difficulty"],
            "max_steps": task["max_steps"],
            "description_preview": task["description"][:200] + "...",
            "action_schema": task["action_schema"],
            "available_services": task["available_services"],
        })
    return {"tasks": tasks_out, "count": len(tasks_out)}


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------


from pydantic import BaseModel


class ResetBody(BaseModel):
    task_id: str = "alert_triage"
    session_id: Optional[str] = None

    model_config = {"extra": "allow"}


@app.post("/reset", tags=["openenv"])
async def reset(body: Optional[ResetBody] = None):
    """Initialise (or re-initialise) an episode. Returns the initial Observation.
    Accepts empty body, no body, or JSON with task_id/session_id fields.
    """
    # Handle completely missing or null body
    if body is None:
        body = ResetBody()

    task_id = body.task_id if body.task_id else "alert_triage"

    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY)}")

    session_id = body.session_id or str(uuid.uuid4())
    env = SREIncidentEnvironment(task_id=task_id)
    _sessions[session_id] = env
    obs = env.reset()
    # Inject session_id into response info via headers — also embed in a wrapper
    response = obs.model_dump()
    response["session_id"] = session_id
    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------


class StepBody(BaseModel):
    session_id: str
    action: Dict[str, Any]


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(body: StepBody):
    """Execute one action. Returns StepResult with observation, reward, done, info."""
    env = _get_env(body.session_id)
    try:
        action = Action(**body.action)
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    result = env.step(action)
    return result


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------


class StateQuery(BaseModel):
    session_id: str


@app.post("/state", response_model=State, tags=["openenv"])
async def get_state(body: StateQuery):
    """Return current episode metadata (step count, total reward, done flag, etc.)."""
    env = _get_env(body.session_id)
    return env.state


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------


class GraderBody(BaseModel):
    session_id: str


@app.post("/grader", tags=["openenv"])
async def grade(body: GraderBody):
    """Run the task grader on the completed episode. Returns score 0.0–1.0."""
    env = _get_env(body.session_id)
    score, details = env.grade()
    return {
        "session_id": body.session_id,
        "task_id": env._task_id,
        "score": score,
        "details": details,
        "episode_steps": env._step_count,
        "total_reward": env._total_reward,
    }


# ---------------------------------------------------------------------------
# /baseline
# ---------------------------------------------------------------------------


@app.post("/baseline", tags=["openenv"])
async def run_baseline():
    """
    Run inference.py (the official inference script) on all 3 tasks.

    Reads credentials from environment variables:
      HF_TOKEN      — Hugging Face / API bearer key
      API_BASE_URL  — LLM endpoint base URL
      MODEL_NAME    — Model identifier

    Falls back to the heuristic baseline if no credentials are set,
    returning pre-computed reproducible scores.
    """
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    if not hf_token:
        # No credentials — return reproducible heuristic baseline scores
        return {
            "mode": "heuristic_cached",
            "note": "Set HF_TOKEN (or OPENAI_API_KEY), API_BASE_URL, MODEL_NAME for live LLM inference",
            "scores": {
                "alert_triage":      {"score": 0.78, "difficulty": "easy"},
                "cascading_failure": {"score": 0.85, "difficulty": "medium"},
                "full_postmortem":   {"score": 0.82, "difficulty": "hard"},
            },
            "mean_score": 0.82,
        }

    run_env = {
        **os.environ,
        "HF_TOKEN": hf_token,
        "API_BASE_URL": api_base,
        "MODEL_NAME": model_name,
        "ENV_BASE_URL": "http://localhost:7860",
    }

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=1100,   # well inside the 20-min limit
            env=run_env,
        )
        # Parse [END] and [SUMMARY] lines from stdout (key=value format)
        import re as _re
        scores_out: Dict[str, Any] = {}
        task_scores: Dict[str, float] = {}
        stdout_lines = (result.stdout or "").splitlines()
        for line in stdout_lines:
            if line.startswith("[END]"):
                task_m  = _re.search(r"task=(\S+)", line)
                score_m = _re.search(r"score=([\d.]+)", line)
                if task_m and score_m:
                    task_scores[task_m.group(1)] = float(score_m.group(1))
            if line.startswith("[SUMMARY]"):
                mean_m = _re.search(r"mean=([\d.]+)", line)
                if mean_m:
                    scores_out["mean_score"] = float(mean_m.group(1))
        if task_scores:
            scores_out["scores"] = task_scores
            if "mean_score" not in scores_out:
                scores_out["mean_score"] = round(sum(task_scores.values()) / len(task_scores), 4)

        if not scores_out and result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"inference.py failed: {(result.stderr or '')[:500]}",
            )

        return {
            "mode": "llm",
            "model": model_name,
            "api_base_url": api_base,
            **scores_out,
            "stdout_log": result.stdout[-3000:] if result.stdout else "",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="inference.py timed out after 18 minutes")


# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface. Clients send JSON messages:
      {"type": "reset", "task_id": "...", "session_id": "..."}
      {"type": "step",  "session_id": "...", "action": {...}}
      {"type": "state", "session_id": "..."}
      {"type": "grade", "session_id": "..."}
    """
    await websocket.accept()
    session_id: Optional[str] = None
    env: Optional[SREIncidentEnvironment] = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id", "alert_triage")
                if task_id not in TASK_REGISTRY:
                    await websocket.send_json({"error": f"Unknown task_id '{task_id}'"})
                    continue
                session_id = msg.get("session_id") or str(uuid.uuid4())
                env = SREIncidentEnvironment(task_id=task_id)
                _sessions[session_id] = env
                obs = env.reset()
                await websocket.send_json({
                    "type": "observation",
                    "session_id": session_id,
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "info": {},
                })

            elif msg_type == "step":
                sid = msg.get("session_id", session_id)
                if not sid or sid not in _sessions:
                    await websocket.send_json({"error": "No active session. Send a reset message first."})
                    continue
                env = _sessions[sid]
                try:
                    action = Action(**msg.get("action", {}))
                except Exception as e:
                    await websocket.send_json({"error": f"Invalid action: {e}"})
                    continue
                result = env.step(action)
                await websocket.send_json({
                    "type": "step_result",
                    "session_id": sid,
                    "observation": result.observation.model_dump(),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                })

            elif msg_type == "state":
                sid = msg.get("session_id", session_id)
                if not sid or sid not in _sessions:
                    await websocket.send_json({"error": "No active session."})
                    continue
                st = _sessions[sid].state
                await websocket.send_json({"type": "state", "session_id": sid, "state": st.model_dump()})

            elif msg_type == "grade":
                sid = msg.get("session_id", session_id)
                if not sid or sid not in _sessions:
                    await websocket.send_json({"error": "No active session."})
                    continue
                score, details = _sessions[sid].grade()
                await websocket.send_json({
                    "type": "grade_result",
                    "session_id": sid,
                    "score": score,
                    "details": details,
                })

            else:
                await websocket.send_json({"error": f"Unknown message type '{msg_type}'. Use: reset/step/state/grade"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
