"""
environment.py — Core SRE Incident Response Environment.

Implements the OpenEnv interface:
  reset()       → Observation
  step(action)  → StepResult
  state         → State (property)

Reward function design (dense / shaped):
  - Every informative action earns a small +0.05 exploration reward
  - Investigating the root-cause service earns +0.10
  - A correct hypothesis earns up to +0.20
  - A correct mitigation earns up to +0.15
  - A valid resolve earns up to +0.15
  - Exceeding max_steps penalises by −0.02 per extra step
  - Loop detection: same action twice in a row → −0.05

This produces a non-sparse reward surface ideal for RL training.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    LogEntry,
    Observation,
    Reward,
    State,
    StepResult,
)
from tasks.scenarios import TASK_REGISTRY, _keyword_score, _fuzzy_match


class SREIncidentEnvironment:
    """
    OpenEnv-compliant SRE Incident Response Environment.

    Parameters
    ----------
    task_id : str
        One of "alert_triage", "cascading_failure", "full_postmortem".
    """

    MAX_EPISODE_STEPS_HARD_LIMIT = 30  # absolute ceiling regardless of task

    def __init__(self, task_id: str = "alert_triage"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY)}")
        self._task_id = task_id
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False

        # Episode-specific mutable state
        self._investigated_services: List[str] = []
        self._visible_logs: List[LogEntry] = []
        self._hypotheses: List[str] = []
        self._mitigations: List[str] = []
        self._resolved: bool = False
        self._postmortem_written: bool = False
        self._classified_alerts: Dict[str, Dict] = {}
        self._team_responses: List[str] = []
        self._last_action_type: Optional[ActionType] = None

        # Episode history for end-of-episode grading
        self._history: List[Tuple[Action, StepResult]] = []

        # Load task spec
        self._task = TASK_REGISTRY[task_id]
        self._max_steps = self._task["max_steps"]

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Initialise a new episode and return the initial observation."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._investigated_services = []
        self._hypotheses = []
        self._mitigations = []
        self._resolved = False
        self._postmortem_written = False
        self._classified_alerts = {}
        self._team_responses = []
        self._last_action_type = None
        self._history = []

        # Seed visible logs from task (postmortem task reveals all logs upfront)
        self._visible_logs = list(deepcopy(self._task.get("visible_logs", [])))

        return self._build_observation(feedback="Episode started. Good luck, engineer! 🚨")

    def step(self, action: Action) -> StepResult:
        """Execute one agent action, update state, compute reward, return StepResult."""
        if self._done:
            obs = self._build_observation(feedback="Episode already done. Call reset() to start a new episode.")
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "episode_done"})

        self._step_count += 1
        reward_components: Dict[str, float] = {}
        feedback_parts: List[str] = []

        # ---- Loop detection ----
        if action.action_type == self._last_action_type and action.action_type not in (
            ActionType.CLASSIFY, ActionType.HYPOTHESIZE
        ):
            reward_components["loop_penalty"] = -0.05
            feedback_parts.append("⚠️ Repeated action type detected — diversify your approach.")

        self._last_action_type = action.action_type

        # ---- Validate task_id ----
        if action.task_id and action.task_id != self._task_id:
            obs = self._build_observation(feedback=f"❌ Wrong task_id '{action.task_id}'. Active task is '{self._task_id}'.")
            return StepResult(observation=obs, reward=-0.05, done=False, info={"error": "wrong_task_id"})

        # ---- Dispatch action ----
        action_reward, action_feedback = self._dispatch_action(action)
        reward_components["action"] = action_reward
        feedback_parts.extend(action_feedback)

        # ---- Step limit enforcement ----
        if self._step_count >= self._max_steps:
            reward_components["timeout_penalty"] = -0.02
            self._done = True
            feedback_parts.append(f"⏱️ Reached max steps ({self._max_steps}). Episode ending.")

        if self._step_count >= self.MAX_EPISODE_STEPS_HARD_LIMIT:
            self._done = True

        # ---- Natural episode completion ----
        if self._resolved or self._postmortem_written:
            self._done = True

        # ---- Total reward ----
        step_reward = round(sum(reward_components.values()), 4)
        self._total_reward += step_reward

        obs = self._build_observation(feedback=" | ".join(feedback_parts))
        result = StepResult(
            observation=obs,
            reward=step_reward,
            done=self._done,
            info={
                "step": self._step_count,
                "total_reward": self._total_reward,
                "reward_components": reward_components,
                "episode_id": self._episode_id,
            },
        )
        self._history.append((action, result))
        return result

    @property
    def state(self) -> State:
        """Return current episode metadata."""
        return State(
            episode_id=self._episode_id or "",
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self._max_steps,
            total_reward=self._total_reward,
            is_done=self._done,
            investigations_performed=list(self._investigated_services),
            hypotheses_submitted=list(self._hypotheses),
            mitigations_applied=list(self._mitigations),
            resolved=self._resolved,
            postmortem_written=self._postmortem_written,
        )

    # -----------------------------------------------------------------------
    # Grader
    # -----------------------------------------------------------------------

    def grade(self) -> Tuple[float, Dict[str, Any]]:
        """Run the task grader on the completed episode. Returns (score, details)."""
        grader_fn = self._task["grader"]
        return grader_fn(self._history)

    # -----------------------------------------------------------------------
    # Internal: action dispatch
    # -----------------------------------------------------------------------

    def _dispatch_action(self, action: Action) -> Tuple[float, List[str]]:
        """Route to per-action-type handler. Returns (reward, feedback_list)."""
        at = action.action_type
        p = action.payload

        if at == ActionType.INVESTIGATE:
            return self._handle_investigate(p)
        elif at == ActionType.CLASSIFY:
            return self._handle_classify(p)
        elif at == ActionType.HYPOTHESIZE:
            return self._handle_hypothesize(p)
        elif at == ActionType.MITIGATE:
            return self._handle_mitigate(p)
        elif at == ActionType.ESCALATE:
            return self._handle_escalate(p)
        elif at == ActionType.RESOLVE:
            return self._handle_resolve(p)
        elif at == ActionType.WRITE_POSTMORTEM:
            return self._handle_postmortem(p)
        else:
            return -0.02, [f"❌ Unknown action type: {at}"]

    def _handle_investigate(self, payload: Dict) -> Tuple[float, List[str]]:
        svc = payload.get("service_name", "").strip()
        available = self._task.get("available_services", [])

        if svc not in available:
            return -0.02, [f"❌ Service '{svc}' not found. Available: {available}"]

        if svc in self._investigated_services:
            return -0.03, [f"⚠️ Already investigated '{svc}'. Investigation logs unchanged."]

        self._investigated_services.append(svc)
        hidden = self._task.get("hidden_logs", {})
        new_logs = hidden.get(svc, [])
        self._visible_logs.extend(new_logs)

        feedback = [f"🔍 Investigating {svc}... Retrieved {len(new_logs)} log entries."]
        reward = 0.05  # exploration reward

        # Extra reward for investigating the root-cause service
        rc_svc = self._task.get("grader", None)
        gt = TASK_REGISTRY[self._task_id]
        if self._task_id == "cascading_failure":
            if svc == "redis-cluster":
                reward += 0.10
                feedback.append("💡 Redis cluster logs look suspicious — investigate further!")
            elif svc == "payment-service":
                reward += 0.05
                feedback.append("📋 payment-service logs show Redis connection errors.")
        return reward, feedback

    def _handle_classify(self, payload: Dict) -> Tuple[float, List[str]]:
        """Partial credit classification for alert triage task."""
        if self._task_id != "alert_triage":
            return 0.0, ["ℹ️ Classify action is most useful in the alert_triage task."]

        from tasks.scenarios import ALERT_TRIAGE_GROUND_TRUTH, ALERT_TRIAGE_ALERTS

        # Determine which alert is being classified
        matched_id = None
        payload_str = str(payload)
        for alert in ALERT_TRIAGE_ALERTS:
            if alert.alert_id in payload_str or _fuzzy_match(
                payload.get("affected_service", ""), alert.service
            ):
                matched_id = alert.alert_id
                break

        if matched_id is None:
            return -0.02, ["❌ Could not match classification to a known alert. Include alert_id or exact service name."]

        if matched_id in self._classified_alerts:
            return -0.03, [f"⚠️ Alert {matched_id} already classified."]

        self._classified_alerts[matched_id] = payload
        gt = ALERT_TRIAGE_GROUND_TRUTH[matched_id]

        reward = 0.0
        feedback = []

        # Severity feedback
        sev = str(payload.get("severity", "")).upper()
        if sev == gt["severity"].value:
            reward += 0.12
            feedback.append(f"✅ Severity {sev} correct for {matched_id}!")
        else:
            feedback.append(f"⚠️ Severity mismatch for {matched_id} (guessed {sev}, check again)")

        # Team feedback
        team = payload.get("affected_team", "")
        if _fuzzy_match(team, gt["affected_team"], 0.6):
            reward += 0.08
            feedback.append(f"✅ Team assignment correct!")
        else:
            feedback.append(f"⚠️ Team assignment may be off.")

        return reward, feedback

    def _handle_hypothesize(self, payload: Dict) -> Tuple[float, List[str]]:
        rc = payload.get("root_cause", "")
        self._hypotheses.append(rc[:120])
        feedback = [f"📝 Hypothesis recorded: '{rc[:80]}...'"]

        if self._task_id == "cascading_failure":
            from tasks.scenarios import CASCADING_FAILURE_GROUND_TRUTH as gt
            score = _keyword_score(rc, gt["root_cause_keywords"][:3])
            if score > 0.6:
                reward = 0.15 + score * 0.05
                feedback.append("🎯 Strong hypothesis! Looks like you're on the right track.")
            elif score > 0.3:
                reward = 0.05
                feedback.append("🤔 Partial match — look deeper into the cache/database layer.")
            else:
                reward = 0.02
                feedback.append("💭 Hypothesis logged. Keep investigating.")
            return reward, feedback
        return 0.03, feedback  # small reward for engaging with the task

    def _handle_mitigate(self, payload: Dict) -> Tuple[float, List[str]]:
        action_text = payload.get("action", "")
        self._mitigations.append(action_text[:120])
        feedback = [f"🔧 Mitigation proposed: '{action_text[:80]}'"]

        if self._task_id == "cascading_failure":
            from tasks.scenarios import CASCADING_FAILURE_GROUND_TRUTH as gt
            score = _keyword_score(action_text, gt["mitigation_keywords"][:3])
            if score > 0.5:
                reward = 0.12
                feedback.append("✅ Good mitigation strategy! This should restore service.")
                self._team_responses.append("Infrastructure team: Executing mitigation. Redis cluster restarting...")
            else:
                reward = 0.03
                feedback.append("🤔 Mitigation logged. Consider focusing on the root cause service.")
            return reward, feedback
        return 0.03, feedback

    def _handle_escalate(self, payload: Dict) -> Tuple[float, List[str]]:
        team = payload.get("team", "")
        reason = payload.get("reason", "")
        self._team_responses.append(f"{team} notified: {reason[:60]}")
        return 0.03, [f"📞 Escalated to {team}. They will join shortly."]

    def _handle_resolve(self, payload: Dict) -> Tuple[float, List[str]]:
        if self._task_id == "alert_triage":
            return 0.0, ["ℹ️ Alert triage doesn't require a resolve action — just classify the alerts."]

        self._resolved = True
        rc = payload.get("root_cause", "")
        feedback = [f"✅ Incident resolved! Root cause stated: '{rc[:80]}'"]

        if self._task_id == "cascading_failure":
            from tasks.scenarios import CASCADING_FAILURE_GROUND_TRUTH as gt
            score = _keyword_score(rc, gt["resolve_keywords"][:2])
            reward = 0.08 + score * 0.07
            feedback.append(f"📊 Resolution quality score: {score:.0%}")
            return reward, feedback
        return 0.05, feedback

    def _handle_postmortem(self, payload: Dict) -> Tuple[float, List[str]]:
        if self._task_id != "full_postmortem":
            return 0.0, ["ℹ️ write_postmortem action is only scored in the full_postmortem task."]

        self._postmortem_written = True
        title = payload.get("title", "Untitled Post-mortem")

        # Preview scoring
        from tasks.scenarios import POSTMORTEM_GROUND_TRUTH as gt, grade_postmortem
        timeline = payload.get("timeline", [])
        root_cause = payload.get("root_cause", "")
        rc_score = _keyword_score(root_cause, gt["root_cause_keywords"][:3])

        feedback = [
            f"📝 Post-mortem '{title}' submitted.",
            f"Timeline entries: {len(timeline)}",
            f"Root cause keyword coverage: {rc_score:.0%}",
        ]
        reward = 0.05 + rc_score * 0.10
        return reward, feedback

    # -----------------------------------------------------------------------
    # Internal: observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self, feedback: str = "") -> Observation:
        task = self._task
        return Observation(
            task_id=self._task_id,
            task_description=task["description"],
            step_count=self._step_count,
            max_steps=self._max_steps,
            current_alerts=list(task["alerts"]),
            available_services=list(task["available_services"]),
            visible_logs=list(self._visible_logs),
            visible_metrics=list(task.get("metrics", [])),
            team_responses=list(self._team_responses),
            incident_timeline=list(task.get("incident_timeline", [])),
            investigations_done=list(self._investigated_services),
            feedback=feedback,
            is_done=self._done,
        )
