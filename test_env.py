"""
test_env.py — Unit tests for the SRE Incident Response OpenEnv environment.

Run with:  python -m pytest test_env.py -v
  or just: python test_env.py

Tests:
  - reset() returns a valid Observation
  - step() returns StepResult with correct types
  - state() returns State with correct fields
  - Reward shaping (positive for good actions, negative for loops)
  - alert_triage grader correctness
  - cascading_failure grader correctness
  - full_postmortem grader correctness
  - Episode termination conditions
  - Wrong task_id validation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import SREIncidentEnvironment
from models import Action, ActionType, Observation, State, StepResult
from tasks.scenarios import (
    TASK_REGISTRY,
    grade_alert_triage,
    grade_cascading_failure,
    grade_postmortem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task_id: str) -> SREIncidentEnvironment:
    env = SREIncidentEnvironment(task_id=task_id)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Test: reset()
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    for task_id in TASK_REGISTRY:
        env = SREIncidentEnvironment(task_id=task_id)
        obs = env.reset()
        assert isinstance(obs, Observation), f"{task_id}: reset() should return Observation"
        assert obs.task_id == task_id
        assert obs.step_count == 0
        assert obs.is_done is False
        assert obs.max_steps > 0
        print(f"  ✅ reset() [{task_id}] OK")


def test_reset_twice_produces_clean_state():
    env = SREIncidentEnvironment(task_id="alert_triage")
    env.reset()
    env.step(Action(action_type=ActionType.CLASSIFY, payload={"severity": "P1", "affected_service": "api-gateway", "affected_team": "platform-team", "summary": "test", "alert_id": "alert-001"}))
    assert env._step_count == 1
    env.reset()
    assert env._step_count == 0
    assert env._total_reward == 0.0
    assert env._investigated_services == []
    assert env._classified_alerts == {}
    print("  ✅ reset() twice produces clean state")


# ---------------------------------------------------------------------------
# Test: step()
# ---------------------------------------------------------------------------

def test_step_returns_step_result():
    env = make_env("alert_triage")
    action = Action(action_type=ActionType.CLASSIFY, payload={
        "alert_id": "alert-001",
        "severity": "P1",
        "affected_service": "api-gateway",
        "affected_team": "platform-team",
        "summary": "High 5xx error rate on api-gateway.",
    })
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.observation, Observation)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert isinstance(result.info, dict)
    assert result.observation.step_count == 1
    print("  ✅ step() returns StepResult with correct types")


def test_step_after_done_returns_zero_reward():
    env = make_env("alert_triage")
    env._done = True
    action = Action(action_type=ActionType.CLASSIFY, payload={"alert_id": "alert-001", "severity": "P1", "affected_service": "api-gateway", "affected_team": "platform-team", "summary": "x"})
    result = env.step(action)
    assert result.reward == 0.0
    assert result.done is True
    print("  ✅ step() after done returns reward=0.0")


def test_wrong_task_id_returns_negative_reward():
    env = make_env("alert_triage")
    action = Action(action_type=ActionType.CLASSIFY, task_id="cascading_failure", payload={
        "alert_id": "alert-001", "severity": "P1", "affected_service": "api-gateway",
        "affected_team": "platform-team", "summary": "x"
    })
    result = env.step(action)
    assert result.reward < 0
    print("  ✅ Wrong task_id returns negative reward")


# ---------------------------------------------------------------------------
# Test: state()
# ---------------------------------------------------------------------------

def test_state_returns_correct_metadata():
    env = make_env("cascading_failure")
    s = env.state
    assert isinstance(s, State)
    assert s.task_id == "cascading_failure"
    assert s.step_count == 0
    assert s.is_done is False
    assert s.total_reward == 0.0
    print("  ✅ state() returns correct metadata")


# ---------------------------------------------------------------------------
# Test: Reward shaping
# ---------------------------------------------------------------------------

def test_investigation_gives_positive_reward():
    env = make_env("cascading_failure")
    action = Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "payment-service", "log_type": "application"})
    result = env.step(action)
    assert result.reward > 0, f"Investigation should give positive reward, got {result.reward}"
    print(f"  ✅ Investigation reward = {result.reward:+.3f} (positive)")


def test_root_cause_investigation_gives_higher_reward():
    env = make_env("cascading_failure")
    # Investigate non-root-cause first
    r1 = env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "api-gateway"})).reward
    env2 = make_env("cascading_failure")
    # Investigate root cause
    r2 = env2.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "redis-cluster"})).reward
    assert r2 > r1, f"Root cause service should give higher reward: {r2:.3f} > {r1:.3f}"
    print(f"  ✅ Root cause investigation reward ({r2:+.3f}) > non-root ({r1:+.3f})")


def test_loop_penalty():
    env = make_env("cascading_failure")
    env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "api-gateway"}))
    # Repeat same action type (not classify/hypothesize)
    env._investigated_services = []  # reset to allow re-investigation
    env._last_action_type = ActionType.INVESTIGATE
    result = env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "api-gateway"}))
    # Should have a loop penalty component
    assert "loop_penalty" in result.info.get("reward_components", {}), "Loop penalty should be applied"
    print(f"  ✅ Loop penalty applied: {result.info['reward_components'].get('loop_penalty')}")


def test_re_investigation_negative_reward():
    env = make_env("cascading_failure")
    env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "redis-cluster"}))
    # Investigate same service again
    result = env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "redis-cluster"}))
    assert result.reward < 0, f"Re-investigation should be penalised, got {result.reward}"
    print(f"  ✅ Re-investigation penalty = {result.reward:+.3f}")


# ---------------------------------------------------------------------------
# Test: Alert triage grader
# ---------------------------------------------------------------------------

def test_alert_triage_grader_perfect_score():
    env = SREIncidentEnvironment(task_id="alert_triage")
    env.reset()
    for action in [
        Action(action_type=ActionType.CLASSIFY, payload={
            "alert_id": "alert-001", "severity": "P1",
            "affected_service": "api-gateway", "affected_team": "platform-team",
            "summary": "API gateway 5xx error rate at 18.3% — upstream payment-service ECONNREFUSED."}),
        Action(action_type=ActionType.CLASSIFY, payload={
            "alert_id": "alert-002", "severity": "P1",
            "affected_service": "redis-cluster", "affected_team": "infrastructure-team",
            "summary": "Redis memory OOM at 94% with noeviction policy — cache writes failing."}),
        Action(action_type=ActionType.CLASSIFY, payload={
            "alert_id": "alert-003", "severity": "P2",
            "affected_service": "postgres-primary", "affected_team": "database-team",
            "summary": "Postgres replication lag at 187s — replica falling behind."}),
    ]:
        env.step(action)
    score, details = env.grade()
    assert score >= 0.75, f"Perfect triage should score ≥ 0.75, got {score:.4f}: {details}"
    print(f"  ✅ Alert triage grader — perfect answer scores {score:.4f} ≥ 0.75")


def test_alert_triage_grader_wrong_severity():
    env = SREIncidentEnvironment(task_id="alert_triage")
    env.reset()
    env.step(Action(action_type=ActionType.CLASSIFY, payload={
        "alert_id": "alert-001", "severity": "P4",  # wrong severity
        "affected_service": "api-gateway", "affected_team": "platform-team",
        "summary": "Some issue"}))
    score, details = env.grade()
    alert_score = details["per_alert"]["alert-001"]["score"]
    assert alert_score < 0.50, f"Wrong severity should reduce score below 0.50, got {alert_score}"
    print(f"  ✅ Alert triage grader — wrong severity scores {alert_score:.4f} < 0.50")


def test_alert_triage_grader_zero_for_no_actions():
    env = SREIncidentEnvironment(task_id="alert_triage")
    env.reset()
    score, _ = env.grade()
    assert score == 0.0, f"No actions should score 0.0, got {score}"
    print("  ✅ Alert triage grader — no actions = 0.0")


# ---------------------------------------------------------------------------
# Test: Cascading failure grader
# ---------------------------------------------------------------------------

def test_cascading_failure_grader_correct_hypothesis():
    env = SREIncidentEnvironment(task_id="cascading_failure")
    env.reset()
    env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "redis-cluster"}))
    env.step(Action(action_type=ActionType.HYPOTHESIZE, payload={
        "root_cause": "Redis sentinel failover failed because the promoted replica had a misconfigured bind address (127.0.0.1), making it unreachable externally. This caused the entire Redis cache to become unavailable.",
        "confidence": 0.90,
        "evidence": ["Redis cluster: cache-prod-02 bind address 127.0.0.1 prevents external connections", "payment-service: ECONNREFUSED to redis"],
    }))
    env.step(Action(action_type=ActionType.RESOLVE, payload={
        "resolution_summary": "Reconfigured Redis bind address to 0.0.0.0 and restarted sentinel.",
        "root_cause": "Redis sentinel failover to misconfigured replica caused cache outage.",
        "time_to_resolve_minutes": 20,
    }))
    score, details = env.grade()
    assert score >= 0.50, f"Good cascading failure diagnosis should score ≥ 0.50, got {score:.4f}"
    print(f"  ✅ Cascading failure grader — correct answer scores {score:.4f} ≥ 0.50")


def test_cascading_failure_grader_zero_for_no_actions():
    env = SREIncidentEnvironment(task_id="cascading_failure")
    env.reset()
    score, _ = env.grade()
    assert score == 0.0
    print("  ✅ Cascading failure grader — no actions = 0.0")


# ---------------------------------------------------------------------------
# Test: Postmortem grader
# ---------------------------------------------------------------------------

def test_postmortem_grader_full_score():
    env = SREIncidentEnvironment(task_id="full_postmortem")
    env.reset()
    env.step(Action(action_type=ActionType.WRITE_POSTMORTEM, payload={
        "title": "P1: payment-service outage due to VACUUM ANALYZE lock contention",
        "severity": "P1",
        "duration_minutes": 13,
        "timeline": [
            {"time": "2024-09-02T02:00:00Z", "event": "DBA runs index rebuild during maintenance."},
            {"time": "2024-09-02T02:23:00Z", "event": "VACUUM ANALYZE started on transactions table (87M rows)."},
            {"time": "2024-09-02T02:27:00Z", "event": "VACUUM acquires AccessShareLock — payment-service writes queue."},
            {"time": "2024-09-02T02:28:00Z", "event": "Alert fires: payment-service p99 latency >2000ms. On-call paged."},
            {"time": "2024-09-02T02:38:00Z", "event": "DBA cancels VACUUM process. pg_cancel_backend called."},
            {"time": "2024-09-02T02:41:00Z", "event": "Incident resolved. All alerts clear."},
        ],
        "root_cause": "VACUUM ANALYZE held an AccessShareLock on the postgres transactions table blocking payment-service write queries and exhausting the connection pool.",
        "contributing_factors": [
            "No lock_timeout configured for VACUUM operations.",
            "Maintenance window did not drain application traffic first.",
            "Connection pool (50) too small to buffer contention.",
        ],
        "impact": "1247 failed payment transactions during 13-minute window. Customers experienced checkout failures.",
        "action_items": [
            {"owner": "DBA team", "action": "Add lock_timeout=5s to all VACUUM operations. Drain traffic before vacuum on production.", "due": "2024-09-09"},
            {"owner": "SRE team", "action": "Write runbook for lock contention incidents and add connection pool alert at 80% saturation.", "due": "2024-09-09"},
        ],
    }))
    score, details = env.grade()
    assert score >= 0.70, f"Full correct post-mortem should score ≥ 0.70, got {score:.4f}: {details}"
    print(f"  ✅ Postmortem grader — full answer scores {score:.4f} ≥ 0.70")


def test_postmortem_grader_wrong_severity():
    env = SREIncidentEnvironment(task_id="full_postmortem")
    env.reset()
    env.step(Action(action_type=ActionType.WRITE_POSTMORTEM, payload={
        "title": "Some incident",
        "severity": "P4",  # wrong
        "duration_minutes": 13,
        "timeline": [{"time": "t1", "event": "e1"}, {"time": "t2", "event": "e2"},
                     {"time": "t3", "event": "e3"}, {"time": "t4", "event": "e4"}, {"time": "t5", "event": "e5"}],
        "root_cause": "vacuum analyze lock postgres transaction",
        "contributing_factors": ["maintenance window"],
        "impact": "1247 failed payment transactions",
        "action_items": [
            {"owner": "DBA", "action": "never vacuum production without draining", "due": "2024-09-09"},
            {"owner": "SRE", "action": "add runbook for lock contention", "due": "2024-09-09"},
        ],
    }))
    score, details = env.grade()
    assert details["components"]["severity"] == 0.0, "Wrong severity (P4) should score 0 on that component"
    print(f"  ✅ Postmortem grader — wrong severity gets 0 on severity component (total={score:.4f})")


def test_postmortem_grader_no_action():
    env = SREIncidentEnvironment(task_id="full_postmortem")
    env.reset()
    score, details = env.grade()
    assert score == 0.0
    assert "No write_postmortem action" in details.get("reason", "")
    print("  ✅ Postmortem grader — no action = 0.0")


# ---------------------------------------------------------------------------
# Test: Grader scores in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_all_grader_scores_in_valid_range():
    """Fuzz test: random garbage actions must never produce scores outside [0.0, 1.0]."""
    for task_id in TASK_REGISTRY:
        for _ in range(5):
            env = SREIncidentEnvironment(task_id=task_id)
            env.reset()
            # Send a mix of valid and semi-valid actions
            for at in [ActionType.CLASSIFY, ActionType.INVESTIGATE, ActionType.HYPOTHESIZE]:
                try:
                    env.step(Action(action_type=at, payload={"garbage_key": "garbage_value", "severity": "P9999"}))
                except Exception:
                    pass
            score, _ = env.grade()
            assert 0.0 <= score <= 1.0, f"{task_id}: grader score {score} out of [0, 1]"
    print("  ✅ All graders produce scores in [0.0, 1.0] even with garbage inputs")


# ---------------------------------------------------------------------------
# Test: Episode max_steps
# ---------------------------------------------------------------------------

def test_episode_terminates_at_max_steps():
    env = SREIncidentEnvironment(task_id="alert_triage")
    env.reset()
    max_steps = TASK_REGISTRY["alert_triage"]["max_steps"]
    for _ in range(max_steps + 3):  # go over
        if env._done:
            break
        env.step(Action(action_type=ActionType.ESCALATE, payload={"team": "nobody", "reason": "test", "priority": "P3"}))
    assert env._done, "Episode should be done after max_steps"
    print(f"  ✅ Episode terminates at max_steps ({max_steps})")


# ---------------------------------------------------------------------------
# Test: Partial observability (cascading failure logs hidden until investigated)
# ---------------------------------------------------------------------------

def test_logs_hidden_until_investigated():
    env = SREIncidentEnvironment(task_id="cascading_failure")
    obs = env.reset()
    assert len(obs.visible_logs) == 0, "No logs should be visible before investigation"
    result = env.step(Action(action_type=ActionType.INVESTIGATE, payload={"service_name": "redis-cluster"}))
    assert len(result.observation.visible_logs) > 0, "Logs should be visible after investigation"
    print(f"  ✅ Partial observability: 0 logs before investigate, {len(result.observation.visible_logs)} after")


# ---------------------------------------------------------------------------
# Test: Task registry completeness
# ---------------------------------------------------------------------------

def test_task_registry_completeness():
    required_keys = {"task_id", "difficulty", "description", "alerts", "available_services", "max_steps", "grader", "action_schema"}
    for task_id, task in TASK_REGISTRY.items():
        for key in required_keys:
            assert key in task, f"{task_id} missing key '{key}'"
        assert task["difficulty"] in {"easy", "medium", "hard"}
        assert callable(task["grader"])
    print("  ✅ All tasks in registry have required keys")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_reset_returns_observation,
        test_reset_twice_produces_clean_state,
        test_step_returns_step_result,
        test_step_after_done_returns_zero_reward,
        test_wrong_task_id_returns_negative_reward,
        test_state_returns_correct_metadata,
        test_investigation_gives_positive_reward,
        test_root_cause_investigation_gives_higher_reward,
        test_loop_penalty,
        test_re_investigation_negative_reward,
        test_alert_triage_grader_perfect_score,
        test_alert_triage_grader_wrong_severity,
        test_alert_triage_grader_zero_for_no_actions,
        test_cascading_failure_grader_correct_hypothesis,
        test_cascading_failure_grader_zero_for_no_actions,
        test_postmortem_grader_full_score,
        test_postmortem_grader_wrong_severity,
        test_postmortem_grader_no_action,
        test_all_grader_scores_in_valid_range,
        test_episode_terminates_at_max_steps,
        test_logs_hidden_until_investigated,
        test_task_registry_completeness,
    ]

    passed = 0
    failed = 0
    print("\n🧪 SRE Incident Response — Test Suite\n" + "="*50)
    for test_fn in tests:
        try:
            print(f"\n▶ {test_fn.__name__}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("✅ All tests passed!\n")
    else:
        print("❌ Some tests failed.\n")
        sys.exit(1)
