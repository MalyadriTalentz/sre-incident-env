"""
client.py — HTTP client for the SRE Incident Response OpenEnv environment.

Usage (sync):
    from client import SREEnvClient, SREEnvAction
    with SREEnvClient(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="alert_triage")
        result = env.step(SREEnvAction(action_type="classify", payload={...}))
        print(result.reward)

Usage (async):
    async with SREEnvClient(base_url="http://localhost:7860") as env:
        result = await env.reset(task_id="cascading_failure")
        result = await env.step(SREEnvAction(action_type="investigate", payload={...}))
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict, Optional

import httpx


class SREEnvAction:
    """Minimal action container for the HTTP client."""

    def __init__(self, action_type: str, payload: Optional[Dict[str, Any]] = None, task_id: Optional[str] = None):
        self.action_type = action_type
        self.payload = payload or {}
        self.task_id = task_id

    def to_dict(self) -> Dict:
        d: Dict[str, Any] = {"action_type": self.action_type, "payload": self.payload}
        if self.task_id:
            d["task_id"] = self.task_id
        return d


class _StepResult:
    """Lightweight result object returned from step() and reset()."""

    def __init__(self, data: Dict):
        self._data = data
        self.observation = type("Observation", (), data.get("observation", data) or {})()
        for k, v in (data.get("observation", data) or {}).items():
            setattr(self.observation, k, v)
        self.reward: float = float(data.get("reward", 0.0))
        self.done: bool = bool(data.get("done", False))
        self.info: Dict = data.get("info", {})
        self.session_id: str = data.get("session_id", "")

    def __repr__(self) -> str:
        return f"StepResult(reward={self.reward}, done={self.done})"


class _SyncWrapper:
    """Synchronous wrapper around the async client."""

    def __init__(self, async_client: "SREEnvClient"):
        self._client = async_client
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._client.__aexit__(*args))
        self._loop.close()

    def reset(self, task_id: str = "alert_triage") -> _StepResult:
        return self._loop.run_until_complete(self._client.reset(task_id=task_id))

    def step(self, action: SREEnvAction) -> _StepResult:
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> Dict:
        return self._loop.run_until_complete(self._client.state())

    def grade(self) -> Dict:
        return self._loop.run_until_complete(self._client.grade())


class SREEnvClient:
    """
    Async HTTP client for the SRE Incident Response OpenEnv environment.

    Parameters
    ----------
    base_url : str
        Base URL of the running server, e.g. "http://localhost:7860"
        or "https://your-space.hf.space".
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._http: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None

    async def __aenter__(self) -> "SREEnvClient":
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=60)
        return self

    async def __aexit__(self, *args) -> None:
        if self._http:
            await self._http.aclose()

    def sync(self) -> _SyncWrapper:
        """Return a synchronous wrapper for use in non-async code or notebooks."""
        return _SyncWrapper(self)

    async def reset(self, task_id: str = "alert_triage") -> _StepResult:
        """Initialise a new episode. Returns an observation wrapped in StepResult."""
        assert self._http, "Use as async context manager: `async with SREEnvClient(...) as env:`"
        r = await self._http.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        data = r.json()
        self._session_id = data.get("session_id", "")
        return _StepResult(data)

    async def step(self, action: SREEnvAction) -> _StepResult:
        """Execute one action. Returns StepResult with observation, reward, done, info."""
        assert self._http, "Use as async context manager"
        assert self._session_id, "Call reset() before step()"
        r = await self._http.post("/step", json={
            "session_id": self._session_id,
            "action": action.to_dict(),
        })
        r.raise_for_status()
        return _StepResult(r.json())

    async def state(self) -> Dict:
        """Return current episode metadata."""
        assert self._http
        r = await self._http.post("/state", json={"session_id": self._session_id})
        r.raise_for_status()
        return r.json()

    async def grade(self) -> Dict:
        """Grade the current episode. Returns score 0.0–1.0 and details."""
        assert self._http
        r = await self._http.post("/grader", json={"session_id": self._session_id})
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        """Alias for __aexit__ — matches OpenEnv convention."""
        await self.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Convenience alias matching OpenEnv naming convention
# ---------------------------------------------------------------------------

SREAction = SREEnvAction
