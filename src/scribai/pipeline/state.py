"""Run artifact directories and resumable pipeline state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


RUN_STATE_FILENAME = "state.json"
RUN_SUBDIRECTORIES = (
    "raw",
    "chunks",
    "map",
    "reduce",
    "final",
    "logs",
)


class StateError(ValueError):
    """Raised when run state cannot be created or loaded."""


def utc_now_iso() -> str:
    """UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class ArtifactStore:
    """Filesystem-backed storage for run metadata and artifacts."""

    def __init__(self, artifacts_root: Path) -> None:
        self.artifacts_root = artifacts_root.expanduser().resolve()

    def resolve_run_id(self, requested_run_id: str | None = None) -> str:
        """Resolve run ID, generating one when set to auto/None."""
        if (
            requested_run_id is None
            or requested_run_id == ""
            or requested_run_id == "auto"
        ):
            ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            return f"run-{ts}-{uuid4().hex[:8]}"
        return requested_run_id

    def run_dir(self, run_id: str) -> Path:
        """Run directory path."""
        return self.artifacts_root / run_id

    def state_path(self, run_id: str) -> Path:
        """Path to state.json for a run."""
        return self.run_dir(run_id) / RUN_STATE_FILENAME

    def init_run(
        self,
        *,
        run_id: str,
        input_path: Path,
        profile_path: Path,
        stage_statuses: dict[str, str],
    ) -> dict[str, Any]:
        """Create run directory structure and initialize state file."""
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            raise StateError(f"Run directory already exists: {run_dir}")

        run_dir.mkdir(parents=True, exist_ok=False)
        for subdir in RUN_SUBDIRECTORIES:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        state: dict[str, Any] = {
            "run_id": run_id,
            "input_path": str(input_path),
            "profile_path": str(profile_path),
            "status": "running",
            "current_stage": None,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "stages": {
                stage: {
                    "status": initial_status,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                }
                for stage, initial_status in stage_statuses.items()
            },
        }
        self.save_state(state)
        return state

    def load_state(self, run_id: str) -> dict[str, Any]:
        """Load state.json for an existing run."""
        path = self.state_path(run_id)
        if not path.exists() or not path.is_file():
            raise StateError(f"Run state file not found: {path}")

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"Run state file is invalid JSON: {path}") from exc

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist run state to disk."""
        run_id = state["run_id"]
        state["updated_at"] = utc_now_iso()
        path = self.state_path(run_id)
        path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def mark_stage_running(self, state: dict[str, Any], stage_name: str) -> None:
        """Mark stage as running."""
        stage = state["stages"][stage_name]
        stage["status"] = "running"
        stage["started_at"] = utc_now_iso()
        stage["completed_at"] = None
        stage["error"] = None
        state["current_stage"] = stage_name
        state["status"] = "running"
        state["error"] = None
        self.save_state(state)

    def mark_stage_completed(self, state: dict[str, Any], stage_name: str) -> None:
        """Mark stage as completed."""
        stage = state["stages"][stage_name]
        stage["status"] = "completed"
        stage["completed_at"] = utc_now_iso()
        stage["error"] = None
        state["current_stage"] = None
        self.save_state(state)

    def mark_stage_failed(
        self,
        state: dict[str, Any],
        stage_name: str,
        error_message: str,
    ) -> None:
        """Mark stage as failed and persist error details."""
        stage = state["stages"][stage_name]
        stage["status"] = "failed"
        stage["completed_at"] = utc_now_iso()
        stage["error"] = error_message
        state["current_stage"] = stage_name
        state["status"] = "failed_runtime"
        state["error"] = error_message
        self.save_state(state)

    def mark_run_completed(
        self,
        state: dict[str, Any],
        *,
        status: str = "completed",
    ) -> None:
        """Mark run as completed."""
        state["status"] = status
        state["current_stage"] = None
        state["error"] = None
        self.save_state(state)
