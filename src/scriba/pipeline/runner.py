"""Pipeline runner with checkpoints and stage dispatch."""

from __future__ import annotations

import json
from urllib.parse import urlparse
import shlex
import shutil
from pathlib import Path
from typing import Any

from scriba.pipeline.backends import BackendError, ModelManager, ModelSession
from scriba.pipeline.profile import PipelineProfile
from scriba.pipeline.state import ArtifactStore, StateError, utc_now_iso
from scriba.pipeline.stages import StageExecutionError, execute_stage


STAGE_ROLE_BINDINGS: dict[str, str] = {
    "normalize_map": "normalize_text",
}


class PipelineError(RuntimeError):
    """Raised when pipeline execution fails."""


class PipelineRunner:
    """Execute enabled pipeline stages with resumable state."""

    def __init__(self, profile: PipelineProfile) -> None:
        self.profile = profile
        self.store = ArtifactStore(profile.artifacts.root)

    def run(
        self,
        *,
        input_path: str | Path,
        run_id: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run the pipeline for a local input file."""
        input_file = Path(input_path).expanduser().resolve()
        if not input_file.exists() or not input_file.is_file():
            raise PipelineError(f"Input file not found: {input_file}")

        enabled_stages = self.profile.enabled_stages()
        if not enabled_stages:
            raise PipelineError("Profile has no enabled stages")

        resolved_run_id = run_id or self.profile.artifacts.run_id
        if resume:
            if not resolved_run_id or resolved_run_id == "auto":
                raise PipelineError("--resume requires an explicit run_id")
            try:
                state = self.store.load_state(resolved_run_id)
            except StateError as exc:
                raise PipelineError(str(exc)) from exc
        else:
            actual_run_id = self.store.resolve_run_id(resolved_run_id)
            stage_statuses = {
                stage_name: ("pending" if stage_config.enabled else "disabled")
                for stage_name, stage_config in self.profile.stages.items()
            }
            try:
                state = self.store.init_run(
                    run_id=actual_run_id,
                    input_path=input_file,
                    profile_path=self.profile.source_path,
                    stage_statuses=stage_statuses,
                )
            except StateError as exc:
                raise PipelineError(str(exc)) from exc

        with ModelManager(self.profile) as model_manager:
            for stage_name in enabled_stages:
                stage_state = state["stages"].get(stage_name)
                if stage_state is None:
                    raise PipelineError(f"Stage '{stage_name}' missing in run state")

                if resume and stage_state["status"] == "completed":
                    continue

                self.store.mark_stage_running(state, stage_name)
                try:
                    details = self._execute_stage(
                        state=state,
                        stage_name=stage_name,
                        model_manager=model_manager,
                    )
                    state["stages"][stage_name]["details"] = details
                except (BackendError, StageExecutionError, Exception) as exc:
                    self.store.mark_stage_failed(state, stage_name, str(exc))
                    raise PipelineError(f"Stage '{stage_name}' failed: {exc}") from exc
                self.store.mark_stage_completed(state, stage_name)

        self.store.mark_run_completed(
            state,
            status=self._resolve_final_run_status(state=state),
        )
        return state

    def status(self, *, run_id: str) -> dict[str, Any]:
        """Load run state by ID."""
        try:
            return self.store.load_state(run_id)
        except StateError as exc:
            raise PipelineError(str(exc)) from exc

    def doctor(self, *, input_path: str) -> dict[str, Any]:
        """Run lightweight preflight checks and include path context."""
        report = run_doctor(self.profile, input_path=input_path)
        report["profile_path"] = str(self.profile.source_path)
        report["input_path"] = str(Path(input_path).expanduser().resolve())
        return report

    def _execute_stage(
        self,
        *,
        state: dict[str, Any],
        stage_name: str,
        model_manager: ModelManager,
    ) -> dict[str, Any]:
        """Execute one stage and return stage details for state/logging."""
        run_dir = self.store.run_dir(state["run_id"])
        log_path = run_dir / "logs" / f"{stage_name}.log"
        role_name = self._role_for_stage(stage_name=stage_name, state=state)
        model_session: ModelSession | None = None

        if role_name and role_name in self.profile.roles:
            model_session = model_manager.acquire(role_name)

        stage_details = execute_stage(
            stage_name=stage_name,
            state=state,
            run_dir=run_dir,
            stage_config=self.profile.stages[stage_name],
            model_session=model_session,
        )

        endpoint = model_session.endpoint if model_session is not None else None
        endpoint_details = (
            {
                "base_url": endpoint.base_url,
                "backend": endpoint.backend_name,
                "model": endpoint.model,
                "role": endpoint.role,
                "adapter": endpoint.adapter,
                "topology": endpoint.topology,
                "provider": endpoint.provider,
                "context_length": endpoint.context_length,
                "context_length_source": endpoint.context_length_source,
                "inference_url": endpoint.inference_url,
            }
            if endpoint
            else {
                "base_url": "none",
                "backend": "none",
                "model": "",
                "role": "",
                "adapter": "",
                "topology": "",
                "provider": "",
                "context_length": None,
                "context_length_source": "",
                "inference_url": "",
            }
        )

        log_payload = {
            "stage": stage_name,
            "run_id": state["run_id"],
            "status": "completed",
            "requires_role": role_name or "none",
            "endpoint": endpoint_details,
            "details": stage_details,
            "timestamp": utc_now_iso(),
        }

        log_path.write_text(
            json.dumps(log_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return stage_details

    def _role_for_stage(self, *, stage_name: str, state: dict[str, Any]) -> str | None:
        if stage_name == "extract":
            input_path = Path(str(state.get("input_path", ""))).expanduser()
            if (
                input_path.suffix.lower() == ".pdf"
                and "ocr_vision" in self.profile.roles
            ):
                return "ocr_vision"
            return None
        if stage_name == "sectionize" and "normalize_text" in self.profile.roles:
            return "normalize_text"
        return STAGE_ROLE_BINDINGS.get(stage_name)

    def _resolve_final_run_status(self, *, state: dict[str, Any]) -> str:
        validate_stage = state.get("stages", {}).get("validate")
        if isinstance(validate_stage, dict):
            details = validate_stage.get("details")
            if isinstance(details, dict):
                hard_error_count = details.get("hard_error_count")
                if isinstance(hard_error_count, int) and hard_error_count > 0:
                    return "completed_with_validation_errors"
        return "completed"


def run_doctor(
    profile: PipelineProfile,
    *,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run lightweight preflight checks for a profile and optional input file."""
    errors: list[str] = []
    warnings: list[str] = []

    enabled_stages = profile.enabled_stages()
    if not enabled_stages:
        errors.append("No stages are enabled in the profile.")

    if input_path is not None:
        candidate = Path(input_path).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            errors.append(f"Input file not found: {candidate}")

    for backend_name, backend in profile.backends.items():
        if backend.topology == "local_spawned" and not backend.command:
            errors.append(
                f"local_spawned backend '{backend_name}' is missing command configuration."
            )
        if backend.topology == "local_spawned" and backend.command:
            tokens = _command_tokens(backend.command)
            executable = tokens[0] if tokens else None
            if executable and not _is_executable_available(executable):
                warnings.append(
                    f"Backend '{backend_name}' executable not found in PATH: {executable}"
                )
        if backend.topology in {"local_attached", "remote"} and backend.command:
            warnings.append(
                f"Backend '{backend_name}' defines command but topology is '{backend.topology}'; command is ignored."
            )

        host = _extract_host(backend.base_url)
        is_local_host = host in {"127.0.0.1", "localhost", "::1"}
        if backend.topology == "remote" and is_local_host:
            warnings.append(
                f"Backend '{backend_name}' topology is remote but base_url points to local host ({host})."
            )
        if backend.topology == "local_attached" and host and not is_local_host:
            warnings.append(
                f"Backend '{backend_name}' topology is local_attached but host is not local ({host})."
            )
        if backend.topology == "remote" and not backend.api_key.strip():
            warnings.append(
                f"Backend '{backend_name}' topology is remote and api_key is empty."
            )

    for role_name, binding in profile.roles.items():
        if binding.backend not in profile.backends:
            errors.append(
                f"Role '{role_name}' references missing backend '{binding.backend}'."
            )
        if not binding.model.strip():
            errors.append(f"Role '{role_name}' has empty model value.")

    for stage_name in enabled_stages:
        if stage_name == "reduce":
            continue
        role_name = STAGE_ROLE_BINDINGS.get(stage_name)
        if role_name and role_name not in profile.roles:
            warnings.append(
                f"Stage '{stage_name}' has no role '{role_name}', using passthrough behavior."
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "enabled_stages": enabled_stages,
        "artifacts_root": str(profile.artifacts.root.expanduser().resolve()),
    }


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _is_executable_available(executable: str) -> bool:
    if "/" in executable:
        return Path(executable).expanduser().exists()
    return shutil.which(executable) is not None


def _extract_host(base_url: str) -> str:
    try:
        return (urlparse(base_url).hostname or "").lower()
    except Exception:
        return ""
