"""YAML profile loading and validation for scriba pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml


DEFAULT_STAGE_ORDER: tuple[str, ...] = (
    "extract",
    "clean",
    "sectionize",
    "normalize_map",
    "reduce",
    "validate",
    "export",
)

SUPPORTED_ADAPTERS = {"litellm"}
SUPPORTED_TOPOLOGIES = {"local_spawned", "local_attached", "remote"}
SUPPORTED_MODEL_ORIGINS = {"local_weights", "hosted_weights", "unknown"}


class ProfileError(ValueError):
    """Raised when a pipeline profile is invalid."""


@dataclass(frozen=True)
class ArtifactsConfig:
    """Configuration for run artifact directories."""

    root: Path = Path("./artifacts")
    run_id: str = "auto"


@dataclass(frozen=True)
class RoleBinding:
    """Logical stage role mapped to backend + model."""

    backend: str
    model: str


@dataclass(frozen=True)
class BackendConfig:
    """Backend server configuration."""

    adapter: str
    topology: str
    provider: str
    base_url: str
    inference_path: str = "/v1/chat/completions"
    health_path: str = "/v1/models"
    health_method: str = "GET"
    startup_timeout_s: int = 180
    command: str | None = None
    api_key: str = ""
    health_headers: dict[str, str] = field(default_factory=dict)
    health_payload: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    model_origin: str = "unknown"


@dataclass(frozen=True)
class StageConfig:
    """Stage execution knobs."""

    enabled: bool = True
    workers: int = 1
    temperature: float | None = None
    target_tokens: int | None = None
    overlap_tokens: int | None = None
    fail_on_hard_errors: bool | None = None
    multi_file: bool | None = None
    request_timeout_s: int | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_exclude: bool | None = None


@dataclass(frozen=True)
class PipelineProfile:
    """Typed profile used by the pipeline runner."""

    version: int
    artifacts: ArtifactsConfig
    roles: dict[str, RoleBinding]
    backends: dict[str, BackendConfig]
    stages: dict[str, StageConfig]
    source_path: Path

    def enabled_stages(self) -> list[str]:
        """Return enabled stages in deterministic execution order."""
        return [
            stage_name
            for stage_name in DEFAULT_STAGE_ORDER
            if self.stages[stage_name].enabled
        ]

    def resolve_role(self, role_name: str) -> RoleBinding | None:
        """Resolve role binding with contract defaults.

        Contract default: `reduce_text` falls back to `normalize_text` unless
        explicitly overridden.
        """
        direct = self.roles.get(role_name)
        if direct is not None:
            return direct
        if role_name == "reduce_text":
            return self.roles.get("normalize_text")
        return None


def load_profile(path: str | Path) -> PipelineProfile:
    """Load and validate a pipeline YAML profile."""
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.exists() or not profile_path.is_file():
        raise ProfileError(f"Profile file not found: {profile_path}")

    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProfileError("Profile root must be a YAML mapping.")

    version = raw.get("version", 1)
    if version != 1:
        raise ProfileError(f"Unsupported profile version: {version}")

    artifacts = _parse_artifacts(raw.get("artifacts", {}))
    backends = _parse_backends(raw.get("backends", {}))
    roles = _parse_roles(raw.get("roles", {}), backends)
    stages = _parse_stages(raw.get("stages", {}))

    return PipelineProfile(
        version=version,
        artifacts=artifacts,
        roles=roles,
        backends=backends,
        stages=stages,
        source_path=profile_path,
    )


def _parse_artifacts(raw: Any) -> ArtifactsConfig:
    if not isinstance(raw, dict):
        raise ProfileError("artifacts must be a mapping")

    root_value = raw.get("root", "./artifacts")
    run_id_value = raw.get("run_id", "auto")
    if not isinstance(root_value, str):
        raise ProfileError("artifacts.root must be a string path")
    if not isinstance(run_id_value, str):
        raise ProfileError("artifacts.run_id must be a string")

    expanded_root = _expand_env_reference(root_value)
    expanded_run_id = _expand_env_reference(run_id_value)
    return ArtifactsConfig(
        root=Path(expanded_root).expanduser(), run_id=expanded_run_id
    )


def _parse_backends(raw: Any) -> dict[str, BackendConfig]:
    if raw in ({}, None):
        return {}
    if not isinstance(raw, dict):
        raise ProfileError("backends must be a mapping")

    parsed: dict[str, BackendConfig] = {}
    for backend_name, backend_raw in raw.items():
        if not isinstance(backend_name, str) or not backend_name:
            raise ProfileError("backend names must be non-empty strings")
        if not isinstance(backend_raw, dict):
            raise ProfileError(f"backend '{backend_name}' must be a mapping")

        adapter = backend_raw.get("adapter")
        topology = backend_raw.get("topology")
        provider = backend_raw.get("provider")
        model_origin = backend_raw.get("model_origin", "unknown")
        base_url_raw = backend_raw.get("base_url", "")
        base_url = (
            _expand_env_reference(base_url_raw)
            if isinstance(base_url_raw, str)
            else base_url_raw
        )
        inference_path_raw = backend_raw.get("inference_path", "/v1/chat/completions")
        health_path_raw = backend_raw.get("health_path", "/v1/models")
        inference_path = (
            _expand_env_reference(inference_path_raw)
            if isinstance(inference_path_raw, str)
            else inference_path_raw
        )
        health_path = (
            _expand_env_reference(health_path_raw)
            if isinstance(health_path_raw, str)
            else health_path_raw
        )
        health_method = backend_raw.get("health_method", "GET")
        startup_timeout_s = backend_raw.get("startup_timeout_s", 180)
        command_raw = backend_raw.get("command")
        command = (
            _expand_env_reference(command_raw)
            if isinstance(command_raw, str)
            else command_raw
        )
        api_key_raw = backend_raw.get("api_key", "")
        api_key = (
            _expand_env_reference(api_key_raw)
            if isinstance(api_key_raw, str)
            else api_key_raw
        )
        health_headers = backend_raw.get("health_headers", {})
        health_payload = backend_raw.get("health_payload")
        env = backend_raw.get("env", {})

        if adapter not in SUPPORTED_ADAPTERS:
            raise ProfileError(
                f"backend '{backend_name}' has unsupported adapter '{adapter}'"
            )
        if topology not in SUPPORTED_TOPOLOGIES:
            raise ProfileError(
                f"backend '{backend_name}' has unsupported topology '{topology}'"
            )
        if not isinstance(provider, str) or not provider.strip():
            raise ProfileError(f"backend '{backend_name}' must define provider")
        if (
            not isinstance(model_origin, str)
            or model_origin not in SUPPORTED_MODEL_ORIGINS
        ):
            raise ProfileError(
                f"backend '{backend_name}' has unsupported model_origin '{model_origin}'"
            )
        if not isinstance(base_url, str):
            raise ProfileError(f"backend '{backend_name}' base_url must be a string")
        if topology in {"local_spawned", "local_attached"} and not base_url:
            raise ProfileError(
                f"backend '{backend_name}' must define base_url for topology '{topology}'"
            )
        if not isinstance(inference_path, str) or not inference_path.startswith("/"):
            raise ProfileError(
                f"backend '{backend_name}' inference_path must start with '/'"
            )
        if not isinstance(health_path, str) or not health_path.startswith("/"):
            raise ProfileError(
                f"backend '{backend_name}' health_path must start with '/'"
            )
        if not isinstance(health_method, str) or health_method.upper() not in {
            "GET",
            "POST",
        }:
            raise ProfileError(
                f"backend '{backend_name}' health_method must be GET or POST"
            )
        if not isinstance(startup_timeout_s, int) or startup_timeout_s <= 0:
            raise ProfileError(
                f"backend '{backend_name}' startup_timeout_s must be > 0"
            )
        if not isinstance(api_key, str):
            raise ProfileError(f"backend '{backend_name}' api_key must be a string")

        if topology == "local_spawned":
            if not isinstance(command, str) or not command.strip():
                raise ProfileError(
                    f"local_spawned backend '{backend_name}' must define command"
                )
        elif command is not None and not isinstance(command, str):
            raise ProfileError(f"backend '{backend_name}' command must be a string")

        if not isinstance(health_headers, dict):
            raise ProfileError(
                f"backend '{backend_name}' health_headers must be a mapping"
            )
        parsed_health_headers: dict[str, str] = {}
        for key, value in health_headers.items():
            if not isinstance(key, str):
                raise ProfileError(
                    f"backend '{backend_name}' health_headers keys must be strings"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise ProfileError(
                    f"backend '{backend_name}' health_headers values must be scalar"
                )
            parsed_health_headers[key] = _expand_env_reference(str(value))

        if health_payload is not None and not isinstance(health_payload, dict):
            raise ProfileError(
                f"backend '{backend_name}' health_payload must be a mapping"
            )

        if not isinstance(env, dict):
            raise ProfileError(f"backend '{backend_name}' env must be a mapping")

        parsed_env: dict[str, str] = {}
        for key, value in env.items():
            if not isinstance(key, str):
                raise ProfileError(f"backend '{backend_name}' env keys must be strings")
            if not isinstance(value, (str, int, float, bool)):
                raise ProfileError(
                    f"backend '{backend_name}' env values must be scalar types"
                )
            parsed_env[key] = _expand_env_reference(str(value))

        parsed[backend_name] = BackendConfig(
            adapter=adapter,
            topology=topology,
            provider=provider,
            base_url=base_url,
            inference_path=inference_path,
            health_path=health_path,
            health_method=health_method.upper(),
            startup_timeout_s=startup_timeout_s,
            command=command,
            api_key=api_key,
            health_headers=parsed_health_headers,
            health_payload=health_payload,
            env=parsed_env,
            model_origin=model_origin,
        )

    return parsed


def _parse_roles(
    raw: Any,
    backends: dict[str, BackendConfig],
) -> dict[str, RoleBinding]:
    if raw in ({}, None):
        return {}
    if not isinstance(raw, dict):
        raise ProfileError("roles must be a mapping")

    parsed: dict[str, RoleBinding] = {}
    for role_name, role_raw in raw.items():
        if not isinstance(role_name, str) or not role_name:
            raise ProfileError("role names must be non-empty strings")
        if not isinstance(role_raw, dict):
            raise ProfileError(f"role '{role_name}' must be a mapping")

        backend_raw = role_raw.get("backend")
        model_raw = role_raw.get("model")
        backend_name = (
            _expand_env_reference(backend_raw)
            if isinstance(backend_raw, str)
            else backend_raw
        )
        model_name = (
            _expand_env_reference(model_raw)
            if isinstance(model_raw, str)
            else model_raw
        )
        if not isinstance(backend_name, str) or not backend_name:
            raise ProfileError(f"role '{role_name}' must define backend")
        if backend_name not in backends:
            raise ProfileError(
                f"role '{role_name}' references unknown backend '{backend_name}'"
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise ProfileError(f"role '{role_name}' must define model")

        parsed[role_name] = RoleBinding(backend=backend_name, model=model_name)
    return parsed


def _parse_stages(raw: Any) -> dict[str, StageConfig]:
    if raw in ({}, None):
        return {stage: StageConfig() for stage in DEFAULT_STAGE_ORDER}
    if not isinstance(raw, dict):
        raise ProfileError("stages must be a mapping")

    parsed: dict[str, StageConfig] = {}
    for stage_name in DEFAULT_STAGE_ORDER:
        stage_raw = raw.get(stage_name, {})
        if not isinstance(stage_raw, dict):
            raise ProfileError(f"stage '{stage_name}' must be a mapping")

        parsed[stage_name] = StageConfig(
            enabled=bool(stage_raw.get("enabled", True)),
            workers=int(stage_raw.get("workers", 1)),
            temperature=_optional_float(stage_raw.get("temperature")),
            target_tokens=_optional_int(stage_raw.get("target_tokens")),
            overlap_tokens=_optional_int(stage_raw.get("overlap_tokens")),
            fail_on_hard_errors=_optional_bool(stage_raw.get("fail_on_hard_errors")),
            multi_file=_optional_bool(stage_raw.get("multi_file")),
            request_timeout_s=_optional_int(stage_raw.get("request_timeout_s")),
            max_output_tokens=_optional_int(stage_raw.get("max_output_tokens")),
            reasoning_effort=_optional_str(stage_raw.get("reasoning_effort")),
            reasoning_exclude=_optional_bool(stage_raw.get("reasoning_exclude")),
        )

    for stage_name in raw.keys():
        if stage_name not in DEFAULT_STAGE_ORDER:
            raise ProfileError(f"Unknown stage name in profile: {stage_name}")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise ProfileError("Expected integer or null")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise ProfileError("Expected float or null")


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ProfileError("Expected bool or null")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    raise ProfileError("Expected string or null")


_ENV_REF_RE = re.compile(
    r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$|^\$([A-Za-z_][A-Za-z0-9_]*)$"
)


def _expand_env_reference(value: str) -> str:
    """Expand `${VAR}` / `$VAR` strings using environment variables.

    If variable is missing, returns empty string.
    """
    match = _ENV_REF_RE.match(value.strip())
    if not match:
        return value
    env_name = match.group(1) or match.group(2)
    if not env_name:
        return value
    return os.getenv(env_name, "")
