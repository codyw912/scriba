"""Tests for profile loading."""

from pathlib import Path

import pytest

from scriba.pipeline import ProfileError, load_profile


def test_load_profile_defaults(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("version: 1\n", encoding="utf-8")

    profile = load_profile(profile_path)
    assert profile.version == 1
    assert profile.artifacts.run_id == "auto"
    assert profile.enabled_stages() == [
        "extract",
        "clean",
        "sectionize",
        "normalize_map",
        "reduce",
        "validate",
        "export",
    ]


def test_load_profile_can_disable_stage(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "version: 1\nstages:\n  normalize_map:\n    enabled: false\n",
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    assert "normalize_map" not in profile.enabled_stages()


def test_load_profile_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ProfileError):
        load_profile(missing)


def test_example_profiles_parse() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_paths = [
        repo_root / "profiles/pipeline.profile.example.yaml",
        repo_root
        / "profiles/local_attached/pipeline.profile.local_attached_litellm.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_litellm.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_litellm.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_litellm_highcap.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_2b_bf16.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_0p8b_bf16.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_9b_bf16_fast.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_qwen35_4b_bf16_fast.example.yaml",
        repo_root
        / "profiles/local_attached/pipeline.profile.local_attached_litellm_qwen35_9b_bf16.example.yaml",
        repo_root
        / "profiles/local_attached/pipeline.profile.local_attached_litellm_qwen35_4b_bf16.example.yaml",
        repo_root
        / "profiles/local_attached/pipeline.profile.local_attached_litellm_qwen35_35b_a3b_mxfp4.example.yaml",
        repo_root
        / "profiles/local_spawned/pipeline.profile.local_spawned_litellm_qwen35_35b_a3b_mxfp4.example.yaml",
        repo_root / "profiles/remote/pipeline.profile.remote_openrouter.example.yaml",
        repo_root
        / "profiles/hybrid/pipeline.profile.hybrid_local_spawned_ocr_remote_text.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_qwen25_7b.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_llama31_8b.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_qwen35_flash.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_qwen3_next_80b_a3b.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_qwen3_coder_next.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_mistral_nemo.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_gpt4o_mini.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_gemini_2_5_flash.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_openrouter_claude_sonnet_4_5.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_cerebras_llama31_8b.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_cerebras_gpt_oss_120b.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_cerebras_qwen3_235b_a22b_instruct_2507.example.yaml",
        repo_root
        / "profiles/remote/pipeline.profile.remote_cerebras_zai_glm_4_7.example.yaml",
    ]

    for path in example_paths:
        profile = load_profile(path)
        assert profile.version == 1


def test_profile_parses_canonical_backend_axes(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  text_backend:",
                "    adapter: litellm",
                "    topology: local_attached",
                "    provider: lmstudio",
                "    model_origin: local_weights",
                "    base_url: http://127.0.0.1:8090",
                "roles:",
                "  normalize_text:",
                "    backend: text_backend",
                "    model: qwen/qwen3.5-35b-a3b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    backend = profile.backends["text_backend"]
    assert backend.adapter == "litellm"
    assert backend.topology == "local_attached"
    assert backend.provider == "lmstudio"
    assert backend.model_origin == "local_weights"


def test_profile_parses_litellm_cerebras_backend(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  text_backend:",
                "    adapter: litellm",
                "    topology: remote",
                "    provider: cerebras",
                "    model_origin: hosted_weights",
                "    base_url: https://api.cerebras.ai",
                "roles:",
                "  normalize_text:",
                "    backend: text_backend",
                "    model: llama3.1-8b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    backend = profile.backends["text_backend"]
    assert backend.adapter == "litellm"
    assert backend.provider == "cerebras"


def test_profile_rejects_removed_legacy_adapter_values(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  openai_like:",
                "    adapter: openai_http",
                "    topology: remote",
                "    provider: openrouter",
                "    base_url: https://openrouter.ai/api/v1",
                "  cerebras_like:",
                "    adapter: cerebras_sdk",
                "    topology: remote",
                "    provider: cerebras",
                "    base_url: https://api.cerebras.ai/v1",
                "roles:",
                "  normalize_text:",
                "    backend: openai_like",
                "    model: qwen/qwen3.5-35b-a3b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="unsupported adapter"):
        load_profile(profile_path)


def test_profile_rejects_removed_legacy_backend_type_aliases(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  local_backend:",
                "    type: local_openai",
                "    base_url: http://127.0.0.1:8091",
                "    command: mlx_lm.server --host 127.0.0.1 --port 8091",
                "  attached_backend:",
                "    type: external_openai",
                "    base_url: http://127.0.0.1:8090",
                "  remote_backend:",
                "    type: external_openai",
                "    base_url: https://openrouter.ai/api",
                "roles:",
                "  normalize_text:",
                "    backend: local_backend",
                "    model: example",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="unsupported adapter"):
        load_profile(profile_path)


def test_profile_role_resolution_defaults_reduce_to_normalize(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  text_backend:",
                "    adapter: litellm",
                "    topology: local_attached",
                "    provider: lmstudio",
                "    base_url: http://127.0.0.1:8090",
                "roles:",
                "  normalize_text:",
                "    backend: text_backend",
                "    model: model-a",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    normalize = profile.resolve_role("normalize_text")
    reduce_role = profile.resolve_role("reduce_text")

    assert normalize is not None
    assert reduce_role is not None
    assert normalize.backend == reduce_role.backend
    assert normalize.model == reduce_role.model


def test_profile_expands_env_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCRIBA_TEST_API_KEY", "secret-token")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "backends:",
                "  remote_backend:",
                "    adapter: litellm",
                "    topology: remote",
                "    provider: openrouter",
                "    model_origin: hosted_weights",
                "    base_url: https://openrouter.ai/api",
                "    api_key: ${SCRIBA_TEST_API_KEY}",
                "roles:",
                "  normalize_text:",
                "    backend: remote_backend",
                "    model: qwen/qwen3.5-35b-a3b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    assert profile.backends["remote_backend"].api_key == "secret-token"


def test_profile_expands_env_for_artifacts_and_role_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCRIBA_TEST_ARTIFACTS", "./artifacts-env")
    monkeypatch.setenv("SCRIBA_TEST_MODEL", "qwen/qwen3.5-flash-02-23")

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "artifacts:",
                "  root: ${SCRIBA_TEST_ARTIFACTS}",
                "  run_id: test-run",
                "backends:",
                "  remote_backend:",
                "    adapter: litellm",
                "    topology: remote",
                "    provider: openrouter",
                "    model_origin: hosted_weights",
                "    base_url: https://openrouter.ai/api",
                "roles:",
                "  normalize_text:",
                "    backend: remote_backend",
                "    model: ${SCRIBA_TEST_MODEL}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_profile(profile_path)
    assert str(profile.artifacts.root) == "artifacts-env"
    assert profile.roles["normalize_text"].model == "qwen/qwen3.5-flash-02-23"
