"""Deterministic end-to-end benchmark/report integration test."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _write_passthrough_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profile.yaml"
    artifacts_root = tmp_path / "artifacts"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "artifacts:",
                f"  root: {artifacts_root}",
                "  run_id: auto",
                "stages:",
                "  extract:",
                "    enabled: true",
                "  clean:",
                "    enabled: true",
                "  sectionize:",
                "    enabled: true",
                "    target_tokens: 5000",
                "    overlap_tokens: 400",
                "  normalize_map:",
                "    enabled: true",
                "    temperature: 0.0",
                "    request_timeout_s: 600",
                "  reduce:",
                "    enabled: true",
                "  validate:",
                "    enabled: true",
                "    fail_on_hard_errors: false",
                "  export:",
                "    enabled: true",
                "    multi_file: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return profile_path


def _write_benchmark_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark_root = tmp_path / "samples" / "benchmarks" / "v1"
    samples_dir = benchmark_root / "generated_pdfs" / "mini_api"
    manifests_dir = benchmark_root / "manifests"
    gold_markdown_dir = benchmark_root / "gold_markdown"
    gold_contract_dir = benchmark_root / "gold_contracts"
    samples_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    gold_markdown_dir.mkdir(parents=True)
    gold_contract_dir.mkdir(parents=True)

    (manifests_dir / "fixtures.json").write_text(
        json.dumps(
            [
                {
                    "fixture_id": "mini_api",
                    "source_markdown": "gold_markdown/mini_api.md",
                    "size_bucket": "small",
                    "doc_type": "api",
                    "has_contract": True,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifests_dir / "variants.jsonl").write_text(
        json.dumps(
            {
                "fixture_id": "mini_api",
                "variant_id": "clean_pdf",
                "variant_family": "clean_pdf",
                "pdf_path": "generated_pdfs/mini_api/clean_pdf.pdf",
                "seed": 1,
                "renderer": "playwright_chromium_pdf",
                "renderer_version": "test",
                "noise_level": "none",
                "transform_params": {},
                "generated_at": "2026-03-12T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (gold_markdown_dir / "mini_api.md").write_text(
        "# Mini API\n\nGET /v1/ping\n", encoding="utf-8"
    )
    (gold_contract_dir / "mini_api.json").write_text(
        json.dumps({"required_endpoints": ["GET /v1/ping"]}, indent=2),
        encoding="utf-8",
    )
    input_path = samples_dir / "clean_pdf.md"
    input_path.write_text("# Mini API\n\nGET /v1/ping\n", encoding="utf-8")
    return benchmark_root, samples_dir, input_path


def test_benchmark_scripts_run_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = _write_passthrough_profile(tmp_path)
    _, samples_dir, _ = _write_benchmark_fixture(tmp_path)

    matrix_log = tmp_path / "matrix_runs.jsonl"
    matrix_report = tmp_path / "matrix_report.md"
    matrix_report_json = tmp_path / "matrix_report.json"

    run_matrix = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_matrix.sh"),
            "--profile",
            str(profile_path),
            "--samples-dir",
            str(samples_dir),
            "--output",
            str(matrix_log),
            "--reset-log",
            "--max-runs",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert matrix_log.exists(), run_matrix.stderr

    rows = [
        json.loads(line) for line in matrix_log.read_text(encoding="utf-8").splitlines()
    ]
    completed_rows = [row for row in rows if row.get("run_id")]
    assert len(completed_rows) == 1
    row = completed_rows[0]
    assert row["fixture_id"] == "mini_api"
    assert row["variant_id"] == "clean_pdf"
    assert row["variant_family"] == "clean_pdf"
    assert row["noise_level"] == "none"
    assert row["source_kind"] == "synthetic"
    run_id = str(row["run_id"])

    render_report = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "render_matrix_report.py"),
            "--matrix-jsonl",
            str(matrix_log),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(matrix_report),
            "--json-output",
            str(matrix_report_json),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert matrix_report.exists(), render_report.stderr
    assert matrix_report_json.exists(), render_report.stderr

    report_json = json.loads(matrix_report_json.read_text(encoding="utf-8"))
    report_row = next(
        item for item in report_json["rows"] if item.get("run_id") == run_id
    )
    assert report_row["fixture_id"] == "mini_api"
    assert report_row["variant_id"] == "clean_pdf"
    assert report_row["source_kind"] == "synthetic"

    lane_rows = report_json["benchmark_lane_rows"]
    lanes = {item["lane"] for item in lane_rows if item.get("run_id") == run_id}
    assert {"ocr_lane", "full_pipeline_lane", "contract_lane"}.issubset(lanes)
    assert report_json["benchmark_lane_summary"]
    assert report_json["benchmark_noise_level_summary"]
    assert report_json["benchmark_size_bucket_summary"]
    assert report_json["benchmark_doc_type_summary"]
