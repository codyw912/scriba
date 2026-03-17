"""Tests for benchmark-aware matrix report rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any, Callable, cast

import pytest


def _load_report_module() -> dict[str, object]:
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[1] / "scripts" / "render_matrix_report.py"
        ),
        run_name="test_render_matrix_report",
    )


def test_benchmark_context_reads_fixture_and_variant_metadata(tmp_path: Path) -> None:
    module = _load_report_module()
    benchmark_context = cast(
        Callable[..., dict[str, Any]], module["_benchmark_context"]
    )

    benchmark_root = tmp_path / "samples" / "benchmarks" / "v1"
    manifests = benchmark_root / "manifests"
    inputs = benchmark_root / "generated_pdfs" / "mini_api"
    gold_markdown = benchmark_root / "gold_markdown"
    gold_contracts = benchmark_root / "gold_contracts"
    manifests.mkdir(parents=True)
    inputs.mkdir(parents=True)
    gold_markdown.mkdir(parents=True)
    gold_contracts.mkdir(parents=True)

    (manifests / "fixtures.json").write_text(
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
    (manifests / "variants.jsonl").write_text(
        json.dumps(
            {
                "fixture_id": "mini_api",
                "variant_id": "clean_pdf",
                "variant_family": "clean_pdf",
                "noise_level": "none",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (gold_markdown / "mini_api.md").write_text("# Mini API\n", encoding="utf-8")
    (gold_contracts / "mini_api.json").write_text("{}\n", encoding="utf-8")
    input_path = inputs / "clean_pdf.md"
    input_path.write_text("# Mini API\n", encoding="utf-8")

    context = benchmark_context(input_ref=str(input_path))

    assert context["fixture_id"] == "mini_api"
    assert context["variant_id"] == "clean_pdf"
    assert context["variant_family"] == "clean_pdf"
    assert context["noise_level"] == "none"
    assert context["source_kind"] == "synthetic"
    assert context["size_bucket"] == "small"
    assert context["doc_type"] == "api"
    assert str(context["gold_markdown_path"]).endswith("gold_markdown/mini_api.md")
    assert str(context["gold_contract_path"]).endswith("gold_contracts/mini_api.json")


def test_benchmark_lane_rows_expand_for_synthetic_run(tmp_path: Path) -> None:
    module = _load_report_module()
    benchmark_lane_rows_for_run = cast(
        Callable[..., list[dict[str, Any]]],
        module["_benchmark_lane_rows_for_run"],
    )

    benchmark_root = tmp_path / "samples" / "benchmarks" / "v1"
    gold_markdown = benchmark_root / "gold_markdown"
    gold_contracts = benchmark_root / "gold_contracts"
    artifacts_root = tmp_path / "artifacts"
    run_id = "bench-run"
    (artifacts_root / run_id / "raw").mkdir(parents=True)
    (artifacts_root / run_id / "final").mkdir(parents=True)
    gold_markdown.mkdir(parents=True)
    gold_contracts.mkdir(parents=True)

    (gold_markdown / "mini_api.md").write_text(
        "# Mini API\n\nGET /v1/ping\n", encoding="utf-8"
    )
    (gold_contracts / "mini_api.json").write_text(
        json.dumps({"required_endpoints": ["GET /v1/ping"]}, indent=2),
        encoding="utf-8",
    )
    (artifacts_root / run_id / "raw" / "extracted.md").write_text(
        "# Mini API\n\nGET /v1/ping\n", encoding="utf-8"
    )
    (artifacts_root / run_id / "final" / "merged.md").write_text(
        "# Mini API\n\nGET /v1/ping\n", encoding="utf-8"
    )

    lane_rows = benchmark_lane_rows_for_run(
        row={
            "run_id": run_id,
            "fixture_id": "mini_api",
            "variant_id": "clean_pdf",
            "variant_family": "clean_pdf",
            "noise_level": "none",
            "source_kind": "synthetic",
            "size_bucket": "small",
            "doc_type": "api",
            "hard_errors": 0,
            "gold_markdown_path": str(gold_markdown / "mini_api.md"),
            "gold_contract_path": str(gold_contracts / "mini_api.json"),
        },
        artifacts_root=artifacts_root,
    )

    lanes = {row["lane"] for row in lane_rows}
    assert lanes == {"ocr_lane", "full_pipeline_lane", "contract_lane"}


def test_quality_snapshot_includes_ocr_metrics(tmp_path: Path) -> None:
    module = _load_report_module()
    quality_snapshot_from_paths = cast(
        Callable[..., dict[str, Any]],
        module["_quality_snapshot_from_paths"],
    )

    source_path = tmp_path / "source.md"
    output_path = tmp_path / "output.md"
    source_path.write_text(
        "\n".join(
            [
                "# Sample",
                "",
                "GET /v1/ping",
                "",
                "```python",
                "print('ok')",
                "```",
                "",
                "| name | value |",
                "| --- | --- |",
                "| ping | ok |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        "\n".join(
            [
                "# Sample",
                "",
                "GET /v1/ping",
                "",
                "## Extra Heading",
                "",
                "```python",
                "print('ok')",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = quality_snapshot_from_paths(
        source_path=source_path,
        output_path=output_path,
        hard_error_count=0,
    )

    assert snapshot["char_error_rate"] > 0
    assert snapshot["word_error_rate"] > 0
    assert snapshot["code_block_integrity_score"] == 1.0
    assert snapshot["table_retention_score"] == 0.0
    assert snapshot["hallucination_rate"] > 0
    assert snapshot["hallucinated_heading_count"] == 1
    assert snapshot["hallucinated_endpoint_count"] == 0


def test_quality_snapshot_uses_none_for_missing_structures(tmp_path: Path) -> None:
    module = _load_report_module()
    quality_snapshot_from_paths = cast(
        Callable[..., dict[str, Any]],
        module["_quality_snapshot_from_paths"],
    )

    source_path = tmp_path / "source.md"
    output_path = tmp_path / "output.md"
    source_path.write_text("# Sample\n\nGET /v1/ping\n", encoding="utf-8")
    output_path.write_text("# Sample\n\nGET /v1/ping\n", encoding="utf-8")

    snapshot = quality_snapshot_from_paths(
        source_path=source_path,
        output_path=output_path,
        hard_error_count=0,
    )

    assert snapshot["char_error_rate"] == pytest.approx(0.0)
    assert snapshot["word_error_rate"] == pytest.approx(0.0)
    assert snapshot["code_block_integrity_score"] is None
    assert snapshot["table_retention_score"] is None
    assert snapshot["hallucination_rate"] == pytest.approx(0.0)
    assert snapshot["omission_severity_bucket"] == "none"
    assert snapshot["omitted_endpoint_count"] == 0
    assert snapshot["omitted_heading_count"] == 0


def test_quality_snapshot_includes_omission_severity_bucket(tmp_path: Path) -> None:
    module = _load_report_module()
    quality_snapshot_from_paths = cast(
        Callable[..., dict[str, Any]],
        module["_quality_snapshot_from_paths"],
    )

    source_path = tmp_path / "source.md"
    output_path = tmp_path / "output.md"
    source_path.write_text(
        "# Sample\n\nGET /v1/ping\n\n## Auth\n\n/health\n\n200\n",
        encoding="utf-8",
    )
    output_path.write_text("# Sample\n\n", encoding="utf-8")

    snapshot = quality_snapshot_from_paths(
        source_path=source_path,
        output_path=output_path,
        hard_error_count=0,
    )

    assert snapshot["omission_severity_bucket"] == "critical"
    assert snapshot["omitted_endpoint_count"] == 1
    assert snapshot["omitted_heading_count"] == 1
    assert snapshot["omitted_path_count"] == 2
    assert snapshot["omitted_number_count"] == 1


def test_benchmark_aggregate_rows_group_by_requested_dimension() -> None:
    module = _load_report_module()
    benchmark_aggregate_rows = cast(
        Callable[..., list[dict[str, Any]]],
        module["_benchmark_aggregate_rows"],
    )

    rows = [
        {
            "lane": "ocr_lane",
            "noise_level": "low",
            "source_kind": "synthetic",
            "quality_score": 90.0,
            "char_error_rate": 0.2,
            "word_error_rate": 0.3,
            "code_block_integrity_score": 1.0,
            "table_retention_score": 0.5,
            "hallucination_rate": 0.1,
            "contract_recall": None,
            "hard_errors": 0,
        },
        {
            "lane": "full_pipeline_lane",
            "noise_level": "low",
            "source_kind": "synthetic",
            "quality_score": 70.0,
            "char_error_rate": 0.4,
            "word_error_rate": 0.5,
            "code_block_integrity_score": 0.0,
            "table_retention_score": 0.0,
            "hallucination_rate": 0.2,
            "contract_recall": 0.8,
            "hard_errors": 1,
        },
        {
            "lane": "ocr_lane",
            "noise_level": "high",
            "source_kind": "synthetic",
            "quality_score": 50.0,
            "char_error_rate": 0.8,
            "word_error_rate": 0.9,
            "code_block_integrity_score": 0.2,
            "table_retention_score": 0.1,
            "hallucination_rate": 0.4,
            "contract_recall": None,
            "hard_errors": 1,
        },
    ]

    aggregates = benchmark_aggregate_rows(
        benchmark_lane_rows=rows,
        group_keys=("noise_level", "source_kind"),
    )

    assert len(aggregates) == 2
    low_row = next(row for row in aggregates if row["noise_level"] == "low")
    assert low_row["rows"] == 2
    assert low_row["avg_quality"] == "80.00"
    assert low_row["avg_char_error_rate"] == "0.300"
    assert low_row["avg_contract_recall"] == "0.800"
    assert low_row["omission_none_rows"] == 0
    assert low_row["omission_low_rows"] == 0
    assert low_row["omission_medium_rows"] == 0
    assert low_row["omission_high_rows"] == 0
    assert low_row["omission_critical_rows"] == 0
    assert low_row["hard_error_runs"] == 1


def test_benchmark_aggregate_rows_count_omission_buckets() -> None:
    module = _load_report_module()
    benchmark_aggregate_rows = cast(
        Callable[..., list[dict[str, Any]]],
        module["_benchmark_aggregate_rows"],
    )

    aggregates = benchmark_aggregate_rows(
        benchmark_lane_rows=[
            {
                "lane": "ocr_lane",
                "source_kind": "synthetic",
                "quality_score": 90.0,
                "char_error_rate": 0.1,
                "word_error_rate": 0.2,
                "code_block_integrity_score": 1.0,
                "table_retention_score": 1.0,
                "hallucination_rate": 0.0,
                "contract_recall": None,
                "omission_severity_bucket": "none",
                "hard_errors": 0,
            },
            {
                "lane": "ocr_lane",
                "source_kind": "synthetic",
                "quality_score": 70.0,
                "char_error_rate": 0.5,
                "word_error_rate": 0.6,
                "code_block_integrity_score": 0.0,
                "table_retention_score": 0.0,
                "hallucination_rate": 0.1,
                "contract_recall": None,
                "omission_severity_bucket": "high",
                "hard_errors": 1,
            },
        ],
        group_keys=("lane", "source_kind"),
    )

    row = aggregates[0]
    assert row["omission_none_rows"] == 1
    assert row["omission_high_rows"] == 1
    assert row["omission_low_rows"] == 0
    assert row["omission_medium_rows"] == 0
    assert row["omission_critical_rows"] == 0


def test_selection_summary_filters_profiles_by_constraints() -> None:
    module = _load_report_module()
    selection_summary = cast(
        Callable[..., dict[str, Any] | None],
        module["_selection_summary"],
    )

    summary = selection_summary(
        profile_summary_rows=[
            {
                "profile": "profiles/a.yaml",
                "provider": "openrouter",
                "avg_quality": 82.0,
                "avg_visible_tok_s": 15.0,
                "avg_tok_s": 12.0,
                "hard_error_rate": 0.0,
                "avg_cost_usd": 0.02,
            },
            {
                "profile": "profiles/b.yaml",
                "provider": "openrouter",
                "avg_quality": 70.0,
                "avg_visible_tok_s": 18.0,
                "avg_tok_s": 16.0,
                "hard_error_rate": 0.0,
                "avg_cost_usd": 0.01,
            },
            {
                "profile": "profiles/c.yaml",
                "provider": "openrouter",
                "avg_quality": 85.0,
                "avg_visible_tok_s": 14.0,
                "avg_tok_s": 13.0,
                "hard_error_rate": 0.2,
                "avg_cost_usd": 0.08,
            },
        ],
        quality_floor=75.0,
        max_cost_usd=0.05,
        throughput_target=10.0,
        max_hard_error_rate=0.1,
    )

    assert summary is not None
    assert summary["eligible_count"] == 1
    assert summary["recommended_profile"] == "profiles/a.yaml"

    rejected = {row["profile"]: row["reasons"] for row in summary["rejected_profiles"]}
    assert rejected["profiles/b.yaml"] == ["quality_below_floor"]
    assert rejected["profiles/c.yaml"] == [
        "cost_above_budget",
        "hard_error_rate_above_limit",
    ]


def test_selection_summary_returns_none_without_constraints() -> None:
    module = _load_report_module()
    selection_summary = cast(
        Callable[..., dict[str, Any] | None],
        module["_selection_summary"],
    )

    summary = selection_summary(
        profile_summary_rows=[
            {
                "profile": "profiles/a.yaml",
                "provider": "openrouter",
                "avg_quality": 82.0,
                "avg_visible_tok_s": 15.0,
                "avg_tok_s": 12.0,
                "hard_error_rate": 0.0,
                "avg_cost_usd": 0.02,
            }
        ],
        quality_floor=None,
        max_cost_usd=None,
        throughput_target=None,
        max_hard_error_rate=None,
    )

    assert summary is None
