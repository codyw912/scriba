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
