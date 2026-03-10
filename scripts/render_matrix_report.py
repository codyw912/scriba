#!/usr/bin/env -S uv run --python 3.12

"""Render markdown report for matrix run outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from scriba.pipeline import load_profile, run_doctor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matrix benchmark markdown report.",
    )
    parser.add_argument(
        "--matrix-jsonl",
        default="samples/matrix_runs.jsonl",
        help="Path to matrix run JSONL log",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Artifacts root path",
    )
    parser.add_argument(
        "--output",
        default="samples/matrix_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to load before doctor snapshots",
    )
    return parser.parse_args()


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def _table_safe(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text.replace("|", "/").replace("\n", " ").strip()


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_completed_status(status: str) -> bool:
    return status in {"completed", "completed_with_validation_errors"}


def main() -> int:
    args = _parse_args()
    matrix_path = Path(args.matrix_jsonl).expanduser().resolve()
    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    env_file_path = Path(args.env_file).expanduser().resolve()

    _load_env_file(env_file_path)

    rows: list[dict[str, object]] = []
    profile_summary: dict[str, dict[str, object]] = {}
    campaign_summary: dict[str, dict[str, object]] = {}
    campaign_profile_tok: dict[str, dict[str, list[float]]] = {}
    profile_context_cache: dict[str, dict[str, object]] = {}
    doctor_snapshot_cache: dict[tuple[str, str], dict[str, object]] = {}
    if matrix_path.exists() and matrix_path.is_file():
        for line in matrix_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            run_id = str(item.get("run_id", "")).strip()
            profile_ref = str(item.get("profile", "")).strip()
            input_ref = str(item.get("input", "")).strip()
            profile_context = _profile_context(
                profile_ref=profile_ref,
                cache=profile_context_cache,
            )
            doctor_snapshot = _doctor_snapshot(
                profile_ref=profile_ref,
                input_ref=input_ref,
                cache=doctor_snapshot_cache,
            )
            telemetry = {}
            if run_id:
                manifest = artifacts_root / run_id / "map" / "manifest.json"
                if manifest.exists() and manifest.is_file():
                    try:
                        data = json.loads(manifest.read_text(encoding="utf-8"))
                        maybe = data.get("processed_telemetry")
                        if isinstance(maybe, dict):
                            telemetry = maybe
                    except json.JSONDecodeError:
                        telemetry = {}

            campaign_id = str(item.get("campaign_id", "")).strip() or "legacy"
            preset = str(item.get("preset", "")).strip() or "custom"
            status = str(item.get("status", ""))
            timestamp = item.get("timestamp")

            rows.append(
                {
                    "timestamp": timestamp,
                    "campaign_id": campaign_id,
                    "preset": preset,
                    "profile": item.get("profile"),
                    "adapter": profile_context.get("adapter"),
                    "topology": profile_context.get("topology"),
                    "provider": profile_context.get("provider"),
                    "input": Path(input_ref).name,
                    "input_ref": input_ref,
                    "run_id": run_id,
                    "status": status,
                    "tok_s": telemetry.get("effective_tokens_per_second"),
                    "latency_s": telemetry.get("latency_s"),
                    "completion_tokens": telemetry.get("completion_tokens"),
                    "output_tokens_est": telemetry.get("output_tokens_est"),
                    "visible_tok_s": _visible_tok_s(telemetry),
                    "completion_output_ratio": _completion_output_ratio(telemetry),
                    "source": (
                        "usage.completion_tokens"
                        if telemetry.get("completion_tokens") is not None
                        else ("output_tokens_est" if telemetry else "n/a")
                    ),
                    "processed": (
                        None
                        if not run_id
                        else _manifest_processed(artifacts_root, run_id)
                    ),
                    "validation_ok": (
                        None if not run_id else _validation_ok(artifacts_root, run_id)
                    ),
                    "hard_errors": (
                        None
                        if not run_id
                        else _validation_hard_error_count(artifacts_root, run_id)
                    ),
                    "missing_endpoints": (
                        None
                        if not run_id
                        else _validation_missing_endpoint_count(artifacts_root, run_id)
                    ),
                    "doctor_warning_count": doctor_snapshot.get("warning_count"),
                    "doctor_warning_preview": _table_safe(
                        doctor_snapshot.get("warning_preview")
                    ),
                    "base_quality_score": (
                        None
                        if not run_id
                        else _quality_score(
                            artifacts_root=artifacts_root, run_id=run_id
                        )
                    ),
                    "quality_score": (
                        None
                        if not run_id
                        else _quality_score(
                            artifacts_root=artifacts_root, run_id=run_id
                        )
                    ),
                    "quality_tier": (
                        None
                        if not run_id
                        else _quality_tier(artifacts_root=artifacts_root, run_id=run_id)
                    ),
                    "endpoint_recall": (
                        None
                        if not run_id
                        else _quality_metric(
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="endpoint_recall",
                        )
                    ),
                    "heading_recall": (
                        None
                        if not run_id
                        else _quality_metric(
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="heading_recall",
                        )
                    ),
                    "endpoint_precision": (
                        None
                        if not run_id
                        else _quality_metric(
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="endpoint_precision",
                        )
                    ),
                    "heading_precision": (
                        None
                        if not run_id
                        else _quality_metric(
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="heading_precision",
                        )
                    ),
                    "content_f1": (
                        None
                        if not run_id
                        else _quality_metric(
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="content_f1",
                        )
                    ),
                    "contract_recall": (
                        None
                        if not run_id
                        else _contract_metric(
                            input_ref=input_ref,
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="contract_recall",
                        )
                    ),
                    "contract_failures": (
                        None
                        if not run_id
                        else _contract_metric(
                            input_ref=input_ref,
                            artifacts_root=artifacts_root,
                            run_id=run_id,
                            key="contract_failures",
                        )
                    ),
                }
            )
            latest_row = rows[-1]

            base_quality = latest_row.get("base_quality_score")
            contract_recall = latest_row.get("contract_recall")
            if isinstance(base_quality, (int, float)) and isinstance(
                contract_recall, (int, float)
            ):
                blended_quality = round(float(base_quality) * float(contract_recall), 2)
                latest_row["quality_score"] = blended_quality
                latest_row["quality_tier"] = _quality_tier_for_score(blended_quality)

            latest_row["quality_gate_ok"] = _quality_gate_ok(latest_row)
            latest_row["reasoning_heavy"] = _reasoning_heavy(latest_row)
            latest_row["speed_gate_ok"] = _speed_gate_ok(latest_row)

            campaign_bucket = campaign_summary.setdefault(
                campaign_id,
                {
                    "campaign_id": campaign_id,
                    "preset": preset,
                    "rows": 0,
                    "completed": 0,
                    "failed": 0,
                    "doctor_failed": 0,
                    "skipped": 0,
                    "tok_values": [],
                    "quality_values": [],
                    "started_at": None,
                    "last_at": None,
                },
            )
            campaign_bucket["rows"] = int(campaign_bucket["rows"]) + 1
            if _is_completed_status(status):
                campaign_bucket["completed"] = int(campaign_bucket["completed"]) + 1
            elif status == "failed_runtime":
                campaign_bucket["failed"] = int(campaign_bucket["failed"]) + 1
            elif status == "doctor_failed":
                campaign_bucket["doctor_failed"] = (
                    int(campaign_bucket["doctor_failed"]) + 1
                )
            elif status.startswith("skipped"):
                campaign_bucket["skipped"] = int(campaign_bucket["skipped"]) + 1

            parsed_ts = _parse_timestamp(timestamp)
            started_at = campaign_bucket.get("started_at")
            if parsed_ts is not None:
                if started_at is None or (
                    isinstance(started_at, datetime) and parsed_ts < started_at
                ):
                    campaign_bucket["started_at"] = parsed_ts
                last_at = campaign_bucket.get("last_at")
                if last_at is None or (
                    isinstance(last_at, datetime) and parsed_ts > last_at
                ):
                    campaign_bucket["last_at"] = parsed_ts

            profile = str(item.get("profile", ""))
            bucket = profile_summary.setdefault(
                profile,
                {
                    "adapter": profile_context.get("adapter"),
                    "topology": profile_context.get("topology"),
                    "provider": profile_context.get("provider"),
                    "rows": 0,
                    "completed": 0,
                    "failed": 0,
                    "doctor_failed": 0,
                    "skipped": 0,
                    "tok_values": [],
                    "quality_values": [],
                    "visible_tok_values": [],
                    "completion_output_ratio_values": [],
                    "endpoint_recall_values": [],
                    "contract_recall_values": [],
                    "content_f1_values": [],
                    "hard_error_runs": 0,
                    "contract_failed_runs": 0,
                    "quality_gate_checks": 0,
                    "quality_gate_passes": 0,
                    "speed_gate_checks": 0,
                    "speed_gate_passes": 0,
                    "visible_tok_comparable_values": [],
                },
            )
            bucket["rows"] = int(bucket["rows"]) + 1
            if _is_completed_status(status):
                bucket["completed"] = int(bucket["completed"]) + 1
            elif status == "failed_runtime":
                bucket["failed"] = int(bucket["failed"]) + 1
            elif status == "doctor_failed":
                bucket["doctor_failed"] = int(bucket["doctor_failed"]) + 1
            elif status.startswith("skipped"):
                bucket["skipped"] = int(bucket["skipped"]) + 1

            counts_as_completed = _is_completed_status(status)

            tok_value = telemetry.get("effective_tokens_per_second")
            if counts_as_completed and isinstance(tok_value, (int, float)):
                tok_values = bucket["tok_values"]
                assert isinstance(tok_values, list)
                tok_values.append(float(tok_value))

                campaign_tok_values = campaign_bucket["tok_values"]
                assert isinstance(campaign_tok_values, list)
                campaign_tok_values.append(float(tok_value))

                campaign_profile_values = campaign_profile_tok.setdefault(
                    campaign_id,
                    {},
                ).setdefault(
                    profile,
                    [],
                )
                campaign_profile_values.append(float(tok_value))

            visible_tok_s = latest_row.get("visible_tok_s")
            if counts_as_completed and isinstance(visible_tok_s, (int, float)):
                visible_tok_values = bucket["visible_tok_values"]
                assert isinstance(visible_tok_values, list)
                visible_tok_values.append(float(visible_tok_s))

            completion_output_ratio = latest_row.get("completion_output_ratio")
            if counts_as_completed and isinstance(
                completion_output_ratio, (int, float)
            ):
                completion_output_ratio_values = bucket[
                    "completion_output_ratio_values"
                ]
                assert isinstance(completion_output_ratio_values, list)
                completion_output_ratio_values.append(float(completion_output_ratio))

            quality_value = latest_row.get("quality_score")
            if counts_as_completed and isinstance(quality_value, (int, float)):
                campaign_quality_values = campaign_bucket["quality_values"]
                assert isinstance(campaign_quality_values, list)
                campaign_quality_values.append(float(quality_value))

                profile_quality_values = bucket["quality_values"]
                assert isinstance(profile_quality_values, list)
                profile_quality_values.append(float(quality_value))

            endpoint_recall_value = latest_row.get("endpoint_recall")
            if counts_as_completed and isinstance(endpoint_recall_value, (int, float)):
                endpoint_recall_values = bucket["endpoint_recall_values"]
                assert isinstance(endpoint_recall_values, list)
                endpoint_recall_values.append(float(endpoint_recall_value))

            contract_recall_value = latest_row.get("contract_recall")
            if counts_as_completed and isinstance(contract_recall_value, (int, float)):
                contract_recall_values = bucket["contract_recall_values"]
                assert isinstance(contract_recall_values, list)
                contract_recall_values.append(float(contract_recall_value))

            content_f1_value = latest_row.get("content_f1")
            if counts_as_completed and isinstance(content_f1_value, (int, float)):
                content_f1_values = bucket["content_f1_values"]
                assert isinstance(content_f1_values, list)
                content_f1_values.append(float(content_f1_value))

            contract_failures_value = latest_row.get("contract_failures")
            if isinstance(contract_failures_value, int) and contract_failures_value > 0:
                bucket["contract_failed_runs"] = int(bucket["contract_failed_runs"]) + 1

            hard_error_value = latest_row.get("hard_errors")
            if isinstance(hard_error_value, int) and hard_error_value > 0:
                bucket["hard_error_runs"] = int(bucket["hard_error_runs"]) + 1

            quality_gate_value = latest_row.get("quality_gate_ok")
            if isinstance(quality_gate_value, bool):
                bucket["quality_gate_checks"] = int(bucket["quality_gate_checks"]) + 1
                if quality_gate_value:
                    bucket["quality_gate_passes"] = (
                        int(bucket["quality_gate_passes"]) + 1
                    )

            speed_gate_value = latest_row.get("speed_gate_ok")
            if isinstance(speed_gate_value, bool):
                bucket["speed_gate_checks"] = int(bucket["speed_gate_checks"]) + 1
                if speed_gate_value:
                    bucket["speed_gate_passes"] = int(bucket["speed_gate_passes"]) + 1
                    comparable = latest_row.get("visible_tok_s")
                    if isinstance(comparable, (int, float)):
                        comparable_values = bucket["visible_tok_comparable_values"]
                        assert isinstance(comparable_values, list)
                        comparable_values.append(float(comparable))

    lines = [
        "# Matrix Report",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Matrix log: `{matrix_path}`",
        f"- Artifacts root: `{artifacts_root}`",
        "",
    ]

    profile_summary_rows: list[dict[str, object]] = []
    campaign_summary_rows: list[dict[str, object]] = []
    ranking_rows_json: list[dict[str, object]] = []
    trend_rows_json: list[dict[str, object]] = []

    if campaign_summary:
        lines.extend(
            [
                "## Campaign Summary",
                "",
                "| campaign_id | preset | rows | completed | failed | doctor_failed | skipped | avg_tok_s | avg_quality | started_at | last_at |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )

        ordered_campaigns = sorted(
            campaign_summary.values(),
            key=lambda bucket: (
                bucket.get("started_at") or datetime.min.replace(tzinfo=UTC),
                str(bucket.get("campaign_id", "")),
            ),
        )

        for bucket in ordered_campaigns:
            tok_values = bucket["tok_values"]
            assert isinstance(tok_values, list)
            avg_tok = (
                f"{sum(tok_values) / len(tok_values):.3f}" if tok_values else "n/a"
            )
            quality_values = bucket["quality_values"]
            assert isinstance(quality_values, list)
            avg_quality = (
                f"{sum(quality_values) / len(quality_values):.2f}"
                if quality_values
                else "n/a"
            )

            started_at = bucket.get("started_at")
            started_label = (
                started_at.isoformat() if isinstance(started_at, datetime) else "n/a"
            )
            last_at = bucket.get("last_at")
            last_label = last_at.isoformat() if isinstance(last_at, datetime) else "n/a"

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _fmt(bucket.get("campaign_id")),
                    _fmt(bucket.get("preset")),
                    _fmt(bucket.get("rows")),
                    _fmt(bucket.get("completed")),
                    _fmt(bucket.get("failed")),
                    _fmt(bucket.get("doctor_failed")),
                    _fmt(bucket.get("skipped")),
                    avg_tok,
                    avg_quality,
                    started_label,
                    last_label,
                )
            )

            campaign_summary_rows.append(
                {
                    "campaign_id": bucket.get("campaign_id"),
                    "preset": bucket.get("preset"),
                    "rows": bucket.get("rows"),
                    "completed": bucket.get("completed"),
                    "failed": bucket.get("failed"),
                    "doctor_failed": bucket.get("doctor_failed"),
                    "skipped": bucket.get("skipped"),
                    "avg_tok_s": avg_tok,
                    "avg_quality": avg_quality,
                    "started_at": started_label,
                    "last_at": last_label,
                }
            )

        lines.append("")

        if len(ordered_campaigns) >= 2:
            latest = ordered_campaigns[-1]
            previous = ordered_campaigns[-2]
            latest_id = str(latest.get("campaign_id", ""))
            previous_id = str(previous.get("campaign_id", ""))

            lines.extend(
                [
                    "## Campaign Trend (Latest vs Previous)",
                    "",
                    f"- Latest: `{latest_id}`",
                    f"- Previous: `{previous_id}`",
                    "",
                    "| profile | latest_avg_tok_s | previous_avg_tok_s | delta_tok_s | delta_pct |",
                    "|---|---:|---:|---:|---:|",
                ]
            )

            latest_profiles = campaign_profile_tok.get(latest_id, {})
            previous_profiles = campaign_profile_tok.get(previous_id, {})
            profile_names = sorted(
                set(latest_profiles.keys()) | set(previous_profiles.keys())
            )

            for profile in profile_names:
                latest_values = latest_profiles.get(profile, [])
                previous_values = previous_profiles.get(profile, [])

                latest_avg = (
                    sum(latest_values) / len(latest_values) if latest_values else None
                )
                previous_avg = (
                    sum(previous_values) / len(previous_values)
                    if previous_values
                    else None
                )

                delta = (
                    latest_avg - previous_avg
                    if latest_avg is not None and previous_avg is not None
                    else None
                )
                delta_pct = (
                    (delta / previous_avg) * 100.0
                    if delta is not None and previous_avg not in (None, 0)
                    else None
                )

                latest_label = "n/a" if latest_avg is None else f"{latest_avg:.3f}"
                previous_label = (
                    "n/a" if previous_avg is None else f"{previous_avg:.3f}"
                )
                delta_label = "n/a" if delta is None else f"{delta:+.3f}"
                delta_pct_label = "n/a" if delta_pct is None else f"{delta_pct:+.1f}%"

                lines.append(
                    "| {} | {} | {} | {} | {} |".format(
                        profile,
                        latest_label,
                        previous_label,
                        delta_label,
                        delta_pct_label,
                    )
                )
                trend_rows_json.append(
                    {
                        "profile": profile,
                        "latest_campaign": latest_id,
                        "previous_campaign": previous_id,
                        "latest_avg_tok_s": latest_label,
                        "previous_avg_tok_s": previous_label,
                        "delta_tok_s": delta_label,
                        "delta_pct": delta_pct_label,
                    }
                )

            if not profile_names:
                lines.append("| n/a | n/a | n/a | n/a | n/a |")

            lines.append("")

    if profile_summary:
        lines.extend(
            [
                "## Profile Summary",
                "",
                "| profile | adapter | topology | provider | rows | completed | failed | doctor_failed | skipped | avg_tok_s | avg_visible_tok_s | avg_visible_tok_s_comparable | avg_completion_output_ratio | avg_quality | avg_content_f1 | avg_endpoint_recall | avg_contract_recall | hard_error_rate | contract_fail_rate | quality_gate_pass_rate | speed_gate_pass_rate | min_tok_s | max_tok_s |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for profile, bucket in sorted(profile_summary.items()):
            tok_values = bucket["tok_values"]
            assert isinstance(tok_values, list)
            if tok_values:
                avg_tok = f"{sum(tok_values) / len(tok_values):.3f}"
                min_tok = f"{min(tok_values):.3f}"
                max_tok = f"{max(tok_values):.3f}"
            else:
                avg_tok = "n/a"
                min_tok = "n/a"
                max_tok = "n/a"

            visible_tok_values = bucket["visible_tok_values"]
            assert isinstance(visible_tok_values, list)
            avg_visible_tok = (
                f"{sum(visible_tok_values) / len(visible_tok_values):.3f}"
                if visible_tok_values
                else "n/a"
            )

            visible_tok_comparable_values = bucket["visible_tok_comparable_values"]
            assert isinstance(visible_tok_comparable_values, list)
            avg_visible_tok_comparable = (
                f"{sum(visible_tok_comparable_values) / len(visible_tok_comparable_values):.3f}"
                if visible_tok_comparable_values
                else "n/a"
            )

            completion_output_ratio_values = bucket["completion_output_ratio_values"]
            assert isinstance(completion_output_ratio_values, list)
            avg_completion_output_ratio = (
                f"{sum(completion_output_ratio_values) / len(completion_output_ratio_values):.3f}"
                if completion_output_ratio_values
                else "n/a"
            )

            quality_values = bucket["quality_values"]
            assert isinstance(quality_values, list)
            avg_quality = (
                f"{sum(quality_values) / len(quality_values):.2f}"
                if quality_values
                else "n/a"
            )

            endpoint_recall_values = bucket["endpoint_recall_values"]
            assert isinstance(endpoint_recall_values, list)
            avg_endpoint_recall = (
                f"{sum(endpoint_recall_values) / len(endpoint_recall_values):.3f}"
                if endpoint_recall_values
                else "n/a"
            )

            contract_recall_values = bucket["contract_recall_values"]
            assert isinstance(contract_recall_values, list)
            avg_contract_recall = (
                f"{sum(contract_recall_values) / len(contract_recall_values):.3f}"
                if contract_recall_values
                else "n/a"
            )

            content_f1_values = bucket["content_f1_values"]
            assert isinstance(content_f1_values, list)
            avg_content_f1 = (
                f"{sum(content_f1_values) / len(content_f1_values):.3f}"
                if content_f1_values
                else "n/a"
            )

            rows_count = int(bucket["rows"])
            hard_error_rate = (
                f"{(int(bucket['hard_error_runs']) / rows_count):.3f}"
                if rows_count > 0
                else "n/a"
            )
            contract_fail_rate = (
                f"{(int(bucket['contract_failed_runs']) / rows_count):.3f}"
                if rows_count > 0
                else "n/a"
            )
            quality_gate_checks = int(bucket.get("quality_gate_checks", 0))
            quality_gate_passes = int(bucket.get("quality_gate_passes", 0))
            quality_gate_pass_rate = (
                f"{(quality_gate_passes / quality_gate_checks):.3f}"
                if quality_gate_checks > 0
                else "n/a"
            )
            speed_gate_checks = int(bucket.get("speed_gate_checks", 0))
            speed_gate_passes = int(bucket.get("speed_gate_passes", 0))
            speed_gate_pass_rate = (
                f"{(speed_gate_passes / speed_gate_checks):.3f}"
                if speed_gate_checks > 0
                else "n/a"
            )

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    profile,
                    _fmt(bucket["adapter"]),
                    _fmt(bucket["topology"]),
                    _fmt(bucket["provider"]),
                    _fmt(bucket["rows"]),
                    _fmt(bucket["completed"]),
                    _fmt(bucket["failed"]),
                    _fmt(bucket["doctor_failed"]),
                    _fmt(bucket["skipped"]),
                    avg_tok,
                    avg_visible_tok,
                    avg_visible_tok_comparable,
                    avg_completion_output_ratio,
                    avg_quality,
                    avg_content_f1,
                    avg_endpoint_recall,
                    avg_contract_recall,
                    hard_error_rate,
                    contract_fail_rate,
                    quality_gate_pass_rate,
                    speed_gate_pass_rate,
                    min_tok,
                    max_tok,
                )
            )
            profile_summary_rows.append(
                {
                    "profile": profile,
                    "adapter": bucket["adapter"],
                    "topology": bucket["topology"],
                    "provider": bucket["provider"],
                    "rows": bucket["rows"],
                    "completed": bucket["completed"],
                    "failed": bucket["failed"],
                    "doctor_failed": bucket["doctor_failed"],
                    "skipped": bucket["skipped"],
                    "avg_tok_s": avg_tok,
                    "avg_visible_tok_s": avg_visible_tok,
                    "avg_visible_tok_s_comparable": avg_visible_tok_comparable,
                    "avg_completion_output_ratio": avg_completion_output_ratio,
                    "avg_quality": avg_quality,
                    "avg_content_f1": avg_content_f1,
                    "avg_endpoint_recall": avg_endpoint_recall,
                    "avg_contract_recall": avg_contract_recall,
                    "hard_error_rate": hard_error_rate,
                    "contract_fail_rate": contract_fail_rate,
                    "quality_gate_pass_rate": quality_gate_pass_rate,
                    "speed_gate_pass_rate": speed_gate_pass_rate,
                    "min_tok_s": min_tok,
                    "max_tok_s": max_tok,
                }
            )

        lines.append("")

        ranking_rows: list[tuple[float, int, str, object, object, str]] = []
        for profile, bucket in profile_summary.items():
            tok_values = bucket["tok_values"]
            assert isinstance(tok_values, list)
            completed = int(bucket["completed"])
            if tok_values:
                avg_tok = sum(tok_values) / len(tok_values)
                avg_tok_label = f"{avg_tok:.3f}"
            else:
                avg_tok = -1.0
                avg_tok_label = "n/a"
            ranking_rows.append(
                (
                    avg_tok,
                    completed,
                    profile,
                    bucket["topology"],
                    bucket["provider"],
                    avg_tok_label,
                )
            )

        ranking_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

        lines.extend(
            [
                "## Ranking",
                "",
                "| rank | profile | topology | provider | completed | avg_tok_s |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        for index, row in enumerate(ranking_rows, start=1):
            _, completed, profile, topology, provider, avg_tok_label = row
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    index,
                    profile,
                    _fmt(topology),
                    _fmt(provider),
                    completed,
                    avg_tok_label,
                )
            )
            ranking_rows_json.append(
                {
                    "rank": index,
                    "profile": profile,
                    "topology": topology,
                    "provider": provider,
                    "completed": completed,
                    "avg_tok_s": avg_tok_label,
                }
            )

        lines.append("")

        visible_ranking_rows: list[tuple[float, int, str, object, object, str]] = []
        for profile, bucket in profile_summary.items():
            values = bucket["visible_tok_comparable_values"]
            assert isinstance(values, list)
            completed = int(bucket["completed"])
            if values:
                avg_visible_float = sum(values) / len(values)
                avg_visible_label = f"{avg_visible_float:.3f}"
            else:
                avg_visible_float = -1.0
                avg_visible_label = "n/a"
            visible_ranking_rows.append(
                (
                    avg_visible_float,
                    completed,
                    profile,
                    bucket["topology"],
                    bucket["provider"],
                    avg_visible_label,
                )
            )

        visible_ranking_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

        lines.extend(
            [
                "## Visible Speed Ranking (Comparable Rows)",
                "",
                "| rank | profile | topology | provider | completed | avg_visible_tok_s |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        for index, row in enumerate(visible_ranking_rows, start=1):
            _, completed, profile, topology, provider, avg_visible_label = row
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    index,
                    profile,
                    _fmt(topology),
                    _fmt(provider),
                    completed,
                    avg_visible_label,
                )
            )

        lines.append("")

        quality_ranking_rows: list[tuple[float, int, str, object, object, str]] = []
        for profile, bucket in profile_summary.items():
            quality_values = bucket["quality_values"]
            assert isinstance(quality_values, list)
            completed = int(bucket["completed"])
            if quality_values:
                avg_quality_float = sum(quality_values) / len(quality_values)
                avg_quality_label = f"{avg_quality_float:.2f}"
            else:
                avg_quality_float = -1.0
                avg_quality_label = "n/a"
            quality_ranking_rows.append(
                (
                    avg_quality_float,
                    completed,
                    profile,
                    bucket["topology"],
                    bucket["provider"],
                    avg_quality_label,
                )
            )

        quality_ranking_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

        lines.extend(
            [
                "## Quality Ranking",
                "",
                "| rank | profile | topology | provider | completed | avg_quality |",
                "|---:|---|---|---|---:|---:|",
            ]
        )

        for index, row in enumerate(quality_ranking_rows, start=1):
            _, completed, profile, topology, provider, avg_quality_label = row
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    index,
                    profile,
                    _fmt(topology),
                    _fmt(provider),
                    completed,
                    avg_quality_label,
                )
            )

        lines.append("")

    lines.extend(
        [
            "## Per Run",
            "",
            "| timestamp | campaign_id | preset | profile | adapter | topology | provider | input | run_id | status | processed | tok_s | visible_tok_s | latency_s | completion_tokens | output_tokens_est | completion_output_ratio | reasoning_heavy | speed_gate_ok | quality | base_quality | content_f1 | endpoint_recall | endpoint_precision | heading_recall | heading_precision | contract_recall | contract_failures | quality_gate_ok | source | doctor_warning_count | doctor_warning_preview | validation_ok | hard_errors | missing_endpoints |",
            "|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---:|---:|",
        ]
    )

    if rows:
        for row in rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _fmt(row["timestamp"]),
                    _fmt(row["campaign_id"]),
                    _fmt(row["preset"]),
                    _fmt(row["profile"]),
                    _fmt(row["adapter"]),
                    _fmt(row["topology"]),
                    _fmt(row["provider"]),
                    _fmt(row["input"]),
                    _fmt(row["run_id"]),
                    _fmt(row["status"]),
                    _fmt(row["processed"]),
                    _fmt(row["tok_s"]),
                    _fmt(row.get("visible_tok_s")),
                    _fmt(row.get("latency_s")),
                    _fmt(row.get("completion_tokens")),
                    _fmt(row.get("output_tokens_est")),
                    _fmt(row.get("completion_output_ratio")),
                    _fmt(row.get("reasoning_heavy")),
                    _fmt(row.get("speed_gate_ok")),
                    _fmt(row["quality_score"]),
                    _fmt(row.get("base_quality_score")),
                    _fmt(row.get("content_f1")),
                    _fmt(row["endpoint_recall"]),
                    _fmt(row.get("endpoint_precision")),
                    _fmt(row["heading_recall"]),
                    _fmt(row.get("heading_precision")),
                    _fmt(row["contract_recall"]),
                    _fmt(row["contract_failures"]),
                    _fmt(row.get("quality_gate_ok")),
                    _fmt(row["source"]),
                    _fmt(row["doctor_warning_count"]),
                    _fmt(row["doctor_warning_preview"]),
                    _fmt(row["validation_ok"]),
                    _fmt(row["hard_errors"]),
                    _fmt(row["missing_endpoints"]),
                )
            )
    else:
        lines.append(
            "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote matrix report: {output_path}")

    if args.json_output:
        json_output_path = Path(args.json_output).expanduser().resolve()
        json_payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "matrix_log": str(matrix_path),
            "artifacts_root": str(artifacts_root),
            "campaign_summary": campaign_summary_rows,
            "profile_summary": profile_summary_rows,
            "ranking": ranking_rows_json,
            "trend_latest_vs_previous": trend_rows_json,
            "rows": rows,
        }
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote matrix JSON summary: {json_output_path}")

    return 0


def _manifest_processed(artifacts_root: Path, run_id: str) -> str | None:
    manifest = artifacts_root / run_id / "map" / "manifest.json"
    if not manifest.exists() or not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    processed = data.get("processed")
    chunk_count = data.get("chunk_count")
    if processed is None or chunk_count is None:
        return None
    return f"{processed}/{chunk_count}"


def _validation_data(artifacts_root: Path, run_id: str) -> dict[str, object] | None:
    report = artifacts_root / run_id / "final" / "validation_report.json"
    if not report.exists() or not report.is_file():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _validation_ok(artifacts_root: Path, run_id: str) -> object:
    data = _validation_data(artifacts_root, run_id)
    if not data:
        return None
    return data.get("ok")


def _validation_hard_error_count(artifacts_root: Path, run_id: str) -> object:
    data = _validation_data(artifacts_root, run_id)
    if not data:
        return None
    hard_errors = data.get("hard_errors")
    if isinstance(hard_errors, list):
        return len(hard_errors)
    return None


def _profile_context(
    *,
    profile_ref: str,
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    if profile_ref in cache:
        return cache[profile_ref]

    context: dict[str, object] = {
        "adapter": None,
        "topology": None,
        "provider": None,
    }

    if not profile_ref:
        cache[profile_ref] = context
        return context

    path = Path(profile_ref)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists() or not path.is_file():
        cache[profile_ref] = context
        return context

    try:
        profile = load_profile(path)
        role = profile.roles.get("normalize_text")
        if role:
            backend = profile.backends.get(role.backend)
            if backend:
                context = {
                    "adapter": backend.adapter,
                    "topology": backend.topology,
                    "provider": backend.provider,
                }
    except Exception:
        context = {
            "adapter": None,
            "topology": None,
            "provider": None,
        }

    cache[profile_ref] = context
    return context


def _doctor_snapshot(
    *,
    profile_ref: str,
    input_ref: str,
    cache: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    key = (profile_ref, input_ref)
    if key in cache:
        return cache[key]

    snapshot: dict[str, object] = {
        "warning_count": None,
        "warning_preview": None,
    }

    if not profile_ref or not input_ref:
        cache[key] = snapshot
        return snapshot

    profile_path = Path(profile_ref)
    if not profile_path.is_absolute():
        profile_path = (Path.cwd() / profile_path).resolve()
    input_path = Path(input_ref)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    if not profile_path.exists() or not profile_path.is_file():
        cache[key] = snapshot
        return snapshot

    try:
        profile = load_profile(profile_path)
        report = run_doctor(profile, input_path=input_path)
        warnings = report.get("warnings", [])
        if isinstance(warnings, list):
            warning_preview = str(warnings[0]) if warnings else None
            snapshot = {
                "warning_count": len(warnings),
                "warning_preview": warning_preview,
            }
    except Exception:
        snapshot = {
            "warning_count": None,
            "warning_preview": None,
        }

    cache[key] = snapshot
    return snapshot


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ) and len(value) >= 2:
            value = value[1:-1]
        os.environ[key] = value


def _validation_missing_endpoint_count(artifacts_root: Path, run_id: str) -> object:
    data = _validation_data(artifacts_root, run_id)
    if not data:
        return None
    checks = data.get("checks")
    if isinstance(checks, dict):
        return checks.get("missing_endpoint_count")
    return None


_QUALITY_CACHE: dict[str, dict[str, object]] = {}
_CONTRACT_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def _quality_metric(*, artifacts_root: Path, run_id: str, key: str) -> object:
    snapshot = _quality_snapshot(artifacts_root=artifacts_root, run_id=run_id)
    return snapshot.get(key)


def _quality_score(*, artifacts_root: Path, run_id: str) -> object:
    return _quality_metric(
        artifacts_root=artifacts_root, run_id=run_id, key="quality_score"
    )


def _quality_tier(*, artifacts_root: Path, run_id: str) -> object:
    return _quality_metric(
        artifacts_root=artifacts_root, run_id=run_id, key="quality_tier"
    )


def _quality_tier_for_score(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "fair"
    return "poor"


def _quality_snapshot(*, artifacts_root: Path, run_id: str) -> dict[str, object]:
    if run_id in _QUALITY_CACHE:
        return _QUALITY_CACHE[run_id]

    source_path = artifacts_root / run_id / "raw" / "cleaned.md"
    if not source_path.exists():
        source_path = artifacts_root / run_id / "raw" / "extracted.md"
    output_path = artifacts_root / run_id / "final" / "merged.md"

    if not source_path.exists() or not output_path.exists():
        snapshot = {
            "quality_score": None,
            "quality_tier": None,
            "content_f1": None,
            "endpoint_recall": None,
            "endpoint_precision": None,
            "endpoint_f1": None,
            "heading_recall": None,
            "heading_precision": None,
            "heading_f1": None,
            "path_recall": None,
            "number_recall": None,
            "length_ratio": None,
        }
        _QUALITY_CACHE[run_id] = snapshot
        return snapshot

    source_text = source_path.read_text(encoding="utf-8")
    output_text = output_path.read_text(encoding="utf-8")

    endpoint_recall = _recall_ratio(
        source=_extract_endpoints(source_text),
        output=_extract_endpoints(output_text),
    )
    endpoint_precision = _precision_ratio(
        source=_extract_endpoints(source_text),
        output=_extract_endpoints(output_text),
    )
    endpoint_f1 = _f1(endpoint_recall, endpoint_precision)
    heading_recall = _recall_ratio(
        source=_extract_headings(source_text),
        output=_extract_headings(output_text),
    )
    heading_precision = _precision_ratio(
        source=_extract_headings(source_text),
        output=_extract_headings(output_text),
    )
    heading_f1 = _f1(heading_recall, heading_precision)
    path_recall = _recall_ratio(
        source=_extract_paths(source_text),
        output=_extract_paths(output_text),
    )
    number_recall = _recall_ratio(
        source=_extract_numbers(source_text),
        output=_extract_numbers(output_text),
    )

    source_chars = len(source_text.strip())
    output_chars = len(output_text.strip())
    length_ratio = (output_chars / source_chars) if source_chars > 0 else 1.0
    length_score = _length_ratio_score(length_ratio)

    validation = _validation_data(artifacts_root, run_id) or {}
    hard_errors = validation.get("hard_errors")
    hard_error_count = len(hard_errors) if isinstance(hard_errors, list) else 0

    weighted = (
        0.45 * endpoint_recall
        + 0.20 * heading_recall
        + 0.15 * path_recall
        + 0.05 * number_recall
        + 0.15 * length_score
    )
    if hard_error_count > 0:
        weighted *= 0.5

    quality_score = round(weighted * 100.0, 2)
    quality_tier = _quality_tier_for_score(quality_score)

    snapshot = {
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "content_f1": round(0.7 * endpoint_f1 + 0.3 * heading_f1, 3),
        "endpoint_recall": round(endpoint_recall, 3),
        "endpoint_precision": round(endpoint_precision, 3),
        "endpoint_f1": round(endpoint_f1, 3),
        "heading_recall": round(heading_recall, 3),
        "heading_precision": round(heading_precision, 3),
        "heading_f1": round(heading_f1, 3),
        "path_recall": round(path_recall, 3),
        "number_recall": round(number_recall, 3),
        "length_ratio": round(length_ratio, 3),
    }
    _QUALITY_CACHE[run_id] = snapshot
    return snapshot


def _extract_endpoints(markdown: str) -> set[str]:
    methods = "GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD"
    pattern = re.compile(
        rf"\b({methods})\b(?:\s|[*_`]|:|-)+(/[A-Za-z0-9._~!$&'()*+,;=:@%/{{}}-]+)",
        re.IGNORECASE,
    )
    results: set[str] = set()
    for method, path in pattern.findall(markdown):
        canonical = _canonical_endpoint(f"{method} {path}")
        if canonical:
            results.add(canonical)
    return results


def _extract_headings(markdown: str) -> set[str]:
    headings: set[str] = set()
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        heading = match.group(1).strip().lower()
        if heading:
            headings.add(heading)
    return headings


def _extract_paths(markdown: str) -> set[str]:
    pattern = re.compile(r"(/[-A-Za-z0-9._~!$&'()*+,;=:@%/{}]+)")
    return {match.group(1).strip() for match in pattern.finditer(markdown)}


def _extract_numbers(markdown: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[._-]\d+)*\b", markdown))


def _contract_metric(
    *,
    input_ref: str,
    artifacts_root: Path,
    run_id: str,
    key: str,
) -> object:
    snapshot = _contract_snapshot(
        input_ref=input_ref,
        artifacts_root=artifacts_root,
        run_id=run_id,
    )
    return snapshot.get(key)


def _contract_snapshot(
    *,
    input_ref: str,
    artifacts_root: Path,
    run_id: str,
) -> dict[str, object]:
    cache_key = (run_id, input_ref)
    if cache_key in _CONTRACT_CACHE:
        return _CONTRACT_CACHE[cache_key]

    contract_path = _contract_path_from_input_ref(input_ref)
    output_path = artifacts_root / run_id / "final" / "merged.md"
    if not contract_path.exists() or not output_path.exists():
        snapshot = {
            "contract_recall": None,
            "contract_failures": None,
            "contract_checks": None,
        }
        _CONTRACT_CACHE[cache_key] = snapshot
        return snapshot

    try:
        contract_raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        snapshot = {
            "contract_recall": None,
            "contract_failures": None,
            "contract_checks": None,
        }
        _CONTRACT_CACHE[cache_key] = snapshot
        return snapshot

    if not isinstance(contract_raw, dict):
        snapshot = {
            "contract_recall": None,
            "contract_failures": None,
            "contract_checks": None,
        }
        _CONTRACT_CACHE[cache_key] = snapshot
        return snapshot

    output_text = output_path.read_text(encoding="utf-8")
    output_lower = output_text.lower()
    output_normalized = _normalize_literal_haystack(output_text)
    output_endpoints = _extract_endpoints(output_text)
    output_headings = _extract_headings(output_text)

    failures = 0
    checks = 0

    required_endpoints = contract_raw.get("required_endpoints", [])
    if isinstance(required_endpoints, list):
        for item in required_endpoints:
            if not isinstance(item, str) or not item.strip():
                continue
            checks += 1
            endpoint = _canonical_endpoint(item)
            if endpoint is None or endpoint not in output_endpoints:
                failures += 1

    required_headings = contract_raw.get("required_headings", [])
    if isinstance(required_headings, list):
        for item in required_headings:
            if not isinstance(item, str) or not item.strip():
                continue
            checks += 1
            heading = item.strip().lower()
            if heading not in output_headings:
                failures += 1

    required_literals = contract_raw.get("required_literals", [])
    if isinstance(required_literals, list):
        for item in required_literals:
            if not isinstance(item, str) or not item.strip():
                continue
            checks += 1
            literal = item.strip()
            literal_normalized = _normalize_literal_needle(literal)
            if (
                literal.lower() not in output_lower
                and literal_normalized not in output_normalized
            ):
                failures += 1

    forbidden_literals = contract_raw.get("forbidden_literals", [])
    if isinstance(forbidden_literals, list):
        for item in forbidden_literals:
            if not isinstance(item, str) or not item.strip():
                continue
            checks += 1
            literal = item.strip()
            literal_normalized = _normalize_literal_needle(literal)
            if (
                literal.lower() in output_lower
                or literal_normalized in output_normalized
            ):
                failures += 1

    contract_recall = 1.0 if checks == 0 else (checks - failures) / checks
    snapshot = {
        "contract_recall": round(contract_recall, 3),
        "contract_failures": failures,
        "contract_checks": checks,
    }
    _CONTRACT_CACHE[cache_key] = snapshot
    return snapshot


def _contract_path_from_input_ref(input_ref: str) -> Path:
    stem = Path(input_ref).name
    stem = Path(stem).stem
    return (Path.cwd() / "samples" / "contracts" / f"{stem}.json").resolve()


def _visible_tok_s(telemetry: dict[str, object]) -> float | None:
    latency = telemetry.get("latency_s")
    output_tokens_est = telemetry.get("output_tokens_est")
    if not isinstance(latency, (int, float)) or latency <= 0:
        return None
    if not isinstance(output_tokens_est, (int, float)):
        return None
    if output_tokens_est < 0:
        return None
    return round(float(output_tokens_est) / float(latency), 3)


def _completion_output_ratio(telemetry: dict[str, object]) -> float | None:
    completion = telemetry.get("completion_tokens")
    output_tokens_est = telemetry.get("output_tokens_est")
    if not isinstance(completion, (int, float)):
        return None
    if not isinstance(output_tokens_est, (int, float)):
        return None
    if output_tokens_est <= 0:
        return None
    return round(float(completion) / float(output_tokens_est), 3)


def _canonical_endpoint(value: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z]+)\s+(/\S+)\s*$", value)
    if not match:
        return None
    method = match.group(1).upper()
    path = match.group(2).strip().lower()
    return f"{method} {path}"


def _f1(recall: float, precision: float) -> float:
    if recall <= 0 or precision <= 0:
        return 0.0
    return (2.0 * recall * precision) / (recall + precision)


def _recall_ratio(*, source: set[str], output: set[str]) -> float:
    if not source:
        return 1.0
    return len(source & output) / len(source)


def _precision_ratio(*, source: set[str], output: set[str]) -> float:
    if not output:
        return 1.0 if not source else 0.0
    return len(source & output) / len(output)


def _normalize_literal_haystack(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"`+", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    lowered = re.sub(r"\s*([=:/,;()\[\]{}<>\-])\s*", r"\1", lowered)
    return lowered


def _normalize_literal_needle(value: str) -> str:
    return _normalize_literal_haystack(value)


def _quality_gate_ok(row: dict[str, object]) -> bool | None:
    quality = row.get("quality_score")
    endpoint_recall = row.get("endpoint_recall")
    endpoint_precision = row.get("endpoint_precision")
    validation_ok = row.get("validation_ok")
    hard_errors = row.get("hard_errors")

    if not isinstance(quality, (int, float)):
        return None
    if not isinstance(endpoint_recall, (int, float)):
        return None
    if not isinstance(endpoint_precision, (int, float)):
        return None
    if not isinstance(validation_ok, bool):
        return None

    hard_error_count = hard_errors if isinstance(hard_errors, int) else 0
    return bool(
        validation_ok
        and hard_error_count == 0
        and float(quality) >= 65.0
        and float(endpoint_recall) >= 0.9
        and float(endpoint_precision) >= 0.9
    )


def _reasoning_heavy(row: dict[str, object]) -> bool | None:
    ratio = row.get("completion_output_ratio")
    if not isinstance(ratio, (int, float)):
        return None
    return float(ratio) >= 3.0


def _speed_gate_ok(row: dict[str, object]) -> bool | None:
    visible = row.get("visible_tok_s")
    output_tokens_est = row.get("output_tokens_est")
    ratio = row.get("completion_output_ratio")

    if not isinstance(visible, (int, float)):
        return None
    if not isinstance(output_tokens_est, (int, float)):
        return None

    if float(output_tokens_est) < 60.0:
        return False
    if isinstance(ratio, (int, float)) and float(ratio) > 3.0:
        return False
    return True


def _length_ratio_score(length_ratio: float) -> float:
    if 0.6 <= length_ratio <= 1.4:
        return 1.0
    if 0.4 <= length_ratio < 0.6 or 1.4 < length_ratio <= 2.0:
        return 0.7
    if 0.25 <= length_ratio < 0.4 or 2.0 < length_ratio <= 3.0:
        return 0.4
    return 0.1


if __name__ == "__main__":
    raise SystemExit(main())
