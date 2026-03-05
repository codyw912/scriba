#!/usr/bin/env -S uv run --python 3.12

"""Render map telemetry table from quick pipeline runs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render quick run telemetry markdown from artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Artifacts root directory",
    )
    parser.add_argument(
        "--pattern",
        default="quick-*",
        help="Run directory glob pattern",
    )
    parser.add_argument(
        "--output",
        default="samples/quick_telemetry.md",
        help="Output markdown file path",
    )
    return parser.parse_args()


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def main() -> int:
    args = _parse_args()
    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows: list[dict[str, object]] = []
    tok_by_profile: dict[str, list[float]] = {}
    runs_by_profile: dict[str, int] = {}

    for run_dir in sorted(artifacts_root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue

        state_path = run_dir / "state.json"
        manifest_path = run_dir / "map" / "manifest.json"
        if not state_path.exists() or not manifest_path.exists():
            continue

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        telemetry = manifest.get("processed_telemetry")
        if not isinstance(telemetry, dict):
            continue

        profile = Path(str(state.get("profile_path", ""))).name
        sample = Path(str(state.get("input_path", ""))).name
        tok_s = telemetry.get("effective_tokens_per_second")
        source = (
            "usage.completion_tokens"
            if telemetry.get("completion_tokens") is not None
            else "output_tokens_est"
        )

        row = {
            "run_id": run_dir.name,
            "profile": profile,
            "sample": sample,
            "processed": f"{manifest.get('processed', 0)}/{manifest.get('chunk_count', 0)}",
            "requests": telemetry.get("requests"),
            "latency_s": telemetry.get("latency_s"),
            "tok_s": tok_s,
            "source": source,
            "usage_chunks": telemetry.get("chunks_with_usage"),
            "prompt_tokens": telemetry.get("prompt_tokens"),
            "completion_tokens": telemetry.get("completion_tokens"),
        }
        rows.append(row)
        runs_by_profile[profile] = runs_by_profile.get(profile, 0) + 1

        tok_value = _as_float(tok_s)
        if tok_value is not None:
            tok_by_profile.setdefault(profile, []).append(tok_value)

    lines = [
        "# Quick Run Telemetry",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Artifacts root: `{artifacts_root}`",
        f"- Run pattern: `{args.pattern}`",
        "",
    ]

    if runs_by_profile:
        lines.extend(
            [
                "## Profile Averages",
                "",
                "| profile | runs | avg_tok_s | min_tok_s | max_tok_s |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for profile in sorted(runs_by_profile.keys()):
            values = tok_by_profile.get(profile, [])
            if values:
                lines.append(
                    "| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                        profile,
                        runs_by_profile[profile],
                        sum(values) / len(values),
                        min(values),
                        max(values),
                    )
                )
            else:
                lines.append(
                    "| {} | {} | n/a | n/a | n/a |".format(
                        profile,
                        runs_by_profile[profile],
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## Per Run",
            "",
            "| run_id | profile | sample | processed | requests | latency_s | tok_s | source | usage_chunks | prompt_tokens | completion_tokens |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )

    if rows:
        for row in rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    row["run_id"],
                    row["profile"],
                    row["sample"],
                    row["processed"],
                    _fmt(row["requests"]),
                    _fmt(row["latency_s"]),
                    _fmt(row["tok_s"]),
                    row["source"],
                    _fmt(row["usage_chunks"]),
                    _fmt(row["prompt_tokens"]),
                    _fmt(row["completion_tokens"]),
                )
            )
    else:
        lines.append(
            "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote telemetry table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
