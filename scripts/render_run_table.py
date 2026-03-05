#!/usr/bin/env -S uv run --python 3.12

"""Render quick run status table from artifact state files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a markdown run table from quick run artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Artifacts root path",
    )
    parser.add_argument(
        "--pattern",
        default="quick-*",
        help="Run directory glob pattern",
    )
    parser.add_argument(
        "--output",
        default="samples/quick_runs.md",
        help="Markdown output file",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows: list[dict[str, str]] = []
    for run_dir in sorted(artifacts_root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        state_path = run_dir / "state.json"
        if not state_path.exists() or not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        rows.append(
            {
                "run_id": str(state.get("run_id", run_dir.name)),
                "status": str(state.get("status", "unknown")),
                "input": Path(str(state.get("input_path", ""))).name,
                "updated_at": str(state.get("updated_at", "")),
                "profile": Path(str(state.get("profile_path", ""))).name,
            }
        )

    lines = [
        "# Quick Runs",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Artifacts root: `{artifacts_root}`",
        f"- Pattern: `{args.pattern}`",
        "",
        "| run_id | status | input | profile | updated_at |",
        "|---|---|---|---|---|",
    ]

    if rows:
        for row in rows:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["run_id"],
                    row["status"],
                    row["input"],
                    row["profile"],
                    row["updated_at"],
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote run table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
