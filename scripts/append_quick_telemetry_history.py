#!/usr/bin/env -S uv run --python 3.12

"""Append quick-run telemetry profile averages into matrix history."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


HISTORY_START = "<!-- quick-telemetry-history:start -->"
HISTORY_END = "<!-- quick-telemetry-history:end -->"


@dataclass(frozen=True)
class AverageRow:
    profile: str
    runs: str
    avg_tok_s: str
    min_tok_s: str
    max_tok_s: str


@dataclass(frozen=True)
class HistoryRow:
    generated_at: str
    profile: str
    runs: str
    avg_tok_s: str
    min_tok_s: str
    max_tok_s: str

    def key(self) -> tuple[str, str]:
        return (self.generated_at, self.profile)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append timestamped quick-run telemetry averages to decision matrix.",
    )
    parser.add_argument(
        "--telemetry",
        default="samples/quick_telemetry.md",
        help="Quick telemetry markdown file",
    )
    parser.add_argument(
        "--matrix",
        default="docs/backend_decision_matrix.md",
        help="Decision matrix markdown file",
    )
    return parser.parse_args()


def _parse_telemetry(path: Path) -> tuple[str, list[AverageRow]]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"Telemetry file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    generated_at = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- Generated:"):
            generated_at = stripped.split(":", 1)[1].strip()
            break

    if not generated_at:
        raise ValueError("Could not find '- Generated:' line in telemetry file")

    start_index = None
    for i, line in enumerate(lines):
        if line.strip() == "## Profile Averages":
            start_index = i
            break

    if start_index is None:
        fallback_rows = _build_averages_from_per_run(lines)
        if not fallback_rows:
            raise ValueError(
                "Could not find '## Profile Averages' section in telemetry file"
            )
        return generated_at, fallback_rows

    rows: list[AverageRow] = []
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        if "profile" in stripped and "avg_tok_s" in stripped:
            continue
        if stripped.startswith("|---"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 5:
            continue

        rows.append(
            AverageRow(
                profile=cells[0],
                runs=cells[1],
                avg_tok_s=cells[2],
                min_tok_s=cells[3],
                max_tok_s=cells[4],
            )
        )

    if not rows:
        fallback_rows = _build_averages_from_per_run(lines)
        if not fallback_rows:
            raise ValueError("No profile average rows found in telemetry file")
        return generated_at, fallback_rows
    return generated_at, rows


def _build_averages_from_per_run(lines: list[str]) -> list[AverageRow]:
    start_index = None
    for i, line in enumerate(lines):
        if line.strip() == "## Per Run":
            start_index = i
            break
    if start_index is None:
        return []

    by_profile: dict[str, dict[str, object]] = {}
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        if "run_id" in stripped and "profile" in stripped and "tok_s" in stripped:
            continue
        if stripped.startswith("|---"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 7:
            continue
        profile = cells[1]
        tok_cell = cells[6]
        if profile == "n/a":
            continue

        bucket = by_profile.setdefault(profile, {"runs": 0, "tok": []})
        bucket["runs"] = int(bucket["runs"]) + 1
        try:
            tok_value = float(tok_cell)
        except ValueError:
            tok_value = None
        if tok_value is not None:
            values = bucket["tok"]
            assert isinstance(values, list)
            values.append(tok_value)

    rows: list[AverageRow] = []
    for profile, bucket in sorted(by_profile.items()):
        runs = int(bucket["runs"])
        tok_values = bucket["tok"]
        assert isinstance(tok_values, list)
        if tok_values:
            rows.append(
                AverageRow(
                    profile=profile,
                    runs=str(runs),
                    avg_tok_s=f"{sum(tok_values) / len(tok_values):.3f}",
                    min_tok_s=f"{min(tok_values):.3f}",
                    max_tok_s=f"{max(tok_values):.3f}",
                )
            )
        else:
            rows.append(
                AverageRow(
                    profile=profile,
                    runs=str(runs),
                    avg_tok_s="n/a",
                    min_tok_s="n/a",
                    max_tok_s="n/a",
                )
            )
    return rows


def _ensure_history_section(lines: list[str]) -> list[str]:
    if HISTORY_START in lines and HISTORY_END in lines:
        return lines

    section = [
        "",
        "## Quick-run telemetry history",
        "",
        "This section is auto-updated by `scripts/append_quick_telemetry_history.py`.",
        HISTORY_START,
        HISTORY_END,
    ]
    return lines + section


def _parse_existing_history(
    lines: list[str],
) -> tuple[list[str], list[HistoryRow], list[str]]:
    if HISTORY_START not in lines or HISTORY_END not in lines:
        return lines, [], []

    start_index = lines.index(HISTORY_START)
    end_index = lines.index(HISTORY_END)
    if end_index <= start_index:
        return lines, [], []

    before = lines[: start_index + 1]
    block = lines[start_index + 1 : end_index]
    after = lines[end_index:]

    existing: list[HistoryRow] = []
    for line in block:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "generated_at" in stripped and "profile" in stripped:
            continue
        if stripped.startswith("|---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 6:
            continue
        existing.append(
            HistoryRow(
                generated_at=cells[0],
                profile=cells[1],
                runs=cells[2],
                avg_tok_s=cells[3],
                min_tok_s=cells[4],
                max_tok_s=cells[5],
            )
        )

    return before, existing, after


def _render_history_table(rows: list[HistoryRow]) -> list[str]:
    rendered = [
        "",
        "| generated_at | profile | runs | avg_tok_s | min_tok_s | max_tok_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        rendered.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row.generated_at,
                row.profile,
                row.runs,
                row.avg_tok_s,
                row.min_tok_s,
                row.max_tok_s,
            )
        )
    rendered.append("")
    return rendered


def main() -> int:
    args = _parse_args()
    telemetry_path = Path(args.telemetry).expanduser().resolve()
    matrix_path = Path(args.matrix).expanduser().resolve()

    generated_at, averages = _parse_telemetry(telemetry_path)

    if not matrix_path.exists() or not matrix_path.is_file():
        raise SystemExit(f"Matrix file not found: {matrix_path}")

    matrix_lines = matrix_path.read_text(encoding="utf-8").splitlines()
    matrix_lines = _ensure_history_section(matrix_lines)
    before, existing, after = _parse_existing_history(matrix_lines)

    merged: dict[tuple[str, str], HistoryRow] = {row.key(): row for row in existing}
    for avg in averages:
        row = HistoryRow(
            generated_at=generated_at,
            profile=avg.profile,
            runs=avg.runs,
            avg_tok_s=avg.avg_tok_s,
            min_tok_s=avg.min_tok_s,
            max_tok_s=avg.max_tok_s,
        )
        merged[row.key()] = row

    final_rows = sorted(
        merged.values(),
        key=lambda row: (row.generated_at, row.profile),
        reverse=True,
    )

    rendered = before + _render_history_table(final_rows) + after
    matrix_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    print(f"Updated matrix history: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
