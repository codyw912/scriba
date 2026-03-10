#!/usr/bin/env -S uv run --python 3.12

"""Render cost/throughput Pareto candidates for hosted OpenRouter runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

from scriba.pipeline import PipelineProfile, load_profile


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class Pricing:
    prompt_per_token_usd: Decimal
    completion_per_token_usd: Decimal


@dataclass(frozen=True)
class CandidateRow:
    run_id: str
    profile: str
    sample: str
    model: str
    tok_s: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    output_tokens_est: int | None
    visible_tok_s: float | None
    completion_output_ratio: float | None
    cost_usd: Decimal
    warning_count: int
    hard_error_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render hosted OpenRouter Pareto report from artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Artifacts root directory",
    )
    parser.add_argument(
        "--pattern",
        default="matrix-*",
        help="Run directory glob pattern",
    )
    parser.add_argument(
        "--output",
        default="samples/hosted_pareto.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--models-cache",
        default="samples/openrouter_models.json",
        help="Cached OpenRouter models JSON path",
    )
    parser.add_argument(
        "--models-url",
        default=OPENROUTER_MODELS_URL,
        help="OpenRouter models endpoint URL",
    )
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="Refresh models cache from network",
    )
    parser.add_argument(
        "--require-validation-ok",
        action="store_true",
        help="Include only rows with validation ok=true",
    )
    parser.add_argument(
        "--matrix-log",
        default="samples/matrix_runs.jsonl",
        help="Matrix JSONL log for optional campaign filtering",
    )
    parser.add_argument(
        "--campaign-id",
        default="",
        help="Optional campaign id to filter run_ids",
    )
    parser.add_argument(
        "--frontier-metric",
        choices=["tok_s", "visible_tok_s"],
        default="tok_s",
        help="Primary metric for main Pareto frontier",
    )
    parser.add_argument(
        "--comparable-min-output-tokens-est",
        type=int,
        default=60,
        help="Minimum estimated output tokens for practical speed comparability",
    )
    parser.add_argument(
        "--comparable-max-completion-output-ratio",
        type=float,
        default=3.0,
        help="Maximum completion/output ratio for practical speed comparability",
    )
    return parser.parse_args()


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


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


def _as_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None


def _load_models_payload(
    *,
    cache_path: Path,
    models_url: str,
    refresh: bool,
) -> tuple[dict[str, Any], str]:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")), "cache"

    request = urllib.request.Request(
        models_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "scriba-hosted-pareto/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        if cache_path.exists() and cache_path.is_file():
            return json.loads(
                cache_path.read_text(encoding="utf-8")
            ), f"cache_fallback ({exc})"
        raise RuntimeError(f"Failed to fetch OpenRouter models: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, "network"


def _extract_pricing_index(payload: dict[str, Any]) -> dict[str, Pricing]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter models payload missing 'data' list")

    index: dict[str, Pricing] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        pricing = item.get("pricing")
        if not isinstance(model_id, str) or not isinstance(pricing, dict):
            continue
        prompt = _as_decimal(pricing.get("prompt"))
        completion = _as_decimal(pricing.get("completion"))
        if prompt is None or completion is None:
            continue
        index[model_id] = Pricing(
            prompt_per_token_usd=prompt,
            completion_per_token_usd=completion,
        )
    return index


def _allowed_run_ids_from_campaign(matrix_log: Path, campaign_id: str) -> set[str]:
    if not campaign_id:
        return set()
    if not matrix_log.exists() or not matrix_log.is_file():
        return set()

    allowed: set[str] = set()
    for line in matrix_log.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if str(row.get("campaign_id", "")) != campaign_id:
            continue
        if str(row.get("status", "")) not in {
            "completed",
            "completed_with_validation_errors",
        }:
            continue
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            allowed.add(run_id)
    return allowed


def _is_hosted_openrouter(
    profile: PipelineProfile, run_dir: Path
) -> tuple[bool, str, str]:
    role = profile.resolve_role("normalize_text")
    if role is None:
        return False, "", ""
    backend = profile.backends.get(role.backend)
    if backend is None:
        return False, "", ""
    if backend.topology != "remote":
        return False, "", ""
    if backend.provider != "openrouter":
        return False, "", ""

    model = ""
    manifest_path = run_dir / "map" / "manifest.json"
    if manifest_path.exists() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = str(manifest.get("model", "")).strip()
        except json.JSONDecodeError:
            model = ""
    if not model:
        model = role.model
    return True, model, Path(profile.source_path).name


def _pareto_frontier(
    rows: list[CandidateRow], *, metric: str = "tok_s"
) -> list[CandidateRow]:
    frontier: list[CandidateRow] = []
    for row in rows:
        row_metric = getattr(row, metric)
        if not isinstance(row_metric, (int, float)):
            continue
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_metric = getattr(other, metric)
            if not isinstance(other_metric, (int, float)):
                continue
            if (
                other.cost_usd <= row.cost_usd
                and other_metric >= row_metric
                and (other.cost_usd < row.cost_usd or other_metric > row_metric)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda item: (
            item.cost_usd,
            -(
                getattr(item, metric)
                if isinstance(getattr(item, metric), (int, float))
                else -1.0
            ),
            item.run_id,
        ),
    )


def _build_rows(
    *,
    artifacts_root: Path,
    pattern: str,
    pricing_index: dict[str, Pricing],
    require_validation_ok: bool,
    allowed_run_ids: set[str],
) -> tuple[list[CandidateRow], dict[str, int]]:
    counters = {
        "run_dirs": 0,
        "with_state": 0,
        "with_manifest": 0,
        "campaign_filtered": 0,
        "not_hosted_openrouter": 0,
        "missing_pricing": 0,
        "missing_usage": 0,
        "missing_throughput": 0,
        "validation_filtered": 0,
    }

    rows: list[CandidateRow] = []
    profile_cache: dict[Path, PipelineProfile] = {}

    for run_dir in sorted(artifacts_root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        counters["run_dirs"] += 1

        if allowed_run_ids and run_dir.name not in allowed_run_ids:
            counters["campaign_filtered"] += 1
            continue

        state_path = run_dir / "state.json"
        manifest_path = run_dir / "map" / "manifest.json"
        if not state_path.exists() or not state_path.is_file():
            continue
        counters["with_state"] += 1
        if not manifest_path.exists() or not manifest_path.is_file():
            continue
        counters["with_manifest"] += 1

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        profile_raw = str(state.get("profile_path", "")).strip()
        if not profile_raw:
            continue
        profile_path = Path(profile_raw).expanduser().resolve()
        if not profile_path.exists() or not profile_path.is_file():
            continue

        profile = profile_cache.get(profile_path)
        if profile is None:
            try:
                profile = load_profile(profile_path)
            except Exception:
                continue
            profile_cache[profile_path] = profile

        hosted_ok, model_id, profile_name = _is_hosted_openrouter(profile, run_dir)
        if not hosted_ok:
            counters["not_hosted_openrouter"] += 1
            continue

        telemetry = manifest.get("all_outputs_telemetry")
        if not isinstance(telemetry, dict):
            telemetry = manifest.get("processed_telemetry")
        if not isinstance(telemetry, dict):
            continue

        prompt_tokens = _as_int(telemetry.get("prompt_tokens"))
        completion_tokens = _as_int(telemetry.get("completion_tokens"))
        output_tokens_est = _as_int(telemetry.get("output_tokens_est"))
        latency_s = _as_float(telemetry.get("latency_s"))
        tok_s = _as_float(telemetry.get("effective_tokens_per_second"))

        if prompt_tokens is None or completion_tokens is None:
            counters["missing_usage"] += 1
            continue
        if latency_s is None or tok_s is None:
            counters["missing_throughput"] += 1
            continue

        pricing = pricing_index.get(model_id)
        if pricing is None:
            counters["missing_pricing"] += 1
            continue

        validation_ok = False
        hard_error_count = 0
        warning_count = 0
        validation_path = run_dir / "final" / "validation_report.json"
        if validation_path.exists() and validation_path.is_file():
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                validation_ok = bool(validation.get("ok", False))
                hard_errors = validation.get("hard_errors")
                warnings = validation.get("warnings")
                if isinstance(hard_errors, list):
                    hard_error_count = len(hard_errors)
                if isinstance(warnings, list):
                    warning_count = len(warnings)
            except json.JSONDecodeError:
                validation_ok = False

        if require_validation_ok and not validation_ok:
            counters["validation_filtered"] += 1
            continue

        cost_usd = pricing.prompt_per_token_usd * Decimal(
            prompt_tokens
        ) + pricing.completion_per_token_usd * Decimal(completion_tokens)

        rows.append(
            CandidateRow(
                run_id=run_dir.name,
                profile=profile_name,
                sample=Path(str(state.get("input_path", ""))).name,
                model=model_id,
                tok_s=tok_s,
                latency_s=latency_s,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                output_tokens_est=output_tokens_est,
                visible_tok_s=(
                    (float(output_tokens_est) / latency_s)
                    if isinstance(output_tokens_est, int) and latency_s > 0
                    else None
                ),
                completion_output_ratio=(
                    (float(completion_tokens) / float(output_tokens_est))
                    if isinstance(output_tokens_est, int) and output_tokens_est > 0
                    else None
                ),
                cost_usd=cost_usd,
                warning_count=warning_count,
                hard_error_count=hard_error_count,
            )
        )

    rows.sort(key=lambda item: (item.cost_usd, -item.tok_s, item.run_id))
    return rows, counters


def _fmt_money(value: Decimal) -> str:
    return format(value, ".6f")


def _fmt_float(value: float) -> str:
    return format(value, ".3f")


def _fmt_opt_float(value: float | None) -> str:
    return "n/a" if value is None else format(value, ".3f")


def _render_rows(rows: list[CandidateRow]) -> list[str]:
    lines = [
        "| run_id | profile | model | sample | tok_s | visible_tok_s | completion_output_ratio | cost_usd | prompt_tokens | completion_tokens | output_tokens_est | hard_errors | warnings |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.run_id,
                row.profile,
                row.model,
                row.sample,
                _fmt_float(row.tok_s),
                _fmt_opt_float(row.visible_tok_s),
                _fmt_opt_float(row.completion_output_ratio),
                _fmt_money(row.cost_usd),
                row.prompt_tokens,
                row.completion_tokens,
                row.output_tokens_est if row.output_tokens_est is not None else "n/a",
                row.hard_error_count,
                row.warning_count,
            )
        )
    return lines


def _is_speed_comparable(
    row: CandidateRow,
    *,
    min_output_tokens_est: int,
    max_completion_output_ratio: float,
) -> bool:
    if row.output_tokens_est is None:
        return False
    if row.output_tokens_est < min_output_tokens_est:
        return False
    if row.visible_tok_s is None:
        return False
    if row.completion_output_ratio is not None:
        if row.completion_output_ratio > max_completion_output_ratio:
            return False
    return True


def main() -> int:
    args = _parse_args()
    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    models_cache_path = Path(args.models_cache).expanduser().resolve()
    matrix_log_path = Path(args.matrix_log).expanduser().resolve()

    models_payload, models_source = _load_models_payload(
        cache_path=models_cache_path,
        models_url=args.models_url,
        refresh=bool(args.refresh_models),
    )
    pricing_index = _extract_pricing_index(models_payload)
    allowed_run_ids = _allowed_run_ids_from_campaign(
        matrix_log=matrix_log_path,
        campaign_id=str(args.campaign_id or "").strip(),
    )

    rows, counters = _build_rows(
        artifacts_root=artifacts_root,
        pattern=args.pattern,
        pricing_index=pricing_index,
        require_validation_ok=bool(args.require_validation_ok),
        allowed_run_ids=allowed_run_ids,
    )
    frontier = _pareto_frontier(rows, metric=str(args.frontier_metric))
    visible_frontier = _pareto_frontier(rows, metric="visible_tok_s")
    comparable_rows = [
        row
        for row in rows
        if _is_speed_comparable(
            row,
            min_output_tokens_est=max(0, int(args.comparable_min_output_tokens_est)),
            max_completion_output_ratio=float(
                args.comparable_max_completion_output_ratio
            ),
        )
    ]
    practical_frontier = _pareto_frontier(comparable_rows, metric="visible_tok_s")

    lines = [
        "# Hosted OpenRouter Pareto Report",
        "",
        f"- Generated: {datetime.now(UTC).isoformat()}",
        f"- Artifacts root: `{artifacts_root}`",
        f"- Run pattern: `{args.pattern}`",
        f"- Campaign filter: `{args.campaign_id or 'none'}`",
        f"- Models source: `{models_source}`",
        f"- Models cache: `{models_cache_path}`",
        f"- OpenRouter models indexed: `{len(pricing_index)}`",
        f"- Require validation ok: `{bool(args.require_validation_ok)}`",
        f"- Frontier metric: `{args.frontier_metric}`",
        f"- Comparable rows filter: `output_tokens_est>={int(args.comparable_min_output_tokens_est)}` and `completion_output_ratio<={float(args.comparable_max_completion_output_ratio):.3f}`",
        "",
        "## Candidate scan",
        "",
        f"- run_dirs: {counters['run_dirs']}",
        f"- with_state: {counters['with_state']}",
        f"- with_manifest: {counters['with_manifest']}",
        f"- campaign_filtered: {counters['campaign_filtered']}",
        f"- filtered_not_hosted_openrouter: {counters['not_hosted_openrouter']}",
        f"- filtered_missing_pricing: {counters['missing_pricing']}",
        f"- filtered_missing_usage: {counters['missing_usage']}",
        f"- filtered_missing_throughput: {counters['missing_throughput']}",
        f"- filtered_validation: {counters['validation_filtered']}",
        f"- eligible_rows: {len(rows)}",
        f"- comparable_rows: {len(comparable_rows)}",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "No eligible hosted rows found.",
                "",
                "Run a hosted OpenRouter matrix campaign first, then regenerate this report.",
            ]
        )
    else:
        lines.extend([f"## Pareto frontier (`{args.frontier_metric}`)", ""])
        lines.extend(_render_rows(frontier))
        lines.extend(["", "## Pareto frontier (`visible_tok_s`)", ""])
        lines.extend(_render_rows(visible_frontier))
        lines.extend(
            ["", "## Practical frontier (`visible_tok_s`, comparable rows)", ""]
        )
        lines.extend(_render_rows(practical_frontier))
        lines.extend(["", "## All eligible rows", ""])
        lines.extend(_render_rows(rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote hosted pareto report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
