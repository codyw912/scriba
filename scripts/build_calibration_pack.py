#!/usr/bin/env -S uv run --python 3.12

"""Build a model calibration pack from funnel buckets.

The pack includes generated profiles, a manifest, and runnable commands for
doctor checks and a first calibration campaign.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_TRADEOFFS = "samples/model_selection/model_tradeoffs.json"
DEFAULT_TEMPLATE_PROFILE = (
    "profiles/remote/pipeline.profile.remote_openrouter_qwen25_7b.example.yaml"
)
DEFAULT_OUTPUT_DIR = "samples/model_selection/calibration"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a calibration model pack from candidate funnel rows."
    )
    parser.add_argument(
        "--tradeoffs",
        default=DEFAULT_TRADEOFFS,
        help="Path to model_tradeoffs.json",
    )
    parser.add_argument(
        "--template-profile",
        default=DEFAULT_TEMPLATE_PROFILE,
        help="Template profile used for generated model profiles",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Calibration pack root directory",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Pack label (default: UTC timestamp)",
    )
    parser.add_argument(
        "--campaign-id",
        default="",
        help="Campaign id for run_matrix command (default: calib-<label>)",
    )
    parser.add_argument(
        "--strong-prior",
        type=int,
        default=3,
        help="Rows to take from strong_prior bucket",
    )
    parser.add_argument(
        "--promising",
        type=int,
        default=3,
        help="Rows to take from promising_but_uncertain bucket",
    )
    parser.add_argument(
        "--high-capability-unknown-speed",
        type=int,
        default=2,
        help="Rows to take from high_capability_unknown_speed bucket",
    )
    parser.add_argument(
        "--fixtures-per-model",
        type=int,
        default=3,
        help="Assumed fixtures count for max-runs recommendation",
    )
    parser.add_argument(
        "--samples-dir",
        default="samples/docs",
        help="Samples dir used in generated run commands",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="Model id to skip (repeatable)",
    )
    parser.add_argument(
        "--keep-reasoning-fields",
        action="store_true",
        help="Keep reasoning_effort/reasoning_exclude fields from template",
    )
    return parser.parse_args()


def _default_label(raw: str) -> str:
    label = raw.strip()
    if not label:
        label = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return label or datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")


def _slugify_model_id(model_id: str) -> str:
    slug = model_id.strip().lower()
    slug = slug.replace("/", "__").replace(":", "_")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _replace_normalize_model(template_text: str, model_id: str) -> str:
    lines = template_text.splitlines()
    in_roles = False
    in_normalize = False
    replaced = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r"^roles:\s*$", line):
            in_roles = True
            in_normalize = False
            continue

        if in_roles and re.match(r"^[A-Za-z0-9_]+:\s*$", stripped):
            # e.g., "normalize_text:" inside roles
            in_normalize = stripped == "normalize_text:"
            continue

        if in_roles and in_normalize and re.match(r"^\s*model:\s*", line):
            indent = re.match(r"^(\s*)", line)
            prefix = indent.group(1) if indent else ""
            lines[idx] = f"{prefix}model: {model_id}"
            replaced = True
            break

        # Exit roles block when indentation returns to top-level key.
        if (
            in_roles
            and not line.startswith(" ")
            and re.match(r"^[A-Za-z0-9_]+:\s*$", line)
        ):
            in_roles = False
            in_normalize = False

    if not replaced:
        raise RuntimeError(
            "Could not find normalize_text model field in template profile"
        )

    return "\n".join(lines) + "\n"


def _strip_reasoning_fields(profile_text: str) -> str:
    filtered: list[str] = []
    for line in profile_text.splitlines():
        if re.match(r"^\s*reasoning_effort:\s*", line):
            continue
        if re.match(r"^\s*reasoning_exclude:\s*", line):
            continue
        filtered.append(line)
    return "\n".join(filtered) + "\n"


def _bucket_rows(payload: dict[str, Any], bucket_name: str) -> list[dict[str, Any]]:
    funnel = payload.get("candidate_funnel")
    if not isinstance(funnel, dict):
        return []
    bucket = funnel.get(bucket_name)
    if not isinstance(bucket, dict):
        return []
    rows = bucket.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def main() -> int:
    args = _parse_args()

    tradeoffs_path = Path(args.tradeoffs).expanduser().resolve()
    template_profile_path = Path(args.template_profile).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not tradeoffs_path.exists():
        raise RuntimeError(f"Tradeoffs file not found: {tradeoffs_path}")
    if not template_profile_path.exists():
        raise RuntimeError(f"Template profile not found: {template_profile_path}")

    payload = json.loads(tradeoffs_path.read_text(encoding="utf-8"))
    template_text = template_profile_path.read_text(encoding="utf-8")

    label = _default_label(args.label)
    campaign_id = args.campaign_id.strip() or f"calib-{label}"

    strong_rows = _bucket_rows(payload, "strong_prior")[: max(0, args.strong_prior)]
    promising_rows = _bucket_rows(payload, "promising_but_uncertain")[
        : max(0, args.promising)
    ]
    highcap_rows = _bucket_rows(payload, "high_capability_unknown_speed")[
        : max(0, args.high_capability_unknown_speed)
    ]

    selected_entries: list[dict[str, Any]] = []
    seen_models: set[str] = set(
        model.strip() for model in args.skip_model if model.strip()
    )

    for bucket_name, rows in (
        ("strong_prior", strong_rows),
        ("promising_but_uncertain", promising_rows),
        ("high_capability_unknown_speed", highcap_rows),
    ):
        for row in rows:
            model_id = row.get("model_id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            if model_id in seen_models:
                continue
            seen_models.add(model_id)
            selected_entries.append(
                {"bucket": bucket_name, "row": row, "model_id": model_id}
            )

    if not selected_entries:
        raise RuntimeError("No models selected for calibration pack")

    pack_dir = output_root / label
    profiles_dir = pack_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    generated_profiles: list[dict[str, Any]] = []
    profile_paths_for_cmd: list[str] = []
    for idx, entry in enumerate(selected_entries, start=1):
        model_id = entry["model_id"]
        bucket_name = entry["bucket"]
        slug = _slugify_model_id(model_id)
        profile_name = f"pipeline.profile.remote_openrouter.calib.{idx:02d}.{slug}.yaml"
        profile_path = profiles_dir / profile_name
        profile_text = _replace_normalize_model(template_text, model_id)
        if not args.keep_reasoning_fields:
            profile_text = _strip_reasoning_fields(profile_text)
        profile_path.write_text(profile_text, encoding="utf-8")

        rel_profile = str(profile_path.relative_to(Path.cwd()))
        profile_paths_for_cmd.append(rel_profile)
        generated_profiles.append(
            {
                "bucket": bucket_name,
                "model_id": model_id,
                "profile_path": rel_profile,
                "source_scores": entry["row"].get("capability_source_scores", {}),
                "speed_score": entry["row"].get("speed_score"),
                "capability_score": entry["row"].get("capability_score"),
                "balanced_score": entry["row"].get("balanced_score"),
            }
        )

    max_runs = len(generated_profiles) * max(1, args.fixtures_per_model)
    profile_args = " ".join(f"--profile {path}" for path in profile_paths_for_cmd)
    doctor_cmd = (
        f"bash scripts/run_matrix.sh --doctor-only --campaign-id {campaign_id}-doctor "
        f"--samples-dir {args.samples_dir} {profile_args}"
    )
    run_cmd = (
        f"bash scripts/run_matrix.sh --campaign-id {campaign_id} --samples-dir {args.samples_dir} "
        f"--max-runs {max_runs} {profile_args}"
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "label": label,
        "campaign_id": campaign_id,
        "tradeoffs_path": str(tradeoffs_path.relative_to(Path.cwd())),
        "template_profile": str(template_profile_path.relative_to(Path.cwd())),
        "counts": {
            "total_models": len(generated_profiles),
            "strong_prior": sum(
                1 for item in generated_profiles if item["bucket"] == "strong_prior"
            ),
            "promising_but_uncertain": sum(
                1
                for item in generated_profiles
                if item["bucket"] == "promising_but_uncertain"
            ),
            "high_capability_unknown_speed": sum(
                1
                for item in generated_profiles
                if item["bucket"] == "high_capability_unknown_speed"
            ),
        },
        "profiles": generated_profiles,
        "commands": {
            "doctor": doctor_cmd,
            "run": run_cmd,
            "max_runs": max_runs,
        },
        "generation_options": {
            "keep_reasoning_fields": bool(args.keep_reasoning_fields),
        },
    }

    manifest_path = pack_dir / "calibration_pack.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cmd_path = pack_dir / "commands.sh"
    cmd_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"# Generated calibration commands for {label}\n"
        f"{doctor_cmd}\n\n"
        f"{run_cmd}\n",
        encoding="utf-8",
    )

    summary_lines = [
        f"# Calibration Pack `{label}`",
        "",
        f"- campaign_id: `{campaign_id}`",
        f"- models selected: `{len(generated_profiles)}`",
        f"- strong_prior: `{manifest['counts']['strong_prior']}`",
        f"- promising_but_uncertain: `{manifest['counts']['promising_but_uncertain']}`",
        f"- high_capability_unknown_speed: `{manifest['counts']['high_capability_unknown_speed']}`",
        "",
        "## Profiles",
        "",
        "| bucket | model_id | profile_path |",
        "|---|---|---|",
    ]
    for item in generated_profiles:
        summary_lines.append(
            f"| {item['bucket']} | {item['model_id']} | {item['profile_path']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- doctor: `{doctor_cmd}`",
            f"- run: `{run_cmd}`",
            "",
        ]
    )
    (pack_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote calibration pack manifest: {manifest_path}")
    print(f"Wrote calibration pack commands: {cmd_path}")
    print(f"Wrote calibration pack summary: {pack_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
