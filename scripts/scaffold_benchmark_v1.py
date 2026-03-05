#!/usr/bin/env -S uv run --python 3.12

"""Scaffold benchmark v1 dataset directories and manifests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create samples/benchmarks/v1 scaffold and optional fixtures.",
    )
    parser.add_argument(
        "--root",
        default="samples/benchmarks/v1",
        help="Benchmark root directory",
    )
    parser.add_argument(
        "--bootstrap-from",
        action="append",
        default=[],
        help=(
            "Copy markdown fixtures into gold_markdown from directory globs "
            "(repeatable, e.g. samples/docs/*.md)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()

    dirs = [
        root / "gold_markdown",
        root / "gold_contracts",
        root / "generated_pdfs",
        root / "manifests",
        root / "real_paired" / "pdf",
        root / "real_paired" / "source_markdown",
        root / "real_unpaired" / "pdf",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    fixtures_manifest = root / "manifests" / "fixtures.json"
    if not fixtures_manifest.exists():
        fixtures_manifest.write_text("[]\n", encoding="utf-8")

    variants_manifest = root / "manifests" / "variants.jsonl"
    if not variants_manifest.exists():
        variants_manifest.write_text("", encoding="utf-8")

    copied = _bootstrap_markdown(
        root=root,
        sources=args.bootstrap_from,
    )
    if copied:
        _merge_fixtures_manifest(fixtures_manifest=fixtures_manifest, copied=copied)

    print(f"Scaffold ready: {root}")
    print(f"fixtures manifest: {fixtures_manifest}")
    print(f"variants manifest: {variants_manifest}")
    if copied:
        print(f"bootstrapped fixtures: {len(copied)}")
    return 0


def _bootstrap_markdown(*, root: Path, sources: list[str]) -> list[str]:
    copied: list[str] = []
    if not sources:
        return copied

    target_dir = root / "gold_markdown"
    for source_pattern in sources:
        for source_path in sorted(Path.cwd().glob(source_pattern)):
            if not source_path.is_file() or source_path.suffix.lower() != ".md":
                continue
            target_path = target_dir / source_path.name
            shutil.copyfile(source_path, target_path)
            copied.append(target_path.stem)
            _copy_matching_contract(root=root, source_markdown_path=source_path)
    return copied


def _merge_fixtures_manifest(*, fixtures_manifest: Path, copied: list[str]) -> None:
    existing_payload = json.loads(fixtures_manifest.read_text(encoding="utf-8"))
    existing = existing_payload if isinstance(existing_payload, list) else []

    seen = {
        item.get("fixture_id")
        for item in existing
        if isinstance(item, dict) and isinstance(item.get("fixture_id"), str)
    }

    for fixture_id in copied:
        contract_path = (
            fixtures_manifest.parent.parent / "gold_contracts" / f"{fixture_id}.json"
        )
        if fixture_id in seen:
            for item in existing:
                if not isinstance(item, dict):
                    continue
                if item.get("fixture_id") == fixture_id:
                    item["has_contract"] = contract_path.exists()
            continue
        existing.append(
            {
                "fixture_id": fixture_id,
                "source_markdown": f"gold_markdown/{fixture_id}.md",
                "size_bucket": "small",
                "doc_type": "api",
                "has_contract": contract_path.exists(),
            }
        )
        seen.add(fixture_id)

    fixtures_manifest.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_matching_contract(*, root: Path, source_markdown_path: Path) -> None:
    fixture_id = source_markdown_path.stem
    candidate_contract = Path.cwd() / "samples" / "contracts" / f"{fixture_id}.json"
    if not candidate_contract.exists() or not candidate_contract.is_file():
        return
    target_contract = root / "gold_contracts" / candidate_contract.name
    shutil.copyfile(candidate_contract, target_contract)


if __name__ == "__main__":
    raise SystemExit(main())
