#!/usr/bin/env -S uv run --python 3.12

"""Generate deterministic noisy PDF variants for benchmark v1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    variant_family: str
    noise_level: str
    base_dpi: int
    grayscale: bool
    blur_radius: float
    contrast: float
    brightness: float
    jpeg_quality: int | None
    rotate_deg: float
    speckle_points_per_mpx: int


VARIANT_SPECS: dict[str, VariantSpec] = {
    "layout_light": VariantSpec(
        variant_id="layout_light",
        variant_family="layout_light",
        noise_level="low",
        base_dpi=220,
        grayscale=False,
        blur_radius=0.0,
        contrast=1.0,
        brightness=1.0,
        jpeg_quality=None,
        rotate_deg=0.0,
        speckle_points_per_mpx=0,
    ),
    "scan_light": VariantSpec(
        variant_id="scan_light",
        variant_family="scan_light",
        noise_level="low",
        base_dpi=180,
        grayscale=True,
        blur_radius=0.4,
        contrast=0.95,
        brightness=1.0,
        jpeg_quality=85,
        rotate_deg=0.35,
        speckle_points_per_mpx=90,
    ),
    "scan_medium": VariantSpec(
        variant_id="scan_medium",
        variant_family="scan_medium",
        noise_level="medium",
        base_dpi=150,
        grayscale=True,
        blur_radius=0.9,
        contrast=0.86,
        brightness=0.95,
        jpeg_quality=65,
        rotate_deg=0.9,
        speckle_points_per_mpx=260,
    ),
    "scan_hard": VariantSpec(
        variant_id="scan_hard",
        variant_family="scan_hard",
        noise_level="high",
        base_dpi=120,
        grayscale=True,
        blur_radius=1.4,
        contrast=0.74,
        brightness=0.9,
        jpeg_quality=45,
        rotate_deg=2.0,
        speckle_points_per_mpx=650,
    ),
    "mixed_content": VariantSpec(
        variant_id="mixed_content",
        variant_family="mixed_content",
        noise_level="medium",
        base_dpi=165,
        grayscale=True,
        blur_radius=0.75,
        contrast=0.9,
        brightness=0.97,
        jpeg_quality=72,
        rotate_deg=1.1,
        speckle_points_per_mpx=200,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark PDF noise/layout variants.",
    )
    parser.add_argument(
        "--root",
        default="samples/benchmarks/v1",
        help="Benchmark root directory",
    )
    parser.add_argument(
        "--variants-manifest",
        default="manifests/variants.jsonl",
        help="Variants manifest path relative to root",
    )
    parser.add_argument(
        "--fixture-id",
        action="append",
        default=[],
        help="Generate only selected fixture IDs (repeatable)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant ID to generate (repeatable). Defaults to all non-clean variants.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing variant PDFs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    variants_manifest_path = root / args.variants_manifest

    available_rows = _load_variants_jsonl(variants_manifest_path)
    clean_rows = [
        row
        for row in available_rows
        if row.get("variant_id") == "clean_pdf"
        and isinstance(row.get("fixture_id"), str)
    ]
    if not clean_rows:
        raise SystemExit(
            "No clean_pdf rows found. Generate clean PDFs first with:\n"
            "  uv run scripts/generate_benchmark_pdfs.py"
        )

    selected_fixtures = set(args.fixture_id) if args.fixture_id else None
    selected_variants = set(args.variant) if args.variant else set(VARIANT_SPECS.keys())
    unknown = selected_variants - set(VARIANT_SPECS.keys())
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(sorted(unknown))}")

    fitz = _import_fitz()
    pil = _import_pillow()

    generated_rows: list[dict[str, Any]] = []
    for clean_row in clean_rows:
        fixture_id = str(clean_row["fixture_id"])
        if selected_fixtures is not None and fixture_id not in selected_fixtures:
            continue

        clean_pdf_rel = str(clean_row.get("pdf_path", ""))
        clean_pdf_path = root / clean_pdf_rel
        if not clean_pdf_path.exists():
            raise SystemExit(
                f"Missing clean PDF for fixture {fixture_id}: {clean_pdf_path}"
            )

        fixture_output_dir = root / "generated_pdfs" / fixture_id
        fixture_output_dir.mkdir(parents=True, exist_ok=True)

        for variant_id in sorted(selected_variants):
            spec = VARIANT_SPECS[variant_id]
            variant_pdf_path = fixture_output_dir / f"{variant_id}.pdf"
            variant_seed = _variant_seed(fixture_id=fixture_id, variant_id=variant_id)

            if not variant_pdf_path.exists() or args.force:
                _render_variant_pdf(
                    fitz=fitz,
                    pil=pil,
                    input_pdf=clean_pdf_path,
                    output_pdf=variant_pdf_path,
                    spec=spec,
                    seed=variant_seed,
                )

            generated_rows.append(
                {
                    "fixture_id": fixture_id,
                    "variant_id": spec.variant_id,
                    "variant_family": spec.variant_family,
                    "pdf_path": str(variant_pdf_path.relative_to(root)),
                    "seed": variant_seed,
                    "renderer": "playwright_chromium_pdf",
                    "renderer_version": str(
                        clean_row.get("renderer_version", "unknown")
                    ),
                    "noise_level": spec.noise_level,
                    "transform_params": {
                        "source_variant": "clean_pdf",
                        "base_dpi": spec.base_dpi,
                        "grayscale": spec.grayscale,
                        "blur_radius": spec.blur_radius,
                        "contrast": spec.contrast,
                        "brightness": spec.brightness,
                        "jpeg_quality": spec.jpeg_quality,
                        "rotate_deg": spec.rotate_deg,
                        "speckle_points_per_mpx": spec.speckle_points_per_mpx,
                    },
                    "generated_at": datetime.now(UTC).isoformat(),
                }
            )
            print(f"generated: {fixture_id} -> {variant_pdf_path.relative_to(root)}")

    if not generated_rows:
        print("No variants generated.")
        return 0

    _upsert_variants_manifest(
        variants_path=variants_manifest_path,
        generated_rows=generated_rows,
    )
    print(f"updated variants manifest: {variants_manifest_path}")
    return 0


def _variant_seed(*, fixture_id: str, variant_id: str) -> int:
    digest = hashlib.sha256(f"{fixture_id}:{variant_id}".encode("utf-8")).hexdigest()[
        :8
    ]
    return int(digest, 16)


def _import_fitz() -> Any:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'pymupdf'. Install dev deps first with:\n"
            "  uv sync --group dev"
        ) from exc
    return fitz


def _import_pillow() -> Any:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'pillow'. Install dev deps first with:\n"
            "  uv sync --group dev"
        ) from exc

    return {
        "Image": Image,
        "ImageEnhance": ImageEnhance,
        "ImageFilter": ImageFilter,
        "ImageOps": ImageOps,
    }


def _render_variant_pdf(
    *,
    fitz: Any,
    pil: dict[str, Any],
    input_pdf: Path,
    output_pdf: Path,
    spec: VariantSpec,
    seed: int,
) -> None:
    rng = random.Random(seed)
    image_cls = pil["Image"]
    image_enhance = pil["ImageEnhance"]
    image_filter = pil["ImageFilter"]
    image_ops = pil["ImageOps"]

    source = fitz.open(input_pdf)
    target = fitz.open()
    try:
        for page_index, page in enumerate(source):
            page_rng = random.Random(seed + page_index * 104729)
            dpi = _page_dpi(spec, page_rng)
            scale = dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB"
            image = image_cls.frombytes(mode, [pix.width, pix.height], pix.samples)

            if spec.grayscale:
                image = image.convert("L")

            image = _apply_geometric_variation(
                image=image,
                spec=spec,
                image_ops=image_ops,
                page_rng=page_rng,
            )

            if spec.blur_radius > 0:
                image = image.filter(image_filter.GaussianBlur(radius=spec.blur_radius))

            if spec.contrast != 1.0:
                image = image_enhance.Contrast(image).enhance(spec.contrast)

            if spec.brightness != 1.0:
                image = image_enhance.Brightness(image).enhance(spec.brightness)

            if spec.speckle_points_per_mpx > 0:
                image = _apply_speckle_noise(
                    image=image,
                    points_per_mpx=spec.speckle_points_per_mpx,
                    page_rng=page_rng,
                )

            if spec.jpeg_quality is not None:
                image = _jpeg_roundtrip(
                    image=image,
                    quality=spec.jpeg_quality,
                    image_cls=image_cls,
                )

            if image.mode != "RGB":
                image = image.convert("RGB")

            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page_rect = page.rect
            out_page = target.new_page(width=page_rect.width, height=page_rect.height)
            out_page.insert_image(page_rect, stream=stream.getvalue())

        target.save(
            output_pdf,
            garbage=3,
            deflate=True,
            clean=True,
        )
    finally:
        source.close()
        target.close()


def _page_dpi(spec: VariantSpec, page_rng: random.Random) -> int:
    if spec.variant_id == "scan_hard":
        return int(max(96, spec.base_dpi + page_rng.randint(-20, 30)))
    if spec.variant_id == "mixed_content":
        return int(max(110, spec.base_dpi + page_rng.randint(-15, 15)))
    return spec.base_dpi


def _apply_geometric_variation(
    *, image: Any, spec: VariantSpec, image_ops: Any, page_rng: random.Random
) -> Any:
    if spec.rotate_deg <= 0:
        return image

    angle = page_rng.uniform(-spec.rotate_deg, spec.rotate_deg)
    if abs(angle) < 0.01:
        return image

    rotated = image.rotate(
        angle, expand=True, fillcolor=255 if image.mode == "L" else (255, 255, 255)
    )
    return image_ops.fit(rotated, image.size)


def _apply_speckle_noise(
    *, image: Any, points_per_mpx: int, page_rng: random.Random
) -> Any:
    image = image.copy()
    width, height = image.size
    pixel_count = width * height
    points = int((pixel_count / 1_000_000.0) * points_per_mpx)
    if points <= 0:
        return image

    pixels = image.load()
    if pixels is None:
        return image

    for _ in range(points):
        x = page_rng.randrange(0, width)
        y = page_rng.randrange(0, height)
        salt = page_rng.random() < 0.5
        if image.mode == "L":
            pixels[x, y] = 255 if salt else 0
        else:
            pixels[x, y] = (255, 255, 255) if salt else (0, 0, 0)
    return image


def _jpeg_roundtrip(*, image: Any, quality: int, image_cls: Any) -> Any:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    reopened = image_cls.open(buffer)
    return reopened.copy()


def _load_variants_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _upsert_variants_manifest(
    *,
    variants_path: Path,
    generated_rows: list[dict[str, Any]],
) -> None:
    existing_rows = _load_variants_jsonl(variants_path)
    replace_keys = {
        (str(row.get("fixture_id")), str(row.get("variant_id")))
        for row in generated_rows
    }
    filtered = [
        row
        for row in existing_rows
        if (str(row.get("fixture_id")), str(row.get("variant_id"))) not in replace_keys
    ]
    filtered.extend(generated_rows)

    lines = [json.dumps(row, sort_keys=True) for row in filtered]
    content = "\n".join(lines)
    if content:
        content += "\n"
    variants_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
