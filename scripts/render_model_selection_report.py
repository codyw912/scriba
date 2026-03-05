#!/usr/bin/env -S uv run --python 3.12

"""Build rough model tradeoff inputs for OpenRouter universe selection.

This script is intentionally non-prescriptive: it prepares comparable inputs
for cost/speed/capability tradeoff discussions without selecting specific
models.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_RANKINGS_URL = "https://openrouter.ai/rankings#performance"
AA_LLM_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
AA_ATTRIBUTION_URL = "https://artificialanalysis.ai/"
AA_ATTRIBUTION_NOTE = (
    "Artificial Analysis data used under their free API terms; attribution required."
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
HF_LEADERBOARD_ROWS_URL_TEMPLATE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=open-llm-leaderboard/contents"
    "&config=default&split=train&offset={offset}&length={length}"
)
LMSYS_SPACE_TREE_URL = (
    "https://huggingface.co/api/spaces/lmsys/chatbot-arena-leaderboard/tree/main"
    "?recursive=1"
)
LMSYS_SPACE_RAW_URL_TEMPLATE = (
    "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/{path}"
)


COST_PRESETS: dict[str, dict[str, int]] = {
    "light": {"input_tokens_per_page": 500, "output_tokens_per_page": 100},
    "typical": {"input_tokens_per_page": 800, "output_tokens_per_page": 180},
    "dense": {"input_tokens_per_page": 1200, "output_tokens_per_page": 320},
}


CAPABILITY_WEIGHTS = {
    "intelligence": 0.7,
    "coding": 0.15,
    "agentic": 0.15,
}


HF_CAPABILITY_WEIGHTS = {
    "ifeval": 0.3,
    "mmlu_pro": 0.2,
    "gpqa": 0.2,
    "bbh": 0.1,
    "math_lvl5": 0.1,
    "average": 0.1,
}


LMSYS_CAPABILITY_WEIGHTS = {
    "mt_bench": 0.7,
    "mmlu": 0.3,
}


CAPABILITY_SOURCE_WEIGHTS = {
    "aa_api": 0.45,
    "aa_rankings": 0.35,
    "hf_open_llm": 0.1,
    "lmsys_arena": 0.1,
}


AA_API_CAPABILITY_WEIGHTS = {
    "artificial_analysis_intelligence_index": 0.45,
    "artificial_analysis_coding_index": 0.2,
    "mmlu_pro": 0.1,
    "gpqa": 0.1,
    "livecodebench": 0.05,
    "scicode": 0.05,
    "hle": 0.05,
}


BALANCED_WEIGHTS = {
    "cost_efficiency": 0.3,
    "speed": 0.3,
    "capability": 0.4,
}

ECONOMY_WEIGHTS = {
    "cost_efficiency": 0.5,
    "speed": 0.2,
    "capability": 0.3,
}

PERFORMANCE_WEIGHTS = {
    "cost_efficiency": 0.2,
    "speed": 0.5,
    "capability": 0.3,
}


@dataclass(frozen=True)
class CacheLoad:
    payload: Any
    source: str
    cache_path: Path


@dataclass(frozen=True)
class OpenRouterModelMap:
    by_id: dict[str, dict[str, Any]]
    canonical_to_id: dict[str, str]
    normalized_id_to_id: dict[str, str]
    normalized_canonical_to_id: dict[str, str]
    huggingface_to_ids: dict[str, list[str]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render non-prescriptive model tradeoff inputs for OpenRouter models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="samples/model_selection",
        help="Output directory for report and JSON artifacts",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Refresh remote sources even when cache files exist",
    )
    parser.add_argument(
        "--models-dev-url",
        default=MODELS_DEV_URL,
        help="models.dev API URL",
    )
    parser.add_argument(
        "--openrouter-models-url",
        default=OPENROUTER_MODELS_URL,
        help="OpenRouter models API URL",
    )
    parser.add_argument(
        "--openrouter-rankings-url",
        default=OPENROUTER_RANKINGS_URL,
        help="OpenRouter rankings URL (performance section)",
    )
    parser.add_argument(
        "--aa-llm-models-url",
        default=AA_LLM_MODELS_URL,
        help="Artificial Analysis LLM models API URL",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file used to load AA_API_KEY",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent for remote source fetches",
    )
    parser.add_argument(
        "--hf-leaderboard-rows-url-template",
        default=HF_LEADERBOARD_ROWS_URL_TEMPLATE,
        help=("Hugging Face dataset-server rows URL template with {offset}/{length}"),
    )
    parser.add_argument(
        "--lmsys-tree-url",
        default=LMSYS_SPACE_TREE_URL,
        help="LMSYS HF space tree API URL",
    )
    parser.add_argument(
        "--lmsys-raw-url-template",
        default=LMSYS_SPACE_RAW_URL_TEMPLATE,
        help="LMSYS HF space raw file URL template with {path}",
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Write dated snapshot copies under snapshots directory",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="samples/model_selection/snapshots",
        help="Snapshot root directory",
    )
    parser.add_argument(
        "--snapshot-label",
        default="",
        help="Optional snapshot label (default: UTC timestamp)",
    )
    return parser.parse_args()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
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


def _normalize_model_id(value: str) -> str:
    # Handles slugs such as qwen/model-name-20260224.
    return re.sub(r"-20\d{6}$", "", value.strip())


def _normalize_hf_repo_id(value: str) -> str:
    return value.strip().lower()


def _normalize_lmsys_key(value: str) -> str:
    text = value.strip().lower().replace("_", "-").replace(" ", "-")
    # Common naming normalization between arena keys and OpenRouter IDs.
    text = text.replace("claude-3-5", "claude-3.5")
    text = text.replace("claude-3-7", "claude-3.7")
    text = re.sub(r"[^a-z0-9.+-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _flatten_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _strip_match_date_tail(value: str) -> str:
    text = value.strip()
    text = re.sub(r"-20\d{6}$", "", text)
    text = re.sub(r"-\d{8}$", "", text)
    text = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", text)
    text = re.sub(r"-\d{4}-\d{2}$", "", text)
    return text


def _openrouter_aliases_for_matching(
    model_id: str, canonical_slug: str | None
) -> set[str]:
    aliases: set[str] = set()

    for source in (model_id, canonical_slug or ""):
        if not source:
            continue
        source = source.strip()
        if not source:
            continue

        tail = source.split("/", 1)[1] if "/" in source else source
        tail_without_tier = re.sub(r":[a-z0-9_-]+$", "", tail)

        candidates = {
            source,
            _strip_match_date_tail(source),
            tail,
            _strip_match_date_tail(tail),
            tail_without_tier,
            _strip_match_date_tail(tail_without_tier),
        }

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            normalized = _normalize_lmsys_key(candidate)
            aliases.add(normalized)
            aliases.add(_flatten_alnum(normalized))

            for suffix in ("-preview", "-instruct", "-chat"):
                if normalized.endswith(suffix):
                    base = normalized[: -len(suffix)]
                    if base:
                        aliases.add(base)
                        aliases.add(_flatten_alnum(base))

    return {alias for alias in aliases if alias}


def _build_openrouter_alias_index(
    *,
    universe_ids: set[str],
    model_map: OpenRouterModelMap,
) -> dict[str, set[str]]:
    alias_to_ids: dict[str, set[str]] = {}
    for model_id in universe_ids:
        item = model_map.by_id.get(model_id)
        canonical_slug = None
        if isinstance(item, dict):
            raw = item.get("canonical_slug")
            if isinstance(raw, str) and raw.strip():
                canonical_slug = raw.strip()
        for alias in _openrouter_aliases_for_matching(model_id, canonical_slug):
            alias_to_ids.setdefault(alias, set()).add(model_id)
    return alias_to_ids


def _map_lmsys_key_to_openrouter_id(
    *,
    lmsys_key: str,
    alias_to_ids: dict[str, set[str]],
) -> tuple[str | None, str, float]:
    raw_key = lmsys_key.strip()
    if not raw_key:
        return None, "empty", 0.0

    normalized = _normalize_lmsys_key(raw_key)
    stripped = _normalize_lmsys_key(_strip_match_date_tail(raw_key))
    candidates = [
        (normalized, "lmsys_key_normalized", 0.75),
        (_flatten_alnum(normalized), "lmsys_key_flat", 0.7),
        (stripped, "lmsys_key_stripped", 0.65),
        (_flatten_alnum(stripped), "lmsys_key_stripped_flat", 0.6),
    ]

    for candidate, method, confidence in candidates:
        if not candidate:
            continue
        matched_ids = alias_to_ids.get(candidate, set())
        if len(matched_ids) == 1:
            return next(iter(matched_ids)), method, confidence

    return None, "unmapped", 0.0


def _load_json_with_cache(
    *,
    cache_path: Path,
    url: str,
    refresh: bool,
    user_agent: str,
) -> CacheLoad:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return CacheLoad(payload=payload, source="cache", cache_path=cache_path)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if cache_path.exists() and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return CacheLoad(
                payload=payload,
                source=f"cache_fallback ({exc})",
                cache_path=cache_path,
            )
        raise RuntimeError(f"Failed to fetch JSON from {url}: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CacheLoad(payload=payload, source="network", cache_path=cache_path)


def _load_text_with_cache(
    *,
    cache_path: Path,
    url: str,
    refresh: bool,
    user_agent: str,
) -> CacheLoad:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        payload = cache_path.read_text(encoding="utf-8")
        return CacheLoad(payload=payload, source="cache", cache_path=cache_path)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError) as exc:
        if cache_path.exists() and cache_path.is_file():
            payload = cache_path.read_text(encoding="utf-8")
            return CacheLoad(
                payload=payload,
                source=f"cache_fallback ({exc})",
                cache_path=cache_path,
            )
        raise RuntimeError(f"Failed to fetch text from {url}: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(payload, encoding="utf-8")
    return CacheLoad(payload=payload, source="network", cache_path=cache_path)


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _load_aa_llm_models_with_cache(
    *,
    cache_path: Path,
    url: str,
    api_key: str,
    refresh: bool,
    user_agent: str,
) -> CacheLoad:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return CacheLoad(payload=payload, source="cache", cache_path=cache_path)

    if not api_key:
        if cache_path.exists() and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return CacheLoad(
                payload=payload, source="cache_no_key", cache_path=cache_path
            )
        return CacheLoad(
            payload={"data": []}, source="missing_api_key", cache_path=cache_path
        )

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
            "x-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if cache_path.exists() and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return CacheLoad(
                payload=payload,
                source=f"cache_fallback ({exc})",
                cache_path=cache_path,
            )
        return CacheLoad(
            payload={"data": [], "error": str(exc)},
            source=f"error ({exc})",
            cache_path=cache_path,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CacheLoad(payload=payload, source="network", cache_path=cache_path)


def _load_hf_leaderboard_with_cache(
    *,
    cache_path: Path,
    rows_url_template: str,
    refresh: bool,
    user_agent: str,
    page_size: int = 100,
) -> CacheLoad:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return CacheLoad(payload=payload, source="cache", cache_path=cache_path)

    rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] | None = None
    num_rows_total: int | None = None
    offset = 0

    try:
        while True:
            url = rows_url_template.format(offset=offset, length=page_size)
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": user_agent,
                },
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                page = json.loads(response.read().decode("utf-8"))

            if features is None and isinstance(page.get("features"), list):
                features = page.get("features")

            page_rows = page.get("rows")
            if not isinstance(page_rows, list) or not page_rows:
                break

            rows.extend(page_rows)

            total = page.get("num_rows_total")
            if isinstance(total, int):
                num_rows_total = total

            offset += len(page_rows)
            if len(page_rows) < page_size:
                break
            if num_rows_total is not None and offset >= num_rows_total:
                break
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if cache_path.exists() and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return CacheLoad(
                payload=payload,
                source=f"cache_fallback ({exc})",
                cache_path=cache_path,
            )
        raise RuntimeError(
            f"Failed to fetch Hugging Face leaderboard rows: {exc}"
        ) from exc

    payload = {
        "source": "open-llm-leaderboard/contents",
        "num_rows_total": num_rows_total,
        "fetched_rows": len(rows),
        "features": features or [],
        "rows": rows,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CacheLoad(payload=payload, source="network", cache_path=cache_path)


def _load_lmsys_leaderboard_with_cache(
    *,
    cache_path: Path,
    tree_url: str,
    raw_url_template: str,
    refresh: bool,
    user_agent: str,
) -> CacheLoad:
    if cache_path.exists() and cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return CacheLoad(payload=payload, source="cache", cache_path=cache_path)

    try:
        tree_request = urllib.request.Request(
            tree_url,
            headers={
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
        )
        with urllib.request.urlopen(tree_request, timeout=45) as response:
            entries = json.loads(response.read().decode("utf-8"))

        if not isinstance(entries, list):
            raise RuntimeError("LMSYS tree payload is not a list")

        table_paths = [
            str(entry.get("path", ""))
            for entry in entries
            if isinstance(entry, dict)
            and re.match(r"leaderboard_table_\d{8}\.csv$", str(entry.get("path", "")))
        ]
        if not table_paths:
            raise RuntimeError("No leaderboard_table_YYYYMMDD.csv file found")

        selected_path = sorted(table_paths)[-1]
        raw_url = raw_url_template.format(path=selected_path)
        csv_request = urllib.request.Request(
            raw_url,
            headers={
                "Accept": "text/csv,text/plain,*/*",
                "User-Agent": user_agent,
            },
        )
        with urllib.request.urlopen(csv_request, timeout=45) as response:
            csv_text = response.read().decode("utf-8", errors="ignore")

        rows = list(csv.DictReader(io.StringIO(csv_text)))
        payload = {
            "selected_path": selected_path,
            "tree_entry_count": len(entries),
            "rows_count": len(rows),
            "rows": rows,
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        if cache_path.exists() and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return CacheLoad(
                payload=payload,
                source=f"cache_fallback ({exc})",
                cache_path=cache_path,
            )
        raise RuntimeError(f"Failed to fetch LMSYS leaderboard data: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CacheLoad(payload=payload, source="network", cache_path=cache_path)


def _build_openrouter_model_map(payload: dict[str, Any]) -> OpenRouterModelMap:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter models payload missing 'data' list")

    by_id: dict[str, dict[str, Any]] = {}
    canonical_to_id: dict[str, str] = {}
    normalized_id_to_id: dict[str, str] = {}
    normalized_canonical_groups: dict[str, set[str]] = {}
    huggingface_to_ids_groups: dict[str, set[str]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        by_id[model_id] = item

        canonical_slug = item.get("canonical_slug")
        if isinstance(canonical_slug, str) and canonical_slug.strip():
            canonical_slug = canonical_slug.strip()
            canonical_to_id[canonical_slug] = model_id
            normalized_canonical = _normalize_model_id(canonical_slug)
            normalized_canonical_groups.setdefault(normalized_canonical, set()).add(
                model_id
            )

        normalized_id = _normalize_model_id(model_id)
        normalized_id_to_id.setdefault(normalized_id, model_id)

        hugging_face_id = item.get("hugging_face_id")
        if isinstance(hugging_face_id, str) and hugging_face_id.strip():
            hf_key = _normalize_hf_repo_id(hugging_face_id)
            huggingface_to_ids_groups.setdefault(hf_key, set()).add(model_id)

    normalized_canonical_to_id: dict[str, str] = {}
    for key, candidates in normalized_canonical_groups.items():
        if len(candidates) == 1:
            normalized_canonical_to_id[key] = next(iter(candidates))

    huggingface_to_ids = {
        key: sorted(values) for key, values in huggingface_to_ids_groups.items()
    }

    return OpenRouterModelMap(
        by_id=by_id,
        canonical_to_id=canonical_to_id,
        normalized_id_to_id=normalized_id_to_id,
        normalized_canonical_to_id=normalized_canonical_to_id,
        huggingface_to_ids=huggingface_to_ids,
    )


def _map_slug_to_openrouter_id(
    *,
    slug: str,
    model_map: OpenRouterModelMap,
) -> tuple[str | None, str, float]:
    key = slug.strip()
    if not key:
        return None, "empty", 0.0

    if key in model_map.by_id:
        return key, "openrouter_id_exact", 1.0
    if key in model_map.canonical_to_id:
        return model_map.canonical_to_id[key], "canonical_exact", 0.95

    normalized = _normalize_model_id(key)
    if normalized in model_map.by_id:
        return normalized, "openrouter_id_normalized", 0.85
    if normalized in model_map.normalized_id_to_id:
        return (
            model_map.normalized_id_to_id[normalized],
            "openrouter_id_lookup_normalized",
            0.8,
        )
    if normalized in model_map.normalized_canonical_to_id:
        return (
            model_map.normalized_canonical_to_id[normalized],
            "canonical_normalized",
            0.75,
        )

    return None, "unmapped", 0.0


def _extract_rankings_performance(html: str) -> list[dict[str, Any]]:
    object_pattern = re.compile(r"\{\\\"id\\\":\\\".*?\\\"provider_count\\\":[0-9]+\}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for match in object_pattern.finditer(html):
        candidate = match.group(0).replace('\\"', '"')
        try:
            row = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if model_id in seen_ids:
            continue
        if row.get("p50_throughput") is None or row.get("p50_latency") is None:
            continue
        seen_ids.add(model_id)
        rows.append(row)

    return rows


def _extract_rankings_benchmarks(html: str) -> list[dict[str, Any]]:
    categories = ("intelligence", "coding", "agentic")
    markers: list[tuple[int, str]] = []
    for category in categories:
        for match in re.finditer(rf'\\"{category}\\":\[', html):
            markers.append((match.start(), category))
    markers.sort()
    marker_offsets = [item[0] for item in markers]

    object_pattern = re.compile(r"\{\\\"permaslug\\\":\\\".*?\\\"score\\\":[0-9.]+\}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in object_pattern.finditer(html):
        marker_index = bisect.bisect_right(marker_offsets, match.start()) - 1
        if marker_index < 0:
            category = "unknown"
        else:
            category = markers[marker_index][1]

        candidate = match.group(0).replace('\\"', '"')
        try:
            row = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        permaslug = row.get("permaslug")
        score = _to_float(row.get("score"))
        if not isinstance(permaslug, str) or not permaslug.strip() or score is None:
            continue

        key = (category, permaslug)
        if key in seen:
            continue
        seen.add(key)

        row["category"] = category
        row["score"] = score
        rows.append(row)

    return rows


def _extract_hf_leaderboard_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []

    by_fullname: dict[str, dict[str, Any]] = {}

    for wrapped in rows:
        if not isinstance(wrapped, dict):
            continue
        row = wrapped.get("row")
        if not isinstance(row, dict):
            continue

        fullname = row.get("fullname")
        if not isinstance(fullname, str) or not fullname.strip():
            continue
        fullname = fullname.strip()
        fullname_key = _normalize_hf_repo_id(fullname)

        metrics = {
            "average": _to_float(row.get("Average ⬆️")),
            "ifeval": _to_float(row.get("IFEval")),
            "gpqa": _to_float(row.get("GPQA")),
            "mmlu_pro": _to_float(row.get("MMLU-PRO")),
            "bbh": _to_float(row.get("BBH")),
            "math_lvl5": _to_float(row.get("MATH Lvl 5")),
        }
        if all(value is None for value in metrics.values()):
            continue

        bucket = by_fullname.setdefault(
            fullname_key,
            {
                "hf_fullname": fullname,
                "row_count": 0,
                "metrics": {
                    "average": None,
                    "ifeval": None,
                    "gpqa": None,
                    "mmlu_pro": None,
                    "bbh": None,
                    "math_lvl5": None,
                },
            },
        )

        bucket["row_count"] = int(bucket["row_count"]) + 1
        metric_bucket = bucket["metrics"]
        if not isinstance(metric_bucket, dict):
            continue

        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            prev = _to_float(metric_bucket.get(metric_name))
            if prev is None or metric_value > prev:
                metric_bucket[metric_name] = metric_value

    return list(by_fullname.values())


def _extract_lmsys_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []

    extracted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key")
        if not isinstance(key, str) or not key.strip():
            continue

        mt_bench = _to_float(row.get("MT-bench (score)"))
        mmlu = _to_float(row.get("MMLU"))
        if mt_bench is None and mmlu is None:
            continue

        extracted.append(
            {
                "key": key.strip(),
                "model_name": row.get("Model"),
                "mt_bench": mt_bench,
                "mmlu": mmlu,
                "organization": row.get("Organization"),
                "date": row.get("date"),
                "link": row.get("Link"),
            }
        )

    return extracted


def _slugify_for_matching(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9.+-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _extract_aa_llm_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        slug = item.get("slug")
        name = item.get("name")
        creator = item.get("model_creator")
        evaluations = item.get("evaluations")
        if not isinstance(slug, str) or not slug.strip():
            continue
        if not isinstance(evaluations, dict):
            continue

        creator_slug = None
        if isinstance(creator, dict):
            raw_creator_slug = creator.get("slug")
            if isinstance(raw_creator_slug, str) and raw_creator_slug.strip():
                creator_slug = raw_creator_slug.strip()

        rows.append(
            {
                "id": item.get("id"),
                "slug": slug.strip(),
                "name": name if isinstance(name, str) else None,
                "creator_slug": creator_slug,
                "evaluations": evaluations,
                "pricing": item.get("pricing")
                if isinstance(item.get("pricing"), dict)
                else {},
                "median_output_tokens_per_second": _to_float(
                    item.get("median_output_tokens_per_second")
                ),
                "median_time_to_first_token_seconds": _to_float(
                    item.get("median_time_to_first_token_seconds")
                ),
            }
        )

    return rows


def _minmax_normalize(
    values: dict[str, float], *, invert: bool = False
) -> dict[str, float]:
    if not values:
        return {}
    numbers = list(values.values())
    low = min(numbers)
    high = max(numbers)
    if math.isclose(low, high):
        return {key: 0.5 for key in values}

    result: dict[str, float] = {}
    for key, value in values.items():
        normalized = (value - low) / (high - low)
        if invert:
            normalized = 1.0 - normalized
        result[key] = max(0.0, min(1.0, normalized))
    return result


def _confidence_label(value: float | None) -> str:
    if value is None:
        return "none"
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium"
    if value > 0:
        return "low"
    return "none"


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_currency(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"${value:.{digits}f}"


def _snapshot_label(raw_label: str) -> str:
    label = raw_label.strip()
    if not label:
        return datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", label)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Minimize cost, maximize speed/capability.
    frontier: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["cost_typical"] <= row["cost_typical"]
                and other["speed_score"] >= row["speed_score"]
                and other["capability_score"] >= row["capability_score"]
                and (
                    other["cost_typical"] < row["cost_typical"]
                    or other["speed_score"] > row["speed_score"]
                    or other["capability_score"] > row["capability_score"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(
        key=lambda item: (
            item["cost_typical"],
            -item["balanced_score"],
            item["model_id"],
        )
    )
    return frontier


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "cache"
    env_file = Path(args.env_file).expanduser().resolve()
    _load_env_file(env_file)

    models_dev_cache = cache_dir / "models_dev_api.json"
    openrouter_models_cache = cache_dir / "openrouter_models_api.json"
    openrouter_rankings_cache = cache_dir / "openrouter_rankings_performance.html"
    hf_leaderboard_cache = cache_dir / "hf_open_llm_leaderboard_rows.json"
    lmsys_leaderboard_cache = cache_dir / "lmsys_arena_leaderboard.json"
    aa_llm_models_cache = cache_dir / "aa_llm_models.json"

    aa_api_key = os.environ.get("AA_API_KEY", "").strip()
    if aa_api_key.startswith("op://"):
        aa_api_key = ""
    user_agent = str(args.user_agent).strip() or DEFAULT_USER_AGENT

    models_dev_load = _load_json_with_cache(
        cache_path=models_dev_cache,
        url=args.models_dev_url,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )
    openrouter_models_load = _load_json_with_cache(
        cache_path=openrouter_models_cache,
        url=args.openrouter_models_url,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )
    rankings_load = _load_text_with_cache(
        cache_path=openrouter_rankings_cache,
        url=args.openrouter_rankings_url,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )
    hf_leaderboard_load = _load_hf_leaderboard_with_cache(
        cache_path=hf_leaderboard_cache,
        rows_url_template=args.hf_leaderboard_rows_url_template,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )
    lmsys_leaderboard_load = _load_lmsys_leaderboard_with_cache(
        cache_path=lmsys_leaderboard_cache,
        tree_url=args.lmsys_tree_url,
        raw_url_template=args.lmsys_raw_url_template,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )
    aa_llm_models_load = _load_aa_llm_models_with_cache(
        cache_path=aa_llm_models_cache,
        url=args.aa_llm_models_url,
        api_key=aa_api_key,
        refresh=bool(args.refresh_sources),
        user_agent=user_agent,
    )

    models_dev_payload = models_dev_load.payload
    openrouter_provider = models_dev_payload.get("openrouter")
    if not isinstance(openrouter_provider, dict):
        raise RuntimeError("models.dev payload missing top-level openrouter provider")

    models_dev_models = openrouter_provider.get("models")
    if not isinstance(models_dev_models, dict):
        raise RuntimeError("models.dev openrouter payload missing models object")

    openrouter_map = _build_openrouter_model_map(openrouter_models_load.payload)
    performance_rows = _extract_rankings_performance(rankings_load.payload)
    benchmark_rows = _extract_rankings_benchmarks(rankings_load.payload)
    hf_models = _extract_hf_leaderboard_models(hf_leaderboard_load.payload)
    lmsys_models = _extract_lmsys_models(lmsys_leaderboard_load.payload)
    aa_api_models = _extract_aa_llm_models(aa_llm_models_load.payload)

    universe_rows: list[dict[str, Any]] = []
    universe_by_id: dict[str, dict[str, Any]] = {}

    for model_id in sorted(models_dev_models):
        raw = models_dev_models[model_id]
        if not isinstance(raw, dict):
            continue

        limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
        cost = raw.get("cost") if isinstance(raw.get("cost"), dict) else {}
        modalities = (
            raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
        )

        input_cost_per_1m = _to_float(cost.get("input"))
        output_cost_per_1m = _to_float(cost.get("output"))
        input_cost_per_token = (
            None if input_cost_per_1m is None else input_cost_per_1m / 1_000_000.0
        )
        output_cost_per_token = (
            None if output_cost_per_1m is None else output_cost_per_1m / 1_000_000.0
        )

        costs_by_preset: dict[str, float | None] = {}
        for preset_name, preset in COST_PRESETS.items():
            if input_cost_per_token is None or output_cost_per_token is None:
                costs_by_preset[preset_name] = None
                continue
            costs_by_preset[preset_name] = (
                input_cost_per_token * preset["input_tokens_per_page"]
                + output_cost_per_token * preset["output_tokens_per_page"]
            )

        openrouter_match_id, openrouter_mapping_method, openrouter_mapping_conf = (
            _map_slug_to_openrouter_id(slug=model_id, model_map=openrouter_map)
        )
        canonical_slug = None
        hugging_face_id = None
        if openrouter_match_id and openrouter_match_id in openrouter_map.by_id:
            openrouter_item = openrouter_map.by_id[openrouter_match_id]
            canonical_slug = openrouter_item.get("canonical_slug")
            raw_hf = openrouter_item.get("hugging_face_id")
            if isinstance(raw_hf, str) and raw_hf.strip():
                hugging_face_id = raw_hf.strip()

        row = {
            "model_id": model_id,
            "name": raw.get("name"),
            "family": raw.get("family"),
            "open_weights": bool(raw.get("open_weights", False)),
            "release_date": raw.get("release_date"),
            "knowledge_cutoff": raw.get("knowledge"),
            "context_tokens": _to_float(limit.get("context")),
            "output_limit_tokens": _to_float(limit.get("output")),
            "reasoning": bool(raw.get("reasoning", False)),
            "structured_output": bool(raw.get("structured_output", False)),
            "tool_call": bool(raw.get("tool_call", False)),
            "input_modalities": modalities.get("input", []),
            "output_modalities": modalities.get("output", []),
            "cost_input_per_1m_usd": input_cost_per_1m,
            "cost_output_per_1m_usd": output_cost_per_1m,
            "cost_input_per_token_usd": input_cost_per_token,
            "cost_output_per_token_usd": output_cost_per_token,
            "cost_per_page_light_usd": costs_by_preset["light"],
            "cost_per_page_typical_usd": costs_by_preset["typical"],
            "cost_per_page_dense_usd": costs_by_preset["dense"],
            "openrouter_id": openrouter_match_id,
            "openrouter_canonical_slug": canonical_slug,
            "openrouter_hugging_face_id": hugging_face_id,
            "openrouter_mapping_method": openrouter_mapping_method,
            "openrouter_mapping_confidence": openrouter_mapping_conf,
        }
        universe_rows.append(row)
        universe_by_id[model_id] = row

    alias_to_ids = _build_openrouter_alias_index(
        universe_ids=set(universe_by_id),
        model_map=openrouter_map,
    )

    speed_signals: dict[str, dict[str, Any]] = {}
    speed_mapping_stats = {
        "parsed_rows": 0,
        "mapped_openrouter": 0,
        "mapped_universe": 0,
    }

    for raw in performance_rows:
        speed_mapping_stats["parsed_rows"] += 1
        ranking_id = str(raw.get("id", "")).strip()
        mapped_id, mapping_method, mapping_conf = _map_slug_to_openrouter_id(
            slug=ranking_id,
            model_map=openrouter_map,
        )
        if mapped_id is None:
            continue
        speed_mapping_stats["mapped_openrouter"] += 1
        if mapped_id not in universe_by_id:
            continue
        speed_mapping_stats["mapped_universe"] += 1

        request_count = _to_float(raw.get("request_count")) or 0.0
        if request_count >= 500_000:
            request_conf = 0.95
        elif request_count >= 250_000:
            request_conf = 0.85
        elif request_count >= 100_000:
            request_conf = 0.75
        else:
            request_conf = 0.6

        confidence = min(1.0, (0.6 * request_conf) + (0.4 * mapping_conf))

        speed_signals[mapped_id] = {
            "ranking_id": ranking_id,
            "mapping_method": mapping_method,
            "mapping_confidence": mapping_conf,
            "p50_latency_ms": _to_float(raw.get("p50_latency")),
            "p50_throughput_tok_s": _to_float(raw.get("p50_throughput")),
            "request_count": int(request_count),
            "best_latency_provider": raw.get("best_latency_provider"),
            "best_throughput_provider": raw.get("best_throughput_provider"),
            "best_latency_price_per_1m": _to_float(raw.get("best_latency_price")),
            "best_throughput_price_per_1m": _to_float(raw.get("best_throughput_price")),
            "provider_count": int(_to_float(raw.get("provider_count")) or 0),
            "speed_confidence": confidence,
        }

    throughput_values = {
        model_id: signal["p50_throughput_tok_s"]
        for model_id, signal in speed_signals.items()
        if signal.get("p50_throughput_tok_s") is not None
    }
    latency_values = {
        model_id: signal["p50_latency_ms"]
        for model_id, signal in speed_signals.items()
        if signal.get("p50_latency_ms") is not None
    }
    throughput_norm = _minmax_normalize(throughput_values, invert=False)
    latency_inv_norm = _minmax_normalize(latency_values, invert=True)

    for model_id, signal in speed_signals.items():
        t_norm = throughput_norm.get(model_id)
        l_norm = latency_inv_norm.get(model_id)
        if t_norm is None or l_norm is None:
            signal["speed_score"] = None
            continue
        signal["speed_score"] = (0.7 * t_norm) + (0.3 * l_norm)

    benchmark_mapping_stats = {
        "parsed_rows": 0,
        "mapped_openrouter": 0,
        "mapped_universe": 0,
    }
    benchmark_category_counts: dict[str, int] = {}
    benchmark_rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for raw in benchmark_rows:
        benchmark_mapping_stats["parsed_rows"] += 1
        category = str(raw.get("category", "unknown"))
        benchmark_category_counts[category] = (
            benchmark_category_counts.get(category, 0) + 1
        )

        candidates: list[str] = []
        for key in ("openrouter_slug", "permaslug", "heuristic_openrouter_slug"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        mapped_id: str | None = None
        mapping_method = "unmapped"
        mapping_conf = 0.0
        for candidate in candidates:
            maybe_id, method, conf = _map_slug_to_openrouter_id(
                slug=candidate,
                model_map=openrouter_map,
            )
            if maybe_id is not None:
                mapped_id = maybe_id
                mapping_method = method
                mapping_conf = conf
                break

        if mapped_id is None:
            continue
        benchmark_mapping_stats["mapped_openrouter"] += 1
        if mapped_id not in universe_by_id:
            continue
        benchmark_mapping_stats["mapped_universe"] += 1

        benchmark_rows_by_model.setdefault(mapped_id, []).append(
            {
                "category": category,
                "score": _to_float(raw.get("score")),
                "aa_name": raw.get("aa_name"),
                "mapping_method": mapping_method,
                "mapping_confidence": mapping_conf,
                "permaslug": raw.get("permaslug"),
                "openrouter_slug": raw.get("openrouter_slug"),
            }
        )

    hf_mapping_stats = {
        "parsed_rows": len(hf_leaderboard_load.payload.get("rows", []))
        if isinstance(hf_leaderboard_load.payload, dict)
        else 0,
        "parsed_unique_models": len(hf_models),
        "mapped_hf_models": 0,
        "mapped_universe": 0,
    }
    hf_rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for hf_model in hf_models:
        hf_fullname = hf_model.get("hf_fullname")
        if not isinstance(hf_fullname, str) or not hf_fullname.strip():
            continue
        hf_key = _normalize_hf_repo_id(hf_fullname)
        mapped_ids = openrouter_map.huggingface_to_ids.get(hf_key, [])
        if not mapped_ids:
            continue
        hf_mapping_stats["mapped_hf_models"] = (
            int(hf_mapping_stats["mapped_hf_models"]) + 1
        )

        mapped_in_universe = False
        for mapped_id in mapped_ids:
            if mapped_id not in universe_by_id:
                continue
            mapped_in_universe = True
            hf_rows_by_model.setdefault(mapped_id, []).append(
                {
                    "hf_fullname": hf_fullname,
                    "row_count": int(hf_model.get("row_count", 0)),
                    "metrics": hf_model.get("metrics", {}),
                    "mapping_method": "openrouter_hugging_face_id",
                    "mapping_confidence": 0.95 if len(mapped_ids) == 1 else 0.85,
                    "ambiguous_mapped_ids": len(mapped_ids),
                }
            )

        if mapped_in_universe:
            hf_mapping_stats["mapped_universe"] = (
                int(hf_mapping_stats["mapped_universe"]) + 1
            )

    lmsys_mapping_stats = {
        "selected_table": str(lmsys_leaderboard_load.payload.get("selected_path", ""))
        if isinstance(lmsys_leaderboard_load.payload, dict)
        else "",
        "parsed_rows": len(lmsys_models),
        "mapped_openrouter": 0,
        "mapped_universe": 0,
    }
    lmsys_rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for row in lmsys_models:
        key = row.get("key")
        if not isinstance(key, str) or not key.strip():
            continue

        mapped_id, mapping_method, mapping_conf = _map_lmsys_key_to_openrouter_id(
            lmsys_key=key,
            alias_to_ids=alias_to_ids,
        )
        if mapped_id is None:
            continue

        lmsys_mapping_stats["mapped_openrouter"] = (
            int(lmsys_mapping_stats["mapped_openrouter"]) + 1
        )

        if mapped_id not in universe_by_id:
            continue
        lmsys_mapping_stats["mapped_universe"] = (
            int(lmsys_mapping_stats["mapped_universe"]) + 1
        )

        lmsys_rows_by_model.setdefault(mapped_id, []).append(
            {
                "key": key,
                "model_name": row.get("model_name"),
                "mt_bench": _to_float(row.get("mt_bench")),
                "mmlu": _to_float(row.get("mmlu")),
                "organization": row.get("organization"),
                "date": row.get("date"),
                "mapping_method": mapping_method,
                "mapping_confidence": mapping_conf,
            }
        )

    aa_api_mapping_stats = {
        "parsed_rows": len(aa_api_models),
        "mapped_openrouter": 0,
        "mapped_universe": 0,
    }
    aa_api_rows_by_model: dict[str, list[dict[str, Any]]] = {}

    for row in aa_api_models:
        slug = row.get("slug")
        creator_slug = row.get("creator_slug")
        name = row.get("name")

        candidates: list[str] = []
        if (
            isinstance(creator_slug, str)
            and creator_slug
            and isinstance(slug, str)
            and slug
        ):
            candidates.append(f"{creator_slug}/{slug}")
        if isinstance(slug, str) and slug:
            candidates.append(slug)
        if isinstance(name, str) and name.strip():
            slugified_name = _slugify_for_matching(name)
            candidates.append(slugified_name)
            if isinstance(creator_slug, str) and creator_slug:
                candidates.append(f"{creator_slug}/{slugified_name}")

        mapped_id: str | None = None
        mapping_method = "unmapped"
        mapping_conf = 0.0

        for candidate in candidates:
            maybe_id, method, conf = _map_slug_to_openrouter_id(
                slug=candidate,
                model_map=openrouter_map,
            )
            if maybe_id is not None:
                mapped_id = maybe_id
                mapping_method = method
                mapping_conf = conf
                break

        if mapped_id is None:
            for candidate in candidates:
                maybe_id, method, conf = _map_lmsys_key_to_openrouter_id(
                    lmsys_key=candidate,
                    alias_to_ids=alias_to_ids,
                )
                if maybe_id is not None:
                    mapped_id = maybe_id
                    mapping_method = f"aa_name_{method}"
                    mapping_conf = conf * 0.9
                    break

        if mapped_id is None:
            continue

        aa_api_mapping_stats["mapped_openrouter"] = (
            int(aa_api_mapping_stats["mapped_openrouter"]) + 1
        )

        if mapped_id not in universe_by_id:
            continue
        aa_api_mapping_stats["mapped_universe"] = (
            int(aa_api_mapping_stats["mapped_universe"]) + 1
        )

        aa_api_rows_by_model.setdefault(mapped_id, []).append(
            {
                "aa_id": row.get("id"),
                "aa_slug": row.get("slug"),
                "aa_name": row.get("name"),
                "creator_slug": row.get("creator_slug"),
                "evaluations": row.get("evaluations", {}),
                "pricing": row.get("pricing", {}),
                "median_output_tokens_per_second": _to_float(
                    row.get("median_output_tokens_per_second")
                ),
                "median_time_to_first_token_seconds": _to_float(
                    row.get("median_time_to_first_token_seconds")
                ),
                "mapping_method": mapping_method,
                "mapping_confidence": mapping_conf,
            }
        )

    per_category_values: dict[str, dict[str, float]] = {
        "intelligence": {},
        "coding": {},
        "agentic": {},
    }
    for model_id, rows in benchmark_rows_by_model.items():
        for row in rows:
            category = row.get("category")
            score = _to_float(row.get("score"))
            if not isinstance(category, str) or score is None:
                continue
            if category not in per_category_values:
                continue
            per_category_values[category][model_id] = score

    normalized_category_scores: dict[str, dict[str, float]] = {}
    for category, values in per_category_values.items():
        normalized_category_scores[category] = _minmax_normalize(values, invert=False)

    hf_metric_values: dict[str, dict[str, float]] = {
        "average": {},
        "ifeval": {},
        "gpqa": {},
        "mmlu_pro": {},
        "bbh": {},
        "math_lvl5": {},
    }
    for model_id, rows in hf_rows_by_model.items():
        metrics_by_name: dict[str, float] = {}
        for row in rows:
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for metric_name in hf_metric_values:
                metric_value = _to_float(metrics.get(metric_name))
                if metric_value is None:
                    continue
                prev = metrics_by_name.get(metric_name)
                if prev is None or metric_value > prev:
                    metrics_by_name[metric_name] = metric_value
        for metric_name, metric_value in metrics_by_name.items():
            hf_metric_values[metric_name][model_id] = metric_value

    normalized_hf_metric_scores: dict[str, dict[str, float]] = {}
    for metric_name, values in hf_metric_values.items():
        normalized_hf_metric_scores[metric_name] = _minmax_normalize(
            values, invert=False
        )

    aa_capability_signals: dict[str, dict[str, Any]] = {}
    for model_id, rows in benchmark_rows_by_model.items():
        category_scores: dict[str, float] = {}
        mapping_confidences: list[float] = []
        for row in rows:
            category = row.get("category")
            if not isinstance(category, str):
                continue
            if (
                category in normalized_category_scores
                and model_id in normalized_category_scores[category]
            ):
                category_scores[category] = normalized_category_scores[category][
                    model_id
                ]
            mapping_confidences.append(float(row.get("mapping_confidence", 0.0)))

        weighted_sum = 0.0
        weighted_denom = 0.0
        for category, weight in CAPABILITY_WEIGHTS.items():
            score = category_scores.get(category)
            if score is None:
                continue
            weighted_sum += score * weight
            weighted_denom += weight

        capability_score = (
            None if weighted_denom == 0 else (weighted_sum / weighted_denom)
        )
        if capability_score is None:
            continue

        mapping_conf = (
            sum(mapping_confidences) / len(mapping_confidences)
            if mapping_confidences
            else 0.0
        )
        category_coverage = len(category_scores) / max(1, len(CAPABILITY_WEIGHTS))
        intelligence_bonus = 0.1 if "intelligence" in category_scores else 0.0
        capability_confidence = min(
            1.0,
            0.35
            + (0.4 * category_coverage)
            + (0.25 * mapping_conf)
            + intelligence_bonus,
        )

        aa_capability_signals[model_id] = {
            "capability_score": capability_score,
            "capability_confidence": capability_confidence,
            "category_scores": category_scores,
            "category_count": len(category_scores),
        }

    hf_capability_signals: dict[str, dict[str, Any]] = {}
    for model_id, rows in hf_rows_by_model.items():
        metric_scores: dict[str, float] = {}
        mapping_confidences: list[float] = []

        for metric_name in HF_CAPABILITY_WEIGHTS:
            normalized = normalized_hf_metric_scores.get(metric_name, {})
            if model_id in normalized:
                metric_scores[metric_name] = normalized[model_id]

        for row in rows:
            mapping_confidences.append(float(row.get("mapping_confidence", 0.0)))

        weighted_sum = 0.0
        weighted_denom = 0.0
        for metric_name, weight in HF_CAPABILITY_WEIGHTS.items():
            score = metric_scores.get(metric_name)
            if score is None:
                continue
            weighted_sum += score * weight
            weighted_denom += weight

        capability_score = (
            None if weighted_denom == 0 else (weighted_sum / weighted_denom)
        )
        if capability_score is None:
            continue

        mapping_conf = (
            sum(mapping_confidences) / len(mapping_confidences)
            if mapping_confidences
            else 0.0
        )
        metric_coverage = len(metric_scores) / max(1, len(HF_CAPABILITY_WEIGHTS))
        capability_confidence = min(
            1.0,
            0.35 + (0.3 * metric_coverage) + (0.3 * mapping_conf),
        )

        hf_capability_signals[model_id] = {
            "capability_score": capability_score,
            "capability_confidence": capability_confidence,
            "metric_scores": metric_scores,
            "metric_count": len(metric_scores),
        }

    lmsys_mt_values: dict[str, float] = {}
    lmsys_mmlu_values: dict[str, float] = {}
    for model_id, rows in lmsys_rows_by_model.items():
        best_mt: float | None = None
        best_mmlu: float | None = None
        for row in rows:
            mt = _to_float(row.get("mt_bench"))
            mmlu = _to_float(row.get("mmlu"))
            if mt is not None and (best_mt is None or mt > best_mt):
                best_mt = mt
            if mmlu is not None and (best_mmlu is None or mmlu > best_mmlu):
                best_mmlu = mmlu
        if best_mt is not None:
            lmsys_mt_values[model_id] = best_mt
        if best_mmlu is not None:
            lmsys_mmlu_values[model_id] = best_mmlu

    lmsys_mt_norm = _minmax_normalize(lmsys_mt_values, invert=False)
    lmsys_mmlu_norm = _minmax_normalize(lmsys_mmlu_values, invert=False)

    lmsys_capability_signals: dict[str, dict[str, Any]] = {}
    for model_id, rows in lmsys_rows_by_model.items():
        metric_scores: dict[str, float] = {}
        mapping_confidences: list[float] = []
        if model_id in lmsys_mt_norm:
            metric_scores["mt_bench"] = lmsys_mt_norm[model_id]
        if model_id in lmsys_mmlu_norm:
            metric_scores["mmlu"] = lmsys_mmlu_norm[model_id]

        for row in rows:
            mapping_confidences.append(float(row.get("mapping_confidence", 0.0)))

        weighted_sum = 0.0
        weighted_denom = 0.0
        for metric_name, weight in LMSYS_CAPABILITY_WEIGHTS.items():
            score = metric_scores.get(metric_name)
            if score is None:
                continue
            weighted_sum += score * weight
            weighted_denom += weight

        capability_score = (
            None if weighted_denom == 0 else (weighted_sum / weighted_denom)
        )
        if capability_score is None:
            continue

        mapping_conf = (
            sum(mapping_confidences) / len(mapping_confidences)
            if mapping_confidences
            else 0.0
        )
        metric_coverage = len(metric_scores) / max(1, len(LMSYS_CAPABILITY_WEIGHTS))
        capability_confidence = min(
            1.0,
            0.25 + (0.25 * metric_coverage) + (0.35 * mapping_conf),
        )

        lmsys_capability_signals[model_id] = {
            "capability_score": capability_score,
            "capability_confidence": capability_confidence,
            "metric_scores": metric_scores,
            "metric_count": len(metric_scores),
        }

    aa_api_metric_values: dict[str, dict[str, float]] = {
        metric_name: {} for metric_name in AA_API_CAPABILITY_WEIGHTS
    }
    for model_id, rows in aa_api_rows_by_model.items():
        best_by_metric: dict[str, float] = {}
        for row in rows:
            evaluations = row.get("evaluations")
            if not isinstance(evaluations, dict):
                continue
            for metric_name in AA_API_CAPABILITY_WEIGHTS:
                value = _to_float(evaluations.get(metric_name))
                if value is None:
                    continue
                prev = best_by_metric.get(metric_name)
                if prev is None or value > prev:
                    best_by_metric[metric_name] = value
        for metric_name, metric_value in best_by_metric.items():
            aa_api_metric_values[metric_name][model_id] = metric_value

    normalized_aa_api_scores: dict[str, dict[str, float]] = {}
    for metric_name, values in aa_api_metric_values.items():
        normalized_aa_api_scores[metric_name] = _minmax_normalize(values, invert=False)

    aa_api_capability_signals: dict[str, dict[str, Any]] = {}
    for model_id, rows in aa_api_rows_by_model.items():
        metric_scores: dict[str, float] = {}
        mapping_confidences: list[float] = []
        for metric_name in AA_API_CAPABILITY_WEIGHTS:
            normalized = normalized_aa_api_scores.get(metric_name, {})
            if model_id in normalized:
                metric_scores[metric_name] = normalized[model_id]

        for row in rows:
            mapping_confidences.append(float(row.get("mapping_confidence", 0.0)))

        weighted_sum = 0.0
        weighted_denom = 0.0
        for metric_name, weight in AA_API_CAPABILITY_WEIGHTS.items():
            score = metric_scores.get(metric_name)
            if score is None:
                continue
            weighted_sum += score * weight
            weighted_denom += weight

        capability_score = (
            None if weighted_denom == 0 else (weighted_sum / weighted_denom)
        )
        if capability_score is None:
            continue

        mapping_conf = (
            sum(mapping_confidences) / len(mapping_confidences)
            if mapping_confidences
            else 0.0
        )
        metric_coverage = len(metric_scores) / max(1, len(AA_API_CAPABILITY_WEIGHTS))
        capability_confidence = min(
            1.0,
            0.4 + (0.35 * metric_coverage) + (0.25 * mapping_conf),
        )

        aa_api_capability_signals[model_id] = {
            "capability_score": capability_score,
            "capability_confidence": capability_confidence,
            "metric_scores": metric_scores,
            "metric_count": len(metric_scores),
        }

    capability_signals: dict[str, dict[str, Any]] = {}
    for model_id in universe_by_id:
        source_scores: dict[str, float] = {}
        source_confidences: dict[str, float] = {}
        category_scores = {}
        aa_api_metric_scores = {}
        hf_metric_scores = {}
        lmsys_metric_scores = {}

        aa_api_signal = aa_api_capability_signals.get(model_id)
        if aa_api_signal:
            aa_api_score = _to_float(aa_api_signal.get("capability_score"))
            aa_api_conf = _to_float(aa_api_signal.get("capability_confidence"))
            if aa_api_score is not None:
                source_scores["aa_api"] = aa_api_score
                source_confidences["aa_api"] = (
                    aa_api_conf if aa_api_conf is not None else 0.0
                )
                aa_api_metric_scores = aa_api_signal.get("metric_scores", {})

        aa_signal = aa_capability_signals.get(model_id)
        if aa_signal:
            aa_score = _to_float(aa_signal.get("capability_score"))
            aa_conf = _to_float(aa_signal.get("capability_confidence"))
            if aa_score is not None:
                source_scores["aa_rankings"] = aa_score
                source_confidences["aa_rankings"] = (
                    aa_conf if aa_conf is not None else 0.0
                )
                category_scores = aa_signal.get("category_scores", {})

        hf_signal = hf_capability_signals.get(model_id)
        if hf_signal:
            hf_score = _to_float(hf_signal.get("capability_score"))
            hf_conf = _to_float(hf_signal.get("capability_confidence"))
            if hf_score is not None:
                source_scores["hf_open_llm"] = hf_score
                source_confidences["hf_open_llm"] = (
                    hf_conf if hf_conf is not None else 0.0
                )
                hf_metric_scores = hf_signal.get("metric_scores", {})

        lmsys_signal = lmsys_capability_signals.get(model_id)
        if lmsys_signal:
            lmsys_score = _to_float(lmsys_signal.get("capability_score"))
            lmsys_conf = _to_float(lmsys_signal.get("capability_confidence"))
            if lmsys_score is not None:
                source_scores["lmsys_arena"] = lmsys_score
                source_confidences["lmsys_arena"] = (
                    lmsys_conf if lmsys_conf is not None else 0.0
                )
                lmsys_metric_scores = lmsys_signal.get("metric_scores", {})

        if not source_scores:
            continue

        weighted_sum = 0.0
        weighted_denom = 0.0
        confidence_sum = 0.0
        confidence_denom = 0.0
        for source_name, score in source_scores.items():
            weight = CAPABILITY_SOURCE_WEIGHTS.get(source_name, 0.0)
            if weight <= 0:
                continue
            weighted_sum += score * weight
            weighted_denom += weight
            confidence_sum += float(source_confidences.get(source_name, 0.0)) * weight
            confidence_denom += weight

        if weighted_denom == 0:
            continue

        capability_score = weighted_sum / weighted_denom
        base_conf = confidence_sum / confidence_denom if confidence_denom > 0 else 0.0
        source_count_bonus = 0.05 if len(source_scores) > 1 else 0.0
        capability_confidence = min(1.0, base_conf + source_count_bonus)

        capability_signals[model_id] = {
            "capability_score": capability_score,
            "capability_confidence": capability_confidence,
            "category_scores": category_scores,
            "aa_api_metric_scores": aa_api_metric_scores,
            "hf_metric_scores": hf_metric_scores,
            "lmsys_metric_scores": lmsys_metric_scores,
            "source_scores": source_scores,
            "source": "+".join(sorted(source_scores.keys())),
        }

    tradeoff_rows: list[dict[str, Any]] = []
    for model_id in sorted(universe_by_id):
        base = universe_by_id[model_id]
        speed = speed_signals.get(model_id)
        capability = capability_signals.get(model_id)

        speed_score = speed.get("speed_score") if speed else None
        capability_score = capability.get("capability_score") if capability else None

        row = {
            "model_id": model_id,
            "name": base.get("name"),
            "family": base.get("family"),
            "context_tokens": base.get("context_tokens"),
            "structured_output": base.get("structured_output"),
            "reasoning": base.get("reasoning"),
            "tool_call": base.get("tool_call"),
            "cost_per_page_light_usd": base.get("cost_per_page_light_usd"),
            "cost_per_page_typical_usd": base.get("cost_per_page_typical_usd"),
            "cost_per_page_dense_usd": base.get("cost_per_page_dense_usd"),
            "speed_score": speed_score,
            "p50_throughput_tok_s": speed.get("p50_throughput_tok_s")
            if speed
            else None,
            "p50_latency_ms": speed.get("p50_latency_ms") if speed else None,
            "speed_request_count": speed.get("request_count") if speed else None,
            "speed_confidence": speed.get("speed_confidence") if speed else 0.0,
            "capability_score": capability_score,
            "capability_confidence": capability.get("capability_confidence")
            if capability
            else 0.0,
            "capability_category_scores": capability.get("category_scores")
            if capability
            else {},
            "capability_aa_api_metric_scores": capability.get("aa_api_metric_scores")
            if capability
            else {},
            "capability_hf_metric_scores": capability.get("hf_metric_scores")
            if capability
            else {},
            "capability_lmsys_metric_scores": capability.get("lmsys_metric_scores")
            if capability
            else {},
            "capability_source_scores": capability.get("source_scores")
            if capability
            else {},
            "capability_source": capability.get("source") if capability else None,
            "has_cost": base.get("cost_per_page_typical_usd") is not None,
            "has_speed": speed_score is not None,
            "has_capability": capability_score is not None,
            "speed_missing_reason": (
                None
                if speed_score is not None
                else "no_openrouter_rankings_performance_signal"
            ),
            "capability_missing_reason": (
                None
                if capability_score is not None
                else "no_outsourced_benchmark_signal"
            ),
            "openrouter_id": base.get("openrouter_id"),
            "openrouter_canonical_slug": base.get("openrouter_canonical_slug"),
            "openrouter_mapping_method": base.get("openrouter_mapping_method"),
        }
        tradeoff_rows.append(row)

    cost_typical_values = {
        row["model_id"]: row["cost_per_page_typical_usd"]
        for row in tradeoff_rows
        if isinstance(row.get("cost_per_page_typical_usd"), (int, float))
    }
    cost_efficiency = _minmax_normalize(cost_typical_values, invert=True)

    for row in tradeoff_rows:
        model_id = row["model_id"]
        row["cost_efficiency_score"] = cost_efficiency.get(model_id)

        cost_score = row.get("cost_efficiency_score")
        speed_score = row.get("speed_score")
        capability_score = row.get("capability_score")
        if cost_score is None or speed_score is None or capability_score is None:
            row["balanced_score"] = None
            row["economy_score"] = None
            row["performance_score"] = None
            continue

        row["balanced_score"] = (
            BALANCED_WEIGHTS["cost_efficiency"] * cost_score
            + BALANCED_WEIGHTS["speed"] * speed_score
            + BALANCED_WEIGHTS["capability"] * capability_score
        )
        row["economy_score"] = (
            ECONOMY_WEIGHTS["cost_efficiency"] * cost_score
            + ECONOMY_WEIGHTS["speed"] * speed_score
            + ECONOMY_WEIGHTS["capability"] * capability_score
        )
        row["performance_score"] = (
            PERFORMANCE_WEIGHTS["cost_efficiency"] * cost_score
            + PERFORMANCE_WEIGHTS["speed"] * speed_score
            + PERFORMANCE_WEIGHTS["capability"] * capability_score
        )

    eligible_for_frontier = [
        {
            "model_id": row["model_id"],
            "name": row.get("name"),
            "cost_typical": row["cost_per_page_typical_usd"],
            "speed_score": row["speed_score"],
            "capability_score": row["capability_score"],
            "balanced_score": row["balanced_score"],
        }
        for row in tradeoff_rows
        if row.get("balanced_score") is not None
    ]
    pareto_rows = _pareto_frontier(eligible_for_frontier)

    balanced_top = sorted(
        [row for row in tradeoff_rows if row.get("balanced_score") is not None],
        key=lambda item: (
            -float(item["balanced_score"]),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:15]
    economy_top = sorted(
        [row for row in tradeoff_rows if row.get("economy_score") is not None],
        key=lambda item: (
            -float(item["economy_score"]),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:15]
    performance_top = sorted(
        [row for row in tradeoff_rows if row.get("performance_score") is not None],
        key=lambda item: (
            -float(item["performance_score"]),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:15]

    strong_prior_rows = sorted(
        [
            row
            for row in tradeoff_rows
            if row.get("has_speed")
            and row.get("has_capability")
            and float(row.get("speed_confidence", 0.0)) >= 0.75
            and float(row.get("capability_confidence", 0.0)) >= 0.75
            and row.get("balanced_score") is not None
        ],
        key=lambda item: (
            -float(item["balanced_score"]),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:20]

    promising_but_uncertain_rows: list[dict[str, Any]] = []
    for row in tradeoff_rows:
        if not row.get("has_speed") or not row.get("has_cost"):
            continue
        if (
            row.get("has_capability")
            and float(row.get("capability_confidence", 0.0)) >= 0.75
        ):
            continue

        speed_score = _to_float(row.get("speed_score"))
        cost_eff = _to_float(row.get("cost_efficiency_score"))
        if speed_score is None or cost_eff is None:
            continue

        uncertainty_reason = (
            "missing_capability"
            if not row.get("has_capability")
            else "low_capability_confidence"
        )
        enriched = dict(row)
        enriched["exploration_score"] = (0.6 * speed_score) + (0.4 * cost_eff)
        enriched["uncertainty_reason"] = uncertainty_reason
        promising_but_uncertain_rows.append(enriched)

    promising_but_uncertain_rows = sorted(
        promising_but_uncertain_rows,
        key=lambda item: (
            -float(item["exploration_score"]),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:20]

    high_capability_unknown_speed_rows = sorted(
        [
            row
            for row in tradeoff_rows
            if row.get("has_capability")
            and not row.get("has_speed")
            and float(row.get("capability_confidence", 0.0)) >= 0.7
        ],
        key=lambda item: (
            -float(item.get("capability_score") or 0.0),
            item["cost_per_page_typical_usd"],
            item["model_id"],
        ),
    )[:20]

    candidate_funnel = {
        "strong_prior": {
            "description": (
                "Models with both speed and capability priors at high confidence."
            ),
            "count": len(strong_prior_rows),
            "rows": strong_prior_rows,
        },
        "promising_but_uncertain": {
            "description": (
                "Cost/speed-attractive models with missing or low-confidence capability priors."
            ),
            "count": len(promising_but_uncertain_rows),
            "rows": promising_but_uncertain_rows,
        },
        "high_capability_unknown_speed": {
            "description": (
                "Models with stronger capability priors but missing throughput priors."
            ),
            "count": len(high_capability_unknown_speed_rows),
            "rows": high_capability_unknown_speed_rows,
        },
    }

    coverage = {
        "universe_models": len(tradeoff_rows),
        "has_cost": sum(1 for row in tradeoff_rows if row["has_cost"]),
        "has_speed": sum(1 for row in tradeoff_rows if row["has_speed"]),
        "has_capability": sum(1 for row in tradeoff_rows if row["has_capability"]),
        "has_speed_and_capability": sum(
            1 for row in tradeoff_rows if row["has_speed"] and row["has_capability"]
        ),
        "eligible_for_weighted_views": sum(
            1 for row in tradeoff_rows if row.get("balanced_score") is not None
        ),
    }

    aa_data_used = bool(
        int(benchmark_mapping_stats.get("mapped_universe", 0)) > 0
        or int(aa_api_mapping_stats.get("mapped_universe", 0)) > 0
    )

    generated_at = datetime.now(UTC).isoformat()
    snapshot_label = _snapshot_label(args.snapshot_label) if args.write_snapshot else ""
    snapshot_dir = (
        Path(args.snapshot_dir).expanduser().resolve() / snapshot_label
        if args.write_snapshot
        else None
    )

    universe_output = {
        "generated_at": generated_at,
        "source_summary": {
            "models_dev": {
                "url": args.models_dev_url,
                "source": models_dev_load.source,
                "cache": str(models_dev_load.cache_path),
            },
            "openrouter_models": {
                "url": args.openrouter_models_url,
                "source": openrouter_models_load.source,
                "cache": str(openrouter_models_load.cache_path),
            },
            "openrouter_rankings": {
                "url": args.openrouter_rankings_url,
                "source": rankings_load.source,
                "cache": str(rankings_load.cache_path),
            },
            "aa_llm_models_api": {
                "url": args.aa_llm_models_url,
                "source": aa_llm_models_load.source,
                "cache": str(aa_llm_models_load.cache_path),
                "api_key_present": bool(aa_api_key),
            },
            "hf_open_llm_leaderboard": {
                "url_template": args.hf_leaderboard_rows_url_template,
                "source": hf_leaderboard_load.source,
                "cache": str(hf_leaderboard_load.cache_path),
            },
            "lmsys_arena_leaderboard": {
                "tree_url": args.lmsys_tree_url,
                "raw_url_template": args.lmsys_raw_url_template,
                "selected_table": str(
                    lmsys_leaderboard_load.payload.get("selected_path", "")
                )
                if isinstance(lmsys_leaderboard_load.payload, dict)
                else "",
                "source": lmsys_leaderboard_load.source,
                "cache": str(lmsys_leaderboard_load.cache_path),
            },
        },
        "attribution": {
            "artificial_analysis": {
                "required": aa_data_used,
                "url": AA_ATTRIBUTION_URL,
                "note": AA_ATTRIBUTION_NOTE,
                "sources": [
                    "aa_llm_models_api",
                    "openrouter_rankings_benchmark_slices",
                ],
            }
        },
        "snapshot": {
            "enabled": bool(args.write_snapshot),
            "label": snapshot_label,
            "path": str(snapshot_dir) if snapshot_dir else None,
        },
        "cost_presets": COST_PRESETS,
        "rows": universe_rows,
    }

    tradeoff_output = {
        "generated_at": generated_at,
        "coverage": coverage,
        "weights": {
            "capability_categories": CAPABILITY_WEIGHTS,
            "aa_api_capability_metrics": AA_API_CAPABILITY_WEIGHTS,
            "hf_capability_metrics": HF_CAPABILITY_WEIGHTS,
            "lmsys_capability_metrics": LMSYS_CAPABILITY_WEIGHTS,
            "capability_source_blend": CAPABILITY_SOURCE_WEIGHTS,
            "balanced": BALANCED_WEIGHTS,
            "economy": ECONOMY_WEIGHTS,
            "performance": PERFORMANCE_WEIGHTS,
        },
        "mapping_diagnostics": {
            "openrouter_models_api_count": len(openrouter_map.by_id),
            "models_dev_openrouter_universe_count": len(tradeoff_rows),
            "performance": speed_mapping_stats,
            "benchmarks": {
                **benchmark_mapping_stats,
                "category_counts": benchmark_category_counts,
            },
            "aa_api": aa_api_mapping_stats,
            "hf_open_llm": hf_mapping_stats,
            "lmsys_arena": lmsys_mapping_stats,
        },
        "attribution": {
            "artificial_analysis": {
                "required": aa_data_used,
                "url": AA_ATTRIBUTION_URL,
                "note": AA_ATTRIBUTION_NOTE,
            }
        },
        "snapshot": {
            "enabled": bool(args.write_snapshot),
            "label": snapshot_label,
            "path": str(snapshot_dir) if snapshot_dir else None,
        },
        "candidate_funnel": candidate_funnel,
        "pareto_frontier": pareto_rows,
        "weighted_views": {
            "balanced_top": balanced_top,
            "economy_top": economy_top,
            "performance_top": performance_top,
        },
        "rows": tradeoff_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = output_dir / "model_universe_openrouter.json"
    tradeoff_path = output_dir / "model_tradeoffs.json"
    markdown_path = output_dir / "model_tradeoffs.md"

    universe_path.write_text(
        json.dumps(universe_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tradeoff_path.write_text(
        json.dumps(tradeoff_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    coverage_rows = [
        ["universe_models", str(coverage["universe_models"])],
        ["has_cost", str(coverage["has_cost"])],
        ["has_speed", str(coverage["has_speed"])],
        ["has_capability", str(coverage["has_capability"])],
        ["has_speed_and_capability", str(coverage["has_speed_and_capability"])],
        ["eligible_for_weighted_views", str(coverage["eligible_for_weighted_views"])],
    ]

    mapping_rows = [
        [
            "performance",
            str(speed_mapping_stats["parsed_rows"]),
            str(speed_mapping_stats["mapped_openrouter"]),
            str(speed_mapping_stats["mapped_universe"]),
        ],
        [
            "aa_benchmarks",
            str(benchmark_mapping_stats["parsed_rows"]),
            str(benchmark_mapping_stats["mapped_openrouter"]),
            str(benchmark_mapping_stats["mapped_universe"]),
        ],
        [
            "aa_api",
            str(aa_api_mapping_stats["parsed_rows"]),
            str(aa_api_mapping_stats["mapped_openrouter"]),
            str(aa_api_mapping_stats["mapped_universe"]),
        ],
        [
            "hf_open_llm",
            str(hf_mapping_stats["parsed_unique_models"]),
            str(hf_mapping_stats["mapped_hf_models"]),
            str(hf_mapping_stats["mapped_universe"]),
        ],
        [
            "lmsys_arena",
            str(lmsys_mapping_stats["parsed_rows"]),
            str(lmsys_mapping_stats["mapped_openrouter"]),
            str(lmsys_mapping_stats["mapped_universe"]),
        ],
    ]

    pareto_table = [
        [
            row["model_id"],
            _format_currency(row["cost_typical"], 6),
            _format_float(row["speed_score"], 3),
            _format_float(row["capability_score"], 3),
            _format_float(row["balanced_score"], 3),
        ]
        for row in pareto_rows[:20]
    ]

    def _weighted_rows(rows: list[dict[str, Any]], score_key: str) -> list[list[str]]:
        return [
            [
                row["model_id"],
                _format_currency(row["cost_per_page_typical_usd"], 6),
                _format_float(row.get("speed_score"), 3),
                _format_float(row.get("capability_score"), 3),
                _format_float(row.get(score_key), 3),
                _confidence_label(
                    min(
                        float(row.get("speed_confidence", 0.0)),
                        float(row.get("capability_confidence", 0.0)),
                    )
                ),
            ]
            for row in rows
        ]

    strong_prior_table = [
        [
            row["model_id"],
            _format_currency(row.get("cost_per_page_typical_usd"), 6),
            _format_float(row.get("speed_score"), 3),
            _format_float(row.get("capability_score"), 3),
            _format_float(row.get("balanced_score"), 3),
            _confidence_label(
                min(
                    float(row.get("speed_confidence", 0.0)),
                    float(row.get("capability_confidence", 0.0)),
                )
            ),
        ]
        for row in strong_prior_rows[:12]
    ]

    promising_uncertain_table = [
        [
            row["model_id"],
            _format_currency(row.get("cost_per_page_typical_usd"), 6),
            _format_float(row.get("speed_score"), 3),
            _format_float(row.get("capability_score"), 3),
            _format_float(row.get("exploration_score"), 3),
            str(row.get("uncertainty_reason", "")),
        ]
        for row in promising_but_uncertain_rows[:12]
    ]

    high_capability_unknown_speed_table = [
        [
            row["model_id"],
            _format_currency(row.get("cost_per_page_typical_usd"), 6),
            _format_float(row.get("capability_score"), 3),
            _format_float(row.get("capability_confidence"), 3),
            str(row.get("speed_missing_reason") or "unknown"),
        ]
        for row in high_capability_unknown_speed_rows[:12]
    ]

    missing_speed = [row["model_id"] for row in tradeoff_rows if not row["has_speed"]]
    missing_capability = [
        row["model_id"] for row in tradeoff_rows if not row["has_capability"]
    ]

    lines: list[str] = [
        "# Model Tradeoffs (OpenRouter Universe via models.dev)",
        "",
        "- This report is non-prescriptive and intended for planning tradeoff space.",
        f"- Generated: `{generated_at}`",
        f"- models.dev source: `{models_dev_load.source}`",
        f"- OpenRouter models source: `{openrouter_models_load.source}`",
        f"- OpenRouter rankings source: `{rankings_load.source}`",
        f"- Artificial Analysis API source: `{aa_llm_models_load.source}`",
        f"- HF Open LLM leaderboard source: `{hf_leaderboard_load.source}`",
        f"- LMSYS Arena leaderboard source: `{lmsys_leaderboard_load.source}`",
        "",
        "## Assumptions",
        "",
        "- Cost source: `models.dev` openrouter `cost.input` / `cost.output` (treated as USD per 1M tokens).",
        "- Speed source: OpenRouter rankings performance panel (`p50_throughput`, `p50_latency`).",
        "- Capability sources: Artificial Analysis API benchmark fields, OpenRouter rankings benchmark slices, HF Open LLM leaderboard metrics, and LMSYS Arena table metrics.",
        "- Missing speed/capability priors remain `null` (no imputation).",
        "",
        "## Attribution",
        "",
        f"- Artificial Analysis attribution: `{AA_ATTRIBUTION_URL}`",
        f"- {AA_ATTRIBUTION_NOTE}",
        "",
        "### Token/Page presets",
        "",
    ]

    if snapshot_dir:
        lines.insert(11, f"- Snapshot path: `{snapshot_dir}`")

    lines.extend(
        _render_table(
            ["preset", "input_tokens_per_page", "output_tokens_per_page"],
            [
                [
                    name,
                    str(values["input_tokens_per_page"]),
                    str(values["output_tokens_per_page"]),
                ]
                for name, values in COST_PRESETS.items()
            ],
        )
    )

    lines.extend(["", "## Coverage", ""])
    lines.extend(_render_table(["metric", "count"], coverage_rows))

    lines.extend(["", "## Candidate funnel", ""])
    lines.append(
        "- `strong_prior`: high-confidence speed+capability priors for immediate calibration batch candidates."
    )
    lines.append(
        "- `promising_but_uncertain`: cost/speed-attractive rows that need capability validation."
    )
    lines.append(
        "- `high_capability_unknown_speed`: stronger capability priors needing throughput sampling."
    )

    lines.extend(["", "### strong_prior", ""])
    if strong_prior_table:
        lines.extend(
            _render_table(
                [
                    "model_id",
                    "typical_cost_per_page",
                    "speed_score",
                    "capability_score",
                    "balanced_score",
                    "confidence",
                ],
                strong_prior_table,
            )
        )
    else:
        lines.append("No rows currently meet strong-prior criteria.")

    lines.extend(["", "### promising_but_uncertain", ""])
    if promising_uncertain_table:
        lines.extend(
            _render_table(
                [
                    "model_id",
                    "typical_cost_per_page",
                    "speed_score",
                    "capability_score",
                    "exploration_score",
                    "uncertainty_reason",
                ],
                promising_uncertain_table,
            )
        )
    else:
        lines.append("No rows currently meet promising-but-uncertain criteria.")

    lines.extend(["", "### high_capability_unknown_speed", ""])
    if high_capability_unknown_speed_table:
        lines.extend(
            _render_table(
                [
                    "model_id",
                    "typical_cost_per_page",
                    "capability_score",
                    "capability_confidence",
                    "speed_status",
                ],
                high_capability_unknown_speed_table,
            )
        )
    else:
        lines.append("No rows currently meet high-capability-unknown-speed criteria.")

    lines.extend(["", "## Mapping diagnostics", ""])
    lines.extend(
        _render_table(
            ["source", "parsed_rows", "mapped_openrouter", "mapped_universe"],
            mapping_rows,
        )
    )

    lines.extend(["", "## Pareto frontier (cost vs speed vs capability)", ""])
    if pareto_table:
        lines.extend(
            _render_table(
                [
                    "model_id",
                    "typical_cost_per_page",
                    "speed_score",
                    "capability_score",
                    "balanced_score",
                ],
                pareto_table,
            )
        )
    else:
        lines.append("No models currently have all three metrics available.")

    lines.extend(["", "## Weighted view snapshots", ""])
    lines.append("### Balanced (0.3 cost, 0.3 speed, 0.4 capability)")
    lines.append("")
    lines.extend(
        _render_table(
            [
                "model_id",
                "typical_cost_per_page",
                "speed_score",
                "capability_score",
                "balanced_score",
                "confidence",
            ],
            _weighted_rows(balanced_top, "balanced_score"),
        )
    )

    lines.append("")
    lines.append("### Economy (0.5 cost, 0.2 speed, 0.3 capability)")
    lines.append("")
    lines.extend(
        _render_table(
            [
                "model_id",
                "typical_cost_per_page",
                "speed_score",
                "capability_score",
                "economy_score",
                "confidence",
            ],
            _weighted_rows(economy_top, "economy_score"),
        )
    )

    lines.append("")
    lines.append("### Performance (0.2 cost, 0.5 speed, 0.3 capability)")
    lines.append("")
    lines.extend(
        _render_table(
            [
                "model_id",
                "typical_cost_per_page",
                "speed_score",
                "capability_score",
                "performance_score",
                "confidence",
            ],
            _weighted_rows(performance_top, "performance_score"),
        )
    )

    lines.extend(
        [
            "",
            "## Missing priors",
            "",
            f"- models missing speed prior: `{len(missing_speed)}`",
            f"- models missing capability prior: `{len(missing_capability)}`",
            f"- speed missing sample: `{', '.join(missing_speed[:15])}`",
            f"- capability missing sample: `{', '.join(missing_capability[:15])}`",
            "",
            "Raw machine-readable outputs:",
            f"- `{universe_path}`",
            f"- `{tradeoff_path}`",
            "",
        ]
    )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    snapshot_manifest_path: Path | None = None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_universe_path = snapshot_dir / universe_path.name
        snapshot_tradeoff_path = snapshot_dir / tradeoff_path.name
        snapshot_markdown_path = snapshot_dir / markdown_path.name

        snapshot_universe_path.write_text(
            universe_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        snapshot_tradeoff_path.write_text(
            tradeoff_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        snapshot_markdown_path.write_text(
            markdown_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
        snapshot_manifest = {
            "generated_at": generated_at,
            "label": snapshot_label,
            "source_output_dir": str(output_dir),
            "coverage": coverage,
            "candidate_funnel_counts": {
                "strong_prior": candidate_funnel["strong_prior"]["count"],
                "promising_but_uncertain": candidate_funnel["promising_but_uncertain"][
                    "count"
                ],
                "high_capability_unknown_speed": candidate_funnel[
                    "high_capability_unknown_speed"
                ]["count"],
            },
            "files": {
                "model_universe_openrouter": str(snapshot_universe_path),
                "model_tradeoffs_json": str(snapshot_tradeoff_path),
                "model_tradeoffs_md": str(snapshot_markdown_path),
            },
        }
        snapshot_manifest_path.write_text(
            json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote universe JSON: {universe_path}")
    print(f"Wrote tradeoff JSON: {tradeoff_path}")
    print(f"Wrote markdown report: {markdown_path}")
    if snapshot_manifest_path:
        print(f"Wrote snapshot manifest: {snapshot_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
