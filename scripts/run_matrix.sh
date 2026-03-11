#!/usr/bin/env bash

set -euo pipefail

DOCTOR_ONLY=0
SAMPLES_DIR="samples/docs"
OUTPUT_JSONL="samples/matrix_runs.jsonl"
MAX_RUNS=6
MAX_RUNS_SET=0
MAX_FILE_BYTES=1500000
MAX_FILE_BYTES_SET=0
STOP_PROFILE_ON_DOCTOR_FAIL=1
RESET_LOG=0
PRESET=""
CAMPAIGN_ID="campaign-$(date -u +%Y%m%dT%H%M%SZ)"

PROFILES=(
	profiles/pipeline.profile.example.yaml
)

while [[ $# -gt 0 ]]; do
	case "$1" in
	--doctor-only)
		DOCTOR_ONLY=1
		shift
		;;
	--profile)
		PROFILES+=("$2")
		shift 2
		;;
	--samples-dir)
		SAMPLES_DIR="$2"
		shift 2
		;;
	--output)
		OUTPUT_JSONL="$2"
		shift 2
		;;
	--max-runs)
		MAX_RUNS="$2"
		MAX_RUNS_SET=1
		shift 2
		;;
	--max-file-bytes)
		MAX_FILE_BYTES="$2"
		MAX_FILE_BYTES_SET=1
		shift 2
		;;
	--allow-large)
		MAX_FILE_BYTES=0
		MAX_FILE_BYTES_SET=1
		shift
		;;
	--preset)
		PRESET="$2"
		shift 2
		;;
	--campaign-id)
		CAMPAIGN_ID="$2"
		shift 2
		;;
	--continue-on-doctor-fail)
		STOP_PROFILE_ON_DOCTOR_FAIL=0
		shift
		;;
	--reset-log)
		RESET_LOG=1
		shift
		;;
	-h | --help)
		cat <<'EOF'
Usage: bash scripts/run_matrix.sh [options]

Options:
  --doctor-only       Run doctor checks only (no pipeline runs)
  --profile PATH      Add a profile (repeatable). Default includes profiles/pipeline.profile.example.yaml
  --preset NAME       Use a preset profile set (fast-iterate, quality-check)
  --campaign-id ID    Campaign id for grouping rows in output log
  --samples-dir PATH  Sample docs directory (default: samples/docs)
  --output PATH       Matrix output jsonl path (default: samples/matrix_runs.jsonl)
  --max-runs N        Max pipeline runs before stopping (default: 6)
  --max-file-bytes N  Skip files larger than N bytes (default: 1500000)
  --allow-large       Disable max-file-bytes guardrail
  --continue-on-doctor-fail  Continue same profile after doctor failure
  --reset-log         Truncate output JSONL before running
  -h, --help          Show this help text
EOF
		exit 0
		;;
	*)
		echo "Unknown option: $1" >&2
		exit 2
		;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
	set -a
	# shellcheck disable=SC1091
	. ".env"
	set +a
fi

if [[ -n "$PRESET" ]]; then
	case "$PRESET" in
	fast-iterate)
		PROFILES=(
			profiles/local_attached/pipeline.profile.local_attached_litellm.example.yaml
			profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_litellm.example.yaml
		)
		if [[ "$MAX_RUNS_SET" -eq 0 ]]; then
			MAX_RUNS=6
		fi
		if [[ "$MAX_FILE_BYTES_SET" -eq 0 ]]; then
			MAX_FILE_BYTES=1500000
		fi
		;;
	quality-check)
		PROFILES=(
			profiles/local_attached/pipeline.profile.local_attached_litellm.example.yaml
			profiles/local_spawned/pipeline.profile.local_spawned_llama_cpp_litellm_highcap.example.yaml
			profiles/remote/pipeline.profile.remote_openrouter.example.yaml
		)
		if [[ "$MAX_RUNS_SET" -eq 0 ]]; then
			MAX_RUNS=12
		fi
		if [[ "$MAX_FILE_BYTES_SET" -eq 0 ]]; then
			MAX_FILE_BYTES=5000000
		fi
		;;
	*)
		echo "Unknown preset: $PRESET" >&2
		echo "Supported presets: fast-iterate, quality-check" >&2
		exit 2
		;;
	esac
fi

# Remove duplicated default profile if caller supplied explicit profiles.
if [[ "${#PROFILES[@]}" -gt 1 ]]; then
	filtered=()
	for profile in "${PROFILES[@]}"; do
		if [[ "$profile" == "profiles/pipeline.profile.example.yaml" ]]; then
			continue
		fi
		filtered+=("$profile")
	done
	PROFILES=("${filtered[@]}")
fi

if [[ "${#PROFILES[@]}" -eq 0 ]]; then
	echo "No profiles provided." >&2
	exit 1
fi

for profile in "${PROFILES[@]}"; do
	if [[ ! -f "$profile" ]]; then
		echo "Profile not found: $profile" >&2
		exit 1
	fi
done

shopt -s nullglob
SAMPLES=("$SAMPLES_DIR"/*)
shopt -u nullglob

if [[ "${#SAMPLES[@]}" -eq 0 ]]; then
	echo "No sample files found in $SAMPLES_DIR" >&2
	exit 1
fi

mkdir -p "$(dirname "$OUTPUT_JSONL")"
if [[ "$RESET_LOG" -eq 1 ]]; then
	: >"$OUTPUT_JSONL"
	echo "Reset matrix log: $OUTPUT_JSONL"
fi

echo "Matrix campaign: id=$CAMPAIGN_ID preset=${PRESET:-custom}"
RUN_COUNT=0

resolve_benchmark_metadata() {
	local input_path="$1"
	INPUT_PATH="$input_path" uv run python -c 'import json, os; from pathlib import Path

def benchmark_root_for_path(path: Path) -> Path | None:
	parts = path.parts
	for idx in range(len(parts) - 2):
		if parts[idx:idx+3] == ("samples", "benchmarks", "v1"):
			return Path(*parts[:idx+3])
	return None

path = Path(os.environ["INPUT_PATH"]).expanduser().resolve()
root = benchmark_root_for_path(path)
result = {
	"fixture_id": None,
	"variant_id": None,
	"variant_family": None,
	"noise_level": None,
	"source_kind": None,
	"size_bucket": None,
	"doc_type": None,
}
if root is not None:
	try:
		rel = path.relative_to(root)
	except ValueError:
		rel = None
	if rel is not None:
		fixture_id = None
		variant_id = None
		source_kind = None
		if len(rel.parts) >= 3 and rel.parts[0] == "generated_pdfs":
			fixture_id = rel.parts[1]
			variant_id = Path(rel.parts[-1]).stem
			source_kind = "synthetic"
		elif len(rel.parts) >= 3 and rel.parts[0] == "real_paired" and rel.parts[1] == "pdf":
			fixture_id = path.stem
			source_kind = "real_paired"
		elif len(rel.parts) >= 3 and rel.parts[0] == "real_unpaired" and rel.parts[1] == "pdf":
			fixture_id = path.stem
			source_kind = "real_unpaired"
		if fixture_id is not None:
			result["fixture_id"] = fixture_id
			result["variant_id"] = variant_id
			result["source_kind"] = source_kind
			fixtures_path = root / "manifests" / "fixtures.json"
			if fixtures_path.exists():
				try:
					fixtures_raw = json.loads(fixtures_path.read_text(encoding="utf-8"))
				except json.JSONDecodeError:
					fixtures_raw = []
				if isinstance(fixtures_raw, list):
					for item in fixtures_raw:
						if isinstance(item, dict) and item.get("fixture_id") == fixture_id:
							result["size_bucket"] = item.get("size_bucket")
							result["doc_type"] = item.get("doc_type")
							break
			if variant_id is not None:
				variants_path = root / "manifests" / "variants.jsonl"
				if variants_path.exists():
					for line in variants_path.read_text(encoding="utf-8").splitlines():
						if not line.strip():
							continue
						try:
							item = json.loads(line)
						except json.JSONDecodeError:
							continue
						if isinstance(item, dict) and item.get("fixture_id") == fixture_id and item.get("variant_id") == variant_id:
							result["variant_family"] = item.get("variant_family")
							result["noise_level"] = item.get("noise_level")
							break
print(json.dumps(result, separators=(",", ":")))' 2>/dev/null || printf '{}'
}

write_matrix_row() {
	local status="$1"
	local input_path="$2"
	local profile_path="$3"
	local run_id="$4"
	local extra_json="${5-}"
	local metadata_json="${6-}"
	if [[ -z "$extra_json" ]]; then
		extra_json='{}'
	fi
	if [[ -z "$metadata_json" ]]; then
		metadata_json='{}'
	fi
	TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	CAMPAIGN_ID_VALUE="$CAMPAIGN_ID" \
	PRESET_VALUE="$PRESET" \
	PROFILE_VALUE="$profile_path" \
	INPUT_VALUE="$input_path" \
	RUN_ID_VALUE="$run_id" \
	STATUS_VALUE="$status" \
	EXTRA_JSON="$extra_json" \
	METADATA_JSON="$metadata_json" \
		uv run python -c 'import json, os, sys
row = {
	"timestamp": os.environ["TIMESTAMP"],
	"campaign_id": os.environ["CAMPAIGN_ID_VALUE"],
	"preset": os.environ["PRESET_VALUE"],
	"profile": os.environ["PROFILE_VALUE"],
	"input": os.environ["INPUT_VALUE"],
	"status": os.environ["STATUS_VALUE"],
}
run_id = os.environ.get("RUN_ID_VALUE", "").strip()
if run_id:
	row["run_id"] = run_id
for env_key in ("METADATA_JSON", "EXTRA_JSON"):
	raw = os.environ.get(env_key, "{}")
	try:
		payload = json.loads(raw)
	except json.JSONDecodeError:
		payload = {}
	if isinstance(payload, dict):
		for key, value in payload.items():
			row[key] = value
sys.stdout.write(json.dumps(row, separators=(",", ":")) + "\n")' >>"$OUTPUT_JSONL"
}

resolve_run_status() {
	local profile="$1"
	local run_id="$2"
	uv run scribai status --profile "$profile" --run-id "$run_id" 2>/dev/null |
		uv run python -c 'import json,sys; print(str(json.load(sys.stdin).get("status", "failed_runtime")))' 2>/dev/null
}

for profile in "${PROFILES[@]}"; do
	profile_failed=0
	for input in "${SAMPLES[@]}"; do
		if [[ ! -f "$input" ]]; then
			continue
		fi

		metadata_json="$(resolve_benchmark_metadata "$input")"

		if [[ "$MAX_FILE_BYTES" -gt 0 ]]; then
			file_bytes="$(wc -c <"$input" | tr -d '[:space:]')"
			if [[ "$file_bytes" -gt "$MAX_FILE_BYTES" ]]; then
				write_matrix_row \
					"skipped_large" \
					"$input" \
					"$profile" \
					"" \
					"{\"file_bytes\":$file_bytes,\"max_file_bytes\":$MAX_FILE_BYTES}" \
					"$metadata_json"
				echo "Skip large file: $input (${file_bytes} bytes > ${MAX_FILE_BYTES})"
				continue
			fi
		fi

		echo "Doctor: profile=$profile input=$input"
		if ! uv run scribai doctor --profile "$profile" --input "$input"; then
			write_matrix_row "doctor_failed" "$input" "$profile" "" "{}" "$metadata_json"
			profile_failed=1
			if [[ "$STOP_PROFILE_ON_DOCTOR_FAIL" -eq 1 ]]; then
				echo "Stopping profile after doctor failure: $profile"
				break
			fi
			continue
		fi

		if [[ "$DOCTOR_ONLY" -eq 1 ]]; then
			write_matrix_row "doctor_ok" "$input" "$profile" "" "{}" "$metadata_json"
			continue
		fi

		if [[ "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then
			write_matrix_row \
				"skipped_limit" \
				"$input" \
				"$profile" \
				"" \
				"{\"max_runs\":$MAX_RUNS}" \
				"$metadata_json"
			echo "Run limit reached (${MAX_RUNS}). Stopping matrix execution early."
			echo "Matrix run log updated: $OUTPUT_JSONL"
			exit 0
		fi

		sample_base="$(basename "$input")"
		sample_base="${sample_base%.*}"
		profile_base="$(basename "$profile")"
		profile_base="${profile_base%.yaml}"
		stamp="$(date +%Y%m%d-%H%M%S)"
		run_id="matrix-${profile_base}-${sample_base}-${stamp}"

		echo "Run: profile=$profile input=$input run_id=$run_id"
		status="failed_runtime"
		if uv run scribai run --profile "$profile" --input "$input" --run-id "$run_id"; then
			resolved_status="$(resolve_run_status "$profile" "$run_id")"
			if [[ -n "$resolved_status" ]]; then
				status="$resolved_status"
			else
				status="completed"
			fi
		else
			resolved_status="$(resolve_run_status "$profile" "$run_id")"
			if [[ -n "$resolved_status" ]]; then
				status="$resolved_status"
			fi
		fi
		RUN_COUNT=$((RUN_COUNT + 1))

		write_matrix_row "$status" "$input" "$profile" "$run_id" "{}" "$metadata_json"
	done

	if [[ "$profile_failed" -eq 1 && "$STOP_PROFILE_ON_DOCTOR_FAIL" -eq 1 ]]; then
		continue
	fi
done

echo "Matrix run log updated: $OUTPUT_JSONL"
