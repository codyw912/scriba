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

resolve_run_status() {
	local profile="$1"
	local run_id="$2"
	uv run scriba status --profile "$profile" --run-id "$run_id" 2>/dev/null |
		uv run python -c 'import json,sys; print(str(json.load(sys.stdin).get("status", "failed_runtime")))' 2>/dev/null
}

for profile in "${PROFILES[@]}"; do
	profile_failed=0
	for input in "${SAMPLES[@]}"; do
		if [[ ! -f "$input" ]]; then
			continue
		fi

		if [[ "$MAX_FILE_BYTES" -gt 0 ]]; then
			file_bytes="$(wc -c <"$input" | tr -d '[:space:]')"
			if [[ "$file_bytes" -gt "$MAX_FILE_BYTES" ]]; then
				now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
				printf '{"timestamp":"%s","campaign_id":"%s","preset":"%s","profile":"%s","input":"%s","status":"skipped_large","file_bytes":%s,"max_file_bytes":%s}\n' \
					"$now" "$CAMPAIGN_ID" "$PRESET" "$profile" "$input" "$file_bytes" "$MAX_FILE_BYTES" >>"$OUTPUT_JSONL"
				echo "Skip large file: $input (${file_bytes} bytes > ${MAX_FILE_BYTES})"
				continue
			fi
		fi

		echo "Doctor: profile=$profile input=$input"
		if ! uv run scriba doctor --profile "$profile" --input "$input"; then
			now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
			printf '{"timestamp":"%s","campaign_id":"%s","preset":"%s","profile":"%s","input":"%s","status":"doctor_failed"}\n' \
				"$now" "$CAMPAIGN_ID" "$PRESET" "$profile" "$input" >>"$OUTPUT_JSONL"
			profile_failed=1
			if [[ "$STOP_PROFILE_ON_DOCTOR_FAIL" -eq 1 ]]; then
				echo "Stopping profile after doctor failure: $profile"
				break
			fi
			continue
		fi

		if [[ "$DOCTOR_ONLY" -eq 1 ]]; then
			now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
			printf '{"timestamp":"%s","campaign_id":"%s","preset":"%s","profile":"%s","input":"%s","status":"doctor_ok"}\n' \
				"$now" "$CAMPAIGN_ID" "$PRESET" "$profile" "$input" >>"$OUTPUT_JSONL"
			continue
		fi

		if [[ "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then
			now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
			printf '{"timestamp":"%s","campaign_id":"%s","preset":"%s","profile":"%s","input":"%s","status":"skipped_limit","max_runs":%s}\n' \
				"$now" "$CAMPAIGN_ID" "$PRESET" "$profile" "$input" "$MAX_RUNS" >>"$OUTPUT_JSONL"
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
		if uv run scriba run --profile "$profile" --input "$input" --run-id "$run_id"; then
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

		now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		printf '{"timestamp":"%s","campaign_id":"%s","preset":"%s","profile":"%s","input":"%s","run_id":"%s","status":"%s"}\n' \
			"$now" "$CAMPAIGN_ID" "$PRESET" "$profile" "$input" "$run_id" "$status" >>"$OUTPUT_JSONL"
	done

	if [[ "$profile_failed" -eq 1 && "$STOP_PROFILE_ON_DOCTOR_FAIL" -eq 1 ]]; then
		continue
	fi
done

echo "Matrix run log updated: $OUTPUT_JSONL"
