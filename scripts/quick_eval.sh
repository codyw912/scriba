#!/usr/bin/env bash

set -euo pipefail

DOCTOR_ONLY=0
PROFILE="profiles/pipeline.profile.example.yaml"
SAMPLES_DIR="samples/docs"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--doctor-only)
		DOCTOR_ONLY=1
		shift
		;;
	--profile)
		PROFILE="$2"
		shift 2
		;;
	--samples-dir)
		SAMPLES_DIR="$2"
		shift 2
		;;
	-h | --help)
		cat <<'EOF'
Usage: bash scripts/quick_eval.sh [options]

Options:
  --doctor-only       Run doctor checks only
  --profile PATH      Profile path (default: profiles/pipeline.profile.example.yaml)
  --samples-dir PATH  Sample docs directory (default: samples/docs)
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

if [[ ! -f "$PROFILE" ]]; then
	echo "Profile not found: $PROFILE" >&2
	exit 1
fi

shopt -s nullglob
SAMPLES=("$SAMPLES_DIR"/*)
shopt -u nullglob

if [[ "${#SAMPLES[@]}" -eq 0 ]]; then
	echo "No sample files found in $SAMPLES_DIR" >&2
	echo "Add fixtures to samples/docs/ and retry." >&2
	exit 1
fi

for input in "${SAMPLES[@]}"; do
	if [[ ! -f "$input" ]]; then
		continue
	fi

	echo "Doctor: $input"
	uv run scribai doctor --profile "$PROFILE" --input "$input"

	if [[ "$DOCTOR_ONLY" -eq 0 ]]; then
		base="$(basename "$input")"
		stamp="$(date +%Y%m%d-%H%M%S)"
		run_id="quick-${base}-${stamp}"
		echo "Run: $input (run_id=$run_id)"
		uv run scribai run --profile "$PROFILE" --input "$input" --run-id "$run_id"
	fi
done

echo "Done."
