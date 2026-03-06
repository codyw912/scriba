#!/usr/bin/env bash

set -euo pipefail

DOCTOR_ONLY=0
MAX_RUNS=12
CAMPAIGN_ID="cerebras-direct-$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--doctor-only)
		DOCTOR_ONLY=1
		shift
		;;
	--max-runs)
		MAX_RUNS="$2"
		shift 2
		;;
	--campaign-id)
		CAMPAIGN_ID="$2"
		shift 2
		;;
	-h | --help)
		cat <<'EOF'
Usage: bash scripts/quick_cerebras_bench.sh [options]

Options:
  --doctor-only       Run doctor checks only (no pipeline runs)
  --max-runs N        Max runs for matrix command (default: 12)
  --campaign-id ID    Campaign id for matrix report grouping
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

if [[ -z "${CEREBRAS_API_KEY:-}" ]]; then
	echo "CEREBRAS_API_KEY is not set. Set it in .env or environment first." >&2
	exit 2
fi

PROFILES=(
	profiles/remote/pipeline.profile.remote_cerebras_zai_glm_4_7.example.yaml
	profiles/remote/pipeline.profile.remote_cerebras_gpt_oss_120b.example.yaml
	profiles/remote/pipeline.profile.remote_cerebras_qwen3_235b_a22b_instruct_2507.example.yaml
	profiles/remote/pipeline.profile.remote_cerebras_llama31_8b.example.yaml
)

CMD=(bash scripts/run_matrix.sh --reset-log --campaign-id "$CAMPAIGN_ID" --max-runs "$MAX_RUNS")
for profile in "${PROFILES[@]}"; do
	CMD+=(--profile "$profile")
done

if [[ "$DOCTOR_ONLY" -eq 1 ]]; then
	CMD+=(--doctor-only)
fi

echo "Running campaign: $CAMPAIGN_ID"
"${CMD[@]}"

echo "Rendering matrix reports..."
scripts/render_matrix_report.py --json-output samples/matrix_report.json

echo "Done."
