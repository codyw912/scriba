#!/usr/bin/env bash

set -euo pipefail

DOCTOR_ONLY=0
INCLUDE_BASELINE=0
MAX_RUNS=18
CAMPAIGN_ID="hosted-lowcost-$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--doctor-only)
		DOCTOR_ONLY=1
		shift
		;;
	--include-baseline)
		INCLUDE_BASELINE=1
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
Usage: bash scripts/quick_openrouter_bench.sh [options]

Options:
  --doctor-only       Run doctor checks only (no pipeline runs)
  --include-baseline  Include higher-cost baseline profile (qwen3.5-35b-a3b)
  --max-runs N        Max runs for matrix command (default: 18)
  --campaign-id ID    Campaign id for matrix + pareto filtering
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

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
	echo "OPENROUTER_API_KEY is not set. Set it in .env or environment first." >&2
	exit 2
fi

PROFILES=(
	profiles/remote/pipeline.profile.remote_openrouter_qwen25_7b.example.yaml
	profiles/remote/pipeline.profile.remote_openrouter_qwen35_flash.example.yaml
	profiles/remote/pipeline.profile.remote_openrouter_qwen3_coder_next.example.yaml
	profiles/remote/pipeline.profile.remote_openrouter_llama31_8b.example.yaml
	profiles/remote/pipeline.profile.remote_openrouter_mistral_nemo.example.yaml
	profiles/remote/pipeline.profile.remote_openrouter_gpt4o_mini.example.yaml
)

if [[ "$INCLUDE_BASELINE" -eq 1 ]]; then
	PROFILES+=(profiles/remote/pipeline.profile.remote_openrouter.example.yaml)
fi

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

if [[ "$DOCTOR_ONLY" -eq 0 ]]; then
	echo "Rendering hosted Pareto report..."
	scripts/render_hosted_pareto.py --campaign-id "$CAMPAIGN_ID" --require-validation-ok
fi

echo "Done."
