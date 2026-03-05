#!/usr/bin/env bash

set -euo pipefail

ARTIFACTS_ROOT="artifacts"
PATTERN="quick-*"
OUTPUT="samples/quick_runs.md"
TELEMETRY_OUTPUT="samples/quick_telemetry.md"
MATRIX_PATH="docs/backend_decision_matrix.md"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--artifacts-root)
		ARTIFACTS_ROOT="$2"
		shift 2
		;;
	--pattern)
		PATTERN="$2"
		shift 2
		;;
	--output)
		OUTPUT="$2"
		shift 2
		;;
	--telemetry-output)
		TELEMETRY_OUTPUT="$2"
		shift 2
		;;
	--matrix)
		MATRIX_PATH="$2"
		shift 2
		;;
	-h | --help)
		cat <<'EOF'
Usage: bash scripts/update_quick_eval.sh [options]

Options:
  --artifacts-root PATH  Artifacts root (default: artifacts)
  --pattern GLOB         Run glob pattern (default: quick-*)
  --output PATH          Output markdown path (default: samples/quick_runs.md)
  --telemetry-output PATH  Telemetry markdown path (default: samples/quick_telemetry.md)
  --matrix PATH          Decision matrix path (default: docs/backend_decision_matrix.md)
  -h, --help             Show this help text
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

scripts/render_run_table.py \
	--artifacts-root "$ARTIFACTS_ROOT" \
	--pattern "$PATTERN" \
	--output "$OUTPUT"

scripts/render_quick_telemetry.py \
	--artifacts-root "$ARTIFACTS_ROOT" \
	--pattern "$PATTERN" \
	--output "$TELEMETRY_OUTPUT"

scripts/append_quick_telemetry_history.py \
	--telemetry "$TELEMETRY_OUTPUT" \
	--matrix "$MATRIX_PATH"

echo "Updated: $OUTPUT"
echo "Updated: $TELEMETRY_OUTPUT"
echo "Updated: $MATRIX_PATH"
