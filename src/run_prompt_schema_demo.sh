#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILE="${PROMPT_SCHEMA_DEMO_PROFILE:-UMCGCohortsStaging}"
BASE_PROMPTS="${PROMPT_SCHEMA_DEMO_BASE_PROMPTS:-${REPO_ROOT}/prompts_cohort.toml}"
BASE_ROOT="${PROMPT_SCHEMA_DEMO_BASE_ROOT:-${REPO_ROOT}/tests/prompt_schema_demo/base_repo}"
VARIANT_ROOT="${PROMPT_SCHEMA_DEMO_VARIANT_ROOT:-${REPO_ROOT}/tests/prompt_schema_demo/variant_repo}"
OUT_DIR="${PROMPT_SCHEMA_DEMO_OUT_DIR:-${REPO_ROOT}/tmp/prompt_schema_demo}"
WITH_LLM="${PROMPT_SCHEMA_DEMO_WITH_LLM:-0}"
LLM_CONFIG="${PROMPT_SCHEMA_DEMO_LLM_CONFIG:-${REPO_ROOT}/config.final.toml}"
LLM_MODEL="${PROMPT_SCHEMA_DEMO_LLM_MODEL:-}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_DIR}"

echo "[1/4] Exporting base and variant schema CSVs..."
python3 "${REPO_ROOT}/src/emx2_dynamic_runtime.py" export-schema-csv \
  --profile "${PROFILE}" \
  --local-root "${BASE_ROOT}" \
  --output "${OUT_DIR}/base_schema.csv" >/dev/null

python3 "${REPO_ROOT}/src/emx2_dynamic_runtime.py" export-schema-csv \
  --profile "${PROFILE}" \
  --local-root "${VARIANT_ROOT}" \
  --output "${OUT_DIR}/variant_schema.csv" >/dev/null

echo "[2/4] Generating dynamic prompts from fixture CSVs..."
python3 "${REPO_ROOT}/src/cohort_dynamic_prompts.py" \
  --profile "${PROFILE}" \
  --local-root "${BASE_ROOT}" \
  --output "${OUT_DIR}/base_dynamic.toml" >/dev/null

python3 "${REPO_ROOT}/src/cohort_dynamic_prompts.py" \
  --profile "${PROFILE}" \
  --local-root "${VARIANT_ROOT}" \
  --output "${OUT_DIR}/variant_dynamic.toml" >/dev/null

echo "[3/4] Updating existing prompts against the variant..."
update_cmd=(
  python3 "${REPO_ROOT}/src/cohort_prompt_schema_updater.py"
  --base-prompts "${BASE_PROMPTS}"
  --old-schema-csv "${OUT_DIR}/base_schema.csv"
  --new-schema-csv "${OUT_DIR}/variant_schema.csv"
  --old-local-root "${BASE_ROOT}"
  --new-local-root "${VARIANT_ROOT}"
  --output "${OUT_DIR}/updated_from_existing.toml"
  --report-json "${OUT_DIR}/prompt_update.report.json"
  --comparison-json "${OUT_DIR}/prompt_update.compare.json"
  --comparison-md "${OUT_DIR}/prompt_update.compare.md"
)

if [[ "${WITH_LLM}" == "1" ]]; then
  update_cmd+=(
    --rewrite-changed-with-llm
    --llm-config "${LLM_CONFIG}"
  )
  if [[ -n "${LLM_MODEL}" ]]; then
    update_cmd+=(--llm-model "${LLM_MODEL}")
  fi
  update_cmd+=(--llm-report-json "${OUT_DIR}/prompt_update.llm_report.json")
fi

"${update_cmd[@]}" >/dev/null

echo "[4/4] Done."
echo "  Base schema: ${OUT_DIR}/base_schema.csv"
echo "  Variant schema: ${OUT_DIR}/variant_schema.csv"
echo "  Base dynamic prompt: ${OUT_DIR}/base_dynamic.toml"
echo "  Variant dynamic prompt: ${OUT_DIR}/variant_dynamic.toml"
echo "  Updated prompt from existing TOML: ${OUT_DIR}/updated_from_existing.toml"
echo "  Prompt comparison: ${OUT_DIR}/prompt_update.compare.md"
echo "  Prompt report: ${OUT_DIR}/prompt_update.report.json"
if [[ "${WITH_LLM}" == "1" ]]; then
  echo "  LLM report: ${OUT_DIR}/prompt_update.llm_report.json"
fi
