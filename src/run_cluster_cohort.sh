#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server"
MODEL_PATH="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/gemma-4-31B-it-Q4_K_M.gguf"

MODEL_GPU0="$MODEL_PATH"
MODEL_GPU1="$MODEL_PATH"

DEFAULT_PORT_LB=18000
DEFAULT_PORT_GPU0=18080
DEFAULT_PORT_GPU1=18081
DEFAULT_PORT_OCR=18090

PORT_LB="${PORT_LB:-$DEFAULT_PORT_LB}"
PORT_GPU0="${PORT_GPU0:-$DEFAULT_PORT_GPU0}"
PORT_GPU1="${PORT_GPU1:-$DEFAULT_PORT_GPU1}"
PORT_OCR="${PORT_OCR:-$DEFAULT_PORT_OCR}"
AUTO_PORTS="${AUTO_PORTS:-1}"

CTX="${CTX:-20000}"
SLOTS="${SLOTS:-1}"
NGL="${NGL:-999}"
LLM_DISABLE_THINKING="${LLM_DISABLE_THINKING:-0}"
LLM_REASONING_BUDGET="${LLM_REASONING_BUDGET:-0}"

# Optional runtime overrides for [llm] config values (applied to config.runtime.toml).
LLM_CHUNKING_ENABLED="${LLM_CHUNKING_ENABLED:-}"
LLM_LONG_TEXT_THRESHOLD_CHARS="${LLM_LONG_TEXT_THRESHOLD_CHARS:-}"
LLM_CHUNK_SIZE_CHARS="${LLM_CHUNK_SIZE_CHARS:-}"
LLM_CHUNK_OVERLAP_CHARS="${LLM_CHUNK_OVERLAP_CHARS:-}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}}"
RUN_DIR="${RUN_DIR:-${PWD}/logs/runs/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"
STATUS_LOG="${RUN_DIR}/status.jsonl"

# Sync destination:
# - If LOCAL_RSYNC_HOST is empty, destination is interpreted on the cluster node.
# - If LOCAL_RSYNC_HOST is set, destination is interpreted on that remote host.
LOCAL_RSYNC_DEST="/Users/p.jansma/Documents/cluster_data/"
LOCAL_RSYNC_HOST=""
SYNC_OUTPUT_ENABLE="${SYNC_OUTPUT_ENABLE:-1}"
SYNC_REQUIRED="${SYNC_REQUIRED:-0}"

# Optional local vision OCR endpoint for PDF text fallback.
# Defaults below are set for this cluster/user setup.
# You can still override them via environment variables if needed.
OCR_VLM_ENABLE="${OCR_VLM_ENABLE:-1}"
OCR_VLM_MODEL_PATH="${OCR_VLM_MODEL_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf}"
OCR_VLM_MMPROJ_PATH="${OCR_VLM_MMPROJ_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf}"
OCR_VLM_ALIAS="${OCR_VLM_ALIAS:-glm-ocr}"
if [[ "${OCR_VLM_PORT+x}" == "x" ]]; then
  OCR_VLM_PORT_ENV_SET=1
else
  OCR_VLM_PORT_ENV_SET=0
fi
OCR_VLM_PORT="${OCR_VLM_PORT:-$PORT_OCR}"
OCR_VLM_CTX="${OCR_VLM_CTX:-8192}"
OCR_VLM_NGL="${OCR_VLM_NGL:-999}"
OCR_VLM_GPU="${OCR_VLM_GPU:-0,1}"
OCR_VLM_USE_TENSOR_SPLIT="${OCR_VLM_USE_TENSOR_SPLIT:-1}"
OCR_VLM_TENSOR_SPLIT="${OCR_VLM_TENSOR_SPLIT:-}"
OCR_VLM_SPLIT_MODE="${OCR_VLM_SPLIT_MODE:-layer}"
OCR_VLM_MAIN_GPU="${OCR_VLM_MAIN_GPU:-0}"
OCR_VLM_LLAMA_BIN="${OCR_VLM_LLAMA_BIN:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp-glmtest/build/bin/llama-server}"
OCR_VLM_PREFETCH_MODE="${OCR_VLM_PREFETCH_MODE:-1}"
OCR_PREFETCH_DIR="${OCR_PREFETCH_DIR:-${RUN_DIR}/ocr_prefetch}"
# Strip trailing references/bibliography block from extracted paper text.
# 1 = enabled (recommended for lower context usage), 0 = disabled.
STRIP_REFERENCES="${STRIP_REFERENCES:-1}"

# Auto-fetch latest ontology/model CSV files from molgenis-emx2.
AUTO_FETCH_EMX2_ONTOLOGIES="${AUTO_FETCH_EMX2_ONTOLOGIES:-1}"
MOLGENIS_EMX2_REPO="${MOLGENIS_EMX2_REPO:-molgenis/molgenis-emx2}"
MOLGENIS_EMX2_REF="${MOLGENIS_EMX2_REF:-main}"
EMX2_CACHE_DIR="${EMX2_CACHE_DIR:-${RUN_DIR}/emx2_cache}"
EMX2_REPO_ROOT="${EMX2_REPO_ROOT:-${EMX2_CACHE_DIR}/repo}"
COHORT_DYNAMIC_EMX2_RUNTIME="${COHORT_DYNAMIC_EMX2_RUNTIME:-1}"
COHORT_DYNAMIC_PROMPTS="${COHORT_DYNAMIC_PROMPTS:-0}"
COHORT_PROMPT_SCHEMA_SYNC="${COHORT_PROMPT_SCHEMA_SYNC:-1}"
COHORT_PROMPT_SCHEMA_SYNC_LLM="${COHORT_PROMPT_SCHEMA_SYNC_LLM:-1}"
COHORT_PROMPT_SCHEMA_BASE_CSV="${COHORT_PROMPT_SCHEMA_BASE_CSV:-${PWD}/schemas/molgenis_UMCGCohortsStaging.csv}"
COHORT_PROMPT_SCHEMA_STATE_DIR="${COHORT_PROMPT_SCHEMA_STATE_DIR:-${PWD}/tmp/cohort_prompt_schema_sync_state}"
COHORT_PROMPT_SCHEMA_HISTORY_DIR="${COHORT_PROMPT_SCHEMA_HISTORY_DIR:-${PWD}/prompts/history}"
COHORT_PROMPT_SCHEMA_PUSH_GIT="${COHORT_PROMPT_SCHEMA_PUSH_GIT:-0}"
COHORT_PROMPT_SCHEMA_PUSH_DIR="${COHORT_PROMPT_SCHEMA_PUSH_DIR:-prompts/history}"
COHORT_PROMPT_SCHEMA_PUSH_REMOTE="${COHORT_PROMPT_SCHEMA_PUSH_REMOTE:-origin}"
COHORT_PROMPT_SCHEMA_PUSH_BRANCH="${COHORT_PROMPT_SCHEMA_PUSH_BRANCH:-}"

# ------------------------------------------------------------------------------
# CLI passthrough:
# - run without args => defaults to main_cohort.py -p all -o final_result_cohort.xlsx
# - run with args    => forwards args to main_cohort.py
# - extra script-only flags:
#     --ocr        force OCR prefetch for all selected PDFs
#     --ocr-force-use  force OCR prefetch and prefer OCR text over pypdf text
#     --ocr-dump   write pypdf/ocr/diff/summary text files per paper
# Example:
#   bash src/run_cluster_cohort.sh
#   bash src/run_cluster_cohort.sh -p A -o out.xlsx --pdfs data/concrete.pdf data/oncolifes.pdf
#   bash src/run_cluster_cohort.sh --ocr-force-use --ocr-dump -p A --pdfs data/concrete.pdf -o out.xlsx
# ------------------------------------------------------------------------------
DEFAULT_ARGS=(-p all -o final_result_cohort.xlsx)
if [[ $# -gt 0 ]]; then
  INPUT_ARGS=("$@")
else
  INPUT_ARGS=("${DEFAULT_ARGS[@]}")
fi

OCR_FORCE_ALL="${OCR_FORCE_ALL:-0}"
OCR_FORCE_USE_PREFETCH="${OCR_FORCE_USE_PREFETCH:-0}"
OCR_DUMP_COMPARE=0
OCR_DUMP_DIR="${OCR_DUMP_DIR:-}"
RUN_ARGS=()
for ((i=0; i<${#INPUT_ARGS[@]}; i++)); do
  arg="${INPUT_ARGS[$i]}"
  case "$arg" in
    --ocr)
      OCR_FORCE_ALL=1
      ;;
    --ocr-force-use)
      OCR_FORCE_ALL=1
      OCR_FORCE_USE_PREFETCH=1
      ;;
    --ocr-dump)
      OCR_DUMP_COMPARE=1
      ;;
    --ocr-dump-dir)
      OCR_DUMP_COMPARE=1
      if (( i + 1 >= ${#INPUT_ARGS[@]} )); then
        echo "❌ ERROR: --ocr-dump-dir vereist een pad."
        exit 1
      fi
      i=$((i + 1))
      OCR_DUMP_DIR="${INPUT_ARGS[$i]}"
      ;;
    *)
      RUN_ARGS+=("$arg")
      ;;
  esac
done

OUTPUT_FILE="final_result_cohort.xlsx"
for ((i=0; i<${#RUN_ARGS[@]}; i++)); do
  case "${RUN_ARGS[$i]}" in
    -o|--output)
      OUTPUT_FILE="${RUN_ARGS[$((i+1))]:-final_result_cohort.xlsx}"
      ;;
  esac
done

status_event() {
  local state="$1"
  local message="${2:-}"
  local now
  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$STATUS_LOG" "$now" "$state" "$message" "$RUN_ID" "$OUTPUT_FILE" <<'PY'
import json
import os
import sys

status_log, ts, state, msg, run_id, output_file = sys.argv[1:]
os.makedirs(os.path.dirname(status_log), exist_ok=True)
event = {
    "timestamp_utc": ts,
    "run_id": run_id,
    "state": state,
    "message": msg,
    "output_file": output_file,
}
with open(status_log, "a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=True) + "\n")
PY
}

flag_enabled() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_prompt_history_baseline() {
  local history_root="$1"
  local baseline_source="$2"

  if [[ -z "$baseline_source" || ! -f "$baseline_source" ]]; then
    return 0
  fi

  local baseline_dir="${history_root}/baseline"
  local baseline_file="${baseline_dir}/prompts_cohort.toml"
  mkdir -p "$baseline_dir"

  if [[ ! -f "$baseline_file" ]]; then
    cp -f "$baseline_source" "$baseline_file"
  fi
}

archive_prompt_schema_change() {
  local history_root="$1"
  local before_after_md="$2"

  if [[ -z "$before_after_md" || ! -f "$before_after_md" ]]; then
    return 1
  fi

  local year_dir
  local date_dir
  local stamp
  year_dir="$(date +"%Y")"
  date_dir="$(date +"%Y-%m-%d")"
  stamp="$(date +"%Y%m%dT%H%M%S")"

  local out_dir="${history_root}/${year_dir}/${date_dir}"
  local out_file="${out_dir}/${stamp}_run${RUN_ID}.prompt_change.md"
  mkdir -p "$out_dir"
  cp -f "$before_after_md" "$out_file"
  printf '%s\n' "$out_file"
}

write_prompt_unified_diff() {
  local old_file="$1"
  local new_file="$2"
  local out_file="$3"
  local old_label="${4:-old_prompt}"
  local new_label="${5:-new_prompt}"

  python3 - "$old_file" "$new_file" "$out_file" "$old_label" "$new_label" <<'PY'
import difflib
import sys
from pathlib import Path

old_path = Path(sys.argv[1]).resolve()
new_path = Path(sys.argv[2]).resolve()
out_path = Path(sys.argv[3]).resolve()
old_label = sys.argv[4]
new_label = sys.argv[5]

old_lines = old_path.read_text(encoding="utf-8").splitlines(keepends=True)
new_lines = new_path.read_text(encoding="utf-8").splitlines(keepends=True)
diff = difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("".join(diff), encoding="utf-8")
PY
}

write_tree_manifest() {
  local root_dir="$1"
  local out_file="$2"

  python3 - "$root_dir" "$out_file" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

files = []
if root.exists():
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": rel, "sha256": digest})

payload = {"root": str(root), "files": files}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

refresh_tree_snapshot() {
  local src_root="$1"
  local dest_root="$2"

  python3 - "$src_root" "$dest_root" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)
if src.exists():
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dest / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
PY
}

push_prompt_schema_diff_to_git() {
  local diff_file="$1"
  local changed_tasks="${2:-0}"

  if ! flag_enabled "$COHORT_PROMPT_SCHEMA_PUSH_GIT"; then
    return 0
  fi
  if [[ ! -f "$diff_file" ]]; then
    echo "⚠️  Git push overgeslagen: prompt diff ontbreekt: $diff_file"
    status_event "warning" "git push skipped; prompt diff missing"
    return 0
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "⚠️  Git push overgeslagen: git niet beschikbaar."
    status_event "warning" "git push skipped; git not available"
    return 0
  fi

  local repo_root
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -z "$repo_root" ]]; then
    echo "⚠️  Git push overgeslagen: geen git repository gevonden."
    status_event "warning" "git push skipped; not inside git repository"
    return 0
  fi

  local push_dir_abs="${repo_root}/${COHORT_PROMPT_SCHEMA_PUSH_DIR}"
  mkdir -p "$push_dir_abs"
  local diff_file_abs
  diff_file_abs="$(cd "$(dirname "$diff_file")" && pwd -P)/$(basename "$diff_file")"
  local push_file_abs="$diff_file_abs"
  local push_rel=""
  if [[ "$diff_file_abs" == "${repo_root}/"* ]]; then
    push_rel="${diff_file_abs#${repo_root}/}"
  else
    local base_name
    base_name="$(basename "$diff_file")"
    push_file_abs="${push_dir_abs}/${base_name}"
    cp -f "$diff_file" "$push_file_abs"
    push_rel="${COHORT_PROMPT_SCHEMA_PUSH_DIR}/${base_name}"
  fi

  local baseline_abs="${repo_root}/${COHORT_PROMPT_SCHEMA_PUSH_DIR}/baseline/prompts_cohort.toml"
  local baseline_rel=""
  if [[ -f "$baseline_abs" ]]; then
    baseline_rel="${baseline_abs#${repo_root}/}"
  fi
  local commit_msg="Archive prompt change $(basename "$push_file_abs") (${changed_tasks} changed tasks)"

  (
    cd "$repo_root"
    if [[ -n "$baseline_rel" ]]; then
      git add -- "$baseline_rel"
    fi
    git add -- "$push_rel"
    if [[ -n "$baseline_rel" ]]; then
      if git diff --cached --quiet -- "$push_rel" "$baseline_rel"; then
        echo "  Git push: geen nieuwe wijzigingen voor ${push_rel}"
        exit 0
      fi
      git commit -m "$commit_msg" -- "$push_rel" "$baseline_rel" >/dev/null
    else
      if git diff --cached --quiet -- "$push_rel"; then
        echo "  Git push: geen nieuwe wijzigingen voor ${push_rel}"
        exit 0
      fi
      git commit -m "$commit_msg" -- "$push_rel" >/dev/null
    fi
    if [[ -n "$COHORT_PROMPT_SCHEMA_PUSH_BRANCH" ]]; then
      git push "$COHORT_PROMPT_SCHEMA_PUSH_REMOTE" "HEAD:${COHORT_PROMPT_SCHEMA_PUSH_BRANCH}" >/dev/null
    else
      git push "$COHORT_PROMPT_SCHEMA_PUSH_REMOTE" HEAD >/dev/null
    fi
  )
  echo "  Prompt wijziging gepusht naar git: ${push_rel}"
  status_event "prompt_schema_sync_git_pushed" "prompt change pushed to git at ${push_rel}"
}

fetch_emx2_csv() {
  local rel_path="$1"
  local target_var="${2:-}"
  local current_val=""
  if [[ -n "$target_var" ]]; then
    current_val="${!target_var:-}"
  fi

  if [[ -n "$target_var" && -n "$current_val" && -f "$current_val" ]]; then
    return 0
  fi
  if [[ "$AUTO_FETCH_EMX2_ONTOLOGIES" != "1" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p "$EMX2_CACHE_DIR"
  local safe_name
  safe_name="$(echo "$rel_path" | tr '/ ' '__')"
  local out_file="${EMX2_CACHE_DIR}/${safe_name}"
  local repo_file="${EMX2_REPO_ROOT}/${rel_path}"
  local tmp_file="${out_file}.tmp"
  local refs=("$MOLGENIS_EMX2_REF" "main" "master")
  local seen="|"
  local ref
  for ref in "${refs[@]}"; do
    [[ -z "$ref" ]] && continue
    if [[ "$seen" == *"|${ref}|"* ]]; then
      continue
    fi
    seen="${seen}${ref}|"
    local url_path="${rel_path// /%20}"
    local url="https://raw.githubusercontent.com/${MOLGENIS_EMX2_REPO}/${ref}/${url_path}"
    if curl -fsSL "$url" -o "$tmp_file" 2>/dev/null; then
      mkdir -p "$(dirname "$repo_file")"
      cp "$tmp_file" "$out_file"
      mv "$tmp_file" "$repo_file"
      if [[ -n "$target_var" ]]; then
        export "$target_var=$repo_file"
        echo "  ${target_var}=${repo_file} (fetched ${ref}:${rel_path})"
        status_event "ontology_fetched" "${target_var} fetched from ${ref}:${rel_path}"
      else
        status_event "model_fetched" "${rel_path} fetched from ${ref}:${rel_path}"
      fi
      return 0
    fi
  done

  rm -f "$tmp_file" 2>/dev/null || true
  return 0
}

sync_emx2_shared_model_dir() {
  if [[ "$AUTO_FETCH_EMX2_ONTOLOGIES" != "1" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  local refs=("$MOLGENIS_EMX2_REF" "main" "master")
  local seen="|"
  local ref=""
  local tmp_json="${EMX2_CACHE_DIR}/shared_models_index.json.tmp"
  mkdir -p "$EMX2_CACHE_DIR"

  for ref in "${refs[@]}"; do
    [[ -z "$ref" ]] && continue
    if [[ "$seen" == *"|${ref}|"* ]]; then
      continue
    fi
    seen="${seen}${ref}|"
    local api_url="https://api.github.com/repos/${MOLGENIS_EMX2_REPO}/contents/data/_models/shared?ref=${ref}"
    if ! curl -fsSL "$api_url" -o "$tmp_json" 2>/dev/null; then
      continue
    fi
    python3 - "$tmp_json" <<'PY' | while IFS= read -r rel_path; do
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(payload, list):
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip() != "file":
            continue
        path = str(item.get("path") or "").strip()
        if path.endswith(".csv"):
            print(path)
PY
      [[ -z "$rel_path" ]] && continue
      fetch_emx2_csv "$rel_path"
    done
    rm -f "$tmp_json" 2>/dev/null || true
    status_event "shared_model_sync" "shared model directory synced from ${MOLGENIS_EMX2_REPO}@${ref}"
    return 0
  done

  rm -f "$tmp_json" 2>/dev/null || true
  return 0
}

if [[ "$AUTO_PORTS" == "1" ]]; then
  read -r PORT_LB PORT_GPU0 PORT_GPU1 PORT_OCR < <(
    python3 - <<'PY'
import socket

socks = []
ports = []
for _ in range(4):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    socks.append(s)
    ports.append(s.getsockname()[1])
for s in socks:
    s.close()
print(*ports)
PY
  )
  if [[ "$OCR_VLM_PORT_ENV_SET" != "1" ]]; then
    OCR_VLM_PORT="$PORT_OCR"
  fi
fi

echo "[Run] RUN_ID=$RUN_ID"
echo "[Run] RUN_DIR=$RUN_DIR"
echo "[Run] Ports LB/GPU0/GPU1/OCR: $PORT_LB / $PORT_GPU0 / $PORT_GPU1 / $OCR_VLM_PORT"
echo "[Run] STRIP_REFERENCES=$STRIP_REFERENCES"
status_event "initializing" "run bootstrap started"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load GCCcore/11.3.0 || true
  module load CUDA/12.2.0 || true
  module load Python/3.10.4-GCCcore-11.3.0 || true
fi

if [[ -f ".venv/bin/activate" ]]; then
  echo "[Setup] Activating Virtual Environment (.venv)..."
  source .venv/bin/activate
else
  echo "❌ ERROR: Geen .venv gevonden!"
  echo "   Maak er een aan met:"
  echo "   module load Python/3.10.4-GCCcore-11.3.0"
  echo "   python3 -m venv .venv"
  echo "   source .venv/bin/activate"
  echo "   pip3 install -e . --no-build-isolation"
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "❌ ERROR: llama-server niet gevonden op $LLAMA_BIN"
  exit 1
fi

LLAMA_HELP="$("$LLAMA_BIN" --help 2>&1 || true)"
llama_supports_flag() {
  local flag="$1"
  grep -Fq -- "$flag" <<<"$LLAMA_HELP"
}
LLM_SERVER_EXTRA_ARGS=()
if flag_enabled "$LLM_DISABLE_THINKING"; then
  if llama_supports_flag "--reasoning-budget"; then
    LLM_SERVER_EXTRA_ARGS+=(--reasoning-budget "$LLM_REASONING_BUDGET")
    echo "[LLM] Thinking disabled via --reasoning-budget=${LLM_REASONING_BUDGET}"
  elif llama_supports_flag "--chat-template-kwargs"; then
    LLM_SERVER_EXTRA_ARGS+=(--chat-template-kwargs '{"enable_thinking":false}')
    echo "[LLM] Thinking disabled via --chat-template-kwargs enable_thinking=false"
  elif llama_supports_flag "--reasoning-format"; then
    LLM_SERVER_EXTRA_ARGS+=(--reasoning-format none)
    echo "[LLM] Thinking-disable fallback via --reasoning-format none"
  else
    echo "[LLM] Waarschuwing: geen bekende thinking-disable flags gevonden in deze llama-server build."
  fi
fi

if [[ ! -f "config.cohort.toml" ]]; then
  echo "❌ ERROR: config.cohort.toml ontbreekt in ${PWD}"
  echo "   main_cohort.py verwacht standaard config.cohort.toml (of zet PDF_EXTRACT_CONFIG)"
  exit 1
fi

PROMPTS_SRC="${PDF_EXTRACT_PROMPTS:-prompts/prompts_cohort.toml}"
DYNAMIC_RUNTIME_FLAG="$(printf '%s' "${COHORT_DYNAMIC_EMX2_RUNTIME:-1}" | tr '[:upper:]' '[:lower:]')"
DYNAMIC_PROMPTS_FLAG="$(printf '%s' "${COHORT_DYNAMIC_PROMPTS:-1}" | tr '[:upper:]' '[:lower:]')"
PROMPTS_OPTIONAL=0
if flag_enabled "$DYNAMIC_RUNTIME_FLAG"; then
  if flag_enabled "$DYNAMIC_PROMPTS_FLAG"; then
    PROMPTS_OPTIONAL=1
  fi
fi
if [[ ! -f "$PROMPTS_SRC" ]]; then
  if [[ "$PROMPTS_OPTIONAL" == "1" ]]; then
    echo "ℹ️  Geen prompts bestand gevonden; main_cohort.py genereert de task-prompts dynamisch uit EMX2."
    PROMPTS_SRC=""
  else
    echo "❌ ERROR: prompts bestand ontbreekt: $PROMPTS_SRC"
    echo "   Zet PDF_EXTRACT_PROMPTS of plaats prompts/prompts_cohort.toml in ${PWD}"
    exit 1
  fi
fi

echo "[0/4] Runtime config maken met base_url via Load Balancer..."
RUNTIME_CFG="${RUN_DIR}/config.runtime.toml"
cp -f "config.cohort.toml" "$RUNTIME_CFG"
RUNTIME_PROMPTS=""
SCHEMA_SYNC_BASE_PROMPTS=""
SCHEMA_SYNC_ACTIVE=0
if [[ -n "$PROMPTS_SRC" ]]; then
  RUNTIME_PROMPTS="${RUN_DIR}/prompts.runtime.toml"
  cp -f "$PROMPTS_SRC" "$RUNTIME_PROMPTS"
  SCHEMA_SYNC_BASE_PROMPTS="$RUNTIME_PROMPTS"
fi
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"
if [[ -n "$LLM_CHUNKING_ENABLED" ]]; then
  sed -i -E "s|^(chunking_enabled[[:space:]]*=[[:space:]]*).*$|\1${LLM_CHUNKING_ENABLED}|g" "$RUNTIME_CFG"
fi
if [[ -n "$LLM_LONG_TEXT_THRESHOLD_CHARS" ]]; then
  sed -i -E "s|^(long_text_threshold_chars[[:space:]]*=[[:space:]]*).*$|\1${LLM_LONG_TEXT_THRESHOLD_CHARS}|g" "$RUNTIME_CFG"
fi
if [[ -n "$LLM_CHUNK_SIZE_CHARS" ]]; then
  sed -i -E "s|^(chunk_size_chars[[:space:]]*=[[:space:]]*).*$|\1${LLM_CHUNK_SIZE_CHARS}|g" "$RUNTIME_CFG"
fi
if [[ -n "$LLM_CHUNK_OVERLAP_CHARS" ]]; then
  sed -i -E "s|^(chunk_overlap_chars[[:space:]]*=[[:space:]]*).*$|\1${LLM_CHUNK_OVERLAP_CHARS}|g" "$RUNTIME_CFG"
fi
export PDF_EXTRACT_CONFIG="$RUNTIME_CFG"
if [[ -n "$RUNTIME_PROMPTS" ]]; then
  export PDF_EXTRACT_PROMPTS="$RUNTIME_PROMPTS"
else
  unset PDF_EXTRACT_PROMPTS || true
fi
if [[ -n "$SCHEMA_SYNC_BASE_PROMPTS" && -f "$SCHEMA_SYNC_BASE_PROMPTS" ]] && flag_enabled "$COHORT_PROMPT_SCHEMA_SYNC"; then
  SCHEMA_SYNC_ACTIVE=1
  export COHORT_DYNAMIC_PROMPTS=0
fi
export COHORT_DYNAMIC_EMX2_RUNTIME
export COHORT_DYNAMIC_PROMPTS
export COHORT_PROMPT_SCHEMA_SYNC
export COHORT_PROMPT_SCHEMA_SYNC_LLM
export MOLGENIS_EMX2_LOCAL_ROOT="$EMX2_REPO_ROOT"
export STRIP_REFERENCES
export OCR_FORCE_USE_PREFETCH
status_event "runtime_config_ready" "runtime config prepared"

if [[ "$OCR_DUMP_COMPARE" == "1" ]]; then
  if [[ -z "$OCR_DUMP_DIR" ]]; then
    OCR_DUMP_DIR="${LOG_DIR}/text_compare"
  fi
  mkdir -p "$OCR_DUMP_DIR"
  export OCR_COMPARE_DUMP_DIR="$OCR_DUMP_DIR"
  echo "[OCR] Compare dumps enabled: $OCR_COMPARE_DUMP_DIR"
fi

if [[ "$OCR_FORCE_USE_PREFETCH" == "1" ]]; then
  echo "[OCR] OCR_FORCE_USE_PREFETCH=1: prefetched OCR text krijgt voorrang op pypdf."
fi

pids=()
LB_PID=""
OCR_PID=""
WEAK_PDFS_FILE="${LOG_DIR}/ocr_weak_pdfs.txt"
OCR_PREFETCH_FAILED_FILE="${LOG_DIR}/ocr_prefetch_failed.txt"
PDF_TARGETS=()

cleanup() {
  local exit_code=$?
  echo
  echo "[CLEANUP] Stoppen..."
  if [[ -n "${OCR_PID}" ]]; then
    kill "${OCR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${LB_PID}" ]]; then
    kill "${LB_PID}" 2>/dev/null || true
  fi
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
  if [[ "$exit_code" -eq 0 ]]; then
    status_event "completed" "run finished successfully"
  else
    status_event "failed" "run failed with exit_code=${exit_code}"
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  local gpu="$1" model="$2" port="$3" log="$4"
  local model_name
  model_name="$(basename "$model")"
  echo "[START] Model ${model_name} op GPU ${gpu} (Poort ${port})..."
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_BIN" \
    -m "$model" -fa on \
    -ngl "$NGL" -c "$CTX" --parallel "$SLOTS" \
    "${LLM_SERVER_EXTRA_ARGS[@]}" \
    --host 127.0.0.1 --port "$port" >>"$log" 2>&1 &
  local pid=$!
  pids+=("$pid")
}

wait_health_ok() {
  local port="$1"
  echo -n "  ⏳ Wachten tot /health status=ok op poort $port"

  for i in {1..600}; do
    if curl -sS -m 5 "http://127.0.0.1:${port}/health" \
      | tr -d '\n' \
      | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      echo " ✅ READY"
      return 0
    fi
    sleep 2
    echo -n "."
  done

  echo " ❌ TIMEOUT"
  return 1
}

check_ocr_render_deps() {
  if [[ "$OCR_VLM_ENABLE" != "1" ]]; then
    return 0
  fi

  if python3 - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("pypdfium2")
importlib.import_module("PIL")
PY
  then
    echo "[OCR] Renderer deps OK via pypdfium2 + Pillow."
    return 0
  fi

  if command -v pdftoppm >/dev/null 2>&1; then
    echo "[OCR] Renderer deps OK via pdftoppm."
    return 0
  fi

  echo "❌ ERROR: OCR_VLM_ENABLE=1, maar geen PDF page renderer beschikbaar."
  echo "   Installeer in .venv: pip3 install pypdfium2 pillow"
  echo "   Of zorg dat pdftoppm in PATH staat."
  exit 1
}

start_ocr_server_if_enabled() {
  if [[ "$OCR_VLM_ENABLE" != "1" ]]; then
    return 0
  fi

  if [[ ! -x "$OCR_VLM_LLAMA_BIN" ]]; then
    echo "❌ ERROR: OCR llama-server niet gevonden op $OCR_VLM_LLAMA_BIN"
    exit 1
  fi

  if [[ -z "$OCR_VLM_MODEL_PATH" ]]; then
    echo "❌ ERROR: OCR_VLM_ENABLE=1 maar OCR_VLM_MODEL_PATH is leeg."
    echo "   Bijvoorbeeld:"
    echo "   export OCR_VLM_MODEL_PATH=/path/to/vision-model.gguf"
    exit 1
  fi

  if [[ ! -f "$OCR_VLM_MODEL_PATH" ]]; then
    echo "❌ ERROR: OCR model niet gevonden: $OCR_VLM_MODEL_PATH"
    exit 1
  fi

  local llama_help=""
  llama_help="$("$OCR_VLM_LLAMA_BIN" --help 2>&1 || true)"
  supports_flag() {
    local flag="$1"
    grep -Fq -- "$flag" <<<"$llama_help"
  }
  build_tensor_split() {
    local split="$OCR_VLM_TENSOR_SPLIT"
    if [[ -n "$split" ]]; then
      echo "$split"
      return 0
    fi
    local IFS=','
    local gpus=()
    read -r -a gpus <<<"$OCR_VLM_GPU"
    local n="${#gpus[@]}"
    if [[ "$n" -le 1 ]]; then
      echo ""
      return 0
    fi
    local out=""
    for ((i=0; i<n; i++)); do
      if [[ -z "$out" ]]; then
        out="1"
      else
        out="${out},1"
      fi
    done
    echo "$out"
  }

  local cmd=(
    "$OCR_VLM_LLAMA_BIN"
    -m "$OCR_VLM_MODEL_PATH"
    --alias "$OCR_VLM_ALIAS"
    --host 127.0.0.1
    --port "$OCR_VLM_PORT"
    -c "$OCR_VLM_CTX"
    -ngl "$OCR_VLM_NGL"
  )

  if [[ -n "$OCR_VLM_MMPROJ_PATH" ]]; then
    if [[ ! -f "$OCR_VLM_MMPROJ_PATH" ]]; then
      echo "❌ ERROR: OCR mmproj niet gevonden: $OCR_VLM_MMPROJ_PATH"
      exit 1
    fi
    cmd+=(--mmproj "$OCR_VLM_MMPROJ_PATH")
  fi

  if [[ "$OCR_VLM_USE_TENSOR_SPLIT" == "1" && "$OCR_VLM_GPU" == *","* ]]; then
    local split
    split="$(build_tensor_split)"
    if [[ -n "$split" ]]; then
      if supports_flag "--tensor-split"; then
        cmd+=(--tensor-split "$split")
        if supports_flag "--split-mode"; then
          cmd+=(--split-mode "$OCR_VLM_SPLIT_MODE")
        fi
        if supports_flag "--main-gpu"; then
          cmd+=(--main-gpu "$OCR_VLM_MAIN_GPU")
        fi
        echo "[OCR] Multi-GPU tensor split actief: tensor_split=$split split_mode=$OCR_VLM_SPLIT_MODE main_gpu=$OCR_VLM_MAIN_GPU"
      else
        echo "[OCR] Waarschuwing: OCR llama-server ondersteunt --tensor-split niet. OCR kan dan vooral op 1 GPU draaien."
      fi
    fi
  fi

  echo "[OCR] Starten vision OCR server op poort $OCR_VLM_PORT..."
  if [[ -n "$OCR_VLM_GPU" ]]; then
    echo "[OCR] CUDA_VISIBLE_DEVICES=$OCR_VLM_GPU"
    CUDA_VISIBLE_DEVICES="$OCR_VLM_GPU" nohup "${cmd[@]}" >>"$LOG_DIR/ocr.log" 2>&1 &
  else
    echo "[OCR] Geen OCR_VLM_GPU gezet; starten zonder GPU pinning."
    nohup "${cmd[@]}" >>"$LOG_DIR/ocr.log" 2>&1 &
  fi
  OCR_PID=$!
  pids+=("$OCR_PID")

  wait_health_ok "$OCR_VLM_PORT" || exit 1

  export OCR_VLM_BASE_URL="http://127.0.0.1:${OCR_VLM_PORT}/v1"
  export OCR_VLM_MODEL="$OCR_VLM_ALIAS"
  echo "[OCR] OCR_VLM_BASE_URL=$OCR_VLM_BASE_URL"
  echo "[OCR] OCR_VLM_MODEL=$OCR_VLM_MODEL"
}

stop_ocr_server_if_running() {
  if [[ -n "${OCR_PID}" ]]; then
    kill "${OCR_PID}" 2>/dev/null || true
    wait "${OCR_PID}" 2>/dev/null || true
    OCR_PID=""
  fi
}

resolve_pdf_targets() {
  PDF_TARGETS=()
  local n="${#RUN_ARGS[@]}"
  for ((i=0; i<n; i++)); do
    if [[ "${RUN_ARGS[$i]}" == "--pdfs" ]]; then
      for ((j=i+1; j<n; j++)); do
        local arg="${RUN_ARGS[$j]}"
        if [[ "$arg" == -* ]]; then
          break
        fi
        PDF_TARGETS+=("$arg")
      done
    fi
  done

  if [[ ${#PDF_TARGETS[@]} -eq 0 ]]; then
    local cfg_pdf
    cfg_pdf="$(
      PDF_EXTRACT_CONFIG="${PDF_EXTRACT_CONFIG}" python3 - <<'PY'
import os
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

cfg = os.environ.get("PDF_EXTRACT_CONFIG", "config.cohort.toml")
with open(cfg, "rb") as f:
    data = toml.load(f)
path = ((data.get("pdf") or {}).get("path") or "").strip()
print(path)
PY
    )"
    if [[ -n "$cfg_pdf" ]]; then
      PDF_TARGETS+=("$cfg_pdf")
    fi
  fi
}

validate_pdf_targets() {
  if [[ ${#PDF_TARGETS[@]} -eq 0 ]]; then
    echo "❌ ERROR: No PDF targets resolved from --pdfs or config [pdf].path."
    status_event "failed" "no pdf targets resolved"
    exit 1
  fi

  local missing=0
  : > "${RUN_DIR}/pdf_targets.resolved.txt"
  for p in "${PDF_TARGETS[@]}"; do
    if [[ ! -f "$p" ]]; then
      echo "❌ ERROR: PDF target not found: $p"
      missing=1
    else
      printf '%s\n' "$p" >> "${RUN_DIR}/pdf_targets.resolved.txt"
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    status_event "failed" "one or more PDF targets missing"
    exit 1
  fi
}

count_nonempty_lines() {
  local f="$1"
  awk 'NF{c++} END{print c+0}' "$f" 2>/dev/null
}

run_ocr_prefetch_if_enabled() {
  if [[ "$OCR_VLM_ENABLE" != "1" ]]; then
    return 0
  fi
  if [[ "$OCR_VLM_PREFETCH_MODE" != "1" ]]; then
    return 0
  fi

  if [[ ${#PDF_TARGETS[@]} -eq 0 ]]; then
    resolve_pdf_targets
  fi
  if [[ ${#PDF_TARGETS[@]} -eq 0 ]]; then
    echo "[OCR] Geen PDF targets gevonden; skip OCR prefetch."
    OCR_VLM_ENABLE=0
    status_event "ocr_prefetch_skipped" "no PDF targets resolved for OCR prefetch"
    return 0
  fi

  status_event "ocr_prefetch_started" "evaluating OCR prefetch targets"
  printf '%s\n' "${PDF_TARGETS[@]}" > "${LOG_DIR}/pdf_targets.txt"

  check_ocr_render_deps

  if [[ "$OCR_FORCE_ALL" == "1" ]]; then
    echo "[OCR] --ocr actief: force OCR voor alle geselecteerde PDFs."
    printf '%s\n' "${PDF_TARGETS[@]}" > "$WEAK_PDFS_FILE"
  else
  echo "[OCR] Scannen welke PDFs echt OCR nodig hebben..."
  PYTHONPATH=src PDF_TARGET_FILE="${LOG_DIR}/pdf_targets.txt" WEAK_PDFS_FILE="$WEAK_PDFS_FILE" python3 - <<'PY'
from pathlib import Path
import os
import extract_pipeline as ep

target_file = Path(os.environ["PDF_TARGET_FILE"])
weak_file = Path(os.environ["WEAK_PDFS_FILE"])
weak = []
for line in target_file.read_text(encoding="utf-8").splitlines():
    p = line.strip()
    if not p:
        continue
    txt = ep._load_pdf_text_pypdf(p, max_pages=None)
    if ep._needs_text_fallback(txt):
        layout = ep._load_pdf_text_pypdf(p, max_pages=None, layout_mode=True)
        best = ep._pick_better_text(txt, layout, "default", "layout")
        if ep._needs_text_fallback(best):
            weak.append(p)
weak_file.write_text("\n".join(weak), encoding="utf-8")
print(f"weak_pdfs={len(weak)}")
PY
  fi

  local weak_count
  weak_count="$(count_nonempty_lines "$WEAK_PDFS_FILE")"
  if [[ "${weak_count:-0}" -eq 0 ]]; then
    echo "[OCR] Geen zwakke PDFs gevonden; OCR server wordt niet gestart."
    OCR_VLM_ENABLE=0
    status_event "ocr_prefetch_skipped" "no weak PDFs needed OCR prefetch"
    return 0
  fi

  echo "[OCR] Zwakke PDFs: $weak_count. OCR prefetch start..."
  status_event "ocr_prefetch_running" "prefetching OCR text for weak PDFs"
  mkdir -p "$OCR_PREFETCH_DIR"
  start_ocr_server_if_enabled

  PYTHONPATH=src WEAK_PDFS_FILE="$WEAK_PDFS_FILE" OCR_PREFETCH_DIR="$OCR_PREFETCH_DIR" OCR_PREFETCH_FAILED_FILE="$OCR_PREFETCH_FAILED_FILE" python3 - <<'PY'
import os
import re
from pathlib import Path
import extract_pipeline as ep

def safe_stem(p: str) -> str:
    stem = Path(p).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "paper"

weak_file = Path(os.environ["WEAK_PDFS_FILE"])
out_dir = Path(os.environ["OCR_PREFETCH_DIR"])
failed_file = Path(os.environ["OCR_PREFETCH_FAILED_FILE"])
out_dir.mkdir(parents=True, exist_ok=True)
failed = []

for line in weak_file.read_text(encoding="utf-8").splitlines():
    p = line.strip()
    if not p:
        continue
    txt = ep._load_pdf_text_with_vlm_ocr(p, max_pages=None)
    out = out_dir / f"{safe_stem(p)}.ocr.txt"
    out.write_text(txt or "", encoding="utf-8")
    if not (txt or "").strip():
        failed.append(p)
    print(f"prefetched\t{Path(p).name}\t{len(txt)}")
failed_file.write_text("\n".join(failed), encoding="utf-8")
PY

  stop_ocr_server_if_running
  local fail_count
  fail_count="$(count_nonempty_lines "$OCR_PREFETCH_FAILED_FILE")"
  if [[ "${fail_count:-0}" -gt 0 ]]; then
    echo "⚠️  OCR prefetch failed for ${fail_count} weak PDF(s). See $OCR_PREFETCH_FAILED_FILE"
    if [[ "$OCR_FORCE_ALL" == "1" ]]; then
      status_event "failed" "forced OCR prefetch failed for ${fail_count} PDF(s)"
      exit 1
    fi
    status_event "warning" "OCR prefetch failed for ${fail_count} weak PDF(s); continuing with pypdf/layout for those files"
  fi
  unset OCR_VLM_BASE_URL OCR_VLM_MODEL
  export OCR_PREFETCH_DIR
  echo "[OCR] Prefetch klaar. OCR_PREFETCH_DIR=$OCR_PREFETCH_DIR"
  status_event "ocr_prefetch_completed" "OCR prefetch completed"

  # Prevent starting live OCR server concurrently with Qwen servers.
  OCR_VLM_ENABLE=0
}

warmup_chat() {
  local port="$1"
  echo "  🔥 Warmup chat op poort $port..."
  local payload
  payload='{"model":"local","messages":[{"role":"user","content":"Warmup: antwoord met OK."}],"temperature":0.0,"max_tokens":32}'
  curl -sS -m 60 \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "http://127.0.0.1:${port}/v1/chat/completions" >/dev/null 2>&1 || true

  payload='{"model":"local","messages":[{"role":"user","content":"Warmup 2: korte test."}],"temperature":0.0,"max_tokens":32}'
  curl -sS -m 60 \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "http://127.0.0.1:${port}/v1/chat/completions" >/dev/null 2>&1 || true
}

wait_lb_ready() {
  local port="$1"
  echo -n "  ⏳ Wachten tot LB verkeer doorstuurt op poort $port"
  for _ in {1..120}; do
    if curl -sS -m 5 "http://127.0.0.1:${port}/health" \
      | tr -d '\n' \
      | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      echo " ✅ READY"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo " ❌ TIMEOUT"
  return 1
}

echo "[0a/4] Resolving and validating PDF targets..."
resolve_pdf_targets
validate_pdf_targets
status_event "pdf_targets_resolved" "resolved ${#PDF_TARGETS[@]} PDF target(s)"

echo "[0b/4] OCR prefetch (indien nodig) vóór Qwen startup..."
run_ocr_prefetch_if_enabled

echo "[1/4] Starten Servers..."
status_event "qwen_starting" "starting Qwen servers"
start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"

echo "[2/4] Wachten tot modellen echt klaar zijn (health=ok)..."
wait_health_ok "$PORT_GPU0" || exit 1
wait_health_ok "$PORT_GPU1" || exit 1

echo "[2b/4] Warmup (voorkomt 503 Loading model bij eerste zware prompt)..."
warmup_chat "$PORT_GPU0"
warmup_chat "$PORT_GPU1"
status_event "qwen_ready" "Qwen servers healthy and warmed"

echo "[3/4] Starten Load Balancer..."
LB_SCRIPT="${RUN_DIR}/tcp_lb.py"
cat > "$LB_SCRIPT" <<EOF
import socket, threading, select, itertools, time

BIND_ADDR = ('0.0.0.0', $PORT_LB)
BACKENDS = [('127.0.0.1', $PORT_GPU0), ('127.0.0.1', $PORT_GPU1)]

rr = itertools.cycle(BACKENDS)

def choose_backend():
    for _ in range(len(BACKENDS) * 2):
        target = next(rr)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(target)
            s.close()
            return target
        except:
            try: s.close()
            except: pass
            time.sleep(0.05)
    return BACKENDS[0]

def handle_conn(client_sock):
    target = choose_backend()
    server_sock = None
    try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect(target)

        sockets = [client_sock, server_sock]
        while True:
            r, _, _ = select.select(sockets, [], [])
            if client_sock in r:
                data = client_sock.recv(4096)
                if not data:
                    break
                server_sock.sendall(data)
            if server_sock in r:
                data = server_sock.recv(4096)
                if not data:
                    break
                client_sock.sendall(data)
    except:
        pass
    finally:
        try: client_sock.close()
        except: pass
        try:
            if server_sock is not None:
                server_sock.close()
        except: pass

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(BIND_ADDR)
    s.listen(128)
    while True:
        conn, _ = s.accept()
        t = threading.Thread(target=handle_conn, args=(conn,))
        t.daemon = True
        t.start()

if __name__ == '__main__':
    main()
EOF

python3 "$LB_SCRIPT" > "$LOG_DIR/lb.log" 2>&1 &
LB_PID=$!
wait_lb_ready "$PORT_LB" || exit 1
status_event "lb_ready" "load balancer ready"

if [[ "$OCR_VLM_ENABLE" == "1" && "$OCR_VLM_PREFETCH_MODE" != "1" ]]; then
  echo "[3b/4] Starten Vision OCR endpoint (live fallback mode)..."
  start_ocr_server_if_enabled
else
  echo "[3b/4] Vision OCR live start overgeslagen (prefetch mode)."
fi

echo "[4/4] Starten main_cohort.py (PDF extractie → Excel)..."
echo "  PDF_EXTRACT_CONFIG=$PDF_EXTRACT_CONFIG"
echo "  PDF_EXTRACT_PROMPTS=$PDF_EXTRACT_PROMPTS"
echo "  AUTO_FETCH_EMX2_ONTOLOGIES=$AUTO_FETCH_EMX2_ONTOLOGIES"
echo "  EMX2 source=${MOLGENIS_EMX2_REPO}@${MOLGENIS_EMX2_REF}"
status_event "emx2_source_selected" "using EMX2 source ${MOLGENIS_EMX2_REPO}@${MOLGENIS_EMX2_REF}"

# Attempt to fetch latest ontology/model sources from molgenis-emx2.
sync_emx2_shared_model_dir
fetch_emx2_csv "data/_ontologies/Countries.csv" COUNTRY_ONTOLOGY_CSV
fetch_emx2_csv "data/_ontologies/Regions.csv" REGION_ONTOLOGY_CSV
fetch_emx2_csv "data/_ontologies/Resources.csv" REF_RESOURCES_CSV
fetch_emx2_csv "data/_ontologies/Organisations.csv" REF_ORGANISATIONS_CSV
fetch_emx2_csv "data/_models/shared/Subpopulations.csv" REF_SUBPOPULATIONS_CSV

if python3 src/emx2_dynamic_runtime.py required-paths --mode cohort >/dev/null 2>&1; then
  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    dyn_var="EMX2_DYNAMIC_$(echo "$rel_path" | tr -c '[:alnum:]' '_' | tr '[:lower:]' '[:upper:]')"
    fetch_emx2_csv "$rel_path" "$dyn_var"
  done < <(python3 src/emx2_dynamic_runtime.py required-paths --mode cohort)
fi

if python3 src/emx2_dynamic_runtime.py model-paths --mode cohort >/dev/null 2>&1; then
  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    dyn_var="EMX2_MODEL_$(echo "$rel_path" | tr -c '[:alnum:]' '_' | tr '[:lower:]' '[:upper:]')"
    fetch_emx2_csv "$rel_path" "$dyn_var"
  done < <(python3 src/emx2_dynamic_runtime.py model-paths --mode cohort)
fi

if [[ "$SCHEMA_SYNC_ACTIVE" == "1" ]]; then
  if [[ ! -f "$COHORT_PROMPT_SCHEMA_BASE_CSV" ]]; then
    echo "⚠️  Skip prompt schema sync: base schema ontbreekt: $COHORT_PROMPT_SCHEMA_BASE_CSV"
    status_event "warning" "prompt schema sync skipped; base schema missing"
  else
    echo "[4a/4] Schema-check voor prompts..."
    LIVE_SCHEMA_CSV="${RUN_DIR}/emx2_schema.live.csv"
    SCHEMA_SYNC_PROMPTS="${RUN_DIR}/prompts.schema_sync.runtime.toml"
    SCHEMA_SYNC_REPORT_JSON="${RUN_DIR}/prompt_schema_sync.report.json"
    SCHEMA_SYNC_COMPARE_JSON="${RUN_DIR}/prompt_schema_sync.compare.json"
    SCHEMA_SYNC_COMPARE_MD="${RUN_DIR}/prompt_schema_sync.compare.md"
    SCHEMA_SYNC_PROMPT_DIFF="${RUN_DIR}/prompt_schema_sync.prompt.diff"
    SCHEMA_SYNC_BEFORE_AFTER_MD="${RUN_DIR}/prompt_schema_sync.before_after.md"
    SCHEMA_SYNC_LLM_PROMPTS="${RUN_DIR}/prompts.schema_sync.llm.runtime.toml"
    SCHEMA_SYNC_LLM_REPORT_JSON="${RUN_DIR}/prompt_schema_sync.llm_report.json"
    SCHEMA_SYNC_LLM_COMPARE_JSON="${RUN_DIR}/prompt_schema_sync.llm.compare.json"
    SCHEMA_SYNC_LLM_COMPARE_MD="${RUN_DIR}/prompt_schema_sync.llm.compare.md"
    SCHEMA_SYNC_LLM_PROMPT_DIFF="${RUN_DIR}/prompt_schema_sync.llm.prompt.diff"
    mkdir -p "$COHORT_PROMPT_SCHEMA_STATE_DIR"
    SCHEMA_SYNC_STATE_SCHEMA="${COHORT_PROMPT_SCHEMA_STATE_DIR}/live_schema.csv"
    SCHEMA_SYNC_STATE_PROMPTS="${COHORT_PROMPT_SCHEMA_STATE_DIR}/prompts.runtime.toml"
    SCHEMA_SYNC_STATE_SOURCE_ROOT="${COHORT_PROMPT_SCHEMA_STATE_DIR}/live_repo"
    SCHEMA_SYNC_STATE_SOURCE_MANIFEST="${COHORT_PROMPT_SCHEMA_STATE_DIR}/source_manifest.json"
    SCHEMA_SYNC_STATE_REPORT_JSON="${COHORT_PROMPT_SCHEMA_STATE_DIR}/report.json"
    SCHEMA_SYNC_STATE_COMPARE_JSON="${COHORT_PROMPT_SCHEMA_STATE_DIR}/compare.json"
    SCHEMA_SYNC_STATE_COMPARE_MD="${COHORT_PROMPT_SCHEMA_STATE_DIR}/compare.md"
    SCHEMA_SYNC_STATE_PROMPT_DIFF="${COHORT_PROMPT_SCHEMA_STATE_DIR}/prompt.diff"
    SCHEMA_SYNC_STATE_BEFORE_AFTER_MD="${COHORT_PROMPT_SCHEMA_STATE_DIR}/before_after.md"
    SCHEMA_SYNC_STATE_LLM_REPORT_JSON="${COHORT_PROMPT_SCHEMA_STATE_DIR}/llm_report.json"
    SCHEMA_SYNC_STATE_LLM_COMPARE_JSON="${COHORT_PROMPT_SCHEMA_STATE_DIR}/llm_compare.json"
    SCHEMA_SYNC_STATE_LLM_COMPARE_MD="${COHORT_PROMPT_SCHEMA_STATE_DIR}/llm_compare.md"
    SCHEMA_SYNC_STATE_LLM_PROMPT_DIFF="${COHORT_PROMPT_SCHEMA_STATE_DIR}/llm_prompt.diff"
    SCHEMA_SYNC_SOURCE_MANIFEST="${RUN_DIR}/emx2_sources.manifest.json"
    SCHEMA_SYNC_OLD_LOCAL_ROOT_ARGS=()
    if [[ -d "$SCHEMA_SYNC_STATE_SOURCE_ROOT" ]]; then
      SCHEMA_SYNC_OLD_LOCAL_ROOT_ARGS=(--old-local-root "$SCHEMA_SYNC_STATE_SOURCE_ROOT")
    fi
    mkdir -p "$COHORT_PROMPT_SCHEMA_HISTORY_DIR"
    ensure_prompt_history_baseline "$COHORT_PROMPT_SCHEMA_HISTORY_DIR" "$SCHEMA_SYNC_BASE_PROMPTS"
    status_event "prompt_schema_sync_started" "checking EMX2 schema against base prompt"

    if python3 src/emx2_dynamic_runtime.py export-schema-csv \
      --profile UMCGCohortsStaging \
      --local-root "$MOLGENIS_EMX2_LOCAL_ROOT" \
      --output "$LIVE_SCHEMA_CSV" >/dev/null; then

      write_tree_manifest "$EMX2_REPO_ROOT" "$SCHEMA_SYNC_SOURCE_MANIFEST"

      if [[ -f "$SCHEMA_SYNC_STATE_SOURCE_MANIFEST" && -f "$SCHEMA_SYNC_STATE_PROMPTS" ]] && cmp -s "$SCHEMA_SYNC_SOURCE_MANIFEST" "$SCHEMA_SYNC_STATE_SOURCE_MANIFEST"; then
        cp -f "$SCHEMA_SYNC_STATE_PROMPTS" "$SCHEMA_SYNC_PROMPTS"
        if [[ -f "$SCHEMA_SYNC_STATE_REPORT_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_REPORT_JSON" "$SCHEMA_SYNC_REPORT_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_COMPARE_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_COMPARE_JSON" "$SCHEMA_SYNC_COMPARE_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_COMPARE_MD" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_COMPARE_MD" "$SCHEMA_SYNC_COMPARE_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_PROMPT_DIFF" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_PROMPT_DIFF" "$SCHEMA_SYNC_PROMPT_DIFF"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_BEFORE_AFTER_MD" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_BEFORE_AFTER_MD" "$SCHEMA_SYNC_BEFORE_AFTER_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_LLM_REPORT_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_LLM_REPORT_JSON" "$SCHEMA_SYNC_LLM_REPORT_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_JSON" "$SCHEMA_SYNC_LLM_COMPARE_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_MD" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_MD" "$SCHEMA_SYNC_LLM_COMPARE_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_STATE_LLM_PROMPT_DIFF" ]]; then
          cp -f "$SCHEMA_SYNC_STATE_LLM_PROMPT_DIFF" "$SCHEMA_SYNC_LLM_PROMPT_DIFF"
        fi
        export PDF_EXTRACT_PROMPTS="$SCHEMA_SYNC_PROMPTS"
        RUNTIME_PROMPTS="$SCHEMA_SYNC_PROMPTS"
        echo "  Prompt schema sync: geen wijziging in live EMX2 sources; cached prompt gebruikt."
        echo "  Vergelijking: $SCHEMA_SYNC_COMPARE_MD"
        if [[ -f "$SCHEMA_SYNC_LLM_PROMPT_DIFF" ]]; then
          echo "  Prompt diff: $SCHEMA_SYNC_LLM_PROMPT_DIFF"
        elif [[ -f "$SCHEMA_SYNC_PROMPT_DIFF" ]]; then
          echo "  Prompt diff: $SCHEMA_SYNC_PROMPT_DIFF"
        fi
        if [[ -f "$SCHEMA_SYNC_BEFORE_AFTER_MD" ]]; then
          echo "  Prompt before/after: $SCHEMA_SYNC_BEFORE_AFTER_MD"
        fi
        status_event "prompt_schema_sync_cached" "live EMX2 sources unchanged; reused cached prompt sync"
      elif python3 src/cohort_prompt_schema_updater.py \
        --base-prompts "$SCHEMA_SYNC_BASE_PROMPTS" \
        --old-schema-csv "$COHORT_PROMPT_SCHEMA_BASE_CSV" \
        --new-schema-csv "$LIVE_SCHEMA_CSV" \
        "${SCHEMA_SYNC_OLD_LOCAL_ROOT_ARGS[@]}" \
        --new-local-root "$EMX2_REPO_ROOT" \
        --output "$SCHEMA_SYNC_PROMPTS" \
        --report-json "$SCHEMA_SYNC_REPORT_JSON" \
        --comparison-json "$SCHEMA_SYNC_COMPARE_JSON" \
        --comparison-md "$SCHEMA_SYNC_COMPARE_MD" \
        --before-after-md "$SCHEMA_SYNC_BEFORE_AFTER_MD" >/dev/null; then

        SCHEMA_SYNC_CHANGED_TASKS="$(
          python3 - "$SCHEMA_SYNC_REPORT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
tasks = report.get("tasks") or {}
count = 0
for task_report in tasks.values():
    if isinstance(task_report, dict) and task_report.get("changed"):
        count += 1
print(count)
PY
        )"

        write_prompt_unified_diff \
          "$SCHEMA_SYNC_BASE_PROMPTS" \
          "$SCHEMA_SYNC_PROMPTS" \
          "$SCHEMA_SYNC_PROMPT_DIFF" \
          "prompts/prompts_cohort.toml" \
          "prompts.schema_sync.runtime.toml"

        export PDF_EXTRACT_PROMPTS="$SCHEMA_SYNC_PROMPTS"
        RUNTIME_PROMPTS="$SCHEMA_SYNC_PROMPTS"
        echo "  Prompt schema sync: changed_tasks=${SCHEMA_SYNC_CHANGED_TASKS}"
        echo "  PDF_EXTRACT_PROMPTS=$PDF_EXTRACT_PROMPTS"
        echo "  Vergelijking: $SCHEMA_SYNC_COMPARE_MD"
        echo "  Prompt diff: $SCHEMA_SYNC_PROMPT_DIFF"
        echo "  Prompt before/after: $SCHEMA_SYNC_BEFORE_AFTER_MD"
        status_event "prompt_schema_sync_ready" "prompt schema sync ready with ${SCHEMA_SYNC_CHANGED_TASKS} changed task(s)"

        if [[ "${SCHEMA_SYNC_CHANGED_TASKS:-0}" -gt 0 ]] && flag_enabled "$COHORT_PROMPT_SCHEMA_SYNC_LLM"; then
          echo "  Schema gewijzigd; Qwen herschrijft alleen de changed tasks..."
          if python3 src/cohort_prompt_schema_updater.py \
            --base-prompts "$SCHEMA_SYNC_BASE_PROMPTS" \
            --old-schema-csv "$COHORT_PROMPT_SCHEMA_BASE_CSV" \
            --new-schema-csv "$LIVE_SCHEMA_CSV" \
            "${SCHEMA_SYNC_OLD_LOCAL_ROOT_ARGS[@]}" \
            --new-local-root "$EMX2_REPO_ROOT" \
            --output "$SCHEMA_SYNC_LLM_PROMPTS" \
            --report-json "$SCHEMA_SYNC_REPORT_JSON" \
            --rewrite-changed-with-llm \
            --llm-config "$RUNTIME_CFG" \
            --llm-report-json "$SCHEMA_SYNC_LLM_REPORT_JSON" \
            --comparison-json "$SCHEMA_SYNC_LLM_COMPARE_JSON" \
            --comparison-md "$SCHEMA_SYNC_LLM_COMPARE_MD" \
            --before-after-md "$SCHEMA_SYNC_BEFORE_AFTER_MD" >/dev/null; then
            export PDF_EXTRACT_PROMPTS="$SCHEMA_SYNC_LLM_PROMPTS"
            RUNTIME_PROMPTS="$SCHEMA_SYNC_LLM_PROMPTS"
            write_prompt_unified_diff \
              "$SCHEMA_SYNC_BASE_PROMPTS" \
              "$SCHEMA_SYNC_LLM_PROMPTS" \
              "$SCHEMA_SYNC_LLM_PROMPT_DIFF" \
              "prompts/prompts_cohort.toml" \
              "prompts.schema_sync.llm.runtime.toml"
            echo "  LLM prompt sync actief: $PDF_EXTRACT_PROMPTS"
            echo "  LLM vergelijking: $SCHEMA_SYNC_LLM_COMPARE_MD"
            echo "  LLM prompt diff: $SCHEMA_SYNC_LLM_PROMPT_DIFF"
            echo "  Prompt before/after: $SCHEMA_SYNC_BEFORE_AFTER_MD"
            status_event "prompt_schema_sync_llm_done" "Qwen rewrote changed prompt tasks"
          else
            echo "⚠️  Qwen prompt sync mislukte; fallback naar deterministische schema-sync prompt."
            status_event "warning" "Qwen prompt sync failed; using deterministic schema-sync prompt"
          fi
        fi

        refresh_tree_snapshot "$EMX2_REPO_ROOT" "$SCHEMA_SYNC_STATE_SOURCE_ROOT"
        cp -f "$SCHEMA_SYNC_SOURCE_MANIFEST" "$SCHEMA_SYNC_STATE_SOURCE_MANIFEST"
        SCHEMA_SYNC_HISTORY_SAVED_AT=""
        if SCHEMA_SYNC_HISTORY_SAVED_AT="$(
          archive_prompt_schema_change \
            "$COHORT_PROMPT_SCHEMA_HISTORY_DIR" \
            "$SCHEMA_SYNC_BEFORE_AFTER_MD"
        )"; then
          echo "  Prompt wijziging opgeslagen: $SCHEMA_SYNC_HISTORY_SAVED_AT"
          status_event "prompt_schema_sync_archived" "prompt change archived at ${SCHEMA_SYNC_HISTORY_SAVED_AT}"
          if ! push_prompt_schema_diff_to_git "$SCHEMA_SYNC_HISTORY_SAVED_AT" "$SCHEMA_SYNC_CHANGED_TASKS"; then
            echo "⚠️  Git push van prompt wijziging mislukte; lokale wijziging blijft behouden."
            status_event "warning" "git push of prompt change failed; local change retained"
          fi
        else
          echo "⚠️  Prompt wijziging kon niet worden opgeslagen; before/after bestand ontbreekt."
          status_event "warning" "prompt change archive skipped; before/after file missing"
        fi

        cp -f "$LIVE_SCHEMA_CSV" "$SCHEMA_SYNC_STATE_SCHEMA"
        cp -f "$RUNTIME_PROMPTS" "$SCHEMA_SYNC_STATE_PROMPTS"
        cp -f "$SCHEMA_SYNC_REPORT_JSON" "$SCHEMA_SYNC_STATE_REPORT_JSON"
        if [[ -f "$SCHEMA_SYNC_COMPARE_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_COMPARE_JSON" "$SCHEMA_SYNC_STATE_COMPARE_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_COMPARE_MD" ]]; then
          cp -f "$SCHEMA_SYNC_COMPARE_MD" "$SCHEMA_SYNC_STATE_COMPARE_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_PROMPT_DIFF" ]]; then
          cp -f "$SCHEMA_SYNC_PROMPT_DIFF" "$SCHEMA_SYNC_STATE_PROMPT_DIFF"
        else
          rm -f "$SCHEMA_SYNC_STATE_PROMPT_DIFF"
        fi
        if [[ -f "$SCHEMA_SYNC_BEFORE_AFTER_MD" ]]; then
          cp -f "$SCHEMA_SYNC_BEFORE_AFTER_MD" "$SCHEMA_SYNC_STATE_BEFORE_AFTER_MD"
        else
          rm -f "$SCHEMA_SYNC_STATE_BEFORE_AFTER_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_LLM_REPORT_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_LLM_REPORT_JSON" "$SCHEMA_SYNC_STATE_LLM_REPORT_JSON"
        else
          rm -f "$SCHEMA_SYNC_STATE_LLM_REPORT_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_LLM_COMPARE_JSON" ]]; then
          cp -f "$SCHEMA_SYNC_LLM_COMPARE_JSON" "$SCHEMA_SYNC_STATE_LLM_COMPARE_JSON"
        else
          rm -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_JSON"
        fi
        if [[ -f "$SCHEMA_SYNC_LLM_COMPARE_MD" ]]; then
          cp -f "$SCHEMA_SYNC_LLM_COMPARE_MD" "$SCHEMA_SYNC_STATE_LLM_COMPARE_MD"
        else
          rm -f "$SCHEMA_SYNC_STATE_LLM_COMPARE_MD"
        fi
        if [[ -f "$SCHEMA_SYNC_LLM_PROMPT_DIFF" ]]; then
          cp -f "$SCHEMA_SYNC_LLM_PROMPT_DIFF" "$SCHEMA_SYNC_STATE_LLM_PROMPT_DIFF"
        else
          rm -f "$SCHEMA_SYNC_STATE_LLM_PROMPT_DIFF"
        fi
      else
        echo "⚠️  Prompt schema sync mislukte; doorgaan met bestaande runtime prompt."
        status_event "warning" "prompt schema sync failed; using existing runtime prompt"
      fi
    else
      echo "⚠️  Prompt schema sync mislukte; doorgaan met bestaande runtime prompt."
      status_event "warning" "live schema export failed; using existing runtime prompt"
    fi
  fi
fi

if [[ -z "${COUNTRY_ONTOLOGY_CSV:-}" ]]; then
  for c in \
    "${PWD}/Countries.csv" \
    "${PWD}/data/ontologies/Countries.csv" \
    "/Users/p.jansma/Downloads/Countries.csv"
  do
    if [[ -f "$c" ]]; then
      export COUNTRY_ONTOLOGY_CSV="$c"
      break
    fi
  done
fi
if [[ -n "${COUNTRY_ONTOLOGY_CSV:-}" ]]; then
  echo "  COUNTRY_ONTOLOGY_CSV=$COUNTRY_ONTOLOGY_CSV"
  if [[ -z "${COUNTRY_MAPPING_LLM_FALLBACK:-}" ]]; then
    export COUNTRY_MAPPING_LLM_FALLBACK="1"
  fi
  echo "  COUNTRY_MAPPING_LLM_FALLBACK=$COUNTRY_MAPPING_LLM_FALLBACK"
  status_event "countries_mapping_enabled" "country ontology mapping enabled"
else
  echo "  COUNTRY_ONTOLOGY_CSV=(not set; country mapping skipped)"
fi

if [[ -z "${REGION_ONTOLOGY_CSV:-}" ]]; then
  for r in \
    "${PWD}/Regions.csv" \
    "${PWD}/data/ontologies/Regions.csv" \
    "/Users/p.jansma/Downloads/Regions.csv"
  do
    if [[ -f "$r" ]]; then
      export REGION_ONTOLOGY_CSV="$r"
      break
    fi
  done
fi
if [[ -n "${REGION_ONTOLOGY_CSV:-}" ]]; then
  echo "  REGION_ONTOLOGY_CSV=$REGION_ONTOLOGY_CSV"
  if [[ -z "${REGION_MAPPING_LLM_FALLBACK:-}" ]]; then
    export REGION_MAPPING_LLM_FALLBACK="1"
  fi
  echo "  REGION_MAPPING_LLM_FALLBACK=$REGION_MAPPING_LLM_FALLBACK"
  status_event "regions_mapping_enabled" "region ontology mapping enabled"
else
  echo "  REGION_ONTOLOGY_CSV=(not set; region mapping skipped)"
fi

if [[ -n "${REF_RESOURCES_CSV:-}" ]]; then
  echo "  REF_RESOURCES_CSV=$REF_RESOURCES_CSV"
fi
if [[ -n "${REF_ORGANISATIONS_CSV:-}" ]]; then
  echo "  REF_ORGANISATIONS_CSV=$REF_ORGANISATIONS_CSV"
fi
if [[ -n "${REF_SUBPOPULATIONS_CSV:-}" ]]; then
  echo "  REF_SUBPOPULATIONS_CSV=$REF_SUBPOPULATIONS_CSV"
fi

export PIPELINE_ISSUES_FILE="${RUN_DIR}/pipeline_issues.json"
status_event "extract_running" "starting main_cohort.py"
python3 src/main_cohort.py "${RUN_ARGS[@]}"
status_event "extract_done" "main_cohort.py completed"

RSYNC_TARGET="$LOCAL_RSYNC_DEST"
if [[ -n "$LOCAL_RSYNC_HOST" ]]; then
  RSYNC_TARGET="${LOCAL_RSYNC_HOST}:${LOCAL_RSYNC_DEST}"
fi

if [[ "$SYNC_OUTPUT_ENABLE" == "1" ]]; then
  echo "[4b/4] Sync output naar lokaal..."
  sync_failed=0
  if [[ -n "$LOCAL_RSYNC_HOST" ]]; then
    # Ensure destination directory exists on remote receiver before syncing.
    if ! rsync -avhP \
      --rsync-path="mkdir -p \"$LOCAL_RSYNC_DEST\" && rsync" \
      "$OUTPUT_FILE" "$RSYNC_TARGET"; then
      sync_failed=1
    fi
  else
    # Guard: on cluster nodes, macOS paths like /Users/... are not local paths.
    if [[ "$LOCAL_RSYNC_DEST" == /Users/* ]]; then
      echo "⚠️  Skip sync: '$LOCAL_RSYNC_DEST' is een macOS pad en bestaat niet op de cluster node."
      echo "   Haal het bestand op vanaf je Mac met:"
      echo "   rsync -avhP tunnel+nibbler:${PWD}/$OUTPUT_FILE \"$LOCAL_RSYNC_DEST\""
      status_event "sync_skipped" "destination appears local macOS path on cluster node"
    else
      # Ensure destination directory exists for local receiver.
      mkdir -p "$LOCAL_RSYNC_DEST"
      if ! rsync -avhP "$OUTPUT_FILE" "$RSYNC_TARGET"; then
        sync_failed=1
      fi
    fi
  fi
  if [[ "$sync_failed" -eq 1 ]]; then
    status_event "warning" "rsync sync failed"
    if [[ "$SYNC_REQUIRED" == "1" ]]; then
      echo "❌ ERROR: sync failed and SYNC_REQUIRED=1"
      exit 1
    fi
    echo "⚠️  Sync failed, but extraction output is available locally at: ${PWD}/${OUTPUT_FILE}"
  else
    status_event "synced" "output synced via rsync"
  fi
else
  echo "[4b/4] Sync skipped (SYNC_OUTPUT_ENABLE=0)."
  status_event "sync_skipped" "SYNC_OUTPUT_ENABLE=0"
fi

echo "✅ Klaar."
