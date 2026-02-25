#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server"
MODEL_PATH="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

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

CTX=20000
SLOTS=1
NGL=999

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

# ------------------------------------------------------------------------------
# CLI passthrough:
# - run without args => defaults to main_final.py -p all -o final_result.xlsx
# - run with args    => forwards args to main_final.py
# - extra script-only flags:
#     --ocr        force OCR prefetch for all selected PDFs
#     --ocr-dump   write pypdf/ocr/diff/summary text files per paper
# Example:
#   bash src/run_cluster_final.sh
#   bash src/run_cluster_final.sh -p A -o out.xlsx --pdfs data/concrete.pdf data/oncolifes.pdf
#   bash src/run_cluster_final.sh --ocr --ocr-dump -p A --pdfs data/concrete.pdf -o out.xlsx
# ------------------------------------------------------------------------------
DEFAULT_ARGS=(-p all -o final_result.xlsx)
if [[ $# -gt 0 ]]; then
  INPUT_ARGS=("$@")
else
  INPUT_ARGS=("${DEFAULT_ARGS[@]}")
fi

OCR_FORCE_ALL="${OCR_FORCE_ALL:-0}"
OCR_DUMP_COMPARE=0
OCR_DUMP_DIR="${OCR_DUMP_DIR:-}"
RUN_ARGS=()
for ((i=0; i<${#INPUT_ARGS[@]}; i++)); do
  arg="${INPUT_ARGS[$i]}"
  case "$arg" in
    --ocr)
      OCR_FORCE_ALL=1
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

OUTPUT_FILE="final_result.xlsx"
for ((i=0; i<${#RUN_ARGS[@]}; i++)); do
  case "${RUN_ARGS[$i]}" in
    -o|--output)
      OUTPUT_FILE="${RUN_ARGS[$((i+1))]:-final_result.xlsx}"
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

if [[ ! -f "config.final.toml" ]]; then
  echo "❌ ERROR: config.final.toml ontbreekt in ${PWD}"
  echo "   main_final.py verwacht standaard config.final.toml (of zet PDF_EXTRACT_CONFIG)"
  exit 1
fi

echo "[0/4] Runtime config maken met base_url via Load Balancer..."
RUNTIME_CFG="${RUN_DIR}/config.runtime.toml"
cp -f "config.final.toml" "$RUNTIME_CFG"
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"
export PDF_EXTRACT_CONFIG="$RUNTIME_CFG"
status_event "runtime_config_ready" "runtime config prepared"

if [[ "$OCR_DUMP_COMPARE" == "1" ]]; then
  if [[ -z "$OCR_DUMP_DIR" ]]; then
    OCR_DUMP_DIR="${LOG_DIR}/text_compare"
  fi
  mkdir -p "$OCR_DUMP_DIR"
  export OCR_COMPARE_DUMP_DIR="$OCR_DUMP_DIR"
  echo "[OCR] Compare dumps enabled: $OCR_COMPARE_DUMP_DIR"
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
  echo "[START] Qwen 32B op GPU ${gpu} (Poort ${port})..."
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_BIN" \
    -m "$model" -fa on \
    -ngl "$NGL" -c "$CTX" --parallel "$SLOTS" \
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

cfg = os.environ.get("PDF_EXTRACT_CONFIG", "config.final.toml")
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
  weak_count="$(grep -cve '^[[:space:]]*$' "$WEAK_PDFS_FILE" 2>/dev/null || echo 0)"
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
  fail_count="$(grep -cve '^[[:space:]]*$' "$OCR_PREFETCH_FAILED_FILE" 2>/dev/null || echo 0)"
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

echo "[4/4] Starten main_final.py (PDF extractie → Excel)..."
echo "  PDF_EXTRACT_CONFIG=$PDF_EXTRACT_CONFIG"
export PIPELINE_ISSUES_FILE="${RUN_DIR}/pipeline_issues.json"
status_event "extract_running" "starting main_final.py"
python3 src/main_final.py "${RUN_ARGS[@]}"
status_event "extract_done" "main_final.py completed"

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
