#!/usr/bin/env bash
set -euo pipefail

# Lightweight OCR fallback test:
# - starts only a vision llama-server
# - picks the smallest PDF in data/ (or uses explicit arg)
# - prints pypdf vs OCR char counts and timings

REPO_DIR="${PWD}"
DATA_DIR="${DATA_DIR:-data}"
TARGET_PDF="${1:-}"

LLAMA_BIN="${LLAMA_BIN:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp-glmtest/build/bin/llama-server}"
OCR_MODEL_PATH="${OCR_MODEL_PATH:-}"
OCR_MMPROJ_PATH="${OCR_MMPROJ_PATH:-}"
OCR_ALIAS="${OCR_ALIAS:-glm-ocr}"
OCR_PORT="${OCR_PORT:-18090}"
OCR_CTX="${OCR_CTX:-16384}"
OCR_NGL="${OCR_NGL:-999}"
OCR_GPU="${OCR_GPU:-0}"
OCR_IMAGE_MAX_SIDE="${OCR_IMAGE_MAX_SIDE:-1024}"
OCR_TIMEOUT_SEC="${OCR_TIMEOUT_SEC:-240}"

LOG_DIR="${REPO_DIR}/logs"
mkdir -p "$LOG_DIR"
OCR_LOG="${LOG_DIR}/ocr_single_compare.log"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load GCCcore/11.3.0 || true
  module load CUDA/12.2.0 || true
  module load Python/3.10.4-GCCcore-11.3.0 || true
fi

if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "❌ ERROR: .venv not found in ${REPO_DIR}"
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "❌ ERROR: llama-server binary not executable: $LLAMA_BIN"
  exit 1
fi

if [[ -z "$OCR_MODEL_PATH" ]]; then
  echo "❌ ERROR: OCR_MODEL_PATH is empty."
  echo "Set for example:"
  echo "export OCR_MODEL_PATH=/groups/.../GLM-4.6V-Flash-Q4_K_M.gguf"
  exit 1
fi
if [[ ! -f "$OCR_MODEL_PATH" ]]; then
  echo "❌ ERROR: OCR model file not found: $OCR_MODEL_PATH"
  exit 1
fi

if [[ -n "$OCR_MMPROJ_PATH" && ! -f "$OCR_MMPROJ_PATH" ]]; then
  echo "❌ ERROR: OCR mmproj file not found: $OCR_MMPROJ_PATH"
  exit 1
fi

if [[ -z "$TARGET_PDF" ]]; then
  TARGET_PDF="$(
    DATA_DIR="$DATA_DIR" python3 - <<'PY'
from pathlib import Path
import os

data_dir = Path(os.environ["DATA_DIR"])
pdfs = sorted(data_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
if not pdfs:
    raise SystemExit("NO_PDFS")
print(str(pdfs[0]))
PY
  )"
fi

if [[ ! -f "$TARGET_PDF" ]]; then
  echo "❌ ERROR: target PDF not found: $TARGET_PDF"
  exit 1
fi

cleanup() {
  if [[ -n "${OCR_PID:-}" ]]; then
    kill "$OCR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cmd=(
  "$LLAMA_BIN"
  -m "$OCR_MODEL_PATH"
  --alias "$OCR_ALIAS"
  --host 127.0.0.1
  --port "$OCR_PORT"
  -c "$OCR_CTX"
  -ngl "$OCR_NGL"
)
if [[ -n "$OCR_MMPROJ_PATH" ]]; then
  cmd+=(--mmproj "$OCR_MMPROJ_PATH")
fi

echo "[INFO] Starting OCR server..."
echo "[INFO] log: $OCR_LOG"
CUDA_VISIBLE_DEVICES="$OCR_GPU" nohup "${cmd[@]}" >"$OCR_LOG" 2>&1 &
OCR_PID=$!

READY=0
for _ in {1..180}; do
  if curl -sS -m 5 "http://127.0.0.1:${OCR_PORT}/health" \
    | tr -d '\n' \
    | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "❌ ERROR: OCR server did not become ready."
  echo "--- last 120 lines of ${OCR_LOG} ---"
  tail -n 120 "$OCR_LOG" || true
  exit 1
fi

export OCR_VLM_BASE_URL="http://127.0.0.1:${OCR_PORT}/v1"
export OCR_VLM_MODEL="$OCR_ALIAS"
export OCR_VLM_TIMEOUT_SEC="$OCR_TIMEOUT_SEC"
export OCR_VLM_MAX_PAGES=0
export OCR_VLM_IMAGE_MAX_SIDE="$OCR_IMAGE_MAX_SIDE"

echo "[INFO] Target PDF: $TARGET_PDF"
echo "[INFO] OCR endpoint: $OCR_VLM_BASE_URL model=$OCR_VLM_MODEL"

PYTHONPATH=src TARGET_PDF="$TARGET_PDF" python3 - <<'PY'
import os
import time
from pathlib import Path
import extract_pipeline as ep

pdf = os.environ["TARGET_PDF"]
size_bytes = Path(pdf).stat().st_size

t0 = time.perf_counter()
txt_pypdf = ep._load_pdf_text_pypdf(pdf, max_pages=None)
t1 = time.perf_counter()

t2 = time.perf_counter()
txt_ocr = ep._load_pdf_text_with_vlm_ocr(pdf, max_pages=None)
t3 = time.perf_counter()

pypdf_chars = len(txt_pypdf)
ocr_chars = len(txt_ocr)

print("paper\tbytes\tpypdf_chars\tocr_chars\tdelta_ocr_minus_pypdf\tpypdf_sec\tocr_sec")
print(
    f"{Path(pdf).name}\t{size_bytes}\t{pypdf_chars}\t{ocr_chars}\t"
    f"{ocr_chars - pypdf_chars}\t{(t1 - t0):.2f}\t{(t3 - t2):.2f}"
)
PY
