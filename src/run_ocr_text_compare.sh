#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/run_ocr_text_compare.sh <pdf_path> [out_dir]"
  echo "Example: bash src/run_ocr_text_compare.sh data/concrete.pdf logs/text_compare_single"
  exit 1
fi

PDF_PATH="$1"
OUT_DIR="${2:-${ROOT_DIR}/logs/text_compare_single}"
LOG_DIR="${ROOT_DIR}/logs"
OCR_LOG="${LOG_DIR}/ocr_text_compare.log"

LLAMA_BIN="${OCR_VLM_LLAMA_BIN:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp-glmtest/build/bin/llama-server}"
OCR_MODEL_PATH="${OCR_VLM_MODEL_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf}"
OCR_MMPROJ_PATH="${OCR_VLM_MMPROJ_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf}"
OCR_ALIAS="${OCR_VLM_ALIAS:-glm-ocr}"
OCR_PORT="${OCR_VLM_PORT:-18090}"
OCR_CTX="${OCR_VLM_CTX:-8192}"
OCR_NGL="${OCR_VLM_NGL:-999}"
OCR_GPU="${OCR_VLM_GPU:-0,1}"
OCR_TEST_MAX_PAGES="${OCR_TEST_MAX_PAGES:-0}" # 0 = all pages

mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ ! -f "$PDF_PATH" ]]; then
  echo "❌ ERROR: PDF not found: $PDF_PATH"
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "❌ ERROR: OCR llama-server not found: $LLAMA_BIN"
  exit 1
fi

if [[ ! -f "$OCR_MODEL_PATH" ]]; then
  echo "❌ ERROR: OCR model not found: $OCR_MODEL_PATH"
  exit 1
fi

if [[ -n "$OCR_MMPROJ_PATH" && ! -f "$OCR_MMPROJ_PATH" ]]; then
  echo "❌ ERROR: OCR mmproj not found: $OCR_MMPROJ_PATH"
  exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("pypdfium2")
importlib.import_module("PIL")
PY
then
  echo "❌ ERROR: Missing OCR render dependencies."
  echo "   Run: pip3 install pypdfium2 pillow"
  exit 1
fi

OCR_PID=""
cleanup() {
  if [[ -n "$OCR_PID" ]]; then
    kill "$OCR_PID" 2>/dev/null || true
    wait "$OCR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_health_ok() {
  local port="$1"
  echo -n "[INFO] Waiting for OCR /health=ok on port ${port}"
  for _ in {1..180}; do
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
echo "[INFO] OCR log: $OCR_LOG"
if [[ -n "$OCR_GPU" ]]; then
  CUDA_VISIBLE_DEVICES="$OCR_GPU" nohup "${cmd[@]}" >"$OCR_LOG" 2>&1 &
else
  nohup "${cmd[@]}" >"$OCR_LOG" 2>&1 &
fi
OCR_PID=$!

wait_health_ok "$OCR_PORT"

export OCR_VLM_BASE_URL="http://127.0.0.1:${OCR_PORT}/v1"
export OCR_VLM_MODEL="$OCR_ALIAS"

echo "[INFO] Comparing extraction for: $PDF_PATH"
echo "[INFO] Output dir: $OUT_DIR"

PYTHONPATH=src PDF_PATH="$PDF_PATH" OUT_DIR="$OUT_DIR" OCR_TEST_MAX_PAGES="$OCR_TEST_MAX_PAGES" python3 - <<'PY'
import os
import re
from pathlib import Path
import extract_pipeline as ep

pdf_path = os.environ["PDF_PATH"]
out_dir = Path(os.environ["OUT_DIR"])
max_pages_raw = int(os.environ.get("OCR_TEST_MAX_PAGES", "0") or "0")
max_pages = None if max_pages_raw <= 0 else max_pages_raw

def safe_stem(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "paper"

pypdf_txt = ep._load_pdf_text_pypdf(pdf_path, max_pages=max_pages)
ocr_txt = ep._load_pdf_text_with_vlm_ocr(pdf_path, max_pages=max_pages)

stem = safe_stem(pdf_path)
pypdf_path = out_dir / f"{stem}.pypdf.txt"
ocr_path = out_dir / f"{stem}.ocr.txt"
pypdf_path.write_text(pypdf_txt or "", encoding="utf-8")
ocr_path.write_text(ocr_txt or "", encoding="utf-8")

print("paper\tpypdf_chars\tocr_chars\tdelta_ocr_minus_pypdf")
print(f"{Path(pdf_path).name}\t{len(pypdf_txt)}\t{len(ocr_txt)}\t{len(ocr_txt)-len(pypdf_txt)}")
print(f"pypdf_file\t{pypdf_path}")
print(f"ocr_file\t{ocr_path}")
PY

echo "[INFO] Done."
