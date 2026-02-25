#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server"
MODEL_PATH="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

MODEL_GPU0="$MODEL_PATH"
MODEL_GPU1="$MODEL_PATH"

PORT_LB=18000
PORT_GPU0=18080
PORT_GPU1=18081
PORT_OCR=18090

CTX=20000
SLOTS=1
NGL=999

LOG_DIR="${PWD}/logs"
mkdir -p "$LOG_DIR"

# Sync destination:
# - If LOCAL_RSYNC_HOST is empty, destination is interpreted on the cluster node.
# - If LOCAL_RSYNC_HOST is set, destination is interpreted on that remote host.
LOCAL_RSYNC_DEST="/Users/p.jansma/Documents/cluster_data/"
LOCAL_RSYNC_HOST=""

# Optional local vision OCR endpoint for PDF text fallback.
# Defaults below are set for this cluster/user setup.
# You can still override them via environment variables if needed.
OCR_VLM_ENABLE="${OCR_VLM_ENABLE:-1}"
OCR_VLM_MODEL_PATH="${OCR_VLM_MODEL_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf}"
OCR_VLM_MMPROJ_PATH="${OCR_VLM_MMPROJ_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf}"
OCR_VLM_ALIAS="${OCR_VLM_ALIAS:-glm-ocr}"
OCR_VLM_PORT="${OCR_VLM_PORT:-$PORT_OCR}"
OCR_VLM_CTX="${OCR_VLM_CTX:-8192}"
OCR_VLM_NGL="${OCR_VLM_NGL:-999}"
OCR_VLM_GPU="${OCR_VLM_GPU:-0,1}"
OCR_VLM_LLAMA_BIN="${OCR_VLM_LLAMA_BIN:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp-glmtest/build/bin/llama-server}"
OCR_VLM_PREFETCH_MODE="${OCR_VLM_PREFETCH_MODE:-1}"
OCR_PREFETCH_DIR="${OCR_PREFETCH_DIR:-${PWD}/logs/ocr_prefetch}"

# ------------------------------------------------------------------------------
# CLI passthrough:
# - run without args => defaults to main_final.py -p all -o final_result.xlsx
# - run with args    => forwards args to main_final.py
# Example:
#   bash src/run_cluster_final.sh
#   bash src/run_cluster_final.sh -p A -o out.xlsx --pdfs data/concrete.pdf data/oncolifes.pdf
# ------------------------------------------------------------------------------
DEFAULT_ARGS=(-p all -o final_result.xlsx)
if [[ $# -gt 0 ]]; then
  RUN_ARGS=("$@")
else
  RUN_ARGS=("${DEFAULT_ARGS[@]}")
fi

OUTPUT_FILE="final_result.xlsx"
for ((i=0; i<${#RUN_ARGS[@]}; i++)); do
  case "${RUN_ARGS[$i]}" in
    -o|--output)
      OUTPUT_FILE="${RUN_ARGS[$((i+1))]:-final_result.xlsx}"
      ;;
  esac
done

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
RUNTIME_CFG="${PWD}/config.runtime.toml"
cp -f "config.final.toml" "$RUNTIME_CFG"
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"
export PDF_EXTRACT_CONFIG="$RUNTIME_CFG"

pids=()
LB_PID=""
OCR_PID=""
WEAK_PDFS_FILE="${LOG_DIR}/ocr_weak_pdfs.txt"

cleanup() {
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
  rm -f tcp_lb.py
  rm -f "$RUNTIME_CFG"
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

run_ocr_prefetch_if_enabled() {
  if [[ "$OCR_VLM_ENABLE" != "1" ]]; then
    return 0
  fi
  if [[ "$OCR_VLM_PREFETCH_MODE" != "1" ]]; then
    return 0
  fi

  resolve_pdf_targets
  if [[ ${#PDF_TARGETS[@]} -eq 0 ]]; then
    echo "[OCR] Geen PDF targets gevonden; skip OCR prefetch."
    OCR_VLM_ENABLE=0
    return 0
  fi

  printf '%s\n' "${PDF_TARGETS[@]}" > "${LOG_DIR}/pdf_targets.txt"

  check_ocr_render_deps

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

  local weak_count
  weak_count="$(wc -l < "$WEAK_PDFS_FILE" | tr -d ' ')"
  if [[ "${weak_count:-0}" -eq 0 ]]; then
    echo "[OCR] Geen zwakke PDFs gevonden; OCR server wordt niet gestart."
    OCR_VLM_ENABLE=0
    return 0
  fi

  echo "[OCR] Zwakke PDFs: $weak_count. OCR prefetch start..."
  mkdir -p "$OCR_PREFETCH_DIR"
  start_ocr_server_if_enabled

  PYTHONPATH=src WEAK_PDFS_FILE="$WEAK_PDFS_FILE" OCR_PREFETCH_DIR="$OCR_PREFETCH_DIR" python3 - <<'PY'
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
out_dir.mkdir(parents=True, exist_ok=True)

for line in weak_file.read_text(encoding="utf-8").splitlines():
    p = line.strip()
    if not p:
        continue
    txt = ep._load_pdf_text_with_vlm_ocr(p, max_pages=None)
    out = out_dir / f"{safe_stem(p)}.ocr.txt"
    out.write_text(txt or "", encoding="utf-8")
    print(f"prefetched\t{Path(p).name}\t{len(txt)}")
PY

  stop_ocr_server_if_running
  unset OCR_VLM_BASE_URL OCR_VLM_MODEL
  export OCR_PREFETCH_DIR
  echo "[OCR] Prefetch klaar. OCR_PREFETCH_DIR=$OCR_PREFETCH_DIR"

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

echo "[0b/4] OCR prefetch (indien nodig) vóór Qwen startup..."
run_ocr_prefetch_if_enabled

echo "[1/4] Starten Servers..."
start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"

echo "[2/4] Wachten tot modellen echt klaar zijn (health=ok)..."
wait_health_ok "$PORT_GPU0" || exit 1
wait_health_ok "$PORT_GPU1" || exit 1

echo "[2b/4] Warmup (voorkomt 503 Loading model bij eerste zware prompt)..."
warmup_chat "$PORT_GPU0"
warmup_chat "$PORT_GPU1"

echo "[3/4] Starten Load Balancer..."
cat > tcp_lb.py <<EOF
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

python3 tcp_lb.py > "$LOG_DIR/lb.log" 2>&1 &
LB_PID=$!

echo "[3b/4] Starten Vision OCR endpoint (optioneel)..."
start_ocr_server_if_enabled

echo "[4/4] Starten main_final.py (PDF extractie → Excel)..."
echo "  PDF_EXTRACT_CONFIG=$PDF_EXTRACT_CONFIG"
python3 src/main_final.py "${RUN_ARGS[@]}"

RSYNC_TARGET="$LOCAL_RSYNC_DEST"
if [[ -n "$LOCAL_RSYNC_HOST" ]]; then
  RSYNC_TARGET="${LOCAL_RSYNC_HOST}:${LOCAL_RSYNC_DEST}"
fi

echo "[4b/4] Sync output naar lokaal..."
if [[ -n "$LOCAL_RSYNC_HOST" ]]; then
  # Ensure destination directory exists on remote receiver before syncing.
  rsync -avhP \
    --rsync-path="mkdir -p \"$LOCAL_RSYNC_DEST\" && rsync" \
    "$OUTPUT_FILE" "$RSYNC_TARGET"
else
  # Guard: on cluster nodes, macOS paths like /Users/... are not local paths.
  if [[ "$LOCAL_RSYNC_DEST" == /Users/* ]]; then
    echo "⚠️  Skip sync: '$LOCAL_RSYNC_DEST' is een macOS pad en bestaat niet op de cluster node."
    echo "   Haal het bestand op vanaf je Mac met:"
    echo "   rsync -avhP tunnel+nibbler:${PWD}/$OUTPUT_FILE \"$LOCAL_RSYNC_DEST\""
  else
    # Ensure destination directory exists for local receiver.
    mkdir -p "$LOCAL_RSYNC_DEST"
    rsync -avhP "$OUTPUT_FILE" "$RSYNC_TARGET"
  fi
fi

echo "✅ Klaar."
