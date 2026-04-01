#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BIN="${LLAMA_BIN:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf}"
MODEL_GPU0="${MODEL_GPU0:-$MODEL_PATH}"
MODEL_GPU1="${MODEL_GPU1:-$MODEL_PATH}"

DEFAULT_PORT_LB=18000
DEFAULT_PORT_GPU0=18080
DEFAULT_PORT_GPU1=18081

PORT_LB="${PORT_LB:-$DEFAULT_PORT_LB}"
PORT_GPU0="${PORT_GPU0:-$DEFAULT_PORT_GPU0}"
PORT_GPU1="${PORT_GPU1:-$DEFAULT_PORT_GPU1}"
AUTO_PORTS="${AUTO_PORTS:-1}"

CTX="${CTX:-20000}"
SLOTS="${SLOTS:-1}"
NGL="${NGL:-999}"

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
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

pids=()
LB_PID=""

cleanup() {
  if [[ -n "${LB_PID}" ]]; then
    kill "${LB_PID}" 2>/dev/null || true
  fi
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$AUTO_PORTS" == "1" ]]; then
  read -r PORT_LB PORT_GPU0 PORT_GPU1 < <(
    python3 - <<'PY'
import socket

socks = []
ports = []
for _ in range(3):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    socks.append(s)
    ports.append(s.getsockname()[1])
for s in socks:
    s.close()
print(*ports)
PY
  )
fi

start_server() {
  local gpu="$1" model="$2" port="$3" log="$4"
  echo "[Qwen] Start GPU ${gpu} op poort ${port}..."
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_BIN" \
    -m "$model" -fa on \
    -ngl "$NGL" -c "$CTX" --parallel "$SLOTS" \
    --host 127.0.0.1 --port "$port" >>"$log" 2>&1 &
  local pid=$!
  pids+=("$pid")
}

wait_health_ok() {
  local port="$1"
  echo -n "  ⏳ Wachten op Qwen health op poort $port"
  for _ in {1..600}; do
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

warmup_chat() {
  local port="$1"
  echo "  🔥 Warmup chat op poort $port..."
  local payload
  payload='{"model":"local","messages":[{"role":"user","content":"Warmup: antwoord met OK."}],"temperature":0.0,"max_tokens":32}'
  curl -sS -m 60 -H "Content-Type: application/json" -d "$payload" \
    "http://127.0.0.1:${port}/v1/chat/completions" >/dev/null 2>&1 || true
}

wait_lb_ready() {
  local port="$1"
  echo -n "  ⏳ Wachten op LB op poort $port"
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

start_qwen_stack() {
  if command -v module >/dev/null 2>&1; then
    module purge || true
    module load GCCcore/11.3.0 || true
    module load CUDA/12.2.0 || true
    module load Python/3.10.4-GCCcore-11.3.0 || true
  fi

  if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "[Setup] Activating Virtual Environment (.venv)..."
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
  else
    echo "❌ ERROR: Geen .venv gevonden in ${REPO_ROOT}"
    exit 1
  fi

  if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "❌ ERROR: llama-server niet gevonden op $LLAMA_BIN"
    exit 1
  fi
  if [[ ! -f "$MODEL_GPU0" || ! -f "$MODEL_GPU1" ]]; then
    echo "❌ ERROR: modelbestand niet gevonden."
    exit 1
  fi
  if [[ ! -f "$LLM_CONFIG" ]]; then
    echo "❌ ERROR: llm config ontbreekt: $LLM_CONFIG"
    exit 1
  fi

  echo "[Qwen] Runtime config maken..."
  RUNTIME_CFG="${OUT_DIR}/config.runtime.toml"
  cp -f "$LLM_CONFIG" "$RUNTIME_CFG"
  sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\\1http://127.0.0.1:${PORT_LB}/v1\\2|g" "$RUNTIME_CFG"
  LLM_CONFIG="$RUNTIME_CFG"

  echo "[Qwen] Starten servers..."
  start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
  start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"
  wait_health_ok "$PORT_GPU0" || exit 1
  wait_health_ok "$PORT_GPU1" || exit 1
  warmup_chat "$PORT_GPU0"
  warmup_chat "$PORT_GPU1"

  echo "[Qwen] Starten load balancer..."
  LB_SCRIPT="${OUT_DIR}/tcp_lb.py"
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
    server_sock = None
    target = choose_backend()
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
  python3 "$LB_SCRIPT" > "${LOG_DIR}/lb.log" 2>&1 &
  LB_PID=$!
  wait_lb_ready "$PORT_LB" || exit 1
}

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
  start_qwen_stack
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
