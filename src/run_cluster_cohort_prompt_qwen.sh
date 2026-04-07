#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BIN="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server"
MODEL_PATH="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

MODEL_GPU0="$MODEL_PATH"
MODEL_GPU1="$MODEL_PATH"

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

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}}"
RUN_DIR="${RUN_DIR:-${PWD}/logs/runs/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"
STATUS_LOG="${RUN_DIR}/status.jsonl"

AUTO_FETCH_EMX2_ONTOLOGIES="${AUTO_FETCH_EMX2_ONTOLOGIES:-1}"
MOLGENIS_EMX2_REPO="${MOLGENIS_EMX2_REPO:-molgenis/molgenis-emx2}"
MOLGENIS_EMX2_REF="${MOLGENIS_EMX2_REF:-main}"
EMX2_CACHE_DIR="${EMX2_CACHE_DIR:-${RUN_DIR}/emx2_cache}"
EMX2_REPO_ROOT="${EMX2_REPO_ROOT:-${EMX2_CACHE_DIR}/repo}"

OUTPUT_PATH="prompts_full_qwen.runtime.toml"
BASE_PROMPTS="${REPO_ROOT}/prompts/prompts_cohort.toml"
PROFILE="UMCGCohortsStaging"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      OUTPUT_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --base-prompts)
      BASE_PROMPTS="${2:?missing value for $1}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:?missing value for $1}"
      shift 2
      ;;
    *)
      echo "❌ ERROR: onbekende optie: $1"
      exit 1
      ;;
  esac
done

status_event() {
  local state="$1"
  local message="${2:-}"
  local now
  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$STATUS_LOG" "$now" "$state" "$message" "$RUN_ID" "$OUTPUT_PATH" <<'PY'
import json
import os
import sys

status_log, ts, state, msg, run_id, output_path = sys.argv[1:]
os.makedirs(os.path.dirname(status_log), exist_ok=True)
event = {
    "timestamp_utc": ts,
    "run_id": run_id,
    "state": state,
    "message": msg,
    "output_file": output_path,
}
with open(status_log, "a", encoding="utf-8") as f:
    f.write(json.dumps(event, ensure_ascii=True) + "\n")
PY
}

fetch_emx2_csv() {
  local rel_path="$1"
  local target_var="$2"
  local current_val="${!target_var:-}"

  if [[ -n "$current_val" && -f "$current_val" ]]; then
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
      export "$target_var=$repo_file"
      echo "  ${target_var}=${repo_file} (fetched ${ref}:${rel_path})"
      status_event "emx2_fetched" "${target_var} fetched from ${ref}:${rel_path}"
      return 0
    fi
  done

  rm -f "$tmp_file" 2>/dev/null || true
  return 0
}

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

echo "[Run] RUN_ID=$RUN_ID"
echo "[Run] RUN_DIR=$RUN_DIR"
echo "[Run] Ports LB/GPU0/GPU1: $PORT_LB / $PORT_GPU0 / $PORT_GPU1"
status_event "initializing" "prompt qwen bootstrap started"

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
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "❌ ERROR: llama-server niet gevonden op $LLAMA_BIN"
  exit 1
fi

if [[ ! -f "config.final.toml" ]]; then
  echo "❌ ERROR: config.final.toml ontbreekt in ${PWD}"
  exit 1
fi

if [[ ! -f "$BASE_PROMPTS" ]]; then
  echo "❌ ERROR: base prompts bestand ontbreekt: $BASE_PROMPTS"
  exit 1
fi

echo "[0/3] Runtime config maken..."
RUNTIME_CFG="${RUN_DIR}/config.runtime.toml"
cp -f "config.final.toml" "$RUNTIME_CFG"
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"
export MOLGENIS_EMX2_LOCAL_ROOT="$EMX2_REPO_ROOT"
status_event "runtime_config_ready" "runtime config prepared"

pids=()
LB_PID=""

cleanup() {
  local exit_code=$?
  echo
  echo "[CLEANUP] Stoppen..."
  if [[ -n "${LB_PID}" ]]; then
    kill "${LB_PID}" 2>/dev/null || true
  fi
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
  if [[ "$exit_code" -eq 0 ]]; then
    status_event "completed" "prompt qwen run finished successfully"
  else
    status_event "failed" "prompt qwen run failed with exit_code=${exit_code}"
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

echo "[1/3] Starten Servers..."
status_event "qwen_starting" "starting Qwen servers for prompt rewrite"
start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"

echo "[1b/3] Wachten tot modellen klaar zijn..."
wait_health_ok "$PORT_GPU0" || exit 1
wait_health_ok "$PORT_GPU1" || exit 1
warmup_chat "$PORT_GPU0"
warmup_chat "$PORT_GPU1"
status_event "qwen_ready" "Qwen servers healthy and warmed"

echo "[2/3] Starten Load Balancer..."
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

echo "[2b/3] Fetch live EMX2 CSV files..."
echo "  EMX2 source=${MOLGENIS_EMX2_REPO}@${MOLGENIS_EMX2_REF}"
status_event "emx2_source_selected" "using EMX2 source ${MOLGENIS_EMX2_REPO}@${MOLGENIS_EMX2_REF}"
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
status_event "emx2_ready" "live EMX2 files resolved"

echo "[3/3] Volledige prompt genereren en herschrijven met Qwen..."
OUTPUT_ABS="$OUTPUT_PATH"
if [[ "$OUTPUT_ABS" != /* ]]; then
  OUTPUT_ABS="${PWD}/${OUTPUT_PATH}"
fi
COMPARE_JSON="${OUTPUT_ABS%.toml}.compare.json"
COMPARE_MD="${OUTPUT_ABS%.toml}.compare.md"
LLM_REPORT_JSON="${OUTPUT_ABS%.toml}.llm_report.json"
status_event "prompt_rewrite_running" "starting full schema prompt rewrite with Qwen"
PYTHONPATH=src python3 src/cohort_prompt_full_schema_qwen.py \
  --base-prompts "$BASE_PROMPTS" \
  --profile "$PROFILE" \
  --local-root "$EMX2_REPO_ROOT" \
  --llm-config "$RUNTIME_CFG" \
  --output "$OUTPUT_ABS" \
  --comparison-json "$COMPARE_JSON" \
  --comparison-md "$COMPARE_MD" \
  --llm-report-json "$LLM_REPORT_JSON"
status_event "prompt_rewrite_done" "full schema prompt rewrite completed"

echo "✅ Klaar."
echo "  Prompt TOML: $OUTPUT_ABS"
echo "  Vergelijking: $COMPARE_MD"
echo "  LLM report: $LLM_REPORT_JSON"
