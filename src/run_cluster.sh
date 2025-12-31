#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp/build/bin/llama-server"
MODEL_PATH="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/Qwen2.5-32B-Instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

MODEL_GPU0="$MODEL_PATH"
MODEL_GPU1="$MODEL_PATH"

PORT_LB=18000
PORT_GPU0=18080
PORT_GPU1=18081

CTX=8192
SLOTS=1
NGL=999

LOG_DIR="${PWD}/logs"
mkdir -p "$LOG_DIR"

DEFAULT_CMD=(python3 src/main.py -p all -o final_result.xlsx)
if [[ $# -gt 0 ]]; then RUN_CMD=("$@"); else RUN_CMD=("${DEFAULT_CMD[@]}"); fi

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
  echo "   pip install pandas openpyxl requests tomli pypdf"
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "❌ ERROR: llama-server niet gevonden op $LLAMA_BIN"
  exit 1
fi

if [[ ! -f "config.toml" ]]; then
  echo "❌ ERROR: config.toml ontbreekt in ${PWD}"
  echo "   main.py verwacht standaard config.toml (of zet PDF_EXTRACT_CONFIG)"
  exit 1
fi

echo "[0/4] Runtime config maken met base_url via Load Balancer..."
RUNTIME_CFG="${PWD}/config.runtime.toml"
cp -f "config.toml" "$RUNTIME_CFG"
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"
export PDF_EXTRACT_CONFIG="$RUNTIME_CFG"

pids=()
LB_PID=""

cleanup() {
  echo "[CLEANUP] Stoppen..."
  if [[ -n "${LB_PID}" ]]; then kill "${LB_PID}" 2>/dev/null || true; fi
  if [[ ${#pids[@]} -gt 0 ]]; then kill "${pids[@]}" 2>/dev/null || true; fi
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

# Wacht totdat chat/completions echt werkt (dus niet alleen /health)
wait_chat_ready() {
  local port="$1"
  echo -n "  ⏳ Wachten tot /v1/chat/completions klaar is op poort $port..."

  local payload
  payload='{"model":"local","messages":[{"role":"user","content":"ping"}],"temperature":0.0,"max_tokens":1}'

  for i in {1..240}; do
    if curl -sS -m 5 \
      -H "Content-Type: application/json" \
      -d "$payload" \
      "http://127.0.0.1:${port}/v1/chat/completions" >/dev/null 2>&1; then
      echo " ✅ READY"
      return 0
    fi
    sleep 2
    echo -n "."
  done

  echo " ❌ TIMEOUT"
  return 1
}

echo "[1/4] Starten Servers..."
start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"

echo "[2/4] Wachten tot modellen echt klaar zijn..."
wait_chat_ready "$PORT_GPU0" || exit 1
wait_chat_ready "$PORT_GPU1" || exit 1

echo "[3/4] Starten Load Balancer (na readiness)..."
cat > tcp_lb.py <<EOF
import socket, threading, select, itertools
BIND_ADDR = ('0.0.0.0', $PORT_LB)
BACKENDS = [('127.0.0.1', $PORT_GPU0), ('127.0.0.1', $PORT_GPU1)]
rr = itertools.cycle(BACKENDS)
def handle_conn(client_sock):
    target = next(rr)  # round-robin (stabieler dan random)
    try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect(target)
        sockets = [client_sock, server_sock]
        while True:
            r, _, _ = select.select(sockets, [], [])
            if client_sock in r:
                data = client_sock.recv(4096)
                if not data: break
                server_sock.sendall(data)
            if server_sock in r:
                data = server_sock.recv(4096)
                if not data: break
                client_sock.sendall(data)
    except:
        pass
    finally:
        try: client_sock.close()
        except: pass
        try: server_sock.close()
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

echo "[4/4] Starten main.py (PDF extractie → Excel)..."
echo "  PDF_EXTRACT_CONFIG=$PDF_EXTRACT_CONFIG"
"${RUN_CMD[@]}"

echo "✅ Klaar."
