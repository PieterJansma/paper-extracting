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
  echo "   pip install pandas openpyxl requests tomli"
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

# Vervang base_url binnen de llm-sectie (simpele, praktische aanpak).
# Verwacht een regel zoals: base_url = "http://..."
# Als jouw config anders is opgebouwd, pas deze sed aan.
sed -i -E "s|^(base_url[[:space:]]*=[[:space:]]*\").*(\"[[:space:]]*)$|\1http://127.0.0.1:${PORT_LB}/v1\2|g" "$RUNTIME_CFG"

export PDF_EXTRACT_CONFIG="$RUNTIME_CFG"

echo "[1/4] Starten Load Balancer..."
cat > tcp_lb.py <<EOF
import socket, threading, select, random
BIND_ADDR = ('0.0.0.0', $PORT_LB)
BACKENDS = [('127.0.0.1', $PORT_GPU0), ('127.0.0.1', $PORT_GPU1)]
def handle_conn(client_sock):
    target = random.choice(BACKENDS)
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
        client_sock.close()
        try: server_sock.close()
        except: pass
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(BIND_ADDR)
    s.listen(10)
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

pids=()
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

cleanup() {
  echo "[CLEANUP] Stoppen..."
  kill $LB_PID "${pids[@]}" 2>/dev/null || true
  rm -f tcp_lb.py
  rm -f "$RUNTIME_CFG"
}
trap cleanup EXIT INT TERM

echo "[2/4] Starten Servers..."
start_server 0 "$MODEL_GPU0" "$PORT_GPU0" "$LOG_DIR/gpu0.log"
start_server 1 "$MODEL_GPU1" "$PORT_GPU1" "$LOG_DIR/gpu1.log"

wait_health() {
  local port="$1"
  echo -n "  ⏳ Wachten op poort $port..."
  for i in {1..60}; do
    if curl -s "http://127.0.0.1:${port}/health" >/dev/null; then
      echo " ✅ OK"
      return 0
    fi
    sleep 2
    echo -n "."
  done
  echo " ❌ TIMEOUT"
  return 1
}

echo "[3/4] Wachten tot modellen geladen zijn..."
wait_health "$PORT_GPU0" || exit 1
wait_health "$PORT_GPU1" || exit 1

echo "[4/4] Starten main.py (PDF extractie → Excel)..."
echo "  PDF_EXTRACT_CONFIG=$PDF_EXTRACT_CONFIG"
"${RUN_CMD[@]}"

echo "✅ Klaar."
