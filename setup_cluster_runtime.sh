#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CURRENT_CLUSTER_USER="${USER:-$(id -un)}"
WORKDIR="${WORKDIR:-/groups/umcg-gcc/tmp02/users/${CURRENT_CLUSTER_USER}}"
SETUP_SUFFIX="${SETUP_SUFFIX:-}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$SCRIPT_DIR/.runtime${SETUP_SUFFIX}}"
REPOS_DIR="${REPOS_DIR:-$RUNTIME_ROOT}"
MODELS_DIR="${MODELS_DIR:-$RUNTIME_ROOT/GGUF}"
LLAMA_DIR="${LLAMA_DIR:-$RUNTIME_ROOT/llama.cpp}"
LLAMA_COMMIT="${LLAMA_COMMIT:-e21cdc11a0461d8b0cbd28cc356d993bf6be7282}"
BUILD_LLAMA="${BUILD_LLAMA:-0}"
BUILD_JOBS="${BUILD_JOBS:-32}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv.cluster${SETUP_SUFFIX}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INSTALL_PROJECT="${PIP_INSTALL_PROJECT:-1}"
PIP_EXTRA_PACKAGES="${PIP_EXTRA_PACKAGES:-pypdfium2 pillow xlsxwriter}"

GEMMA_DIR="${GEMMA_DIR:-$MODELS_DIR}"
GEMMA_NAME="gemma-4-31B-it-Q4_K_M.gguf"
GEMMA_URL="https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf"
GEMMA_SHA256="${GEMMA_SHA256:-4f369f8fe0e1bedc5caee9abb89316887f548f80f3035398a5d222a737e699e6}"

GLMOCR_DIR="${GLMOCR_DIR:-$MODELS_DIR/GLM-OCR}"
GLMOCR_MAIN_NAME="GLM-OCR-Q8_0.gguf"
GLMOCR_MAIN_URL="https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-Q8_0.gguf"
GLMOCR_MAIN_SHA256="${GLMOCR_MAIN_SHA256:-45bc244a6446aff850521dc41f18bc8d7105ad5f0c2c8c28af04e7cc4f4d50b1}"
GLMOCR_MMPROJ_NAME="mmproj-GLM-OCR-Q8_0.gguf"
GLMOCR_MMPROJ_URL="https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/mmproj-GLM-OCR-Q8_0.gguf"
GLMOCR_MMPROJ_SHA256="${GLMOCR_MMPROJ_SHA256:-9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f}"

usage() {
  cat <<'USAGE'
Usage:
  bash setup_cluster_runtime.sh

Environment variables:
  WORKDIR=/groups/.../users/<user>                          Compatibility seed path for existing downloads
  SETUP_SUFFIX=_test                                        Optional isolated suffix for test dirs
  RUNTIME_ROOT=/path/to/repo/.runtime_test                  Optional full runtime root override
  VENV_DIR=/path/to/repo/.venv.cluster_test                 Optional cluster venv override
  BUILD_LLAMA=1                                             Also build llama.cpp after checkout
  BUILD_JOBS=32                                             Parallel build jobs
  LLAMA_COMMIT=e21cdc11a0461d8b0cbd28cc356d993bf6be7282     Exact llama.cpp commit to use
  PYTHON_BIN=python3                                        Python used for the venv
  PIP_EXTRA_PACKAGES="pypdfium2 pillow xlsxwriter"          Extra runtime packages for this repo

What this script does:
  1. Clone ggml-org/llama.cpp into $RUNTIME_ROOT/llama.cpp
  2. Checkout the exact working commit currently used on the cluster
  3. Download gemma-4-31B-it-Q4_K_M.gguf with wget
  4. Verify Gemma against the working sha256 hash
  5. Download GLM-OCR-Q8_0.gguf and mmproj-GLM-OCR-Q8_0.gguf (OCR VLM) with wget
  6. Verify both GLM-OCR files against sha256 hashes
  7. Create/update a dedicated cluster venv for this repo
  8. Install pinned deps from requirements.lock if present (exact snapshot),
     otherwise install latest compatible versions plus OCR/Excel extras
  9. Optionally build llama.cpp if BUILD_LLAMA=1

Important:
  BUILD_LLAMA=1 should be used on a compute node with the required modules loaded.
  Default runtime output is inside this repo: .runtime${SETUP_SUFFIX}
  Default cluster venv is inside this repo: .venv.cluster${SETUP_SUFFIX}
USAGE
}

log() {
  printf '[setup] %s\n' "$1"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1"
    exit 1
  fi
}

copy_if_verified() {
  local source_file="$1"
  local target_file="$2"
  local expected_sha="$3"
  local label="$4"
  if [[ -z "$source_file" || ! -f "$source_file" ]]; then
    return 1
  fi

  local actual
  actual="$(sha256sum "$source_file" | awk '{print $1}')"
  if [[ "$actual" != "$expected_sha" ]]; then
    log "skip unverified source for $label: $source_file"
    return 1
  fi

  mkdir -p "$(dirname "$target_file")"
  if [[ "$source_file" != "$target_file" ]]; then
    log "copy verified $label from $source_file"
    cp -f "$source_file" "$target_file"
  fi
  return 0
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: sha256 mismatch for $label"
    echo "  file:     $file"
    echo "  expected: $expected"
    echo "  actual:   $actual"
    exit 1
  fi
  log "sha256 OK for $label"
}

remove_if_hash_mismatch() {
  local file="$1"
  local expected="$2"
  local label="$3"
  if [[ ! -f "$file" ]]; then
    return 0
  fi

  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    log "remove stale $label with wrong sha256: $file"
    rm -f "$file"
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    log "skip existing file: $out"
    return 0
  fi
  log "download: $url"
  wget -nv --show-progress -O "$out" "$url"
}

seed_gemma_from_existing() {
  local target="$GEMMA_DIR/$GEMMA_NAME"
  local candidates=(
    "${GEMMA_SOURCE_FILE:-}"
    "$SCRIPT_DIR/.runtime/GGUF/$GEMMA_NAME"
    "$WORKDIR/Models/GGUF/$GEMMA_NAME"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if copy_if_verified "$candidate" "$target" "$GEMMA_SHA256" "Gemma GGUF"; then
      return 0
    fi
  done
  return 1
}

seed_glm_ocr_from_existing() {
  local label="$1"                       # "GLM-OCR main" or "GLM-OCR mmproj"
  local filename="$2"
  local expected_sha="$3"
  local source_override="$4"             # may be empty
  local target="$GLMOCR_DIR/$filename"
  local candidates=(
    "$source_override"
    "$SCRIPT_DIR/.runtime/GGUF/GLM-OCR/$filename"
    "$WORKDIR/Models/GGUF/GLM-OCR/$filename"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if copy_if_verified "$candidate" "$target" "$expected_sha" "$label"; then
      return 0
    fi
  done
  return 1
}

clone_llama_cpp() {
  mkdir -p "$REPOS_DIR"
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "cloning llama.cpp into $LLAMA_DIR"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  else
    log "llama.cpp already exists: $LLAMA_DIR"
  fi

  cd "$LLAMA_DIR"
  git fetch --tags
  git checkout "$LLAMA_COMMIT"
  git rev-parse HEAD | tee "$LLAMA_DIR/.checkout"
}

build_llama_cpp() {
  cd "$LLAMA_DIR"

  if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found. Load cluster modules first."
    exit 1
  fi

  export CUDA_HOME
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
  export LDFLAGS="${LDFLAGS:-} -L$CUDA_HOME/lib64/stubs -Wl,-rpath,$CUDA_HOME/lib64"

  log "building llama.cpp at commit $LLAMA_COMMIT"
  cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DLLAMA_CURL=OFF \
    -DCUDAToolkit_ROOT="$CUDA_HOME"

  cmake --build build -j "$BUILD_JOBS" --config Release
}

download_gemma() {
  mkdir -p "$GEMMA_DIR"
  cd "$GEMMA_DIR"

  remove_if_hash_mismatch "$GEMMA_DIR/$GEMMA_NAME" "$GEMMA_SHA256" "Gemma GGUF"
  if [[ -f "$GEMMA_DIR/$GEMMA_NAME" ]]; then
    verify_sha256 "$GEMMA_DIR/$GEMMA_NAME" "$GEMMA_SHA256" "Gemma GGUF"
    return 0
  fi

  if seed_gemma_from_existing; then
    verify_sha256 "$GEMMA_DIR/$GEMMA_NAME" "$GEMMA_SHA256" "Gemma GGUF"
    return 0
  fi

  download_file "$GEMMA_URL" "$GEMMA_NAME"
  verify_sha256 "$GEMMA_DIR/$GEMMA_NAME" "$GEMMA_SHA256" "Gemma GGUF"
}

download_glm_ocr_file() {
  local label="$1"
  local filename="$2"
  local url="$3"
  local expected_sha="$4"
  local source_override="$5"
  local target="$GLMOCR_DIR/$filename"

  remove_if_hash_mismatch "$target" "$expected_sha" "$label"
  if [[ -f "$target" ]]; then
    verify_sha256 "$target" "$expected_sha" "$label"
    return 0
  fi

  if seed_glm_ocr_from_existing "$label" "$filename" "$expected_sha" "$source_override"; then
    verify_sha256 "$target" "$expected_sha" "$label"
    return 0
  fi

  cd "$GLMOCR_DIR"
  download_file "$url" "$filename"
  verify_sha256 "$target" "$expected_sha" "$label"
}

download_glm_ocr() {
  mkdir -p "$GLMOCR_DIR"
  download_glm_ocr_file \
    "GLM-OCR main"   "$GLMOCR_MAIN_NAME"   "$GLMOCR_MAIN_URL"   "$GLMOCR_MAIN_SHA256"   "${GLMOCR_MAIN_SOURCE_FILE:-}"
  download_glm_ocr_file \
    "GLM-OCR mmproj" "$GLMOCR_MMPROJ_NAME" "$GLMOCR_MMPROJ_URL" "$GLMOCR_MMPROJ_SHA256" "${GLMOCR_MMPROJ_SOURCE_FILE:-}"
}

setup_venv() {
  cd "$SCRIPT_DIR"
  local python_bin="$PYTHON_BIN"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "creating venv at $VENV_DIR"
    "$python_bin" -m venv "$VENV_DIR"
  else
    log "venv already exists: $VENV_DIR"
  fi

  local pip_python="$VENV_DIR/bin/python"
  log "upgrading pip/setuptools/wheel in $VENV_DIR"
  "$pip_python" -m pip install --upgrade pip setuptools wheel

  local lock_file="$SCRIPT_DIR/requirements.lock"
  if [[ -f "$lock_file" ]]; then
    log "installing pinned dependencies from $lock_file"
    "$pip_python" -m pip install -r "$lock_file"
    if [[ "$PIP_INSTALL_PROJECT" == "1" ]]; then
      log "installing project from $SCRIPT_DIR into $VENV_DIR (--no-deps; lock controls versions)"
      "$pip_python" -m pip install -e "$SCRIPT_DIR" --no-build-isolation --no-deps
    fi
  else
    log "no requirements.lock found; falling back to unpinned install"
    if [[ "$PIP_INSTALL_PROJECT" == "1" ]]; then
      log "installing project from $SCRIPT_DIR into $VENV_DIR"
      "$pip_python" -m pip install -e "$SCRIPT_DIR" --no-build-isolation
    fi
    if [[ -n "$PIP_EXTRA_PACKAGES" ]]; then
      local extra_packages=()
      read -r -a extra_packages <<<"$PIP_EXTRA_PACKAGES"
      if [[ ${#extra_packages[@]} -gt 0 ]]; then
        log "installing extra runtime packages: $PIP_EXTRA_PACKAGES"
        "$pip_python" -m pip install "${extra_packages[@]}"
      fi
    fi
  fi
}

print_summary() {
  cat <<SUMMARY

[setup] Done.

setup suffix:
  suffix:  ${SETUP_SUFFIX:-<none>}
  root:    $RUNTIME_ROOT

venv:
  path:    $VENV_DIR
  python:  $VENV_DIR/bin/python

llama.cpp:
  repo:    $LLAMA_DIR
  commit:  $(cat "$LLAMA_DIR/.checkout")
  server:  $LLAMA_DIR/build/bin/llama-server

models:
  Gemma:         $GEMMA_DIR/$GEMMA_NAME
  GLM-OCR main:  $GLMOCR_DIR/$GLMOCR_MAIN_NAME
  GLM-OCR mmproj: $GLMOCR_DIR/$GLMOCR_MMPROJ_NAME

verified sha256:
  Gemma:          $GEMMA_SHA256
  GLM-OCR main:   $GLMOCR_MAIN_SHA256
  GLM-OCR mmproj: $GLMOCR_MMPROJ_SHA256

Generated runtime paths:
  VENV_DIR=$VENV_DIR
  LLAMA_BIN=$LLAMA_DIR/build/bin/llama-server
  MODEL_PATH=$GEMMA_DIR/$GEMMA_NAME
  OCR_VLM_MODEL_PATH=$GLMOCR_DIR/$GLMOCR_MAIN_NAME
  OCR_VLM_MMPROJ_PATH=$GLMOCR_DIR/$GLMOCR_MMPROJ_NAME
  RUNTIME_ROOT=$RUNTIME_ROOT

Run with:
  bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o cohort.xlsx
SUMMARY
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  need_cmd git
  need_cmd wget
  need_cmd sha256sum
  need_cmd "$PYTHON_BIN"

  clone_llama_cpp

  if [[ "$BUILD_LLAMA" == "1" ]]; then
    build_llama_cpp
  else
    log "BUILD_LLAMA=0, skipping build step"
    log "If llama-server does not exist yet, rerun this script with BUILD_LLAMA=1 on a compute node."
  fi

  download_gemma
  download_glm_ocr
  setup_venv
  print_summary
}

main "$@"
