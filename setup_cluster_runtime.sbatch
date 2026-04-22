#!/usr/bin/env bash
#SBATCH --job-name=setup_cluster_runtime
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --output=setup_cluster_runtime_%j.out
#SBATCH --error=setup_cluster_runtime_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
WORKDIR="${WORKDIR:-/groups/umcg-gcc/tmp02/users/umcg-pjansma}"
SETUP_SUFFIX="${SETUP_SUFFIX:-}"
BUILD_LLAMA="${BUILD_LLAMA:-1}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load GCCcore/11.3.0 || true
  module load CMake/3.23.1-GCCcore-11.3.0 || true
  module load CUDA/12.2.0 || true
  module load Python/3.10.4-GCCcore-11.3.0 || true
fi

echo "[setup-sbatch] REPO_DIR=$REPO_DIR"
echo "[setup-sbatch] WORKDIR=$WORKDIR"
echo "[setup-sbatch] SETUP_SUFFIX=${SETUP_SUFFIX:-<none>}"
echo "[setup-sbatch] BUILD_LLAMA=$BUILD_LLAMA"
echo "[setup-sbatch] BUILD_JOBS=$BUILD_JOBS"
echo "[setup-sbatch] PYTHON_BIN=$PYTHON_BIN"
echo "[setup-sbatch] HOSTNAME=${HOSTNAME:-unknown}"

WORKDIR="$WORKDIR" \
SETUP_SUFFIX="$SETUP_SUFFIX" \
BUILD_LLAMA="$BUILD_LLAMA" \
BUILD_JOBS="$BUILD_JOBS" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$REPO_DIR/setup_cluster_runtime.sh"
