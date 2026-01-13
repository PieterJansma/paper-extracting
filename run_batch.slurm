#!/usr/bin/env bash
#SBATCH --job-name=llama_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/batch_%j.out
#SBATCH --error=logs/batch_%j.err

set -euo pipefail

echo "=== JOB START ==="
date
hostname

# --- modules ---
module purge
module load GCCcore/11.3.0
module load CUDA/12.2.0
module load Python/3.10.4-GCCcore-11.3.0

# --- env ---
export WORKDIR="/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# --- go to project ---
cd "$WORKDIR/paper-extracting"

# --- activate venv ---
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "❌ .venv not found"
  exit 1
fi

# --- safety ---
which python
nvidia-smi

# --- run your cluster script, single pass example ---
bash src/run_cluster.sh \
  --pdfs data/oncolifes.pdf \
  -o batch_test_13_01_2026.xlsx

echo "=== JOB END ==="
date
