#!/usr/bin/env bash
#SBATCH --job-name=cohort_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:2
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ ! -f "src/run_cluster_cohort.sh" ]]; then
  echo "ERROR: src/run_cluster_cohort.sh not found in ${PWD}"
  exit 1
fi

# Live EMX2 defaults (override with sbatch --export=...).
export MOLGENIS_EMX2_REPO="${MOLGENIS_EMX2_REPO:-molgenis/molgenis-emx2}"
export MOLGENIS_EMX2_REF="${MOLGENIS_EMX2_REF:-main}"
export EMX2_PROFILE="${EMX2_PROFILE:-UMCGCohortsStaging}"
export AUTO_FETCH_EMX2_ONTOLOGIES="${AUTO_FETCH_EMX2_ONTOLOGIES:-1}"

# No args passed? Use a safe default run.
if [[ $# -gt 0 ]]; then
  RUN_ARGS=("$@")
else
  RUN_ARGS=(-p all --pdfs data/oncolifes.pdf -o "cohort_${SLURM_JOB_ID:-manual}.xlsx")
fi

echo "[SBATCH] PWD=${PWD}"
echo "[SBATCH] RUN_ARGS=${RUN_ARGS[*]}"
echo "[SBATCH] EMX2=${MOLGENIS_EMX2_REPO}@${MOLGENIS_EMX2_REF} profile=${EMX2_PROFILE}"

bash src/run_cluster_cohort.sh "${RUN_ARGS[@]}"
