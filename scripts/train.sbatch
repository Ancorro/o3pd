#!/bin/bash
# Cluster-specific flags (account, partition, GPU constraint) are passed on the command line; see private/.
# One training run on a single GPU (student, teacher and vLLM share it). Usage:
#   sbatch [cluster flags] scripts/train.sbatch --loss opd --out runs/4b-1.7b-opd
#SBATCH --job-name=o3pd-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train-%x-%j.out
set -euo pipefail
cd "${O3PD_DIR:-$SLURM_SUBMIT_DIR}"
[ -f private/hpc.env ] && source private/hpc.env  # cluster paths: HF_HOME, SIF (untracked)
export PYTHONUNBUFFERED=1 PYTHONPATH=pylib:.
apptainer exec --nv "${SIF:?set SIF to a vLLM apptainer image}" python3 -m o3pd.train \
  --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B --student-device cuda:0 --teacher-device cuda:0 --vllm-mem 0.3 "$@"
