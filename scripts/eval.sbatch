#!/bin/bash
# Cluster-specific flags (account, partition, GPU constraint) are passed on the command line; see private/.
# Avg@k evaluation as a Slurm array: one task per (benchmark, seed). Usage:
#   sbatch [cluster flags] --export=ALL,MODEL=Qwen/Qwen3-1.7B,LORA=,TAG=base17 scripts/eval.sbatch
#SBATCH --job-name=o3pd-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --requeue
#SBATCH --output=logs/eval-%x-%A_%a.out
set -euo pipefail
cd "${O3PD_DIR:-$SLURM_SUBMIT_DIR}"
[ -f private/hpc.env ] && source private/hpc.env  # cluster paths: HF_HOME, SIF (untracked)
export PYTHONUNBUFFERED=1 PYTHONPATH=pylib:.
B=(hmmt25 aime24 aime25); S=(41 42 43)
b=${B[$((SLURM_ARRAY_TASK_ID / 3))]}; s=${S[$((SLURM_ARRAY_TASK_ID % 3))]}
out=results/eval/${TAG}/${b}_s${s}.json
[ -f "$out" ] && { echo "exists $out"; exit 0; }
mkdir -p "results/eval/${TAG}"
apptainer exec --nv "${SIF:?set SIF to a vLLM apptainer image}" python3 -m o3pd.evaluate \
  --model "$MODEL" ${LORA:+--lora "$LORA"} --benchmarks "$b" --seeds "$s" --n-per-seed "${N:-4}" --out "$out"
