#!/bin/bash
#SBATCH --job-name=laysumm
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Run train.py on one Rangpur A100. Submit from this folder:
#
#   sbatch --job-name=zero_shot --time=02:00:00 slurm_train.sh zero_shot runs/zero_shot
#   sbatch --job-name=lora_r8_s0 slurm_train.sh lora runs/lora_r8_s0 --rank 8 --seed 0
#   sbatch --job-name=full_s0    slurm_train.sh full runs/full_s0 --seed 0
#
# The first two arguments are the mode and the output folder; anything after
# them is passed straight to train.py. A shorter --time helps the scheduler
# fit the job into gaps, so lower it once the real run time is known.
#
# One-off setup on the login node (compute nodes then run offline):
#   - data:  train.parquet and validation.parquet in $DATA_DIR
#   - model: HF_HOME=$HOME/.cache/huggingface HF_HUB_DISABLE_XET=1 python -c \
#            "from huggingface_hub import snapshot_download; snapshot_download('google/flan-t5-base')"

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch slurm_train.sh <zero_shot|lora|full> <out_dir> [train.py options]" >&2
    exit 1
fi
MODE=$1
OUT_DIR=$2
shift 2

DATA_DIR=${DATA_DIR:-$HOME/data/laymanrrg}
CONDA_ENV=${CONDA_ENV:-torch}

# The environment name is passed explicitly: a bare `source` would hand this
# script's own arguments to conda. set -u trips over unset variables inside it.
set +u
source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
set -u

# The model is read from the Hugging Face cache; no network needed. The cache
# path is pinned because Rangpur login shells set XDG_CACHE_HOME=/cache, which
# would otherwise move the cache to a disk the compute nodes may not share.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-.}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  mode $MODE  out $OUT_DIR  extra: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python train.py --mode "$MODE" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" "$@"

echo "finished at $(date)"
