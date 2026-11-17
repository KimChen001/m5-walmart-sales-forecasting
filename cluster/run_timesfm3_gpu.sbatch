#!/bin/bash
#SBATCH -J m5_timesfm3
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
module load miniforge
eval "$(conda shell.bash hook)"
conda activate m5
export PYTHONPATH="$PWD/timesfm3_lib${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from timesfm3 import TimesFM3Evaluator; import scipy, torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"'

run_root="outputs/timesfm3_${SLURM_JOB_ID}"
python -u timesfm3_improvements.py --smoke --out "$run_root/smoke"
python -u timesfm3_improvements.py --out "$run_root/full"
