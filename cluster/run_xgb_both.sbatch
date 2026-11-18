#!/bin/bash
# Full-panel XGBoost: both forecast settings plus scoring on both panels, in one job.
# Submit (from ~/m5-project):  sbatch cluster/run_xgb_both.sbatch
# Follow progress:             tail -f logs/m5_xgb_both_<jobid>.out
#SBATCH -J m5_xgb_both
#SBATCH -p mit_normal
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out

set -e                         # stop if any step fails; never carry on with half-finished results

module load miniforge
eval "$(conda shell.bash hook)"
conda activate m5
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "================ job $SLURM_JOB_ID started $(date) ================"
echo "node $SLURMD_NODENAME | $SLURM_CPUS_PER_TASK cores | $SLURM_MEM_PER_NODE MB"

echo; echo "######## 1/4  horizon 28 (M5 convention, one origin every 28 days) ########"
python xgb_forecast.py --state ALL --cat ALL --device cpu

echo; echo "######## 2/4  horizon 1 (aligned with OLS, re-forecast every day) ########"
python xgb_forecast.py --state ALL --cat ALL --device cpu --horizon 1

for d in outputs/xgb_ALL_ALL outputs/xgb_ALL_ALL_h1; do
    echo; echo "######## 3/4  Panel B (6,638 regular sellers) — $d ########"
    python wrmsse.py "$d" --min-selling-share 0.5
done

for d in outputs/xgb_ALL_ALL outputs/xgb_ALL_ALL_h1; do
    echo; echo "######## 4/4  all metrics on both panels — $d ########"
    python final_metrics.py "$d"
done

echo; echo "================ all done $(date) ================"
