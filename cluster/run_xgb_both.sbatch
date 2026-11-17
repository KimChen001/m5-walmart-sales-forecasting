#!/bin/bash
# 全量 XGBoost, 两种预测口径 + 两套面板打分, 一个作业跑完。
# 提交 (在 ~/m5-project 下):  sbatch cluster/run_xgb_both.sbatch
# 看进度:                     tail -f logs/m5_xgb_both_<jobid>.out
#SBATCH -J m5_xgb_both
#SBATCH -p mit_normal
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out

set -e                         # 任何一步失败就停, 不要带着半截结果往下跑

module load miniforge
eval "$(conda shell.bash hook)"
conda activate m5
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "================ 作业 $SLURM_JOB_ID 开始 $(date) ================"
echo "节点 $SLURMD_NODENAME | $SLURM_CPUS_PER_TASK 核 | $SLURM_MEM_PER_NODE MB"

echo; echo "######## 1/4  horizon 28 (M5 口径, 每 28 天一个起点) ########"
python xgb_forecast.py --state ALL --cat ALL --device cpu

echo; echo "######## 2/4  horizon 1 (对齐 OLS, 每天重新预测) ########"
python xgb_forecast.py --state ALL --cat ALL --device cpu --horizon 1

for d in outputs/xgb_ALL_ALL outputs/xgb_ALL_ALL_h1; do
    echo; echo "######## 3/4  Panel B (6,638 常销品) — $d ########"
    python wrmsse.py "$d" --min-selling-share 0.5
done

for d in outputs/xgb_ALL_ALL outputs/xgb_ALL_ALL_h1; do
    echo; echo "######## 4/4  两套面板全部指标 — $d ########"
    python final_metrics.py "$d"
done

echo; echo "================ 全部完成 $(date) ================"
