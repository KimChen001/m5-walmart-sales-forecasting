#!/bin/bash
# 在 ORCD Engaging 登录节点上运行一次: bash cluster/setup_env.sh
set -e
module load miniforge
eval "$(conda shell.bash hook)"
conda create -y -n m5 python=3.11
conda activate m5
pip install "pandas>=2.2" "numpy>=1.26" "xgboost>=2.1" scikit-learn
python -c "import pandas, numpy, xgboost; print('ok', pandas.__version__, numpy.__version__, xgboost.__version__)"
