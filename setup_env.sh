#!/bin/bash
# 环境配置脚本

echo "=========================================="
echo "人群疏散项目 - 环境配置"
echo "=========================================="

# 检查 Python 版本
echo ""
echo "[1/4] 检查 Python 版本..."
python3 --version

# 创建虚拟环境
echo ""
echo "[2/4] 创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级 pip
echo ""
echo "[3/4] 升级 pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "[4/4] 安装依赖..."
pip install numpy scipy matplotlib pysocialforce scikit-learn xgboost torch pandas tqdm pygame seaborn pyyaml toml gymnasium stable-baselines3 jupyterlab

echo ""
echo "=========================================="
echo "环境配置完成！"
echo ""
echo "激活环境: source venv/bin/activate"
echo "运行演示: python examples/demo_sfm.py"
echo "=========================================="
