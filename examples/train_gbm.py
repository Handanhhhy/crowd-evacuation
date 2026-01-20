"""
训练 GBM 行为预测模型
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml.data_processor import TrajectoryDataProcessor
from ml.gbm_predictor import GBMPredictor


def main():
    print("=" * 50)
    print("GBM 行人行为预测模型训练")
    print("=" * 50)

    # 数据路径
    data_path = project_root / "data" / "raw" / "eth_ucy" / "synthetic_eth.txt"

    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        print("请先运行: python scripts/download_data.py")
        return

    # 1. 数据处理
    print("\n[1/4] 加载和处理数据...")
    processor = TrajectoryDataProcessor(obs_len=8, pred_len=4)
    X, y, feature_names, target_names = processor.prepare_dataset(str(data_path))

    print(f"\n特征列表: {feature_names}")
    print(f"目标列表: {target_names}")

    # 2. 训练模型
    print("\n[2/4] 训练 XGBoost 模型...")
    predictor = GBMPredictor(
        model_type="xgboost",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    results = predictor.fit(
        X, y,
        feature_names=feature_names,
        target_names=target_names,
        validation_split=0.2
    )

    # 3. 特征重要性
    print("\n[3/4] 特征重要性分析...")
    importance = predictor.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 重要特征:")
    for name, imp in sorted_importance[:10]:
        print(f"  {name}: {imp:.4f}")

    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    model_path = project_root / "outputs" / "models" / "gbm_predictor.joblib"
    predictor.save(str(model_path))

    # 5. 可视化
    visualize_results(predictor, X, y, feature_names, importance, project_root)

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)


def visualize_results(predictor, X, y, feature_names, importance, project_root):
    """可视化训练结果"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 特征重要性图
    ax1 = axes[0]
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_importance[:10]]
    values = [item[1] for item in sorted_importance[:10]]

    ax1.barh(names[::-1], values[::-1], color='steelblue')
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 10 Feature Importance')

    # 2. 预测 vs 真实值
    ax2 = axes[1]
    y_pred = predictor.predict(X)

    # 只画目标x的预测结果
    ax2.scatter(y[:, 0], y_pred[:, 0], alpha=0.3, s=10)
    ax2.plot([y[:, 0].min(), y[:, 0].max()],
             [y[:, 0].min(), y[:, 0].max()],
             'r--', label='Perfect prediction')
    ax2.set_xlabel('True target_x')
    ax2.set_ylabel('Predicted target_x')
    ax2.set_title('Prediction vs Ground Truth')
    ax2.legend()

    plt.tight_layout()

    # 保存图片
    fig_path = project_root / "outputs" / "figures" / "gbm_training.png"
    plt.savefig(str(fig_path), dpi=150)
    print(f"\n训练结果图已保存: {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()
