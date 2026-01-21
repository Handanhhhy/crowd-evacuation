"""
梯度提升树(GBM)行为预测模块
用于预测行人下一步的位置/速度
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


class GBMPredictor:
    """梯度提升树行为预测器"""

    def __init__(
        self,
        model_type: str = "xgboost",  # "sklearn" or "xgboost"
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 模型类型，"sklearn" 或 "xgboost"
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            random_state: 随机种子
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = None
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False

    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用于XGBoost"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _create_model(self):
        """创建模型

        自动检测GPU并使用加速:
        - NVIDIA GPU: 使用 tree_method='gpu_hist'
        - 其他: 使用CPU (tree_method='hist')
        """
        if self.model_type == "xgboost":
            # 检测GPU
            use_gpu = self._check_gpu_available()

            if use_gpu:
                # GPU加速配置
                base_model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    tree_method='gpu_hist',  # GPU加速
                    predictor='gpu_predictor',
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                # CPU配置
                base_model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    tree_method='hist',  # CPU快速方法
                    n_jobs=-1,
                    verbosity=0
                )
        else:
            base_model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

        # 使用多输出回归器包装（支持多目标预测）
        return MultiOutputRegressor(base_model)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
        target_names: list = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标矩阵 (n_samples, n_targets)
            feature_names: 特征名列表
            target_names: 目标名列表
            validation_split: 验证集比例

        Returns:
            训练结果字典
        """
        self.feature_names = feature_names
        self.target_names = target_names

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )

        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"特征数: {X_train.shape[1]}")
        print(f"目标数: {y_train.shape[1]}")

        # 创建并训练模型
        print(f"\n使用 {self.model_type} 训练中...")
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        self.is_fitted = True

        # 评估
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        results = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
        }

        print(f"\n训练结果:")
        print(f"  训练集 MSE: {results['train_mse']:.6f}")
        print(f"  训练集 MAE: {results['train_mae']:.6f}")
        print(f"  训练集 R²:  {results['train_r2']:.4f}")
        print(f"  验证集 MSE: {results['val_mse']:.6f}")
        print(f"  验证集 MAE: {results['val_mae']:.6f}")
        print(f"  验证集 R²:  {results['val_r2']:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测结果 (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        return self.model.predict(X)

    def predict_next_position(
        self,
        current_pos: np.ndarray,
        velocity: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        """预测下一步位置

        这是一个便捷方法，用于实时预测

        Args:
            current_pos: 当前位置 (2,)
            velocity: 当前速度 (2,)
            features: 其他特征

        Returns:
            预测的下一步位置 (2,)
        """
        # 组合特征
        X = np.concatenate([current_pos, velocity, features]).reshape(1, -1)
        pred = self.predict(X)[0]

        # 假设预测的是速度增量，计算下一步位置
        next_pos = current_pos + pred[:2]
        return next_pos

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        # 对于多输出模型，取各个子模型特征重要性的平均
        importances = []
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)

        if not importances:
            return {}

        mean_importance = np.mean(importances, axis=0)

        if self.feature_names:
            return dict(zip(self.feature_names, mean_importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(mean_importance)}

    def save(self, path: str) -> None:
        """保存模型"""
        save_dict = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            }
        }
        joblib.dump(save_dict, path)
        print(f"模型已保存到: {path}")

    def load(self, path: str) -> None:
        """加载模型"""
        save_dict = joblib.load(path)
        self.model = save_dict['model']
        self.model_type = save_dict['model_type']
        self.feature_names = save_dict['feature_names']
        self.target_names = save_dict['target_names']
        self.is_fitted = True
        print(f"模型已加载: {path}")
