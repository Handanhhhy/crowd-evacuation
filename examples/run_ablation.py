#!/usr/bin/env python3
"""
消融实验主脚本
Ablation Study Runner

运行所有消融实验组:
- A组: PPO观测空间消融 (16D -> 8D -> 6D)
- B组: PPO奖励函数消融 (逐项移除奖励组件)
- C组: 轨迹预测消融 (神经网络 vs 线性外推)
- D组: 行人仿真消融 (SFM参数 + 行人类型)
- E组: 引导策略消融 (有/无PPO引导)

用法:
    # 运行所有消融实验
    python examples/run_ablation.py

    # 运行指定组
    python examples/run_ablation.py --groups A B

    # 运行单个实验
    python examples/run_ablation.py --experiments A1_16D A2_8D

    # 使用更少训练步数 (快速测试)
    python examples/run_ablation.py --timesteps 10000

    # 只生成汇总报告 (不训练)
    python examples/run_ablation.py --summary-only
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("警告: stable-baselines3不可用，训练功能已禁用")

from simulation.metro_evacuation_env import MetroEvacuationEnv
from sfm.social_force import PedestrianType
from utils.experiment_logger import (
    ExperimentLogger,
    AblationSummaryGenerator,
    ExperimentResult
)


def get_trajectory_device() -> str:
    """自动检测轨迹预测的最佳设备（神经网络用GPU更快）"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"轨迹预测使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("轨迹预测使用 Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("轨迹预测使用 CPU")
    return device


def get_ppo_device() -> str:
    """PPO设备选择（MLP策略用CPU更快）"""
    # PPO使用MlpPolicy时，CPU通常比GPU更快
    # 因为小型MLP网络的GPU数据传输开销大于计算收益
    print("PPO训练使用 CPU（MLP策略在CPU上更快）")
    return "cpu"


class ObservationWrapper(gym.ObservationWrapper):
    """观测空间包装器，用于消融实验A组"""

    def __init__(self, env, observation_config: Dict[str, bool]):
        """
        Args:
            env: 原始环境
            observation_config: 观测空间配置
                - exit_density: 是否包含出口密度 [0-2]
                - exit_congestion: 是否包含出口拥堵度 [3-5]
                - flow_direction: 是否包含人流方向 [6-8]
                - evacuation_rate: 是否包含疏散速率 [9-11]
                - bottleneck_density: 是否包含瓶颈密度 [12-13]
                - remaining_ratio: 是否包含剩余比例 [14]
                - time_ratio: 是否包含时间比例 [15]
        """
        super().__init__(env)
        self.observation_config = observation_config

        # 计算新的观测维度
        self.feature_indices = []
        self.feature_names = []

        if observation_config.get("exit_density", True):
            self.feature_indices.extend([0, 1, 2])
            self.feature_names.append("exit_density")
        if observation_config.get("exit_congestion", True):
            self.feature_indices.extend([3, 4, 5])
            self.feature_names.append("exit_congestion")
        if observation_config.get("flow_direction", True):
            self.feature_indices.extend([6, 7, 8])
            self.feature_names.append("flow_direction")
        if observation_config.get("evacuation_rate", True):
            self.feature_indices.extend([9, 10, 11])
            self.feature_names.append("evacuation_rate")
        if observation_config.get("bottleneck_density", True):
            self.feature_indices.extend([12, 13])
            self.feature_names.append("bottleneck_density")
        if observation_config.get("remaining_ratio", True):
            self.feature_indices.append(14)
            self.feature_names.append("remaining_ratio")
        if observation_config.get("time_ratio", True):
            self.feature_indices.append(15)
            self.feature_names.append("time_ratio")

        self.new_obs_dim = len(self.feature_indices)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.new_obs_dim,),
            dtype=np.float32
        )

        print(f"[ObservationWrapper] 观测维度: 16D -> {self.new_obs_dim}D")
        print(f"[ObservationWrapper] 包含特征: {self.feature_names}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """提取指定的观测特征"""
        return obs[self.feature_indices]


class EarlyStoppingCallback(BaseCallback):
    """
    早停回调，当训练无改善时提前停止

    Args:
        patience: 连续无改善的检查次数后停止
        min_improvement: 最小改善阈值
        check_freq: 检查频率（步数）
        verbose: 日志级别
    """

    def __init__(
        self,
        patience: int = 5,
        min_improvement: float = 0.01,
        check_freq: int = 2000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.check_freq = check_freq
        self.best_mean_reward = -float('inf')
        self.no_improve_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 获取当前平均奖励
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

                # 检查是否有改善
                if mean_reward > self.best_mean_reward + self.min_improvement:
                    self.best_mean_reward = mean_reward
                    self.no_improve_count = 0
                    if self.verbose > 0:
                        print(f"[EarlyStop] 新最佳奖励: {mean_reward:.2f}")
                else:
                    self.no_improve_count += 1
                    if self.verbose > 0:
                        print(f"[EarlyStop] 无改善 ({self.no_improve_count}/{self.patience})")

                # 达到耐心上限，停止训练
                if self.no_improve_count >= self.patience:
                    if self.verbose > 0:
                        print(f"[EarlyStop] 早停触发! 最佳奖励: {self.best_mean_reward:.2f}")
                    return False

        return True


class TrainingCallback(BaseCallback):
    """训练回调，用于记录训练指标"""

    def __init__(self, logger: ExperimentLogger, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.exp_logger = logger
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # 获取训练指标
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                std_reward = np.std([ep["r"] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            else:
                mean_reward = 0.0
                std_reward = 0.0
                mean_length = 0.0

            self.exp_logger.log_training_step(
                timestep=self.num_timesteps,
                mean_reward=mean_reward,
                std_reward=std_reward,
                episode_length=mean_length
            )

        return True


def load_ablation_config(config_path: str = None) -> Dict[str, Any]:
    """加载消融实验配置"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "ablation_configs.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def build_env_kwargs(
    exp_config: Dict[str, Any],
    group: str,
    global_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    构建环境参数字典

    Args:
        exp_config: 实验配置
        group: 实验组 (A/B/C/D/E)
        global_config: 全局配置

    Returns:
        环境参数字典
    """
    env_config = global_config.get("environment", {})

    # 基础环境参数
    env_kwargs = {
        "n_pedestrians": env_config.get("num_pedestrians", 80),
        "max_steps": env_config.get("max_steps", 800),
        "dt": env_config.get("dt", 0.1),
        # GPU加速SFM (4070ti Super等NVIDIA GPU)
        "use_optimized_gpu_sfm": global_config.get("training", {}).get("use_gpu_sfm", True),
        "sfm_device": global_config.get("training", {}).get("sfm_device", "auto"),
    }

    # 根据实验组配置参数
    if group == "B":
        # B组: 奖励函数消融
        env_kwargs["reward_weights"] = exp_config.get("reward_weights", {})

    elif group == "C":
        # C组: 轨迹预测消融
        traj_config = exp_config.get("trajectory_prediction", {})
        env_kwargs["enable_neural_prediction"] = traj_config.get("use_neural_network", True)
        if traj_config.get("model_path"):
            env_kwargs["trajectory_model_path"] = traj_config.get("model_path")

    elif group == "D":
        # D组: 行人仿真消融
        ped_config = exp_config.get("pedestrian_types", {})
        if ped_config.get("use_multi_type", True):
            dist = ped_config.get("type_distribution", {})
            env_kwargs["type_distribution"] = {
                PedestrianType.NORMAL: dist.get("NORMAL", 0.6),
                PedestrianType.ELDERLY: dist.get("ELDERLY", 0.15),
                PedestrianType.CHILD: dist.get("CHILD", 0.1),
                PedestrianType.IMPATIENT: dist.get("IMPATIENT", 0.15),
            }
        else:
            # 只使用NORMAL类型
            env_kwargs["type_distribution"] = {
                PedestrianType.NORMAL: 1.0,
                PedestrianType.ELDERLY: 0.0,
                PedestrianType.CHILD: 0.0,
                PedestrianType.IMPATIENT: 0.0,
            }

        # SFM参数 (D3/D4实验关键参数)
        sfm_config = exp_config.get("sfm_params", {})
        env_kwargs["sfm_A"] = sfm_config.get("A", 2000.0)
        env_kwargs["sfm_B"] = sfm_config.get("B", 0.08)
        env_kwargs["sfm_tau"] = sfm_config.get("tau", 0.5)

        # GBM修正配置
        gbm_config = exp_config.get("gbm_correction", {})
        env_kwargs["enable_enhanced_behaviors"] = gbm_config.get("enabled", True)
        # 禁用GBM时权重设为0
        env_kwargs["gbm_weight"] = gbm_config.get("weight", 0.3) if gbm_config.get("enabled", True) else 0.0

    elif group == "E":
        # E组: 引导策略消融
        # 关键：通过enable_guidance控制是否启用引导逻辑
        guidance_config = exp_config.get("guidance", {})
        env_kwargs["enable_guidance"] = guidance_config.get("enabled", True)

    return env_kwargs


def create_environment(
    exp_config: Dict[str, Any],
    group: str,
    global_config: Dict[str, Any],
    seed: int = 42
) -> gym.Env:
    """
    根据实验配置创建环境

    Args:
        exp_config: 实验配置
        group: 实验组 (A/B/C/D/E)
        global_config: 全局配置
        seed: 随机种子

    Returns:
        配置好的gym环境
    """
    env_kwargs = build_env_kwargs(exp_config, group, global_config)

    # 创建基础环境
    env = MetroEvacuationEnv(**env_kwargs)
    env.reset(seed=seed)

    # A组: 观测空间消融 - 使用包装器
    if group == "A":
        obs_config = exp_config.get("observation_features", {})
        env = ObservationWrapper(env, obs_config)

    return env


def make_env(env_kwargs: Dict[str, Any], seed: int):
    """创建环境工厂函数 (用于SubprocVecEnv)"""
    def _init():
        env = MetroEvacuationEnv(**env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo_model(
    env: gym.Env,
    exp_id: str,
    logger: ExperimentLogger,
    global_config: Dict[str, Any],
    output_dir: Path,
    device: str = "auto",
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> PPO:
    """
    训练PPO模型

    Args:
        env: 训练环境 (单环境，用于DummyVecEnv回退)
        exp_id: 实验ID
        logger: 实验日志记录器
        global_config: 全局配置
        output_dir: 输出目录
        device: 训练设备
        env_kwargs: 环境参数 (用于SubprocVecEnv创建多个环境)
        seed: 随机种子

    Returns:
        训练好的PPO模型
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3不可用")

    train_config = global_config.get("training", {})
    n_envs = train_config.get("n_envs", 4)

    # 创建向量环境 (优先使用SubprocVecEnv并行加速)
    if env_kwargs is not None and n_envs > 1:
        try:
            vec_env = SubprocVecEnv([
                make_env(env_kwargs, seed + i) for i in range(n_envs)
            ])
            print(f"  使用SubprocVecEnv: {n_envs}个并行环境")
        except Exception as e:
            print(f"  SubprocVecEnv创建失败: {e}, 回退到DummyVecEnv")
            vec_env = DummyVecEnv([lambda: env])
    else:
        vec_env = DummyVecEnv([lambda: env])

    # 自动检测GPU (支持CUDA和MPS)
    if torch.cuda.is_available():
        ppo_device = "cuda"
        print(f"  PPO使用NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        ppo_device = "mps"
        print("  PPO使用Apple Silicon GPU (MPS)")
    else:
        ppo_device = "cpu"
        print("  PPO使用CPU")

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=train_config.get("learning_rate", 3e-4),
        n_steps=train_config.get("n_steps", 1024),
        batch_size=train_config.get("batch_size", 128),
        n_epochs=train_config.get("n_epochs", 10),
        gamma=train_config.get("gamma", 0.99),
        gae_lambda=train_config.get("gae_lambda", 0.95),
        clip_range=train_config.get("clip_range", 0.2),
        ent_coef=train_config.get("ent_coef", 0.01),
        verbose=1,
        device=ppo_device
    )

    # 训练回调
    callbacks = [TrainingCallback(logger, log_freq=1000)]

    # 早停回调（可选）
    early_stop_config = train_config.get("early_stopping", {})
    if early_stop_config.get("enabled", False):
        early_stop_callback = EarlyStoppingCallback(
            patience=early_stop_config.get("patience", 5),
            min_improvement=early_stop_config.get("min_improvement", 0.01),
            check_freq=2000,
            verbose=1
        )
        callbacks.append(early_stop_callback)
        print("  启用早停机制")

    # 训练
    total_timesteps = train_config.get("total_timesteps", 100000)
    print(f"\n[训练] 开始训练 {exp_id}, 总步数: {total_timesteps}")

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    training_time = time.time() - start_time

    print(f"[训练] 完成! 耗时: {training_time:.1f}秒")

    # 保存模型
    model_path = output_dir / "model.zip"
    model.save(str(model_path))
    print(f"[保存] 模型: {model_path}")

    return model


def evaluate_model(
    model: Optional[PPO],
    env: gym.Env,
    logger: ExperimentLogger,
    n_episodes: int = 10,
    use_guidance: bool = True
) -> Dict[str, float]:
    """
    评估模型

    Args:
        model: PPO模型 (如果为None，则不使用引导)
        env: 评估环境
        logger: 实验日志记录器
        n_episodes: 评估episode数
        use_guidance: 是否使用PPO引导

    Returns:
        评估指标字典
    """
    print(f"\n[评估] 运行 {n_episodes} 个episodes...")

    all_metrics = {
        "evacuation_rates": [],
        "evacuation_times": [],
        "max_congestions": [],
        "exit_balances": [],
        "cumulative_rewards": [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        max_congestion = 0.0

        while not (done or truncated):
            if model is not None and use_guidance:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 随机动作 (无引导)
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # 跟踪最大拥堵度
            if hasattr(env, 'unwrapped'):
                base_env = env.unwrapped
            else:
                base_env = env

            # 优化: 只检查最新的拥堵值，避免O(T²)每步扫描整个历史
            if hasattr(base_env, 'history') and 'congestion' in base_env.history:
                history = base_env.history['congestion']
                if history:
                    # 只取最新值与当前max比较，而非每步重新扫描整个历史
                    max_congestion = max(max_congestion, history[-1])

        # 计算指标
        if hasattr(env, 'unwrapped'):
            base_env = env.unwrapped
        else:
            base_env = env

        evacuated = getattr(base_env, 'evacuated_count', 0)
        total = getattr(base_env, 'n_pedestrians', 80)
        evacuation_rate = evacuated / total if total > 0 else 0.0

        # 出口均衡度
        if hasattr(base_env, 'evacuated_by_exit'):
            counts = list(base_env.evacuated_by_exit.values())
            if sum(counts) > 0:
                mean_count = sum(counts) / len(counts)
                exit_balance = np.std(counts)
            else:
                exit_balance = 0.0
        else:
            exit_balance = 0.0

        # 记录到logger
        logger.log_evaluation_episode(
            episode_id=ep,
            evacuation_rate=evacuation_rate,
            avg_evacuation_time=float(step_count),
            max_congestion=max_congestion,
            exit_balance=exit_balance,
            cumulative_reward=episode_reward,
            episode_length=step_count
        )

        all_metrics["evacuation_rates"].append(evacuation_rate)
        all_metrics["evacuation_times"].append(step_count)
        all_metrics["max_congestions"].append(max_congestion)
        all_metrics["exit_balances"].append(exit_balance)
        all_metrics["cumulative_rewards"].append(episode_reward)

    # 计算平均值
    summary = {
        "mean_evacuation_rate": np.mean(all_metrics["evacuation_rates"]),
        "std_evacuation_rate": np.std(all_metrics["evacuation_rates"]),
        "mean_evacuation_time": np.mean(all_metrics["evacuation_times"]),
        "mean_max_congestion": np.mean(all_metrics["max_congestions"]),
        "mean_exit_balance": np.mean(all_metrics["exit_balances"]),
        "mean_cumulative_reward": np.mean(all_metrics["cumulative_rewards"]),
    }

    print(f"[评估] 完成! 疏散率: {summary['mean_evacuation_rate']:.1%}, "
          f"平均时间: {summary['mean_evacuation_time']:.1f}步")

    return summary


def run_single_experiment(
    exp_id: str,
    exp_config: Dict[str, Any],
    group: str,
    global_config: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
    device: str = "auto",
    run_timestamp: Optional[str] = None
) -> ExperimentResult:
    """
    运行单个消融实验

    Args:
        exp_id: 实验ID
        exp_config: 实验配置
        group: 实验组
        global_config: 全局配置
        output_dir: 输出目录
        seed: 随机种子
        device: 训练设备
        run_timestamp: 运行时间戳 (用于同一批次实验共享目录)

    Returns:
        实验结果
    """
    exp_name = exp_config.get("name", exp_id)
    print(f"\n{'='*60}")
    print(f"实验: {exp_id} - {exp_name}")
    print(f"组: {group}, 种子: {seed}")
    print(f"{'='*60}")

    # 创建日志记录器 (使用共享时间戳避免覆盖)
    logger = ExperimentLogger(
        experiment_id=exp_id,
        experiment_name=exp_name,
        group=group,
        output_dir=output_dir,
        config=exp_config,
        run_timestamp=run_timestamp
    )

    # 构建环境参数 (用于SubprocVecEnv并行创建)
    env_kwargs = build_env_kwargs(exp_config, group, global_config)

    # 创建环境
    env = create_environment(exp_config, group, global_config, seed=seed)

    # 训练模型 (除了E组的无引导实验)
    model = None
    use_guidance = True

    if group == "E":
        guidance_config = exp_config.get("guidance", {})
        use_guidance = guidance_config.get("enabled", True)

    if use_guidance and SB3_AVAILABLE:
        model = train_ppo_model(
            env=env,
            exp_id=exp_id,
            logger=logger,
            global_config=global_config,
            output_dir=logger.output_dir,
            device=device,
            env_kwargs=env_kwargs,
            seed=seed
        )

    # 评估模型
    eval_config = global_config.get("evaluation", {})
    n_eval_episodes = eval_config.get("n_eval_episodes", 10)

    # 重新创建评估环境
    eval_env = create_environment(exp_config, group, global_config, seed=seed + 1000)

    evaluate_model(
        model=model,
        env=eval_env,
        logger=logger,
        n_episodes=n_eval_episodes,
        use_guidance=use_guidance
    )

    # 保存所有结果
    logger.save_all()

    # 返回实验结果
    return logger.compute_summary()


def run_ablation_group(
    group: str,
    config: Dict[str, Any],
    output_dir: str,
    experiments: Optional[List[str]] = None,
    device: str = "auto",
    run_timestamp: Optional[str] = None
) -> List[ExperimentResult]:
    """
    运行一个消融实验组

    Args:
        group: 实验组名称 (A/B/C/D/E)
        config: 完整配置
        output_dir: 输出目录
        experiments: 指定的实验ID列表 (可选)
        device: 训练设备
        run_timestamp: 运行时间戳 (用于同一批次实验共享目录)

    Returns:
        实验结果列表
    """
    group_key = f"group_{group}"
    if group_key not in config:
        print(f"[警告] 配置中不存在组 {group}")
        return []

    group_config = config[group_key]
    global_config = config.get("global", {}).copy()

    # 合并组级别的训练配置 (支持E组专用30k步数)
    if "training" in group_config:
        group_training = group_config["training"]
        global_config["training"] = {**global_config.get("training", {}), **group_training}

    eval_config = global_config.get("evaluation", {})
    random_seeds = eval_config.get("random_seeds", [42])

    results = []

    for exp_id, exp_config in group_config.get("experiments", {}).items():
        # 检查是否在指定实验列表中
        if experiments and exp_id not in experiments:
            continue

        # 对每个随机种子运行实验
        for seed in random_seeds:
            result = run_single_experiment(
                exp_id=f"{exp_id}_seed{seed}",
                exp_config=exp_config,
                group=group,
                global_config=global_config,
                output_dir=output_dir,
                seed=seed,
                device=device,
                run_timestamp=run_timestamp
            )
            results.append(result)

    return results


def _run_group_worker(args: Tuple) -> List[ExperimentResult]:
    """
    并行执行单个组的工作函数

    Args:
        args: (group, config, output_dir, experiments, device, run_timestamp)

    Returns:
        实验结果列表
    """
    group, config, output_dir, experiments, device, run_timestamp = args
    print(f"\n[组{group}] 开始... (进程 {os.getpid()})")

    results = run_ablation_group(
        group=group,
        config=config,
        output_dir=output_dir,
        experiments=experiments,
        device=device,
        run_timestamp=run_timestamp
    )

    print(f"[组{group}] 完成! 共 {len(results)} 个实验")
    return results


def run_all_ablations(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    groups: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    timesteps: Optional[int] = None,
    device: str = "auto",
    n_parallel: int = 1
) -> Tuple[List[ExperimentResult], str]:
    """
    运行所有消融实验

    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        groups: 指定的实验组列表
        experiments: 指定的实验ID列表
        timesteps: 覆盖训练步数
        device: 训练设备
        n_parallel: 并行运行的组数 (默认1=顺序执行, 0=自动检测CPU核心数)

    Returns:
        (所有实验结果列表, 运行时间戳)
    """
    # 加载配置
    config = load_ablation_config(config_path)

    # 覆盖训练步数
    if timesteps:
        config["global"]["training"]["total_timesteps"] = timesteps

    # 设置输出目录
    if output_dir is None:
        output_dir = config.get("global", {}).get("output", {}).get("base_dir", "outputs/ablation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成本次运行的时间戳 (所有实验共享，避免覆盖历史结果)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 确定要运行的组
    if groups is None:
        groups = ["A", "B", "C", "D", "E"]

    # 确定并行度
    if n_parallel == 0:
        # 自动检测: 使用CPU核心数，但不超过组数
        n_parallel = min(multiprocessing.cpu_count(), len(groups))
    n_parallel = min(n_parallel, len(groups))  # 不超过组数

    print(f"\n{'#'*60}")
    print(f"消融实验开始")
    print(f"配置: {config_path or 'configs/ablation_configs.yaml'}")
    print(f"输出目录: {output_dir / run_timestamp}")
    print(f"实验组: {groups}")
    print(f"并行度: {n_parallel} {'(并行)' if n_parallel > 1 else '(顺序)'}")
    print(f"运行时间戳: {run_timestamp}")
    print(f"{'#'*60}")

    start_time = time.time()
    all_results = []

    if n_parallel > 1:
        # 并行执行
        worker_args = [
            (group, config, str(output_dir), experiments, device, run_timestamp)
            for group in groups
        ]

        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(_run_group_worker, args): args[0] for args in worker_args}

            for future in as_completed(futures):
                group_name = futures[future]
                try:
                    group_results = future.result()
                    all_results.extend(group_results)
                except Exception as e:
                    print(f"[错误] 组{group_name}执行失败: {e}")
    else:
        # 顺序执行
        for group in groups:
            print(f"\n[组{group}] 开始...")
            group_results = run_ablation_group(
                group=group,
                config=config,
                output_dir=str(output_dir),
                experiments=experiments,
                device=device,
                run_timestamp=run_timestamp
            )
            all_results.extend(group_results)

    total_time = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"所有实验完成! 总耗时: {total_time/3600:.2f}小时")
    print(f"结果保存到: {output_dir / run_timestamp}")
    print(f"{'#'*60}")

    return all_results, run_timestamp


def generate_summary(output_dir: str, run_timestamp: Optional[str] = None):
    """
    生成汇总报告

    Args:
        output_dir: 输出根目录
        run_timestamp: 运行时间戳 (可选，用于定位特定运行目录)
    """
    print("\n[汇总] 生成汇总报告...")

    summary_gen = AblationSummaryGenerator(output_dir, run_timestamp=run_timestamp)
    summary_gen.load_results_from_dir()
    summary_gen.generate_all()

    print("[汇总] 完成!")


def main():
    parser = argparse.ArgumentParser(
        description="消融实验运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径 (默认: configs/ablation_configs.yaml)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录 (默认: outputs/ablation)"
    )

    parser.add_argument(
        "--groups", "-g",
        type=str,
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        default=None,
        help="指定要运行的实验组"
    )

    parser.add_argument(
        "--experiments", "-e",
        type=str,
        nargs="+",
        default=None,
        help="指定要运行的实验ID (如 A1_16D A2_8D)"
    )

    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=None,
        help="覆盖训练步数"
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="训练设备"
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只生成汇总报告，不运行实验"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=4,
        help="并行运行的组数 (默认4, 设为1则顺序执行, 设为0则自动检测)"
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 确定输出目录
    output_dir = args.output_dir or "outputs/ablation"

    if args.summary_only:
        # 只生成汇总 (自动查找最新的运行目录)
        generate_summary(output_dir)
    else:
        # 运行实验 (默认并行)
        results, run_timestamp = run_all_ablations(
            config_path=args.config,
            output_dir=output_dir,
            groups=args.groups,
            experiments=args.experiments,
            timesteps=args.timesteps,
            device=args.device,
            n_parallel=args.parallel
        )

        # 生成汇总 (使用本次运行的时间戳)
        if results:
            generate_summary(output_dir, run_timestamp=run_timestamp)

    print("\n完成!")


if __name__ == "__main__":
    main()
