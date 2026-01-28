#!/usr/bin/env python3
"""
统一奖励评估脚本
Unified Reward Evaluation for B-Group Ablation

解决问题：B组消融实验中，每个实验使用不同的奖励权重，导致cumulative_reward不可比。
本脚本用baseline(B1_full)的奖励权重重新评估所有B组实验，使结果可比。

用法:
    # 评估所有B组实验
    python examples/evaluate_with_unified_reward.py

    # 指定实验目录
    python examples/evaluate_with_unified_reward.py --dir outputs/ablation/20260128_162201

    # 指定评估episodes数
    python examples/evaluate_with_unified_reward.py --episodes 10
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from simulation.metro_evacuation_env import MetroEvacuationEnv, REWARD_DEFAULTS


# B1_full baseline 奖励权重 (用于统一评估)
BASELINE_REWARD_WEIGHTS = {
    "evac_per_person": 12.0,
    "congestion_penalty": 3.0,
    "time_penalty": 0.2,
    "completion_bonus": 200.0,
    "balance_penalty": 0.8,
    "flow_efficiency_bonus": 1.5,
    "safety_distance_bonus": 0.5,
    "guidance_penalty": 0.3,
    "evacuation_rate_bonus": 2.0,
    "crush_penalty": 10.0,
}


@dataclass
class EpisodeTrajectory:
    """单个episode的轨迹数据"""
    evacuation_ratios: List[float]      # 每步疏散比例
    congestions: List[float]            # 每步拥堵度
    balance_penalties: List[float]      # 每步均衡惩罚
    safety_rewards: List[float]         # 每步安全奖励
    rate_improvements: List[float]      # 每步速率提升
    is_completed: bool                  # 是否完成疏散
    completion_step: int                # 完成步数
    total_steps: int                    # 总步数
    max_steps: int                      # 最大步数


@dataclass
class UnifiedEvalResult:
    """统一评估结果"""
    experiment_id: str
    original_reward_weights: Dict[str, float]

    # 原始累计奖励 (使用实验自己的权重)
    original_cumulative_reward: float
    original_reward_std: float

    # 归一化累计奖励 (使用baseline权重)
    normalized_cumulative_reward: float
    normalized_reward_std: float

    # 行为指标 (与奖励无关)
    evacuation_rate: float
    evacuation_time: float
    max_congestion: float
    exit_balance: float

    n_episodes: int


def compute_unified_reward(
    trajectory: EpisodeTrajectory,
    reward_weights: Dict[str, float]
) -> float:
    """
    使用指定的奖励权重计算轨迹的累计奖励

    Args:
        trajectory: episode轨迹数据
        reward_weights: 奖励权重字典

    Returns:
        累计奖励
    """
    w = reward_weights
    total_reward = 0.0

    for i in range(len(trajectory.evacuation_ratios)):
        step_reward = 0.0

        # 1. 疏散奖励
        step_reward += trajectory.evacuation_ratios[i] * w["evac_per_person"] * 100

        # 2. 拥堵惩罚
        step_reward -= trajectory.congestions[i] * w["congestion_penalty"]

        # 3. 时间惩罚
        step_reward -= w["time_penalty"]

        # 4. 均衡惩罚
        step_reward -= trajectory.balance_penalties[i] * w["balance_penalty"]

        # 5. 安全奖励
        step_reward += trajectory.safety_rewards[i] * w.get("safety_distance_bonus", 0.5)

        # 6. 速率提升奖励
        if trajectory.rate_improvements[i] > 0:
            step_reward += trajectory.rate_improvements[i] * w.get("evacuation_rate_bonus", 2.0) * 100

        total_reward += step_reward

    # 7. 完成奖励
    if trajectory.is_completed:
        total_reward += w["completion_bonus"]
        # 快速完成额外奖励
        time_bonus = max(0, (trajectory.max_steps - trajectory.completion_step) / trajectory.max_steps * 50)
        total_reward += time_bonus

    return total_reward


def collect_trajectory(
    env: MetroEvacuationEnv,
    model: Optional[Any] = None,
    use_guidance: bool = True
) -> EpisodeTrajectory:
    """
    收集单个episode的轨迹数据

    Args:
        env: 环境
        model: PPO模型 (可选)
        use_guidance: 是否使用引导

    Returns:
        轨迹数据
    """
    obs, info = env.reset()
    done = False
    truncated = False

    evacuation_ratios = []
    congestions = []
    balance_penalties = []
    safety_rewards = []
    rate_improvements = []

    last_evacuated = 0
    evacuation_buffer = [0.0, 0.0, 0.0]

    step = 0
    is_completed = False
    completion_step = env.max_steps

    while not (done or truncated):
        # 选择动作
        if model is not None and use_guidance:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        # 收集轨迹数据
        # 1. 疏散比例
        new_evacuated = env.evacuated_count - last_evacuated
        evac_ratio = new_evacuated / max(env.n_pedestrians, 1)
        evacuation_ratios.append(evac_ratio)
        last_evacuated = env.evacuated_count

        # 更新速率缓冲
        evacuation_buffer.pop(0)
        evacuation_buffer.append(evac_ratio)

        # 2. 拥堵度
        total_congestion = 0
        for exit_obj in env.exits:
            _, congestion = env._compute_exit_metrics(exit_obj)
            total_congestion += congestion
        congestions.append(total_congestion)

        # 3. 均衡惩罚
        counts = list(env.evacuated_by_exit.values())
        total_evacuated = sum(counts)
        if total_evacuated > 0:
            mean_count = total_evacuated / 3
            if mean_count > 0:
                std_count = np.sqrt(sum((c - mean_count) ** 2 for c in counts) / 3)
                cv = std_count / mean_count
                balance_penalty = min(cv, 1.0)
            else:
                balance_penalty = 0.0
        else:
            balance_penalty = 0.0
        balance_penalties.append(balance_penalty)

        # 4. 安全奖励
        safety_reward = env._compute_safety_distance_reward()
        safety_rewards.append(safety_reward)

        # 5. 速率提升
        if len(evacuation_buffer) >= 2:
            rate_improvement = evacuation_buffer[-1] - evacuation_buffer[-2]
        else:
            rate_improvement = 0.0
        rate_improvements.append(rate_improvement)

        # 检查完成
        if env._get_remaining_count() == 0 and not is_completed:
            is_completed = True
            completion_step = step

    return EpisodeTrajectory(
        evacuation_ratios=evacuation_ratios,
        congestions=congestions,
        balance_penalties=balance_penalties,
        safety_rewards=safety_rewards,
        rate_improvements=rate_improvements,
        is_completed=is_completed,
        completion_step=completion_step,
        total_steps=step,
        max_steps=env.max_steps
    )


def evaluate_experiment_unified(
    exp_id: str,
    exp_dir: Path,
    n_episodes: int = 5,
    seed: int = 42
) -> Optional[UnifiedEvalResult]:
    """
    使用统一奖励函数评估单个实验

    Args:
        exp_id: 实验ID
        exp_dir: 实验目录
        n_episodes: 评估episode数
        seed: 随机种子

    Returns:
        统一评估结果
    """
    print(f"\n评估实验: {exp_id}")

    # 加载实验配置 (支持 config.yaml 或 experiment_config.json)
    config_file = exp_dir / "config.yaml"
    if not config_file.exists():
        config_file = exp_dir / "experiment_config.json"
    if not config_file.exists():
        print(f"  配置文件不存在: {exp_dir}")
        return None

    with open(config_file, 'r') as f:
        if config_file.suffix == '.yaml':
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # 获取原始奖励权重
    original_weights = config.get("reward_weights", REWARD_DEFAULTS.copy())

    # 创建环境
    env = MetroEvacuationEnv(
        n_pedestrians=80,
        max_steps=600,
        reward_weights=original_weights  # 使用原始权重创建环境
    )

    # 加载模型 (如果存在)
    model = None
    model_path = exp_dir / "model.zip"
    if model_path.exists() and SB3_AVAILABLE:
        try:
            model = PPO.load(str(model_path))
            print(f"  加载模型: {model_path}")
        except Exception as e:
            print(f"  模型加载失败: {e}")

    # 收集轨迹并计算奖励
    original_rewards = []
    normalized_rewards = []
    evacuation_rates = []
    evacuation_times = []
    max_congestions = []
    exit_balances = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)

        # 收集轨迹
        trajectory = collect_trajectory(env, model, use_guidance=(model is not None))

        # 计算原始奖励 (使用实验自己的权重)
        original_reward = compute_unified_reward(trajectory, original_weights)
        original_rewards.append(original_reward)

        # 计算归一化奖励 (使用baseline权重)
        normalized_reward = compute_unified_reward(trajectory, BASELINE_REWARD_WEIGHTS)
        normalized_rewards.append(normalized_reward)

        # 行为指标
        evacuation_rates.append(env.evacuated_count / env.n_pedestrians)
        evacuation_times.append(trajectory.total_steps)
        max_congestions.append(max(trajectory.congestions) if trajectory.congestions else 0)

        counts = list(env.evacuated_by_exit.values())
        if sum(counts) > 0:
            exit_balances.append(np.std(counts))
        else:
            exit_balances.append(0)

    return UnifiedEvalResult(
        experiment_id=exp_id,
        original_reward_weights=original_weights,
        original_cumulative_reward=np.mean(original_rewards),
        original_reward_std=np.std(original_rewards),
        normalized_cumulative_reward=np.mean(normalized_rewards),
        normalized_reward_std=np.std(normalized_rewards),
        evacuation_rate=np.mean(evacuation_rates),
        evacuation_time=np.mean(evacuation_times),
        max_congestion=np.mean(max_congestions),
        exit_balance=np.mean(exit_balances),
        n_episodes=n_episodes
    )


def find_b_group_experiments(base_dir: Path) -> List[Path]:
    """查找所有B组实验目录"""
    b_experiments = []

    # 遍历所有时间戳目录
    for timestamp_dir in base_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        # 查找B组实验
        for exp_dir in timestamp_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("B"):
                b_experiments.append(exp_dir)

    return sorted(b_experiments)


def generate_unified_report(
    results: List[UnifiedEvalResult],
    output_path: Path
):
    """生成统一评估报告"""

    report = {
        "generated_at": datetime.now().isoformat(),
        "baseline_weights": BASELINE_REWARD_WEIGHTS,
        "description": "B组消融实验统一奖励评估结果",
        "methodology": "使用B1_full的奖励权重重新计算所有B组实验的累计奖励，使结果可比",
        "results": [asdict(r) for r in results],
        "ranking": {
            "by_normalized_reward": [],
            "by_evacuation_rate": [],
            "by_evacuation_time": []
        }
    }

    # 按归一化奖励排序
    sorted_by_reward = sorted(results, key=lambda x: x.normalized_cumulative_reward, reverse=True)
    report["ranking"]["by_normalized_reward"] = [
        {"rank": i+1, "exp_id": r.experiment_id, "normalized_reward": r.normalized_cumulative_reward}
        for i, r in enumerate(sorted_by_reward)
    ]

    # 按疏散率排序
    sorted_by_rate = sorted(results, key=lambda x: x.evacuation_rate, reverse=True)
    report["ranking"]["by_evacuation_rate"] = [
        {"rank": i+1, "exp_id": r.experiment_id, "evacuation_rate": r.evacuation_rate}
        for i, r in enumerate(sorted_by_rate)
    ]

    # 按疏散时间排序 (越小越好)
    sorted_by_time = sorted(results, key=lambda x: x.evacuation_time)
    report["ranking"]["by_evacuation_time"] = [
        {"rank": i+1, "exp_id": r.experiment_id, "evacuation_time": r.evacuation_time}
        for i, r in enumerate(sorted_by_time)
    ]

    # 保存JSON报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("B组统一奖励评估摘要")
    print("="*60)

    print("\n按归一化累计奖励排序 (使用B1_full权重):")
    print("-" * 50)
    for item in report["ranking"]["by_normalized_reward"]:
        print(f"  {item['rank']}. {item['exp_id']}: {item['normalized_reward']:.2f}")

    print("\n按疏散率排序:")
    print("-" * 50)
    for item in report["ranking"]["by_evacuation_rate"]:
        print(f"  {item['rank']}. {item['exp_id']}: {item['evacuation_rate']:.2%}")

    print("\n按疏散时间排序 (越小越好):")
    print("-" * 50)
    for item in report["ranking"]["by_evacuation_time"]:
        print(f"  {item['rank']}. {item['exp_id']}: {item['evacuation_time']:.1f}步")


def main():
    parser = argparse.ArgumentParser(description="B组消融实验统一奖励评估")
    parser.add_argument("--dir", "-d", type=str, default="outputs/ablation",
                        help="消融实验输出目录")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="每个实验的评估episode数")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return

    # 查找B组实验
    b_experiments = find_b_group_experiments(base_dir)
    print(f"找到 {len(b_experiments)} 个B组实验")

    if not b_experiments:
        print("未找到B组实验，请先运行消融实验")
        return

    # 评估所有实验
    results = []
    for exp_dir in b_experiments:
        exp_id = exp_dir.name
        result = evaluate_experiment_unified(
            exp_id=exp_id,
            exp_dir=exp_dir,
            n_episodes=args.episodes,
            seed=args.seed
        )
        if result:
            results.append(result)

    if not results:
        print("没有成功评估的实验")
        return

    # 生成报告
    output_path = base_dir / "unified_reward_evaluation.json"
    generate_unified_report(results, output_path)


if __name__ == "__main__":
    main()
