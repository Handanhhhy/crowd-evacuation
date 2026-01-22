#!/usr/bin/env python3
"""
快速验证脚本 - 20-40分钟内验证PPO引导系统可行性

验证内容:
1. PPO是否能学到有意义的策略
2. 有引导 vs 无引导的疏散效率对比
3. 训练曲线是否收敛

用法:
    python examples/quick_validate.py
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch


class Logger:
    """同时输出到终端和文件"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("错误: 需要安装 stable-baselines3")
    print("运行: pip install stable-baselines3")
    sys.exit(1)

from simulation.metro_evacuation_env import MetroEvacuationEnv
from sfm.social_force import PedestrianType


# ============================================================
# 快速验证配置 (针对30分钟内完成)
# ============================================================
QUICK_CONFIG = {
    # 环境参数 - 减少行人数
    "n_pedestrians": 30,        # 从80减少到30
    "max_steps": 400,           # 从800减少到400

    # 训练参数 - 减少训练量
    "total_timesteps": 20000,   # 从100k减少到20k
    "n_steps": 512,             # 从1024减少到512
    "batch_size": 64,           # 从128减少到64

    # 评估参数
    "n_eval_episodes": 3,       # 从10减少到3
}


class SimpleCallback(BaseCallback):
    """简单的训练回调，显示进度"""

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                print(f"  步数: {self.n_calls:5d} | 平均奖励: {mean_reward:.1f}")
        return True


def create_quick_env(n_peds=30, max_steps=400):
    """创建快速验证环境"""
    env = MetroEvacuationEnv(
        n_pedestrians=n_peds,
        max_steps=max_steps,
        dt=0.1,
        enable_enhanced_behaviors=True,
        enable_neural_prediction=False,  # 禁用神经网络预测加速
    )
    return env


def evaluate_policy(model, env, n_episodes=3, use_model=True):
    """评估策略"""
    results = {
        "evacuation_rates": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        evac_rate = info.get("evacuated", 0) / env.n_pedestrians
        results["evacuation_rates"].append(evac_rate)
        results["episode_lengths"].append(steps)
        results["rewards"].append(total_reward)

    return results


def run_quick_validation():
    """运行快速验证"""
    # 设置日志输出
    output_dir = PROJECT_ROOT / "outputs" / "quick_validate"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"simple_mode_{timestamp}.txt"
    logger = Logger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("PPO引导系统 - 快速可行性验证")
    print(f"日志文件: {log_path}")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 行人数: {QUICK_CONFIG['n_pedestrians']}")
    print(f"  - 训练步数: {QUICK_CONFIG['total_timesteps']}")
    print(f"  - 预计时间: 20-40分钟")
    print()

    start_time = time.time()

    # ========== 第1步: 创建环境 ==========
    print("[1/4] 创建环境...")
    env = create_quick_env(
        n_peds=QUICK_CONFIG["n_pedestrians"],
        max_steps=QUICK_CONFIG["max_steps"]
    )
    vec_env = DummyVecEnv([lambda: env])
    print("  环境创建成功")
    print()

    # ========== 第2步: 训练PPO ==========
    print("[2/4] 训练PPO模型...")
    print(f"  总步数: {QUICK_CONFIG['total_timesteps']}")

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

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=QUICK_CONFIG["n_steps"],
        batch_size=QUICK_CONFIG["batch_size"],
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        device=ppo_device
    )

    callback = SimpleCallback(check_freq=2000)

    train_start = time.time()
    model.learn(
        total_timesteps=QUICK_CONFIG["total_timesteps"],
        callback=callback
    )
    train_time = time.time() - train_start

    print(f"\n  训练完成! 耗时: {train_time:.1f}秒 ({train_time/60:.1f}分钟)")
    print()

    # ========== 第3步: 评估对比 ==========
    print("[3/4] 评估对比...")
    eval_env = create_quick_env(
        n_peds=QUICK_CONFIG["n_pedestrians"],
        max_steps=QUICK_CONFIG["max_steps"]
    )

    # 有PPO引导
    print("  评估: PPO引导...")
    ppo_results = evaluate_policy(
        model, eval_env,
        n_episodes=QUICK_CONFIG["n_eval_episodes"],
        use_model=True
    )

    # 无引导（随机）
    print("  评估: 无引导（随机）...")
    random_results = evaluate_policy(
        model, eval_env,
        n_episodes=QUICK_CONFIG["n_eval_episodes"],
        use_model=False
    )
    print()

    # ========== 第4步: 结果分析 ==========
    print("[4/4] 结果分析")
    print("=" * 60)

    ppo_evac = np.mean(ppo_results["evacuation_rates"]) * 100
    random_evac = np.mean(random_results["evacuation_rates"]) * 100
    improvement = ppo_evac - random_evac

    ppo_steps = np.mean(ppo_results["episode_lengths"])
    random_steps = np.mean(random_results["episode_lengths"])

    print(f"\n{'指标':<20} {'PPO引导':<15} {'无引导':<15} {'差异':<10}")
    print("-" * 60)
    print(f"{'疏散率':<20} {ppo_evac:.1f}%{'':<10} {random_evac:.1f}%{'':<10} {improvement:+.1f}%")
    print(f"{'平均步数':<20} {ppo_steps:.0f}{'':<12} {random_steps:.0f}{'':<12} {ppo_steps-random_steps:+.0f}")
    print(f"{'平均奖励':<20} {np.mean(ppo_results['rewards']):.1f}{'':<10} {np.mean(random_results['rewards']):.1f}")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

    # ========== 结论 ==========
    print("\n" + "=" * 60)
    print("验证结论:")
    print("=" * 60)

    if improvement > 5:
        print(f"✓ PPO引导有效! 疏散率提升 {improvement:.1f}%")
        print("  建议: 可以继续增加训练步数优化效果")
        conclusion = "PASS"
    elif improvement > 0:
        print(f"△ PPO引导略有效果 (+{improvement:.1f}%)")
        print("  建议: 需要更多训练或调整奖励函数")
        conclusion = "MARGINAL"
    else:
        print(f"✗ PPO引导效果不明显 ({improvement:.1f}%)")
        print("  建议: 检查奖励函数设计或观测空间")
        conclusion = "FAIL"

    # 检查训练是否收敛
    if len(callback.rewards) >= 3:
        early_reward = np.mean(callback.rewards[:2])
        late_reward = np.mean(callback.rewards[-2:])
        if late_reward > early_reward:
            print(f"✓ 训练在收敛 (奖励: {early_reward:.1f} -> {late_reward:.1f})")
        else:
            print(f"△ 训练可能不稳定，建议增加步数")

    print("\n" + "=" * 60)

    # 保存模型
    model_path = output_dir / "ppo_quick.zip"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")

    # 保存JSON结果
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "config": QUICK_CONFIG,
        "results": {
            "ppo_evac_rate": ppo_evac,
            "random_evac_rate": random_evac,
            "improvement": improvement,
            "ppo_steps": float(ppo_steps),
            "random_steps": float(random_steps),
            "train_time_sec": train_time,
            "total_time_sec": total_time,
        },
        "conclusion": conclusion,
    }
    result_path = output_dir / "simple_mode_result.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"结果已保存: {result_path}")

    print(f"日志已保存: {log_path}")

    # 关闭日志
    sys.stdout = logger.terminal
    logger.close()

    return conclusion, {
        "ppo_evac_rate": ppo_evac,
        "random_evac_rate": random_evac,
        "improvement": improvement,
        "train_time": train_time,
        "total_time": total_time,
    }


if __name__ == "__main__":
    conclusion, results = run_quick_validation()

    print("\n下一步建议:")
    if conclusion == "PASS":
        print("  1. 增加训练到 50k-100k 步")
        print("  2. 增加行人数到 50-80 人")
        print("  3. 运行完整消融实验")
    elif conclusion == "MARGINAL":
        print("  1. 调整奖励函数权重")
        print("  2. 增加训练步数到 50k")
        print("  3. 检查观测空间是否包含足够信息")
    else:
        print("  1. 检查奖励函数设计")
        print("  2. 简化问题（减少出口数）")
        print("  3. 验证环境是否正常工作")
