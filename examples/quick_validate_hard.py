#!/usr/bin/env python3
"""
快速验证脚本 - 困难模式

增加难度，让随机选择无法达到100%疏散，验证PPO的真正价值

难度调整：
- 行人数: 30 -> 60 (制造拥堵)
- 最大步数: 400 -> 250 (时间压力)
"""

import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from simulation.metro_evacuation_env import MetroEvacuationEnv


# ============================================================
# 困难模式配置
# ============================================================
HARD_CONFIG = {
    # 增加难度
    "n_pedestrians": 60,        # 更多行人 -> 拥堵
    "max_steps": 250,           # 更少时间 -> 压力

    # 训练参数
    "total_timesteps": 30000,
    "n_steps": 512,
    "batch_size": 64,

    # 评估
    "n_eval_episodes": 5,

    # GPU加速SFM（MPS/CUDA）
    "use_gpu_sfm": True,        # 启用GPU加速SFM仿真
}


class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=2000):
        super().__init__()
        self.check_freq = check_freq
        self.rewards = []
        self.best_reward = -float('inf')

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                self.rewards.append(mean_reward)
                marker = " *" if mean_reward > self.best_reward else ""
                self.best_reward = max(self.best_reward, mean_reward)
                print(f"  步数: {self.n_calls:5d} | 奖励: {mean_reward:7.1f}{marker}")
        return True


def create_env(n_peds, max_steps, use_gpu_sfm=False):
    return MetroEvacuationEnv(
        n_pedestrians=n_peds,
        max_steps=max_steps,
        dt=0.1,
        enable_enhanced_behaviors=True,
        enable_neural_prediction=False,
        use_gpu_sfm=use_gpu_sfm,
        sfm_device="auto",  # 自动选择MPS/CUDA/CPU
    )


def evaluate(model, env, n_episodes, use_model=True, verbose=False):
    results = {"evac_rates": [], "steps": [], "rewards": []}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        done, truncated = False, False
        total_reward, steps = 0, 0

        while not (done or truncated):
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        evac_rate = info.get("evacuated", 0) / env.n_pedestrians
        results["evac_rates"].append(evac_rate)
        results["steps"].append(steps)
        results["rewards"].append(total_reward)

        if verbose:
            print(f"    Episode {ep+1}: 疏散率={evac_rate*100:.1f}%, 步数={steps}")

    return results


def main():
    # 设置日志输出
    output_dir = PROJECT_ROOT / "outputs" / "quick_validate"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"hard_mode_{timestamp}.txt"
    logger = Logger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("PPO引导系统 - 困难模式验证")
    print(f"日志文件: {log_path}")
    print("=" * 60)
    print(f"\n配置 (增加难度):")
    print(f"  - 行人数: {HARD_CONFIG['n_pedestrians']} (vs 简单模式30)")
    print(f"  - 最大步数: {HARD_CONFIG['max_steps']} (vs 简单模式400)")
    print(f"  - 训练步数: {HARD_CONFIG['total_timesteps']}")
    print(f"  - GPU加速SFM: {HARD_CONFIG['use_gpu_sfm']}")
    print()

    start_time = time.time()

    # 1. 创建环境
    print("[1/5] 创建环境...")
    env = create_env(HARD_CONFIG["n_pedestrians"], HARD_CONFIG["max_steps"],
                     use_gpu_sfm=HARD_CONFIG["use_gpu_sfm"])
    vec_env = DummyVecEnv([lambda: env])
    print("  完成")

    # 2. 先测试随机基线
    print("\n[2/5] 测试随机基线 (无引导)...")
    eval_env = create_env(HARD_CONFIG["n_pedestrians"], HARD_CONFIG["max_steps"],
                          use_gpu_sfm=HARD_CONFIG["use_gpu_sfm"])
    random_results = evaluate(None, eval_env, HARD_CONFIG["n_eval_episodes"], use_model=False, verbose=True)
    random_evac = np.mean(random_results["evac_rates"]) * 100
    print(f"  随机基线疏散率: {random_evac:.1f}%")

    if random_evac >= 95:
        print("\n  ⚠️ 警告: 随机基线疏散率过高，场景可能仍然太简单")
        print("  建议: 进一步增加行人数或减少步数")

    # 3. 训练PPO
    print(f"\n[3/5] 训练PPO ({HARD_CONFIG['total_timesteps']}步)...")
    # PPO用CPU（MLP太小，GPU反而慢）
    ppo_device = "cpu"
    print("  PPO使用CPU（MLP策略在CPU上更快）")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=HARD_CONFIG["n_steps"],
        batch_size=HARD_CONFIG["batch_size"],
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        device=ppo_device
    )

    callback = ProgressCallback(check_freq=3000)
    train_start = time.time()
    model.learn(total_timesteps=HARD_CONFIG["total_timesteps"], callback=callback)
    train_time = time.time() - train_start
    print(f"  训练完成! 耗时: {train_time:.1f}秒")

    # 4. 评估PPO
    print("\n[4/5] 评估PPO引导...")
    ppo_results = evaluate(model, eval_env, HARD_CONFIG["n_eval_episodes"], use_model=True, verbose=True)
    ppo_evac = np.mean(ppo_results["evac_rates"]) * 100

    # 5. 结果分析
    print("\n[5/5] 结果对比")
    print("=" * 60)

    improvement = ppo_evac - random_evac

    print(f"\n{'指标':<15} {'PPO引导':<12} {'无引导':<12} {'差异':<10}")
    print("-" * 50)
    print(f"{'疏散率':<15} {ppo_evac:.1f}%{'':<7} {random_evac:.1f}%{'':<7} {improvement:+.1f}%")
    print(f"{'平均步数':<15} {np.mean(ppo_results['steps']):.0f}{'':<9} {np.mean(random_results['steps']):.0f}")
    print(f"{'平均奖励':<15} {np.mean(ppo_results['rewards']):.1f}{'':<6} {np.mean(random_results['rewards']):.1f}")

    total_time = time.time() - start_time

    # 结论
    print("\n" + "=" * 60)
    print("验证结论:")
    print("=" * 60)

    if random_evac < 85:
        print(f"✓ 场景难度合适 (随机基线: {random_evac:.1f}%)")
    else:
        print(f"△ 场景可能仍然较简单 (随机基线: {random_evac:.1f}%)")

    if improvement > 5:
        conclusion = "PASS"
        print(f"✓ PPO引导有效! 疏散率提升 {improvement:.1f}%")
    elif improvement > 0:
        conclusion = "MARGINAL"
        print(f"△ PPO略有效果 (+{improvement:.1f}%)")
    else:
        conclusion = "FAIL"
        print(f"✗ PPO效果不佳 ({improvement:.1f}%)")

    print(f"\n总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

    # 保存结果
    output_dir = PROJECT_ROOT / "outputs" / "quick_validate"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "config": HARD_CONFIG,
        "results": {
            "ppo_evac_rate": ppo_evac,
            "random_evac_rate": random_evac,
            "improvement": improvement,
            "ppo_steps": float(np.mean(ppo_results["steps"])),
            "random_steps": float(np.mean(random_results["steps"])),
            "train_time_sec": train_time,
            "total_time_sec": total_time,
        },
        "conclusion": conclusion,
    }

    result_path = output_dir / "hard_mode_result.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n结果已保存: {result_path}")

    model_path = output_dir / "ppo_hard.zip"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")

    print(f"\n日志已保存: {log_path}")

    # 关闭日志
    sys.stdout = logger.terminal
    logger.close()

    return conclusion


if __name__ == "__main__":
    main()
