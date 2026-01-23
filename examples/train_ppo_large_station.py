"""
训练 PPO 模型优化成都东站大型地铁站疏散策略
T形布局: 150m × 80m
支持三种人流量等级: small/medium/large

训练策略:
1. 使用小流量(small)进行快速训练
2. 中流量(medium)验证和微调
3. 大流量(large)最终测试

支持GPU加速:
- macOS: MPS (Apple Silicon)
- Linux/Windows: CUDA (NVIDIA GPU)
- 无GPU时自动回退到CPU
"""

import sys
import platform
from pathlib import Path
from typing import List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# macOS multiprocessing 设置 (必须在导入其他模块之前)
if platform.system() == "Darwin":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from simulation.large_station_env import LargeStationEnv


def get_device():
    """自动检测最佳设备"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用 NVIDIA GPU: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("使用 CPU")
    return device


class LargeStationTrainingCallback(BaseCallback):
    """训练回调，记录训练过程"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.evacuated_counts = []
        self.episode_times = []
        self.exit_distributions = []
        self.danger_counts = []  # 记录踩踏风险

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]
            self.evacuated_counts.append(info.get('evacuated', 0))
            self.episode_times.append(info.get('time', 0))
            self.danger_counts.append(info.get('danger_count', 0))

            # 记录出口分布
            exit_dist = info.get('evacuated_by_exit', {})
            self.exit_distributions.append(exit_dist.copy())

        return True


def train_ppo_large_station(
    flow_level: str = "small",
    n_envs: int = 4,
    ppo_device: str = "cpu",
    total_timesteps: int = 500000,
    show_plots: bool = True,
    save_model: bool = True,
):
    """训练大型地铁站场景PPO模型

    Args:
        flow_level: 人流量等级 ("small", "medium", "large")
        n_envs: 并行环境数量
        ppo_device: PPO训练设备
        total_timesteps: 总训练步数
        show_plots: 是否显示图表
        save_model: 是否保存模型
    """
    print("=" * 70)
    print("成都东站大型地铁站 PPO 疏散策略优化训练")
    print("=" * 70)

    # 人流量配置
    flow_configs = {
        "small": {"upper": 500, "lower": 500, "total": 1000},
        "medium": {"upper": 1000, "lower": 1000, "total": 2000},
        "large": {"upper": 1500, "lower": 1500, "total": 3000},
    }
    flow_config = flow_configs.get(flow_level, flow_configs["small"])

    print(f"\n场景参数:")
    print(f"  - T形布局: 150m × 80m")
    print(f"  - 出口数量: 8 (闸机)")
    print(f"  - 扶梯涌入点: 3")
    print(f"  - 人流量等级: {flow_level}")
    print(f"  - 上层初始: {flow_config['upper']}人")
    print(f"  - 下层涌入: {flow_config['lower']}人")
    print(f"  - 总人数: {flow_config['total']}人")

    # 创建环境
    print(f"\n[1/4] 创建环境...")

    def make_env(rank: int, flow_level: str):
        def _init():
            env = LargeStationEnv(
                flow_level=flow_level,
                scene_size=(150.0, 80.0),  # T形布局
                max_steps=6000,  # 600秒 / 0.1秒
                dt=0.1,
                enable_enhanced_behaviors=True,
                emergency_mode=True,
            )
            env.reset(seed=rank)
            return env
        return _init

    # 创建评估环境
    eval_env = make_env(0, flow_level)()

    # 创建训练环境
    print(f"  创建 {n_envs} 个训练环境...")
    if n_envs > 1:
        try:
            vec_env = SubprocVecEnv([make_env(i, flow_level) for i in range(n_envs)])
            print(f"  成功创建 SubprocVecEnv ({n_envs} 进程)")
        except Exception as e:
            print(f"  警告: SubprocVecEnv 失败 ({type(e).__name__}), 回退到 DummyVecEnv")
            vec_env = DummyVecEnv([make_env(i, flow_level) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, flow_level)])

    # 创建PPO模型
    print(f"\n[2/4] 创建 PPO 模型...")
    print(f"  - 观测空间: 34维")
    print(f"    [0-7]   8个出口密度")
    print(f"    [8-15]  8个出口拥堵度")
    print(f"    [16-23] 8个出口人流方向占比")
    print(f"    [24-28] 5步历史疏散速率")
    print(f"    [29-31] 3个扶梯涌入点密度")
    print(f"    [32-33] 剩余比例 + 时间比例")
    print(f"  - 动作空间: Discrete(8) (选择推荐出口)")
    print(f"  - 训练设备: {ppo_device.upper()}")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=ppo_device
    )

    # 训练
    print(f"\n[3/4] 开始训练 ({total_timesteps}步)...")
    callback = LargeStationTrainingCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False
    )

    # 保存模型
    if save_model:
        print(f"\n[4/4] 保存模型...")
        model_dir = project_root / "outputs" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"ppo_large_station_{flow_level}"
        model.save(str(model_path))
        print(f"模型已保存: {model_path}.zip")

    # 评估和可视化
    evaluate_and_visualize(model, eval_env, callback, project_root, flow_level, show_plots)

    return model


def evaluate_and_visualize(model, env, callback, project_root, flow_level, show_plots=True):
    """评估模型并可视化结果"""
    print("\n" + "=" * 70)
    print("评估训练结果")
    print("=" * 70)

    test_episodes = 5
    test_rewards = []
    test_evacuated = []
    test_times = []
    test_danger_counts = []

    for ep in range(test_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        test_rewards.append(total_reward)
        test_evacuated.append(info['evacuated'])
        test_times.append(info['time'])
        test_danger_counts.append(info.get('danger_count', 0))

        print(f"  Episode {ep+1}: "
              f"Reward={total_reward:.1f}, "
              f"Evacuated={info['evacuated']}/{env.n_pedestrians}, "
              f"Time={info['time']:.1f}s, "
              f"Danger={info.get('danger_count', 0)}")

    print(f"\n平均结果:")
    print(f"  平均奖励: {np.mean(test_rewards):.1f} +/- {np.std(test_rewards):.1f}")
    print(f"  平均疏散: {np.mean(test_evacuated):.1f} +/- {np.std(test_evacuated):.1f}")
    print(f"  平均时间: {np.mean(test_times):.1f}s +/- {np.std(test_times):.1f}s")
    print(f"  平均危险数: {np.mean(test_danger_counts):.1f}")

    # 检查是否达到10分钟目标
    target_time = 600  # 10分钟
    avg_time = np.mean(test_times)
    if avg_time <= target_time:
        print(f"\n  ✓ 达到10分钟疏散目标! (平均 {avg_time:.1f}s)")
    else:
        print(f"\n  ✗ 未达到10分钟目标 (平均 {avg_time:.1f}s > 600s)")

    # 可视化
    if not show_plots and platform.system() == "Windows":
        plt.switch_backend("Agg")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练过程疏散人数
    ax1 = axes[0, 0]
    if callback.evacuated_counts:
        ax1.plot(callback.evacuated_counts, alpha=0.7, color='steelblue')
        ax1.axhline(y=env.n_pedestrians, color='r', linestyle='--',
                   label=f'Target ({env.n_pedestrians})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Evacuated Count')
        ax1.set_title('Training Progress: Evacuation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 训练过程疏散时间
    ax2 = axes[0, 1]
    if callback.episode_times:
        ax2.plot(callback.episode_times, alpha=0.7, color='green')
        ax2.axhline(y=600, color='r', linestyle='--', label='Target (600s)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Evacuation Time (s)')
        ax2.set_title('Training Progress: Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 测试结果
    ax3 = axes[1, 0]
    x = np.arange(test_episodes)
    ax3.bar(x, test_evacuated, color='steelblue', alpha=0.7)
    ax3.axhline(y=env.n_pedestrians, color='r', linestyle='--',
               label=f'Target ({env.n_pedestrians})')
    ax3.set_xlabel('Test Episode')
    ax3.set_ylabel('Evacuated Count')
    ax3.set_title('Test Results: Evacuated')
    ax3.legend()
    ax3.set_xticks(x)

    # 4. 危险事件
    ax4 = axes[1, 1]
    if callback.danger_counts:
        ax4.plot(callback.danger_counts, alpha=0.7, color='red')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Danger Count')
        ax4.set_title('Training Progress: Safety (Crush Risk)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    fig_dir = project_root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"ppo_large_station_{flow_level}_training.png"
    plt.savefig(str(fig_path), dpi=150)
    print(f"\n训练结果图已保存: {fig_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def scale_test(model_path: str, flow_levels: List[str] = None):
    """跨规模测试: 用小规模训练的模型测试大规模场景

    Args:
        model_path: 模型路径
        flow_levels: 要测试的人流量等级列表
    """
    if flow_levels is None:
        flow_levels = ["small", "medium", "large"]

    print("=" * 70)
    print("跨规模泛化性测试")
    print("=" * 70)

    model = PPO.load(model_path)

    results = {}

    for level in flow_levels:
        print(f"\n测试 {level} 流量...")
        env = LargeStationEnv(
            flow_level=level,
            emergency_mode=True,
        )

        test_evacuated = []
        test_times = []
        test_danger = []

        for _ in range(3):  # 每个等级测试3次
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            test_evacuated.append(info['evacuated'])
            test_times.append(info['time'])
            test_danger.append(info.get('danger_count', 0))

        results[level] = {
            'evacuated': np.mean(test_evacuated),
            'time': np.mean(test_times),
            'danger': np.mean(test_danger),
            'total': env.n_pedestrians,
        }

        print(f"  {level}: 疏散={results[level]['evacuated']:.0f}/{results[level]['total']}, "
              f"时间={results[level]['time']:.1f}s, 危险={results[level]['danger']:.1f}")

    print("\n跨规模测试总结:")
    for level, r in results.items():
        evacuation_rate = r['evacuated'] / r['total'] * 100
        within_target = "✓" if r['time'] <= 600 else "✗"
        print(f"  {level}: {evacuation_rate:.1f}% 疏散, {r['time']:.1f}s {within_target}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练大型地铁站场景PPO模型")
    parser.add_argument("--flow-level", default="small",
                        choices=["small", "medium", "large"],
                        help="人流量等级")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数量")
    parser.add_argument("--ppo-device", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"],
                        help="PPO 训练设备")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="总训练步数")
    parser.add_argument("--no-show", action="store_true",
                        help="保存图像但不显示")
    parser.add_argument("--scale-test", type=str, default=None,
                        help="跨规模测试模型路径")
    args = parser.parse_args()

    if args.scale_test:
        scale_test(args.scale_test)
    else:
        if args.ppo_device == "auto":
            args.ppo_device = get_device()

        train_ppo_large_station(
            flow_level=args.flow_level,
            n_envs=args.n_envs,
            ppo_device=args.ppo_device,
            total_timesteps=args.total_timesteps,
            show_plots=not args.no_show,
        )
