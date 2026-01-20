"""
训练 PPO 模型优化地铁站疏散策略
适配成都东客站3出口场景
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from simulation.metro_evacuation_env import MetroEvacuationEnv


class MetroTrainingCallback(BaseCallback):
    """训练回调，记录训练过程"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.evacuated_counts = []
        self.exit_distributions = []  # 记录各出口疏散分布

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]
            self.evacuated_counts.append(info.get('evacuated', 0))

            # 记录各出口分布
            exit_dist = info.get('evacuated_by_exit', {'A': 0, 'B': 0, 'C': 0})
            self.exit_distributions.append(exit_dist.copy())

        return True


def train_ppo_metro():
    """训练地铁站场景PPO模型"""

    print("=" * 60)
    print("成都东客站地铁出站口 PPO 疏散策略优化训练")
    print("=" * 60)
    print("\n场景参数:")
    print("  - 场景尺寸: 60×40 米")
    print("  - 出口数量: 3 (A、B、C)")
    print("  - 行人数量: 80")
    print("  - 闸机通道: 5")
    print("  - 柱子: 6")

    # 创建环境
    print("\n[1/4] 创建地铁站环境...")
    env = MetroEvacuationEnv(
        n_pedestrians=80,
        scene_size=(60.0, 40.0),
        max_steps=800,
        dt=0.1
    )

    # 包装为向量环境
    vec_env = DummyVecEnv([lambda: env])

    # 创建 PPO 模型
    print("\n[2/4] 创建 PPO 模型...")
    print("  - 观测空间: 8维 (3出口密度 + 3出口拥堵度 + 剩余比例 + 时间比例)")
    print("  - 动作空间: Discrete(3) (选择推荐出口A/B/C)")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,        # 增加采样步数
        batch_size=128,      # 增加批次大小
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # 熵系数，鼓励探索
        verbose=1,
        device="cpu"
    )

    # 训练
    print("\n[3/4] 开始训练 (50000步)...")
    callback = MetroTrainingCallback()

    total_timesteps = 50000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False
    )

    # 保存模型
    print("\n[4/4] 保存模型...")
    model_dir = project_root / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppo_metro"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}.zip")

    # 评估和可视化
    evaluate_and_visualize(model, env, callback, project_root)

    return model


def evaluate_and_visualize(model, env, callback, project_root):
    """评估模型并可视化结果"""

    print("\n" + "=" * 60)
    print("评估训练结果")
    print("=" * 60)

    # 运行测试 episode
    test_episodes = 10
    test_rewards = []
    test_evacuated = []
    test_steps = []
    test_exit_dists = []

    for ep in range(test_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        test_rewards.append(total_reward)
        test_evacuated.append(info['evacuated'])
        test_steps.append(steps)
        test_exit_dists.append(info['evacuated_by_exit'])

        print(f"  Episode {ep+1}: Reward={total_reward:.1f}, "
              f"Evacuated={info['evacuated']}/{env.n_pedestrians}, "
              f"Steps={steps}, "
              f"分布=A:{info['evacuated_by_exit']['A']} B:{info['evacuated_by_exit']['B']} C:{info['evacuated_by_exit']['C']}")

    print(f"\n平均结果:")
    print(f"  平均奖励: {np.mean(test_rewards):.1f} +/- {np.std(test_rewards):.1f}")
    print(f"  平均疏散: {np.mean(test_evacuated):.1f} +/- {np.std(test_evacuated):.1f}")
    print(f"  平均步数: {np.mean(test_steps):.1f} +/- {np.std(test_steps):.1f}")

    # 计算平均出口分布
    avg_A = np.mean([d['A'] for d in test_exit_dists])
    avg_B = np.mean([d['B'] for d in test_exit_dists])
    avg_C = np.mean([d['C'] for d in test_exit_dists])
    print(f"  平均出口分布: A={avg_A:.1f}, B={avg_B:.1f}, C={avg_C:.1f}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练过程中的疏散人数
    ax1 = axes[0, 0]
    if callback.evacuated_counts:
        ax1.plot(callback.evacuated_counts, alpha=0.7, color='steelblue')
        ax1.axhline(y=env.n_pedestrians, color='r', linestyle='--',
                   label=f'Max ({env.n_pedestrians})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Evacuated Count')
        ax1.set_title('Training Progress: Evacuation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 测试结果柱状图
    ax2 = axes[0, 1]
    x = np.arange(test_episodes)
    ax2.bar(x, test_evacuated, color='steelblue', alpha=0.7)
    ax2.axhline(y=env.n_pedestrians, color='r', linestyle='--',
               label=f'Target ({env.n_pedestrians})')
    ax2.set_xlabel('Test Episode')
    ax2.set_ylabel('Evacuated Count')
    ax2.set_title('Test Results: Evacuated per Episode')
    ax2.legend()
    ax2.set_xticks(x)

    # 3. 各出口疏散分布
    ax3 = axes[1, 0]
    exits = ['A', 'B', 'C']
    x_pos = np.arange(len(exits))
    avg_counts = [avg_A, avg_B, avg_C]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax3.bar(x_pos, avg_counts, color=colors, alpha=0.8)
    ax3.set_xlabel('Exit')
    ax3.set_ylabel('Average Evacuated')
    ax3.set_title('Average Exit Distribution (PPO Guided)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Exit {e}' for e in exits])

    # 添加数值标签
    for bar, count in zip(bars, avg_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count:.1f}', ha='center', va='bottom')

    # 4. 奖励曲线（滑动平均）
    ax4 = axes[1, 1]
    if callback.episode_rewards or len(callback.evacuated_counts) > 0:
        # 使用疏散数作为代理指标
        if callback.evacuated_counts:
            window = min(20, len(callback.evacuated_counts))
            if window > 0:
                smoothed = np.convolve(callback.evacuated_counts,
                                      np.ones(window)/window, mode='valid')
                ax4.plot(smoothed, color='green', alpha=0.8)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Smoothed Evacuated Count')
                ax4.set_title(f'Training Progress (Moving Avg, window={window})')
                ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    fig_dir = project_root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "ppo_metro_training.png"
    plt.savefig(str(fig_path), dpi=150)
    print(f"\n训练结果图已保存: {fig_path}")

    plt.show()


def demo_trained_model():
    """演示训练好的模型"""
    model_path = project_root / "outputs" / "models" / "ppo_metro.zip"

    if not model_path.exists():
        print("模型不存在，请先运行训练: python train_ppo_metro.py")
        return

    # 加载模型
    print("加载模型...")
    model = PPO.load(str(model_path))

    # 创建环境
    env = MetroEvacuationEnv(
        n_pedestrians=80,
        scene_size=(60.0, 40.0),
        max_steps=800,
        render_mode="human"
    )

    # 运行演示
    print("\n开始演示...")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated

    print(f"\n演示结束:")
    print(f"  总奖励: {total_reward:.1f}")
    print(f"  疏散人数: {info['evacuated']}/{env.n_pedestrians}")
    print(f"  各出口分布: A={info['evacuated_by_exit']['A']}, "
          f"B={info['evacuated_by_exit']['B']}, C={info['evacuated_by_exit']['C']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练地铁站场景PPO模型")
    parser.add_argument("--demo", action="store_true", help="演示训练好的模型")
    args = parser.parse_args()

    if args.demo:
        demo_trained_model()
    else:
        train_ppo_metro()
