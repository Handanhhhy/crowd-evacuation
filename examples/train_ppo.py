"""
训练 PPO 强化学习模型优化疏散策略
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

from simulation.evacuation_env import EvacuationEnv
from utils.device_info import print_device_info, get_device, print_device_selection


class TrainingCallback(BaseCallback):
    """训练回调，记录训练过程"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.evacuated_counts = []

    def _on_step(self) -> bool:
        # 记录每个 episode 结束时的信息
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]
            self.evacuated_counts.append(info.get('evacuated', 0))

        return True


def train_ppo():
    """训练 PPO 模型"""
    
    # 打印设备信息
    print_device_info("系统设备信息")
    
    # 检测设备
    device = get_device("auto")
    print_device_selection(device)

    print("=" * 50)
    print("PPO 疏散策略优化训练")
    print("=" * 50)

    # 创建环境
    print("\n[1/4] 创建环境...")
    env = EvacuationEnv(
        n_pedestrians=30,  # 先用较少的人测试
        scene_size=(30.0, 20.0),
        n_exits=2,
        max_steps=500,
        dt=0.1
    )

    # 包装为向量环境
    vec_env = DummyVecEnv([lambda: env])

    # 创建 PPO 模型
    print("\n[2/4] 创建 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device=device  # 使用自动检测的设备
    )

    # 训练
    print("\n[3/4] 开始训练...")
    callback = TrainingCallback()

    total_timesteps = 20000  # 可以增加以获得更好的结果
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False  # 禁用进度条避免依赖问题
    )

    # 保存模型
    print("\n[4/4] 保存模型...")
    model_path = project_root / "outputs" / "models" / "ppo_evacuation"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")

    # 评估和可视化
    evaluate_and_visualize(model, env, callback, project_root)

    return model


def evaluate_and_visualize(model, env, callback, project_root):
    """评估模型并可视化结果"""

    print("\n" + "=" * 50)
    print("评估训练结果")
    print("=" * 50)

    # 运行几个测试 episode
    test_episodes = 5
    test_rewards = []
    test_evacuated = []
    test_steps = []

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

        print(f"  Episode {ep+1}: Reward={total_reward:.1f}, "
              f"Evacuated={info['evacuated']}/{env.n_pedestrians}, "
              f"Steps={steps}")

    print(f"\n平均结果:")
    print(f"  平均奖励: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"  平均疏散: {np.mean(test_evacuated):.1f} ± {np.std(test_evacuated):.1f}")
    print(f"  平均步数: {np.mean(test_steps):.1f} ± {np.std(test_steps):.1f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 训练过程中的疏散人数
    ax1 = axes[0]
    if callback.evacuated_counts:
        ax1.plot(callback.evacuated_counts, alpha=0.7)
        ax1.axhline(y=env.n_pedestrians, color='r', linestyle='--',
                   label=f'Max ({env.n_pedestrians})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Evacuated Count')
        ax1.set_title('Training Progress: Evacuation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 测试结果
    ax2 = axes[1]
    x = np.arange(test_episodes)
    ax2.bar(x, test_evacuated, color='steelblue', alpha=0.7)
    ax2.axhline(y=env.n_pedestrians, color='r', linestyle='--',
               label=f'Target ({env.n_pedestrians})')
    ax2.set_xlabel('Test Episode')
    ax2.set_ylabel('Evacuated Count')
    ax2.set_title('Test Results')
    ax2.legend()
    ax2.set_xticks(x)

    plt.tight_layout()

    fig_path = project_root / "outputs" / "figures" / "ppo_training.png"
    plt.savefig(str(fig_path), dpi=150)
    print(f"\n训练结果图已保存: {fig_path}")

    plt.show()


def demo_trained_model():
    """演示训练好的模型"""
    model_path = project_root / "outputs" / "models" / "ppo_evacuation.zip"

    if not model_path.exists():
        print("模型不存在，请先训练")
        return

    # 加载模型
    model = PPO.load(str(model_path))

    # 创建环境
    env = EvacuationEnv(
        n_pedestrians=30,
        scene_size=(30.0, 20.0),
        n_exits=2,
        max_steps=500,
        render_mode="human"
    )

    # 运行演示
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

    print(f"\n演示结束: 疏散 {info['evacuated']}/{env.n_pedestrians} 人")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="演示训练好的模型")
    args = parser.parse_args()

    if args.demo:
        demo_trained_model()
    else:
        train_ppo()
