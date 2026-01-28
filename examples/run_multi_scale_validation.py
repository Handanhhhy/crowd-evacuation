#!/usr/bin/env python3
"""
多规模验证实验脚本
Multi-Scale Validation Runner

验证小规模场景(80人)得出的结论在大规模场景下是否成立:
- V1: balance_penalty在不同规模下的价值
- V2: PPO引导在复杂场景的效果
- V3: 轨迹预测方法在高密度场景的优势
- V4: 观测空间维度在大规模场景的表现

用法:
    # 运行所有验证实验
    python examples/run_multi_scale_validation.py

    # 运行指定验证组
    python examples/run_multi_scale_validation.py --groups V1 V2

    # 运行指定规模
    python examples/run_multi_scale_validation.py --scales small medium

    # 快速模式 (减少训练步数)
    python examples/run_multi_scale_validation.py --quick
"""

import os
import sys
import time
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("stable-baselines3不可用，PPO训练功能已禁用")

from simulation.metro_evacuation_env import MetroEvacuationEnv


@dataclass
class ScaleExperimentResult:
    """单个规模实验结果"""
    experiment_id: str
    scale: str
    num_pedestrians: int

    # 性能指标
    evacuation_rate: float
    evacuation_rate_std: float
    evacuation_time: float
    evacuation_time_std: float
    max_congestion: float
    exit_balance: float

    # 训练指标
    training_time: float
    final_reward: float

    n_episodes: int
    random_seeds: List[int]


@dataclass
class ValidationGroupResult:
    """验证组结果"""
    group_id: str
    description: str
    hypothesis: str
    results: List[ScaleExperimentResult]
    conclusion: str = ""


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载多规模验证配置"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "multi_scale_validation.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def create_env_for_scale(
    scale_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    seed: int = 42
) -> MetroEvacuationEnv:
    """
    根据规模和实验配置创建环境

    Args:
        scale_config: 规模配置 (num_pedestrians, max_steps等)
        exp_config: 实验配置 (reward_weights, guidance等)
        seed: 随机种子

    Returns:
        配置好的环境
    """
    env_kwargs = {
        "n_pedestrians": scale_config.get("num_pedestrians", 80),
        "max_steps": scale_config.get("max_steps", 600),
        "dt": 0.1,
        "use_optimized_gpu_sfm": True,
        "sfm_device": "auto",
    }

    # 应用实验配置
    if "reward_weights" in exp_config:
        env_kwargs["reward_weights"] = exp_config["reward_weights"]

    if "guidance" in exp_config:
        env_kwargs["enable_guidance"] = exp_config["guidance"].get("enabled", True)

    env = MetroEvacuationEnv(**env_kwargs)
    env.reset(seed=seed)

    return env


def train_and_evaluate(
    env: MetroEvacuationEnv,
    config: Dict[str, Any],
    exp_id: str,
    use_guidance: bool = True,
    quick_mode: bool = False
) -> Tuple[Optional[Any], Dict[str, float]]:
    """
    训练模型并评估

    Args:
        env: 环境
        config: 全局配置
        exp_id: 实验ID
        use_guidance: 是否使用PPO引导
        quick_mode: 快速模式

    Returns:
        (模型, 评估结果)
    """
    train_config = config.get("global", {}).get("training", {})
    eval_config = config.get("global", {}).get("evaluation", {})

    # 训练步数
    total_timesteps = train_config.get("total_timesteps", 100000)
    if quick_mode:
        total_timesteps = min(total_timesteps, 20000)

    model = None
    training_time = 0.0

    # 训练PPO模型 (如果需要引导)
    if use_guidance and SB3_AVAILABLE:
        print(f"  训练PPO模型 ({total_timesteps}步)...")

        vec_env = DummyVecEnv([lambda: env])

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=train_config.get("learning_rate", 3e-4),
            n_steps=train_config.get("n_steps", 1024),
            batch_size=train_config.get("batch_size", 128),
            n_epochs=train_config.get("n_epochs", 10),
            gamma=train_config.get("gamma", 0.99),
            verbose=0,
            device="cpu"
        )

        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time

        print(f"  训练完成 (耗时: {training_time:.1f}秒)")

    # 评估
    n_episodes = eval_config.get("n_eval_episodes", 10)
    if quick_mode:
        n_episodes = min(n_episodes, 3)

    seeds = eval_config.get("random_seeds", [42])
    if quick_mode:
        seeds = seeds[:2]

    print(f"  评估 ({n_episodes} episodes × {len(seeds)} seeds)...")

    all_rates = []
    all_times = []
    all_congestions = []
    all_balances = []
    all_rewards = []

    for seed in seeds:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep * 100)
            done = False
            truncated = False
            episode_reward = 0.0
            max_cong = 0.0

            while not (done or truncated):
                if model is not None and use_guidance:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                if hasattr(env, 'history') and 'congestion' in env.history:
                    if env.history['congestion']:
                        max_cong = max(max_cong, env.history['congestion'][-1])

            # 收集指标
            evac_rate = env.evacuated_count / env.n_pedestrians
            evac_time = env.current_step

            counts = list(env.evacuated_by_exit.values())
            exit_balance = np.std(counts) if sum(counts) > 0 else 0

            all_rates.append(evac_rate)
            all_times.append(evac_time)
            all_congestions.append(max_cong)
            all_balances.append(exit_balance)
            all_rewards.append(episode_reward)

    results = {
        "evacuation_rate": np.mean(all_rates),
        "evacuation_rate_std": np.std(all_rates),
        "evacuation_time": np.mean(all_times),
        "evacuation_time_std": np.std(all_times),
        "max_congestion": np.mean(all_congestions),
        "exit_balance": np.mean(all_balances),
        "final_reward": np.mean(all_rewards),
        "training_time": training_time,
        "n_episodes": n_episodes * len(seeds),
        "seeds": seeds
    }

    return model, results


def run_validation_group(
    group_id: str,
    group_config: Dict[str, Any],
    scales_config: Dict[str, Any],
    global_config: Dict[str, Any],
    selected_scales: List[str],
    output_dir: Path,
    quick_mode: bool = False
) -> ValidationGroupResult:
    """
    运行单个验证组

    Args:
        group_id: 验证组ID
        group_config: 验证组配置
        scales_config: 规模配置
        global_config: 全局配置
        selected_scales: 选择的规模列表
        output_dir: 输出目录
        quick_mode: 快速模式

    Returns:
        验证组结果
    """
    print(f"\n{'='*60}")
    print(f"验证组: {group_id}")
    print(f"描述: {group_config.get('description', '')}")
    print(f"假设: {group_config.get('hypothesis', '')}")
    print(f"{'='*60}")

    results = []

    for exp_id, exp_config in group_config.get("experiments", {}).items():
        exp_scales = exp_config.get("scales", list(scales_config.keys()))

        for scale_name in exp_scales:
            if scale_name not in selected_scales:
                continue

            scale_config = scales_config.get(scale_name, {})
            full_exp_id = f"{group_id}_{exp_id}_{scale_name}"

            print(f"\n实验: {full_exp_id}")
            print(f"  规模: {scale_config.get('name', scale_name)}")
            print(f"  人数: {scale_config.get('num_pedestrians', 80)}")

            # 创建环境
            env = create_env_for_scale(scale_config, exp_config)

            # 确定是否使用引导
            use_guidance = True
            if "guidance" in exp_config:
                use_guidance = exp_config["guidance"].get("enabled", True)

            # 训练和评估
            model, eval_results = train_and_evaluate(
                env=env,
                config={"global": global_config},
                exp_id=full_exp_id,
                use_guidance=use_guidance,
                quick_mode=quick_mode
            )

            # 创建结果
            result = ScaleExperimentResult(
                experiment_id=full_exp_id,
                scale=scale_name,
                num_pedestrians=scale_config.get("num_pedestrians", 80),
                evacuation_rate=eval_results["evacuation_rate"],
                evacuation_rate_std=eval_results["evacuation_rate_std"],
                evacuation_time=eval_results["evacuation_time"],
                evacuation_time_std=eval_results["evacuation_time_std"],
                max_congestion=eval_results["max_congestion"],
                exit_balance=eval_results["exit_balance"],
                training_time=eval_results["training_time"],
                final_reward=eval_results["final_reward"],
                n_episodes=eval_results["n_episodes"],
                random_seeds=eval_results["seeds"]
            )

            results.append(result)

            print(f"  结果: 疏散率={result.evacuation_rate:.2%}, "
                  f"时间={result.evacuation_time:.1f}步")

    return ValidationGroupResult(
        group_id=group_id,
        description=group_config.get("description", ""),
        hypothesis=group_config.get("hypothesis", ""),
        results=results
    )


def generate_report(
    all_results: List[ValidationGroupResult],
    output_dir: Path,
    config: Dict[str, Any]
):
    """生成验证报告"""
    report_path = output_dir / "multi_scale_validation_report.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "validation_groups": []
    }

    for group_result in all_results:
        group_data = {
            "group_id": group_result.group_id,
            "description": group_result.description,
            "hypothesis": group_result.hypothesis,
            "results": [asdict(r) for r in group_result.results],
            "analysis": analyze_group(group_result)
        }
        report["validation_groups"].append(group_data)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {report_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("多规模验证实验摘要")
    print("="*60)

    for group_result in all_results:
        print(f"\n{group_result.group_id}: {group_result.description}")
        print("-" * 50)

        # 按规模分组显示
        by_scale = {}
        for r in group_result.results:
            if r.scale not in by_scale:
                by_scale[r.scale] = []
            by_scale[r.scale].append(r)

        for scale, results in sorted(by_scale.items()):
            print(f"  {scale} ({results[0].num_pedestrians}人):")
            for r in results:
                exp_name = r.experiment_id.split("_")[-2]  # 提取实验名
                print(f"    {exp_name}: 疏散率={r.evacuation_rate:.2%}, "
                      f"时间={r.evacuation_time:.1f}步")


def analyze_group(group_result: ValidationGroupResult) -> Dict[str, Any]:
    """分析验证组结果"""
    analysis = {
        "scale_comparison": {},
        "conclusion": ""
    }

    # 按规模分组
    by_scale = {}
    for r in group_result.results:
        if r.scale not in by_scale:
            by_scale[r.scale] = []
        by_scale[r.scale].append(r)

    # 分析每个规模下的对比
    conclusions = []
    for scale, results in by_scale.items():
        if len(results) >= 2:
            # 比较两个实验
            r1, r2 = results[0], results[1]
            diff_rate = r1.evacuation_rate - r2.evacuation_rate
            diff_time = r1.evacuation_time - r2.evacuation_time

            analysis["scale_comparison"][scale] = {
                "experiments": [r1.experiment_id, r2.experiment_id],
                "rate_diff": diff_rate,
                "time_diff": diff_time,
                "better_experiment": r1.experiment_id if diff_rate > 0 else r2.experiment_id
            }

            # 判断结论
            if abs(diff_rate) > 0.02:  # 2%以上差异认为显著
                better = "第一个" if diff_rate > 0 else "第二个"
                conclusions.append(f"{scale}规模下{better}实验更优")

    analysis["conclusion"] = "; ".join(conclusions) if conclusions else "各规模下差异不显著"

    return analysis


def main():
    parser = argparse.ArgumentParser(description="多规模验证实验")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--groups", "-g", nargs="+",
                        choices=["V1", "V2", "V3", "V4"],
                        default=None, help="指定验证组")
    parser.add_argument("--scales", "-s", nargs="+",
                        choices=["small", "medium", "large", "extreme"],
                        default=["small", "medium"], help="指定规模")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="快速模式 (减少训练步数)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--force-cpu", action="store_true",
                        help="强制使用CPU运行 (避免GPU内存不足)")
    args = parser.parse_args()

    # 强制CPU模式
    if args.force_cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[配置] 强制使用CPU运行")

    # 加载配置
    config = load_config(args.config)

    # 确定输出目录
    output_dir = Path(args.output) if args.output else Path(
        config.get("global", {}).get("output", {}).get("base_dir", "outputs/multi_scale")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要运行的验证组
    validation_groups = config.get("validation_groups", {})
    if args.groups:
        selected_groups = {f"V{g[1:]}_" if not g.startswith("V") else g: validation_groups.get(g, {})
                          for g in args.groups if g in validation_groups}
        # 修正: 使用正确的key
        selected_groups = {k: v for k, v in validation_groups.items()
                          if any(k.startswith(g) for g in args.groups)}
    else:
        selected_groups = validation_groups

    print(f"多规模验证实验开始")
    print(f"配置: {args.config or 'configs/multi_scale_validation.yaml'}")
    print(f"验证组: {list(selected_groups.keys())}")
    print(f"规模: {args.scales}")
    print(f"输出目录: {output_dir}")
    print(f"快速模式: {args.quick}")

    # 运行验证
    all_results = []
    scales_config = config.get("scales", {})
    global_config = config.get("global", {})

    for group_id, group_config in selected_groups.items():
        result = run_validation_group(
            group_id=group_id,
            group_config=group_config,
            scales_config=scales_config,
            global_config=global_config,
            selected_scales=args.scales,
            output_dir=output_dir,
            quick_mode=args.quick
        )
        all_results.append(result)

    # 生成报告
    generate_report(all_results, output_dir, config)

    print("\n完成!")


if __name__ == "__main__":
    main()
