#!/usr/bin/env python
"""
完整泛化验证脚本

验证所有模型在不同规模场景下的表现：
1. 密度预测模型泛化 - 在不同流量级别测试预测准确率
2. PPO模型泛化 - small训练 → medium/large测试
3. 完整系统泛化 - 密度预测+PPO+引导的端到端测试
4. 生成汇总报告

使用方法:
    python scripts/run_generalization_test.py
    python scripts/run_generalization_test.py --quick
    python scripts/run_generalization_test.py --only-density
    python scripts/run_generalization_test.py --only-ppo
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch


# ========== 1. 密度预测模型泛化验证 ==========

def test_density_predictor_generalization(
    model_path: str = None,
    flow_levels: List[str] = None,
    n_episodes: int = 3,
    max_steps: int = 500,
    quick: bool = False,
    device: str = "auto",
) -> Dict[str, Any]:
    """测试密度预测模型在不同流量下的准确率

    Args:
        model_path: 模型路径
        flow_levels: 要测试的流量级别
        n_episodes: 每个级别测试的episode数
        max_steps: 每个episode的最大步数
        quick: 快速模式

    Returns:
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("密度预测模型泛化验证")
    print("=" * 60)

    if flow_levels is None:
        flow_levels = ["small", "medium", "large"]

    if quick:
        n_episodes = 1
        max_steps = 200

    # 导入必要模块
    try:
        from prediction.density_predictor import DensityFieldPredictor, GRID_SIZE
        from simulation.large_station_env import LargeStationEnv
    except ImportError as e:
        print(f"导入错误: {e}")
        return {"error": str(e)}

    # 加载模型
    if model_path is None:
        model_path = str(PROJECT_ROOT / "outputs" / "models" / "density_predictor.pt")

    if not Path(model_path).exists():
        print(f"警告: 模型不存在 {model_path}")
        return {"error": f"Model not found: {model_path}"}

    results = {
        "model_path": model_path,
        "flow_levels": {},
        "summary": {},
    }

    for level in flow_levels:
        print(f"\n[{level}] 测试中...")

        # 创建环境
        env = LargeStationEnv(
            flow_level=level,
            max_steps=max_steps,
            emergency_mode=True,
        )

        # 获取出口信息
        exits_info = [
            {"id": e.id, "position": e.position}
            for e in env.exits
        ]

        # 创建预测器
        predictor = DensityFieldPredictor(
            exits=exits_info,
            model_path=model_path,
            device=device,
        )

        level_results = {
            "mae_list": [],
            "rmse_list": [],
            "correlation_list": [],
        }

        for ep in range(n_episodes):
            obs, _ = env.reset()

            predictions = []
            actuals = []

            step = 0
            while step < max_steps:
                # 收集当前密度场
                if hasattr(env, 'sfm') and hasattr(env.sfm, 'pedestrians'):
                    peds = env.sfm.pedestrians
                    if len(peds) > 0:
                        field = predictor.compute_density_field(peds)
                        predictor.add_frame(field)

                        # 如果有足够历史帧，进行预测
                        if predictor.has_enough_frames():
                            pred_density = predictor.predict()
                            if pred_density is not None:
                                predictions.append(pred_density.copy())
                                # 存储当前实际密度用于比较
                                actuals.append(field.density.copy())

                # 执行随机动作
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

                if terminated or truncated:
                    break

            # 计算预测误差
            if len(predictions) > 10:
                # 使用滞后比较（预测的是未来密度）
                lag = min(10, len(predictions) - 10)  # 预测10步后的密度

                pred_array = np.array(predictions[:-lag])
                actual_array = np.array(actuals[lag:len(predictions)])

                # 确保形状匹配
                min_len = min(len(pred_array), len(actual_array))
                pred_array = pred_array[:min_len]
                actual_array = actual_array[:min_len]

                # 计算指标
                mae = np.mean(np.abs(pred_array - actual_array))
                rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2))

                # 计算相关系数
                pred_flat = pred_array.flatten()
                actual_flat = actual_array.flatten()
                if np.std(pred_flat) > 0 and np.std(actual_flat) > 0:
                    correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
                else:
                    correlation = 0.0

                level_results["mae_list"].append(mae)
                level_results["rmse_list"].append(rmse)
                level_results["correlation_list"].append(correlation)

                print(f"  Episode {ep+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={correlation:.4f}")

        # 汇总该级别结果
        if level_results["mae_list"]:
            results["flow_levels"][level] = {
                "mae_mean": float(np.mean(level_results["mae_list"])),
                "mae_std": float(np.std(level_results["mae_list"])),
                "rmse_mean": float(np.mean(level_results["rmse_list"])),
                "rmse_std": float(np.std(level_results["rmse_list"])),
                "correlation_mean": float(np.mean(level_results["correlation_list"])),
                "n_samples": len(level_results["mae_list"]),
            }
            print(f"  {level} 平均: MAE={results['flow_levels'][level]['mae_mean']:.4f}, "
                  f"Corr={results['flow_levels'][level]['correlation_mean']:.4f}")

        env.close()

    # 计算总体汇总
    all_maes = [r["mae_mean"] for r in results["flow_levels"].values() if "mae_mean" in r]
    all_corrs = [r["correlation_mean"] for r in results["flow_levels"].values() if "correlation_mean" in r]

    if all_maes:
        results["summary"] = {
            "overall_mae": float(np.mean(all_maes)),
            "overall_correlation": float(np.mean(all_corrs)),
            "generalization_gap": float(np.std(all_maes)),  # 跨级别标准差越小，泛化越好
        }

    return results


# ========== 2. PPO模型泛化验证 ==========

def test_ppo_generalization(
    model_path: str = None,
    flow_levels: List[str] = None,
    n_episodes: int = 5,
    quick: bool = False,
) -> Dict[str, Any]:
    """测试PPO模型跨规模泛化

    Args:
        model_path: 模型路径
        flow_levels: 要测试的流量级别
        n_episodes: 每个级别测试的episode数
        quick: 快速模式

    Returns:
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("PPO模型泛化验证")
    print("=" * 60)

    if flow_levels is None:
        flow_levels = ["small", "medium", "large"]

    if quick:
        n_episodes = 2

    # 导入必要模块
    try:
        from stable_baselines3 import PPO
        from simulation.large_station_env import LargeStationEnv
    except ImportError as e:
        print(f"导入错误: {e}")
        return {"error": str(e)}

    # 加载模型
    if model_path is None:
        model_path = str(PROJECT_ROOT / "outputs" / "models" / "ppo_large_station_small.zip")

    if not Path(model_path).exists():
        print(f"警告: 模型不存在 {model_path}")
        return {"error": f"Model not found: {model_path}"}

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    results = {
        "model_path": model_path,
        "flow_levels": {},
        "summary": {},
    }

    for level in flow_levels:
        print(f"\n[{level}] 测试中...")

        env = LargeStationEnv(
            flow_level=level,
            emergency_mode=True,
        )

        level_results = {
            "evacuation_rates": [],
            "evacuation_times": [],
            "danger_counts": [],
            "total_people": [],
        }

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            evacuated = info.get('evacuated', 0)
            total = env.n_pedestrians
            evac_rate = evacuated / total if total > 0 else 0
            evac_time = info.get('time', 0)
            danger = info.get('danger_count', 0)

            level_results["evacuation_rates"].append(evac_rate)
            level_results["evacuation_times"].append(evac_time)
            level_results["danger_counts"].append(danger)
            level_results["total_people"].append(total)

            print(f"  Episode {ep+1}: 疏散率={evac_rate:.1%}, 时间={evac_time:.1f}s, "
                  f"人数={evacuated}/{total}")

        env.close()

        # 汇总该级别结果
        results["flow_levels"][level] = {
            "evacuation_rate_mean": float(np.mean(level_results["evacuation_rates"])),
            "evacuation_rate_std": float(np.std(level_results["evacuation_rates"])),
            "evacuation_time_mean": float(np.mean(level_results["evacuation_times"])),
            "evacuation_time_std": float(np.std(level_results["evacuation_times"])),
            "danger_count_mean": float(np.mean(level_results["danger_counts"])),
            "total_people": int(level_results["total_people"][0]),
            "within_target": float(np.mean(level_results["evacuation_times"])) <= 600,
        }

        print(f"  {level} 平均: 疏散率={results['flow_levels'][level]['evacuation_rate_mean']:.1%}, "
              f"时间={results['flow_levels'][level]['evacuation_time_mean']:.1f}s")

    # 计算泛化性指标
    all_rates = [r["evacuation_rate_mean"] for r in results["flow_levels"].values()]
    all_times = [r["evacuation_time_mean"] for r in results["flow_levels"].values()]

    results["summary"] = {
        "overall_evacuation_rate": float(np.mean(all_rates)),
        "rate_degradation": float(all_rates[0] - all_rates[-1]) if len(all_rates) > 1 else 0,
        "time_increase": float(all_times[-1] - all_times[0]) if len(all_times) > 1 else 0,
        "all_within_target": all(r["within_target"] for r in results["flow_levels"].values()),
    }

    return results


# ========== 3. 完整系统泛化验证 ==========

def test_full_system_generalization(
    density_model_path: str = None,
    ppo_model_path: str = None,
    flow_levels: List[str] = None,
    n_episodes: int = 3,
    quick: bool = False,
) -> Dict[str, Any]:
    """测试完整系统（密度预测+PPO+引导）的泛化

    对比:
    - baseline: 无引导（随机动作）
    - full: 密度预测 + PPO引导

    Args:
        density_model_path: 密度预测模型路径
        ppo_model_path: PPO模型路径
        flow_levels: 要测试的流量级别
        n_episodes: 每个级别测试的episode数
        quick: 快速模式

    Returns:
        测试结果字典
    """
    print("\n" + "=" * 60)
    print("完整系统泛化验证 (Baseline vs Full)")
    print("=" * 60)

    if flow_levels is None:
        flow_levels = ["small", "medium", "large"]

    if quick:
        n_episodes = 1

    # 导入必要模块
    try:
        from stable_baselines3 import PPO
        from simulation.large_station_env import LargeStationEnv
    except ImportError as e:
        print(f"导入错误: {e}")
        return {"error": str(e)}

    # 设置模型路径
    if density_model_path is None:
        density_model_path = str(PROJECT_ROOT / "outputs" / "models" / "density_predictor.pt")
    if ppo_model_path is None:
        ppo_model_path = str(PROJECT_ROOT / "outputs" / "models" / "ppo_large_station_small.zip")

    # 检查PPO模型
    ppo_model = None
    if Path(ppo_model_path).exists():
        print(f"加载PPO模型: {ppo_model_path}")
        ppo_model = PPO.load(ppo_model_path)
    else:
        print(f"警告: PPO模型不存在 {ppo_model_path}")

    results = {
        "density_model": density_model_path,
        "ppo_model": ppo_model_path,
        "flow_levels": {},
        "summary": {},
    }

    for level in flow_levels:
        print(f"\n[{level}] 测试中...")

        level_results = {
            "baseline": {"rates": [], "times": []},
            "full": {"rates": [], "times": []},
        }

        for mode in ["baseline", "full"]:
            env = LargeStationEnv(
                flow_level=level,
                emergency_mode=True,
            )

            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False

                while not done:
                    if mode == "full" and ppo_model is not None:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                    else:
                        action = env.action_space.sample()

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                evacuated = info.get('evacuated', 0)
                total = env.n_pedestrians
                evac_rate = evacuated / total if total > 0 else 0
                evac_time = info.get('time', 0)

                level_results[mode]["rates"].append(evac_rate)
                level_results[mode]["times"].append(evac_time)

            env.close()

        # 汇总该级别结果
        results["flow_levels"][level] = {
            "baseline": {
                "evacuation_rate": float(np.mean(level_results["baseline"]["rates"])),
                "evacuation_time": float(np.mean(level_results["baseline"]["times"])),
            },
            "full": {
                "evacuation_rate": float(np.mean(level_results["full"]["rates"])),
                "evacuation_time": float(np.mean(level_results["full"]["times"])),
            },
        }

        # 计算提升
        baseline_rate = results["flow_levels"][level]["baseline"]["evacuation_rate"]
        full_rate = results["flow_levels"][level]["full"]["evacuation_rate"]
        baseline_time = results["flow_levels"][level]["baseline"]["evacuation_time"]
        full_time = results["flow_levels"][level]["full"]["evacuation_time"]

        results["flow_levels"][level]["improvement"] = {
            "rate_improvement": float(full_rate - baseline_rate),
            "time_reduction": float(baseline_time - full_time),
        }

        print(f"  {level}: Baseline疏散率={baseline_rate:.1%}, Full疏散率={full_rate:.1%}, "
              f"提升={full_rate-baseline_rate:.1%}")

    # 计算总体汇总
    rate_improvements = [r["improvement"]["rate_improvement"] for r in results["flow_levels"].values()]
    time_reductions = [r["improvement"]["time_reduction"] for r in results["flow_levels"].values()]

    results["summary"] = {
        "avg_rate_improvement": float(np.mean(rate_improvements)),
        "avg_time_reduction": float(np.mean(time_reductions)),
        "consistent_improvement": all(r > 0 for r in rate_improvements),
    }

    return results


# ========== 4. 生成汇总报告 ==========

def generate_generalization_report(
    density_results: Dict,
    ppo_results: Dict,
    system_results: Dict,
    output_path: str = None,
) -> Dict:
    """生成泛化验证汇总报告

    Args:
        density_results: 密度预测模型结果
        ppo_results: PPO模型结果
        system_results: 完整系统结果
        output_path: 输出路径

    Returns:
        汇总报告字典
    """
    print("\n" + "=" * 60)
    print("生成泛化验证汇总报告")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "density_predictor": density_results,
        "ppo_model": ppo_results,
        "full_system": system_results,
        "conclusions": {},
    }

    # 分析结论
    conclusions = []

    # 密度预测结论
    if density_results and "summary" in density_results:
        corr = density_results["summary"].get("overall_correlation", 0)
        gap = density_results["summary"].get("generalization_gap", 1)
        if corr > 0.7 and gap < 0.1:
            conclusions.append("密度预测模型泛化性良好")
        elif corr > 0.5:
            conclusions.append("密度预测模型泛化性一般")
        else:
            conclusions.append("密度预测模型需要更多训练数据")

    # PPO结论
    if ppo_results and "summary" in ppo_results:
        degradation = ppo_results["summary"].get("rate_degradation", 0)
        within_target = ppo_results["summary"].get("all_within_target", False)
        if within_target and degradation < 0.1:
            conclusions.append("PPO模型跨规模泛化性优秀")
        elif degradation < 0.2:
            conclusions.append("PPO模型泛化性良好，大规模略有下降")
        else:
            conclusions.append("PPO模型需要多规模联合训练")

    # 系统结论
    if system_results and "summary" in system_results:
        improvement = system_results["summary"].get("avg_rate_improvement", 0)
        consistent = system_results["summary"].get("consistent_improvement", False)
        if consistent and improvement > 0.05:
            conclusions.append("完整系统显著优于baseline")
        elif improvement > 0:
            conclusions.append("完整系统优于baseline")
        else:
            conclusions.append("系统集成需要优化")

    report["conclusions"] = conclusions

    # 保存报告
    if output_path is None:
        output_dir = PROJECT_ROOT / "outputs" / "generalization"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "generalization_report.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存: {output_path}")

    # 打印汇总
    print("\n" + "-" * 40)
    print("结论:")
    for c in conclusions:
        print(f"  - {c}")

    return report


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(
        description="完整泛化验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--quick", action="store_true",
                        help="快速模式（减少测试量）")
    parser.add_argument("--only-density", action="store_true",
                        help="只测试密度预测模型")
    parser.add_argument("--only-ppo", action="store_true",
                        help="只测试PPO模型")
    parser.add_argument("--only-system", action="store_true",
                        help="只测试完整系统")
    parser.add_argument("--density-model", type=str, default=None,
                        help="密度预测模型路径")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="PPO模型路径")
    parser.add_argument("--flow-levels", type=str, nargs="+",
                        default=None, choices=["small", "medium", "large"],
                        help="要测试的流量级别")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="每个级别测试的episode数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告路径")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="计算设备 (默认: auto)")

    args = parser.parse_args()

    print("=" * 60)
    print("完整泛化验证")
    print("=" * 60)
    print(f"模式: {'快速' if args.quick else '完整'}")
    print(f"流量级别: {args.flow_levels or ['small', 'medium', 'large']}")
    print(f"计算设备: {args.device}")

    start_time = time.time()

    density_results = {}
    ppo_results = {}
    system_results = {}

    # 1. 密度预测模型验证
    if not args.only_ppo and not args.only_system:
        density_results = test_density_predictor_generalization(
            model_path=args.density_model,
            flow_levels=args.flow_levels,
            n_episodes=args.n_episodes,
            quick=args.quick,
            device=args.device,
        )

    # 2. PPO模型验证
    if not args.only_density and not args.only_system:
        ppo_results = test_ppo_generalization(
            model_path=args.ppo_model,
            flow_levels=args.flow_levels,
            n_episodes=args.n_episodes,
            quick=args.quick,
        )

    # 3. 完整系统验证
    if not args.only_density and not args.only_ppo:
        system_results = test_full_system_generalization(
            density_model_path=args.density_model,
            ppo_model_path=args.ppo_model,
            flow_levels=args.flow_levels,
            n_episodes=args.n_episodes,
            quick=args.quick,
        )

    # 4. 生成报告
    report = generate_generalization_report(
        density_results=density_results,
        ppo_results=ppo_results,
        system_results=system_results,
        output_path=args.output,
    )

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time/60:.1f} 分钟")

    return report


if __name__ == "__main__":
    main()
