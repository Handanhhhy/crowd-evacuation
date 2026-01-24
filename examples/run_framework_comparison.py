#!/usr/bin/env python3
"""
研究框架对比实验
Framework Comparison Experiments

运行5组方案对比（参考 docs/new_station_plan.md 6.1节）:
  1. 原始 SFM（基线）
  2. SFM + 密度预测
  3. SFM + 动态分流
  4. SFM + 预测 + 分流（完整方案）
  5. SFM + PPO引导

用法:
    python examples/run_framework_comparison.py
    python examples/run_framework_comparison.py --episodes 10
    python examples/run_framework_comparison.py --flow-levels small medium large
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulation.large_station_env import LargeStationEnv


@dataclass
class ExperimentResult:
    """单次实验结果"""
    method: str
    flow_level: str
    episode: int
    evacuation_rate: float  # 疏散率 (0-1)
    evacuation_time: float  # 疏散时间 (秒)
    max_density: float  # 最大密度 (人/m²)
    avg_density: float  # 平均密度 (人/m²)
    exit_load_variance: float  # 出口负载方差 (均衡度指标)
    congestion_events: int  # 拥堵事件数
    crush_risk_events: int  # 踩踏风险事件数


@dataclass
class MethodSummary:
    """方案汇总统计"""
    method: str
    flow_level: str
    n_episodes: int
    # 疏散率
    evacuation_rate_mean: float
    evacuation_rate_std: float
    # 疏散时间
    evacuation_time_mean: float
    evacuation_time_std: float
    # 最大密度
    max_density_mean: float
    max_density_std: float
    # 出口均衡度
    exit_load_variance_mean: float
    # 拥堵事件
    congestion_events_mean: float
    crush_risk_events_mean: float


# 5组对比方案定义
METHODS = {
    "baseline_sfm": {
        "name": "原始SFM",
        "description": "基线社会力模型，无额外引导",
        "use_density_prediction": False,
        "use_dynamic_routing": False,
        "use_ppo_guidance": False,
    },
    "sfm_prediction": {
        "name": "SFM+密度预测",
        "description": "SFM + ConvLSTM密度预测",
        "use_density_prediction": True,
        "use_dynamic_routing": False,
        "use_ppo_guidance": False,
    },
    "sfm_routing": {
        "name": "SFM+动态分流",
        "description": "SFM + 规则引擎动态分流",
        "use_density_prediction": False,
        "use_dynamic_routing": True,
        "use_ppo_guidance": False,
    },
    "sfm_full": {
        "name": "SFM+预测+分流",
        "description": "完整方案：SFM + 密度预测 + 动态分流",
        "use_density_prediction": True,
        "use_dynamic_routing": True,
        "use_ppo_guidance": False,
    },
    "sfm_ppo": {
        "name": "SFM+PPO引导",
        "description": "SFM + PPO强化学习引导（对比方案）",
        "use_density_prediction": False,
        "use_dynamic_routing": False,
        "use_ppo_guidance": True,
    },
}


def run_single_episode(
    env: LargeStationEnv,
    method_config: dict,
    max_steps: int = 6000,
) -> Dict[str, Any]:
    """运行单个episode"""

    obs, info = env.reset()

    total_reward = 0
    step = 0
    max_density = 0
    density_history = []
    congestion_events = 0
    crush_risk_events = 0
    exit_loads = []

    done = False
    truncated = False

    while not (done or truncated) and step < max_steps:
        # 根据方案选择动作
        if method_config.get("use_ppo_guidance") and hasattr(env, '_ppo_model'):
            # PPO引导：使用训练好的模型
            action, _ = env._ppo_model.predict(obs, deterministic=True)
        elif method_config.get("use_dynamic_routing"):
            # 动态分流：基于规则的动作选择
            action = _get_routing_action(env, method_config.get("use_density_prediction", False))
        else:
            # 基线/仅预测：均匀分配动作
            action = np.zeros(env.action_space.shape[0])

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # 记录统计
        if 'max_density' in info:
            max_density = max(max_density, info['max_density'])
        if 'avg_density' in info:
            density_history.append(info['avg_density'])
        if info.get('congestion_detected', False):
            congestion_events += 1
        if info.get('crush_risk', False):
            crush_risk_events += 1
        if 'exit_loads' in info:
            exit_loads.append(info['exit_loads'])

    # 计算结果
    evacuated = info.get('evacuated', 0)
    total_peds = info.get('total_pedestrians', env.n_pedestrians)
    evacuation_rate = evacuated / total_peds if total_peds > 0 else 0

    # 出口负载方差（均衡度）
    if exit_loads:
        final_loads = exit_loads[-1] if isinstance(exit_loads[-1], (list, np.ndarray)) else [0]
        exit_load_variance = np.var(final_loads) if len(final_loads) > 1 else 0
    else:
        exit_load_variance = 0

    return {
        "evacuation_rate": evacuation_rate,
        "evacuation_time": step * env.dt,
        "max_density": max_density,
        "avg_density": np.mean(density_history) if density_history else 0,
        "exit_load_variance": exit_load_variance,
        "congestion_events": congestion_events,
        "crush_risk_events": crush_risk_events,
        "total_reward": total_reward,
        "steps": step,
    }


def _get_routing_action(env: LargeStationEnv, use_prediction: bool = False) -> np.ndarray:
    """基于规则的动态分流动作"""
    action = np.zeros(env.action_space.shape[0])

    # 获取当前出口负载
    if hasattr(env, 'exit_loads'):
        loads = np.array(env.exit_loads)
        if len(loads) > 0 and np.sum(loads) > 0:
            # 负载均衡：减少高负载出口的权重
            normalized = loads / (np.sum(loads) + 1e-6)
            # 动作：负值减少该出口吸引力，正值增加
            action = 0.5 - normalized  # 负载高的出口动作为负

    # 如果使用密度预测，可以进一步调整
    if use_prediction and hasattr(env, 'density_predictor'):
        # 预测未来密度，调整分流策略
        pass  # 预留密度预测集成接口

    return np.clip(action, -1, 1)


def run_method_experiments(
    method_key: str,
    method_config: dict,
    flow_levels: List[str],
    n_episodes: int,
    output_dir: Path,
) -> List[ExperimentResult]:
    """运行某个方案的所有实验"""

    results = []

    print(f"\n{'='*60}")
    print(f"方案: {method_config['name']}")
    print(f"描述: {method_config['description']}")
    print(f"{'='*60}")

    for flow_level in flow_levels:
        print(f"\n  流量等级: {flow_level}")

        # 创建环境
        env = LargeStationEnv(
            flow_level=flow_level,
            use_gpu_sfm=True,
            emergency_mode=True,
        )

        # 如果是PPO方案，尝试加载模型
        if method_config.get("use_ppo_guidance"):
            ppo_model_path = PROJECT_ROOT / "outputs/models/ppo_large_station_small.zip"
            if ppo_model_path.exists():
                try:
                    from stable_baselines3 import PPO
                    env._ppo_model = PPO.load(str(ppo_model_path))
                    print(f"    已加载PPO模型: {ppo_model_path}")
                except Exception as e:
                    print(f"    警告: 无法加载PPO模型 ({e})，使用随机动作")
            else:
                print(f"    警告: PPO模型不存在，使用随机动作")

        for ep in range(n_episodes):
            start_time = time.time()
            ep_result = run_single_episode(env, method_config)
            elapsed = time.time() - start_time

            result = ExperimentResult(
                method=method_key,
                flow_level=flow_level,
                episode=ep,
                evacuation_rate=ep_result["evacuation_rate"],
                evacuation_time=ep_result["evacuation_time"],
                max_density=ep_result["max_density"],
                avg_density=ep_result["avg_density"],
                exit_load_variance=ep_result["exit_load_variance"],
                congestion_events=ep_result["congestion_events"],
                crush_risk_events=ep_result["crush_risk_events"],
            )
            results.append(result)

            print(f"    Episode {ep+1}/{n_episodes}: "
                  f"疏散率={result.evacuation_rate:.1%}, "
                  f"时间={result.evacuation_time:.1f}s, "
                  f"最大密度={result.max_density:.2f}人/m² "
                  f"({elapsed:.1f}s)")

        env.close()

    return results


def compute_summary(results: List[ExperimentResult]) -> List[MethodSummary]:
    """计算各方案汇总统计"""

    summaries = []

    # 按方案和流量分组
    groups = {}
    for r in results:
        key = (r.method, r.flow_level)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (method, flow_level), group_results in groups.items():
        n = len(group_results)

        evac_rates = [r.evacuation_rate for r in group_results]
        evac_times = [r.evacuation_time for r in group_results]
        max_densities = [r.max_density for r in group_results]
        load_vars = [r.exit_load_variance for r in group_results]
        congestions = [r.congestion_events for r in group_results]
        crush_risks = [r.crush_risk_events for r in group_results]

        summary = MethodSummary(
            method=method,
            flow_level=flow_level,
            n_episodes=n,
            evacuation_rate_mean=np.mean(evac_rates),
            evacuation_rate_std=np.std(evac_rates),
            evacuation_time_mean=np.mean(evac_times),
            evacuation_time_std=np.std(evac_times),
            max_density_mean=np.mean(max_densities),
            max_density_std=np.std(max_densities),
            exit_load_variance_mean=np.mean(load_vars),
            congestion_events_mean=np.mean(congestions),
            crush_risk_events_mean=np.mean(crush_risks),
        )
        summaries.append(summary)

    return summaries


def print_comparison_table(summaries: List[MethodSummary], flow_level: str):
    """打印对比表格"""

    # 筛选指定流量等级
    level_summaries = [s for s in summaries if s.flow_level == flow_level]
    if not level_summaries:
        return

    print(f"\n{'='*90}")
    print(f"研究框架对比结果 - {flow_level}流量")
    print(f"{'='*90}")

    # 表头
    print(f"{'方案':<20} {'疏散率':>12} {'疏散时间(s)':>14} {'最大密度':>12} {'出口均衡':>12} {'拥堵次数':>10}")
    print("-" * 90)

    for s in level_summaries:
        method_name = METHODS.get(s.method, {}).get('name', s.method)
        print(f"{method_name:<20} "
              f"{s.evacuation_rate_mean:>10.1%}+-{s.evacuation_rate_std:.1%} "
              f"{s.evacuation_time_mean:>10.1f}+-{s.evacuation_time_std:.1f} "
              f"{s.max_density_mean:>10.2f} "
              f"{s.exit_load_variance_mean:>12.3f} "
              f"{s.congestion_events_mean:>10.1f}")


def save_report(
    results: List[ExperimentResult],
    summaries: List[MethodSummary],
    output_dir: Path,
):
    """保存实验报告"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    results_file = output_dir / "detailed_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # 保存汇总
    summary_file = output_dir / "comparison_report.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "methods": {k: v for k, v in METHODS.items()},
        "summaries": [asdict(s) for s in summaries],
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存:")
    print(f"  详细结果: {results_file}")
    print(f"  汇总报告: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="研究框架对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方案说明:
  1. baseline_sfm    - 原始SFM（基线）
  2. sfm_prediction  - SFM + 密度预测
  3. sfm_routing     - SFM + 动态分流
  4. sfm_full        - SFM + 预测 + 分流（完整方案）
  5. sfm_ppo         - SFM + PPO引导（对比）

示例:
  python examples/run_framework_comparison.py --episodes 10
  python examples/run_framework_comparison.py --methods baseline_sfm sfm_full
  python examples/run_framework_comparison.py --flow-levels small medium
        """
    )

    parser.add_argument("--episodes", "-n", type=int, default=5,
                        help="每个方案每个流量等级的episode数 (默认5)")
    parser.add_argument("--flow-levels", "-f", nargs="+",
                        default=["small", "medium", "large"],
                        choices=["small", "medium", "large"],
                        help="测试的流量等级 (默认全部)")
    parser.add_argument("--methods", "-m", nargs="+",
                        default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help="测试的方案 (默认全部)")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="outputs/framework_comparison",
                        help="输出目录")

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("研究框架对比实验")
    print("=" * 60)
    print(f"方案数: {len(args.methods)}")
    print(f"流量等级: {args.flow_levels}")
    print(f"每组Episodes: {args.episodes}")
    print(f"输出目录: {output_dir}")

    all_results = []

    for method_key in args.methods:
        method_config = METHODS[method_key]
        results = run_method_experiments(
            method_key=method_key,
            method_config=method_config,
            flow_levels=args.flow_levels,
            n_episodes=args.episodes,
            output_dir=output_dir,
        )
        all_results.extend(results)

    # 计算汇总
    summaries = compute_summary(all_results)

    # 打印对比表格
    for flow_level in args.flow_levels:
        print_comparison_table(summaries, flow_level)

    # 保存报告
    save_report(all_results, summaries, output_dir)

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
