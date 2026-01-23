"""
动态分流测试: SFM + 动态分流规则引擎
对比基线(无分流)的效果

运行: python scripts/routing_test.py --n-peds 100 --max-steps 300
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
from simulation.large_station_env import LargeStationEnv
from routing.dynamic_router import DynamicRouter, ExitInfo
import time
import argparse
import json
import os
from datetime import datetime


def run_routing_test(flow_level='small', max_steps=3000, n_peds=None, verbose=True):
    """运行动态分流测试"""

    if verbose:
        print("=" * 60)
        print("动态分流测试: SFM + 规则引擎")
        print(f"流量等级: {flow_level}" + (f" (自定义人数: {n_peds})" if n_peds else ""))
        print("=" * 60)

    env = LargeStationEnv(flow_level=flow_level)

    # 自定义人数
    if n_peds is not None:
        env.n_pedestrians = n_peds
        env.upper_layer_count = n_peds // 2
        env.lower_layer_count = n_peds - env.upper_layer_count

    obs, _ = env.reset()

    # 初始化动态分流器
    exit_infos = [
        ExitInfo(
            id=exit_obj.id,
            position=exit_obj.position,
            capacity=exit_obj.capacity,
        )
        for exit_obj in env.exits
    ]
    router = DynamicRouter(exits=exit_infos)

    if verbose:
        print(f"\n场景: 150m x 80m (T形)")
        print(f"总人数: {env.n_pedestrians}")
        print(f"涌入点: {env.n_escalators}个")
        print(f"出口: {env.n_exits}个")
        print("-" * 60)

    start_time = time.time()
    step = 0
    total_reward = 0
    reassign_interval = 10  # 每10步重新评估一次

    while step < max_steps:
        # 更新出口状态
        exit_densities = {}
        exit_congestions = {}
        exit_loads = {}

        for exit_obj in env.exits:
            density, congestion = env._compute_exit_metrics(exit_obj)
            exit_densities[exit_obj.id] = density
            exit_congestions[exit_obj.id] = congestion
            # 计算当前朝向该出口的人数
            load = sum(
                1 for ped in env.sfm.pedestrians
                if np.linalg.norm(ped.target - exit_obj.position) < 5.0
            )
            exit_loads[exit_obj.id] = load

        router.update_exit_states(exit_densities, exit_congestions, exit_loads)

        # 动态分流：为每个行人分配最优出口
        for ped in env.sfm.pedestrians:
            # 找到当前目标出口ID
            current_target_id = None
            for exit_obj in env.exits:
                if np.linalg.norm(ped.target - exit_obj.position) < 1.0:
                    current_target_id = exit_obj.id
                    break

            # 获取新分配
            new_exit_id, new_exit_pos = router.assign_exit(
                pedestrian_pos=ped.position,
                current_target_id=current_target_id,
                force_reassign=(step % reassign_interval == 0),
            )

            # 更新行人目标
            ped.target = new_exit_pos

        # 执行仿真步骤（使用随机动作，因为引导已通过router完成）
        action = 0  # 动作不再重要，分流由router完成
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # 输出状态
        if verbose and step % 300 == 0:
            sim_time = step * env.dt
            evac_rate = info['evacuated'] / env.n_pedestrians * 100
            stats = router.get_statistics()
            print(f"[{sim_time:5.1f}s] 疏散: {info['evacuated']:4d} ({evac_rate:5.1f}%) | "
                  f"剩余: {info['remaining']:4d} | 均衡度: {stats['balance_ratio']:.2f}")

        if terminated:
            break

    elapsed = time.time() - start_time
    sim_time = step * env.dt
    evac_rate = info['evacuated'] / env.n_pedestrians * 100
    stats = router.get_statistics()

    if verbose:
        print("-" * 60)
        print(f"仿真时间: {sim_time:.1f}s ({sim_time/60:.2f}分钟)")
        print(f"实际耗时: {elapsed:.1f}s")
        print(f"疏散人数: {info['evacuated']}/{env.n_pedestrians} ({evac_rate:.1f}%)")
        print(f"剩余人数: {info['remaining']}")
        print(f"累计奖励: {total_reward:.1f}")
        print(f"负载均衡度: {stats['balance_ratio']:.2f}")
        print(f"重分配次数: {stats['reassignment_count']}")
        print(f"状态: {'完成' if terminated else '超时'}")

        print("\n各出口疏散人数:")
        for exit_id, count in sorted(info['evacuated_by_exit'].items()):
            if count > 0:
                pct = 100 * count / max(info['evacuated'], 1)
                print(f"  {exit_id}: {count:4d} ({pct:5.1f}%)")

    result = {
        'timestamp': datetime.now().isoformat(),
        'method': 'sfm_dynamic_routing',
        'flow_level': flow_level,
        'total_pedestrians': env.n_pedestrians,
        'evacuated': info['evacuated'],
        'evacuation_rate': evac_rate,
        'remaining': info['remaining'],
        'sim_time': sim_time,
        'real_time': elapsed,
        'total_reward': total_reward,
        'completed': terminated,
        'evacuated_by_exit': info['evacuated_by_exit'],
        'max_steps': max_steps,
        'balance_ratio': stats['balance_ratio'],
        'reassignment_count': stats['reassignment_count'],
    }

    save_result(result)
    return result


def save_result(result: dict):
    """保存测试结果"""
    output_dir = 'outputs/results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/routing_{result['flow_level']}_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {filename}")

    summary_file = f"{output_dir}/routing_summary.jsonl"
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"已追加到汇总: {summary_file}")


def compare_with_baseline():
    """与基线对比"""
    from scripts.baseline_test import run_baseline_test

    print("\n" + "=" * 60)
    print("对比测试: 基线 vs 动态分流")
    print("=" * 60)

    # 基线测试
    print("\n>>> 运行基线测试...")
    baseline = run_baseline_test(n_peds=100, max_steps=300, verbose=False)

    # 动态分流测试
    print("\n>>> 运行动态分流测试...")
    routing = run_routing_test(n_peds=100, max_steps=300, verbose=False)

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'指标':<15} {'基线':<15} {'动态分流':<15} {'提升':<10}")
    print("-" * 60)

    evac_improve = routing['evacuation_rate'] - baseline['evacuation_rate']
    print(f"{'疏散率':<15} {baseline['evacuation_rate']:>6.1f}%        "
          f"{routing['evacuation_rate']:>6.1f}%        {evac_improve:>+.1f}%")

    print(f"{'累计奖励':<15} {baseline['total_reward']:>10.1f}     "
          f"{routing['total_reward']:>10.1f}     {routing['total_reward']-baseline['total_reward']:>+.1f}")

    balance = routing.get('balance_ratio', 0)
    print(f"{'负载均衡度':<15} {'N/A':<15} {balance:>6.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动态分流测试')
    parser.add_argument('--flow-level', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='流量等级')
    parser.add_argument('--max-steps', type=int, default=3000,
                        help='最大步数')
    parser.add_argument('--n-peds', type=int, default=None,
                        help='自定义人数')
    parser.add_argument('--compare', action='store_true',
                        help='与基线对比')

    args = parser.parse_args()

    if args.compare:
        compare_with_baseline()
    else:
        run_routing_test(
            flow_level=args.flow_level,
            max_steps=args.max_steps,
            n_peds=args.n_peds,
        )
