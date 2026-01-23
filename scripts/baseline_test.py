"""
基线测试: 原始SFM疏散效果 (无智能引导)
用于对比后续的密度预测+动态分流方案

运行: .venv/bin/python scripts/baseline_test.py
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from simulation.large_station_env import LargeStationEnv
import time
import argparse


def run_baseline_test(flow_level='small', max_steps=3000, n_peds=None, verbose=True):
    """运行基线测试

    Args:
        flow_level: 流量等级 (small/medium/large)
        max_steps: 最大步数
        n_peds: 自定义人数 (None则使用flow_level默认值)
        verbose: 是否输出详细信息
    """

    if verbose:
        print("=" * 60)
        print(f"基线测试: 原始SFM疏散效果 (无智能引导)")
        print(f"流量等级: {flow_level}" + (f" (自定义人数: {n_peds})" if n_peds else ""))
        print("=" * 60)

    env = LargeStationEnv(flow_level=flow_level)

    # 自定义人数 (用于快速测试)
    if n_peds is not None:
        env.n_pedestrians = n_peds
        env.upper_layer_count = n_peds // 2
        env.lower_layer_count = n_peds - env.upper_layer_count

    obs, _ = env.reset()

    if verbose:
        print(f"\n场景: 150m x 80m (T形)")
        print(f"总人数: {env.n_pedestrians}")
        print(f"上层初始: {env.upper_layer_count}")
        print(f"下层涌入: {env.lower_layer_count}")
        print(f"涌入点: {env.n_escalators}个 (5扶梯+2步梯)")
        print(f"出口: {env.n_exits}个 (8组闸机)")
        print("-" * 60)

    start_time = time.time()
    step = 0
    total_reward = 0

    while step < max_steps:
        # 随机动作 (无引导策略)
        action = np.random.randint(0, env.n_exits)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # 每30秒输出一次状态
        if verbose and step % 300 == 0:
            sim_time = step * env.dt
            evac_rate = info['evacuated'] / env.n_pedestrians * 100
            print(f"[{sim_time:5.1f}s] 疏散: {info['evacuated']:4d} ({evac_rate:5.1f}%) | "
                  f"剩余: {info['remaining']:4d} | 涌入: {info['spawned_from_lower']:4d}")

        if terminated:
            break

    elapsed = time.time() - start_time
    sim_time = step * env.dt
    evac_rate = info['evacuated'] / env.n_pedestrians * 100

    if verbose:
        print("-" * 60)
        print(f"仿真时间: {sim_time:.1f}s ({sim_time/60:.2f}分钟)")
        print(f"实际耗时: {elapsed:.1f}s")
        print(f"疏散人数: {info['evacuated']}/{env.n_pedestrians} ({evac_rate:.1f}%)")
        print(f"剩余人数: {info['remaining']}")
        print(f"累计奖励: {total_reward:.1f}")
        print(f"状态: {'完成' if terminated else '超时'}")

        # 各出口疏散统计
        print("\n各出口疏散人数:")
        for exit_id, count in sorted(info['evacuated_by_exit'].items()):
            if count > 0:
                pct = 100 * count / max(info['evacuated'], 1)
                print(f"  {exit_id}: {count:4d} ({pct:5.1f}%)")

    return {
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
    }


def run_all_levels():
    """测试所有流量等级"""
    print("\n" + "=" * 60)
    print("测试所有流量等级")
    print("=" * 60)

    results = []
    for level in ['small', 'medium', 'large']:
        print(f"\n>>> 测试 {level} 流量...")
        result = run_baseline_test(flow_level=level, verbose=True)
        results.append(result)

    # 汇总表格
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"{'流量等级':<10} {'总人数':<8} {'疏散率':<10} {'仿真时间':<12} {'状态':<8}")
    print("-" * 60)
    for r in results:
        status = '完成' if r['completed'] else '超时'
        print(f"{r['flow_level']:<10} {r['total_pedestrians']:<8} "
              f"{r['evacuation_rate']:>6.1f}%    {r['sim_time']:>6.1f}s      {status:<8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基线测试: 原始SFM疏散效果')
    parser.add_argument('--flow-level', type=str, default='small',
                        choices=['small', 'medium', 'large', 'all'],
                        help='流量等级 (default: small)')
    parser.add_argument('--max-steps', type=int, default=3000,
                        help='最大步数 (default: 3000, 即5分钟)')
    parser.add_argument('--n-peds', type=int, default=None,
                        help='自定义人数 (用于快速测试，如100)')

    args = parser.parse_args()

    if args.flow_level == 'all':
        run_all_levels()
    else:
        run_baseline_test(flow_level=args.flow_level, max_steps=args.max_steps, n_peds=args.n_peds)
