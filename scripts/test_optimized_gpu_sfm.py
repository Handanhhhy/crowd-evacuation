#!/usr/bin/env python3
"""测试优化版GPU SFM性能"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.large_station_env import LargeStationEnv


def test_performance():
    """测试优化版GPU SFM性能"""
    print("=" * 60)
    print("测试优化版GPU SFM性能")
    print("=" * 60)

    # 测试配置
    flow_level = "large"  # 3000人
    n_steps = 100

    print(f"\n配置: flow_level={flow_level}, 测试步数={n_steps}")

    # 创建环境（默认使用优化版GPU SFM）
    print("\n创建环境...")
    env = LargeStationEnv(
        flow_level=flow_level,
        use_optimized_gpu_sfm=True,
        emergency_mode=True,
    )
    print(f"  使用优化版GPU SFM: {env.use_optimized_gpu_sfm}")

    # 重置环境（此时创建SFM）
    print("\n重置环境...")
    obs, info = env.reset()
    print(f"  设备: {env.sfm.device}")
    print(f"  初始观测维度: {obs.shape}")
    print(f"  初始行人数: {env.sfm.n_pedestrians}")

    # 测试性能
    print(f"\n运行 {n_steps} 步仿真...")
    start_time = time.time()

    step_times = []
    for i in range(n_steps):
        step_start = time.time()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_times.append(time.time() - step_start)

        if (i + 1) % 20 == 0:
            active_count = env.sfm.get_active_count()
            avg_step_time = sum(step_times[-20:]) / 20
            print(f"  步骤 {i+1}: 活跃行人={active_count}, "
                  f"疏散={info['evacuated']}, "
                  f"平均步时={avg_step_time*1000:.1f}ms")

        if terminated:
            print(f"\n  所有行人已疏散，共 {i+1} 步")
            break

    total_time = time.time() - start_time
    avg_step_time = total_time / len(step_times)

    print("\n" + "=" * 60)
    print("性能统计")
    print("=" * 60)
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均每步: {avg_step_time*1000:.2f}ms")
    print(f"  最大每步: {max(step_times)*1000:.2f}ms")
    print(f"  最小每步: {min(step_times)*1000:.2f}ms")
    print(f"  总疏散人数: {info['evacuated']}")
    print(f"  疏散率: {info['evacuated'] / env.n_pedestrians * 100:.1f}%")

    # 性能对比预期
    print("\n性能对比:")
    print(f"  原版GPU SFM (3000人): ~14s/步")
    print(f"  优化版GPU SFM (3000人): ~{avg_step_time*1000:.2f}ms/步")
    if avg_step_time < 1.0:  # 小于1秒
        speedup = 14.0 / avg_step_time
        print(f"  加速比: ~{speedup:.0f}x")
        print("\n✅ 优化成功！")
    else:
        print("\n⚠️ 性能未达预期，请检查")

    env.close()


if __name__ == "__main__":
    test_performance()
