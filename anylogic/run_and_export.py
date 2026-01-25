#!/usr/bin/env python
"""
运行疏散仿真并导出 AnyLogic 可视化数据

用法:
    # 默认运行 (medium 流量, full 方法)
    python anylogic/run_and_export.py
    
    # 指定参数
    python anylogic/run_and_export.py --flow large --method ppo
    
    # 快速测试
    python anylogic/run_and_export.py --quick
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from simulation.large_station_env import LargeStationEnv
from anylogic.trajectory_exporter import TrajectoryExporter


def load_ppo_model(model_path: str):
    """加载 PPO 模型"""
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"[PPO] 加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"[PPO] 加载失败: {e}")
        return None


def get_action(method: str, env, obs, model=None):
    """根据方法选择动作
    
    Args:
        method: 仿真方法
        env: 环境
        obs: 观测
        model: PPO 模型 (可选)
        
    Returns:
        动作
    """
    if method == "ppo" and model is not None:
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    elif method in ["routing", "full", "prediction"]:
        # 动态分流策略: 选择负载最小的出口
        n_exits = env.action_space.n
        
        if hasattr(env, 'evacuated_by_exit') and env.evacuated_by_exit:
            loads = np.array([env.evacuated_by_exit.get(i, 0) for i in range(n_exits)])
        else:
            loads = np.zeros(n_exits)
        
        # 选择负载最小的出口 (添加小扰动)
        scores = -loads + np.random.uniform(0, 0.1, n_exits)
        return int(np.argmax(scores))
    
    else:
        # baseline: 随机动作
        return env.action_space.sample()


def run_and_export(
    flow_level: str = "medium",
    method: str = "full",
    model_path: str = None,
    output_dir: str = "anylogic/exported_data",
    max_steps: int = 3000,
    export_interval: int = 1,
    use_gpu: bool = True,
    verbose: bool = True,
):
    """运行仿真并导出数据
    
    Args:
        flow_level: 人流等级 (small/medium/large)
        method: 仿真方法 (baseline/prediction/routing/full/ppo)
        model_path: PPO 模型路径
        output_dir: 输出目录
        max_steps: 最大仿真步数
        export_interval: 导出间隔
        use_gpu: 是否使用 GPU SFM
        verbose: 是否显示详细信息
        
    Returns:
        导出的文件路径字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("AnyLogic 数据导出")
    print("=" * 60)
    print(f"  流量等级: {flow_level}")
    print(f"  仿真方法: {method}")
    print(f"  最大步数: {max_steps}")
    print(f"  导出间隔: {export_interval}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)
    
    # 创建导出器
    exporter = TrajectoryExporter(
        output_dir=output_dir,
        export_interval=export_interval,
    )
    
    # 创建环境
    print("\n[1/4] 初始化环境...")
    env = LargeStationEnv(
        flow_level=flow_level,
        use_gpu_sfm=use_gpu,
        render_mode=None,
    )
    
    # 加载 PPO 模型 (如果需要)
    model = None
    if method == "ppo":
        if model_path is None:
            model_path = str(project_root / "outputs/models/ppo_large_station_small.zip")
        
        if Path(model_path).exists():
            model = load_ppo_model(model_path)
        
        if model is None:
            print("[警告] PPO 模型加载失败, 回退到 full 方法")
            method = "full"
    
    # 设置元数据
    exporter.set_metadata(
        flow_level=flow_level,
        method=method,
        total_pedestrians=env.n_pedestrians,
        dt=env.dt,
        max_steps=max_steps,
        export_interval=export_interval,
        model_path=model_path if method == "ppo" else None,
    )
    
    # 重置环境并提取场景布局
    print("[2/4] 重置环境并提取场景布局...")
    obs, _ = env.reset()
    exporter.set_scene_layout(env)
    
    # 运行仿真
    print(f"[3/4] 运行仿真... (目标: {env.n_pedestrians} 人疏散)")
    start_time = time.time()
    
    step = 0
    done = False
    truncated = False
    
    # 统计
    last_report_step = 0
    report_interval = 300  # 每300步报告一次
    
    while step < max_steps and not (done or truncated):
        sim_time = step * env.dt
        
        # 记录当前帧
        exporter.record_frame(env, sim_time)
        
        # 选择动作
        action = get_action(method, env, obs, model)
        
        # 执行步骤
        obs, reward, done, truncated, info = env.step(action)
        
        # 进度报告
        if verbose and (step - last_report_step >= report_interval or done or truncated):
            elapsed = time.time() - start_time
            evacuated = info.get('evacuated', 0)
            remaining = info.get('remaining', env.n_pedestrians - evacuated)
            evac_rate = evacuated / env.n_pedestrians * 100
            max_density = info.get('max_density', 0)
            
            print(f"  [{sim_time:.1f}s] 疏散: {evacuated}/{env.n_pedestrians} ({evac_rate:.1f}%) | "
                  f"剩余: {remaining} | 最大密度: {max_density:.2f}/m² | "
                  f"耗时: {elapsed:.1f}s")
            last_report_step = step
        
        step += 1
    
    # 仿真完成统计
    total_time = time.time() - start_time
    final_evacuated = info.get('evacuated', 0)
    final_evac_rate = final_evacuated / env.n_pedestrians * 100
    
    print(f"\n仿真完成:")
    print(f"  - 总步数: {step}")
    print(f"  - 仿真时间: {step * env.dt:.1f}s")
    print(f"  - 实际耗时: {total_time:.1f}s")
    print(f"  - 疏散率: {final_evac_rate:.1f}%")
    print(f"  - 疏散人数: {final_evacuated}/{env.n_pedestrians}")
    
    # 更新元数据统计
    exporter.metadata['statistics'] = {
        'evacuation_rate': round(final_evac_rate / 100, 4),
        'total_evacuated': final_evacuated,
        'total_steps': step,
        'simulation_time': round(step * env.dt, 2),
        'real_time': round(total_time, 2),
        'max_density': round(info.get('max_density', 0), 2),
    }
    
    # 导出数据
    print(f"\n[4/4] 导出数据...")
    paths = exporter.export_all()
    
    # 显示摘要
    summary = exporter.get_summary()
    print(f"\n导出摘要:")
    print(f"  - 轨迹记录: {summary['total_records']}")
    print(f"  - 行人数: {summary['unique_pedestrians']}")
    print(f"  - 时间帧: {summary['unique_timestamps']}")
    print(f"  - 疏散事件: {summary['evacuation_events']}")
    
    # 关闭环境
    env.close()
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="运行疏散仿真并导出 AnyLogic 可视化数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认运行
  python anylogic/run_and_export.py
  
  # 大流量 + PPO 方法
  python anylogic/run_and_export.py --flow large --method ppo
  
  # 快速测试
  python anylogic/run_and_export.py --quick
  
  # 降低数据量 (每5步导出一次)
  python anylogic/run_and_export.py --export-interval 5

方法说明:
  baseline   - 基线 SFM (随机动作)
  prediction - SFM + 密度预测
  routing    - SFM + 动态分流
  full       - SFM + 预测 + 分流 (完整方法)
  ppo        - SFM + PPO 引导 (需要训练模型)
        """
    )
    
    parser.add_argument(
        "--flow", 
        default="medium", 
        choices=["small", "medium", "large"],
        help="人流等级 (default: medium)"
    )
    parser.add_argument(
        "--method", 
        default="full",
        choices=["baseline", "prediction", "routing", "full", "ppo"],
        help="仿真方法 (default: full)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="PPO 模型路径 (method=ppo 时使用)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anylogic/exported_data",
        help="输出目录 (default: anylogic/exported_data)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="最大仿真步数 (default: 3000)"
    )
    parser.add_argument(
        "--export-interval",
        type=int,
        default=1,
        help="导出间隔, 每N步导出一次 (default: 1)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式: small 流量, 1000 步, 间隔 5"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用 GPU SFM"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式, 减少输出"
    )
    
    args = parser.parse_args()
    
    # 快速模式覆盖
    if args.quick:
        args.flow = "small"
        args.max_steps = 1000
        args.export_interval = 5
        print("[快速模式] flow=small, max_steps=1000, interval=5")
    
    # 运行导出
    paths = run_and_export(
        flow_level=args.flow,
        method=args.method,
        model_path=args.model,
        output_dir=args.output,
        max_steps=args.max_steps,
        export_interval=args.export_interval,
        use_gpu=not args.no_gpu,
        verbose=not args.quiet,
    )
    
    print("\n" + "=" * 60)
    print("导出完成! 文件列表:")
    for name, path in paths.items():
        if path:
            print(f"  - {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
