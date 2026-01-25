#!/usr/bin/env python
"""
完整训练流水线脚本

执行顺序：
1. 用Jülich真实数据训练密度预测模型（快，必须先做）
2. 研究框架对比实验（依赖Step1）
3. PPO训练（依赖Step1）
4. 完整泛化验证（依赖Step1,3，验证密度预测+PPO+完整系统）
5. 消融实验（独立，放最后，可选，不影响核心流程）

使用方法:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --quick
    python scripts/run_full_pipeline.py --skip-ablation
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 步骤定义
STEPS = [
    {
        "id": 1,
        "name": "density_predictor",
        "description": "用Jülich真实数据训练密度预测模型",
        "script": "scripts/train_density_predictor.py",
        "args_normal": [
            "--data-source", "juelich",
            "--train",
            "--epochs", "100",
            "--pred-horizon", "10",
        ],
        "args_quick": [
            "--data-source", "juelich",
            "--train",
            "--epochs", "20",
            "--pred-horizon", "10",
        ],
        "output_file": "outputs/models/density_predictor.pt",
    },
    {
        "id": 2,
        "name": "framework_comparison",
        "description": "研究框架对比实验（5组方案）",
        "script": "examples/run_framework_comparison.py",
        "args_normal": [
            "--episodes", "5",
        ],
        "args_quick": [
            "--quick",  # baseline vs full, small flow, 2 episodes
        ],
        "output_file": "outputs/framework_comparison/comparison_report.json",
    },
    {
        "id": 3,
        "name": "ppo_training",
        "description": "训练PPO大站策略（小流量）",
        "script": "examples/train_ppo_large_station.py",
        "args_normal": [
            "--flow-level", "small",
            "--total-timesteps", "500000",
        ],
        "args_quick": [
            "--flow-level", "small",
            "--total-timesteps", "100000",
        ],
        "output_file": "outputs/models/ppo_large_station_small.zip",
    },
    {
        "id": 4,
        "name": "generalization_test",
        "description": "完整泛化验证（密度预测+PPO+完整系统）",
        "script": "scripts/run_generalization_test.py",
        "args_normal": [],
        "args_quick": [
            "--quick",
        ],
        "output_file": "outputs/generalization/generalization_report.json",
    },
    {
        "id": 5,
        "name": "ablation",
        "description": "运行消融实验（17组对比，可选）",
        "script": "examples/run_ablation.py",
        "args_normal": [
            "--timesteps", "100000",
        ],
        "args_quick": [
            "--timesteps", "10000",
        ],
        "output_file": "outputs/ablation/summary_report.json",
        "skippable": True,  # 可以通过 --skip-ablation 跳过
    },
]


def run_step(step: dict, quick: bool = False) -> dict:
    """执行单个步骤"""
    args = step.get("args_quick") if quick and step.get("args_quick") else step["args_normal"]
    if args is None:
        args = step["args_normal"]

    cmd = [sys.executable, str(PROJECT_ROOT / step["script"])] + args

    print(f"\n{'='*70}")
    print(f"Step {step['id']}: {step['description']}")
    print(f"{'='*70}")
    print(f"命令: {' '.join(cmd)}")
    print()

    start_time = datetime.now()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    success = result.returncode == 0

    # 检查输出文件
    output_exists = True
    if step.get("output_file"):
        output_path = PROJECT_ROOT / step["output_file"]
        output_exists = output_path.exists()

    return {
        "step_id": step["id"],
        "name": step["name"],
        "description": step["description"],
        "success": success and output_exists,
        "return_code": result.returncode,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "output_file": step.get("output_file"),
        "output_exists": output_exists,
    }


def run_pipeline(
    start_step: int = 1,
    only_step: Optional[int] = None,
    skip_ablation: bool = False,
    quick: bool = False,
) -> dict:
    """运行完整流水线"""

    results = {
        "pipeline_start": datetime.now().isoformat(),
        "mode": "quick" if quick else "normal",
        "skip_ablation": skip_ablation,
        "steps": [],
    }

    for step in STEPS:
        step_id = step["id"]

        # 跳过逻辑
        if only_step is not None and step_id != only_step:
            continue
        if step_id < start_step:
            print(f"跳过 Step {step_id}: {step['name']} (start_step={start_step})")
            continue
        if skip_ablation and step.get("skippable"):
            print(f"跳过 Step {step_id}: {step['name']} (--skip-ablation)")
            continue

        # 执行步骤
        step_result = run_step(step, quick=quick)
        results["steps"].append(step_result)

        # 打印结果
        status = "成功" if step_result["success"] else "失败"
        print(f"\n{status} - 耗时: {step_result['duration_seconds']:.1f}秒")

        # 如果失败，停止流水线
        if not step_result["success"]:
            print(f"\n警告: Step {step_id} 失败，流水线停止")
            break

    results["pipeline_end"] = datetime.now().isoformat()

    # 计算总耗时
    total_duration = sum(s["duration_seconds"] for s in results["steps"])
    results["total_duration_seconds"] = total_duration

    # 保存结果日志
    log_dir = PROJECT_ROOT / "outputs" / "pipeline_results"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_log_{timestamp}.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("流水线执行完成")
    print(f"{'='*70}")
    print(f"总耗时: {total_duration/60:.1f} 分钟")
    print(f"日志保存: {log_file}")

    # 打印汇总
    print("\n步骤汇总:")
    for step_result in results["steps"]:
        status = "[OK]" if step_result["success"] else "[FAIL]"
        print(f"  {status} Step {step_result['step_id']}: {step_result['name']} ({step_result['duration_seconds']:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="完整训练流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_full_pipeline.py                # 完整执行
  python scripts/run_full_pipeline.py --quick        # 快速模式
  python scripts/run_full_pipeline.py --skip-ablation  # 跳过消融实验
  python scripts/run_full_pipeline.py --start-step 2   # 从Step 2开始
  python scripts/run_full_pipeline.py --only-step 1    # 只执行Step 1
        """
    )

    parser.add_argument("--start-step", type=int, default=1,
                        help="从指定步骤开始（默认1）")
    parser.add_argument("--only-step", type=int, default=None,
                        help="只执行指定步骤")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="跳过消融实验（Step 5）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（减少训练步数）")

    args = parser.parse_args()

    run_pipeline(
        start_step=args.start_step,
        only_step=args.only_step,
        skip_ablation=args.skip_ablation,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
