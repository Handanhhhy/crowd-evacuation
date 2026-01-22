"""
实验日志记录器
用于消融实验的结果记录、保存和可视化

功能:
- 记录训练过程中的指标
- 保存评估结果到CSV/JSON
- 生成训练曲线图
- 生成对比图表
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib不可用，图表生成功能已禁用")


@dataclass
class EpisodeMetrics:
    """单个episode的评估指标"""
    episode_id: int
    evacuation_rate: float          # 疏散完成率 (0-1)
    avg_evacuation_time: float      # 平均疏散时间 (steps)
    max_congestion: float           # 最大拥堵度 (人/m²)
    exit_balance: float             # 出口均衡度 (std)
    cumulative_reward: float        # 累计奖励
    episode_length: int             # episode长度 (steps)
    avg_speed: float = 0.0          # 平均行人速度 (m/s)
    guidance_count: int = 0         # 引导次数


@dataclass
class ExperimentResult:
    """实验结果汇总"""
    experiment_id: str
    experiment_name: str
    group: str
    config: Dict[str, Any]

    # 平均指标 (多个episode的平均)
    mean_evacuation_rate: float = 0.0
    std_evacuation_rate: float = 0.0
    mean_evacuation_time: float = 0.0
    std_evacuation_time: float = 0.0
    mean_max_congestion: float = 0.0
    std_max_congestion: float = 0.0
    mean_exit_balance: float = 0.0
    std_exit_balance: float = 0.0
    mean_cumulative_reward: float = 0.0
    std_cumulative_reward: float = 0.0

    # 训练信息
    training_timesteps: int = 0
    training_time_seconds: float = 0.0
    final_mean_reward: float = 0.0

    # 元数据
    timestamp: str = ""
    random_seed: int = 0

    # episode级别的详细数据
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        group: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化实验日志记录器

        Args:
            experiment_id: 实验ID (如 "A1_16D")
            experiment_name: 实验名称
            group: 实验组 (如 "A")
            output_dir: 输出目录
            config: 实验配置
        """
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.group = group
        self.config = config or {}

        # 创建输出目录
        self.output_dir = Path(output_dir) / f"{group}_{experiment_id.split('_')[0]}_{'_'.join(experiment_id.split('_')[1:])}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练日志
        self.training_log: List[Dict[str, Any]] = []

        # 评估结果
        self.episode_metrics: List[EpisodeMetrics] = []

        # 时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[ExperimentLogger] 初始化实验: {experiment_id}")
        print(f"[ExperimentLogger] 输出目录: {self.output_dir}")

    def log_training_step(
        self,
        timestep: int,
        mean_reward: float,
        std_reward: float = 0.0,
        episode_length: float = 0.0,
        **kwargs
    ):
        """
        记录训练步骤的指标

        Args:
            timestep: 当前训练步数
            mean_reward: 平均奖励
            std_reward: 奖励标准差
            episode_length: episode长度
            **kwargs: 其他指标
        """
        log_entry = {
            "timestep": timestep,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episode_length": episode_length,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.training_log.append(log_entry)

    def log_evaluation_episode(
        self,
        episode_id: int,
        evacuation_rate: float,
        avg_evacuation_time: float,
        max_congestion: float,
        exit_balance: float,
        cumulative_reward: float,
        episode_length: int,
        avg_speed: float = 0.0,
        guidance_count: int = 0
    ):
        """
        记录评估episode的指标

        Args:
            episode_id: episode编号
            evacuation_rate: 疏散完成率
            avg_evacuation_time: 平均疏散时间
            max_congestion: 最大拥堵度
            exit_balance: 出口均衡度
            cumulative_reward: 累计奖励
            episode_length: episode长度
            avg_speed: 平均速度
            guidance_count: 引导次数
        """
        metrics = EpisodeMetrics(
            episode_id=episode_id,
            evacuation_rate=evacuation_rate,
            avg_evacuation_time=avg_evacuation_time,
            max_congestion=max_congestion,
            exit_balance=exit_balance,
            cumulative_reward=cumulative_reward,
            episode_length=episode_length,
            avg_speed=avg_speed,
            guidance_count=guidance_count
        )
        self.episode_metrics.append(metrics)

        print(f"  [Episode {episode_id}] 疏散率: {evacuation_rate:.1%}, "
              f"时间: {avg_evacuation_time:.1f}步, "
              f"奖励: {cumulative_reward:.1f}")

    def compute_summary(self) -> ExperimentResult:
        """
        计算实验结果汇总

        Returns:
            ExperimentResult: 实验结果汇总
        """
        if not self.episode_metrics:
            print("[警告] 没有评估数据，返回空结果")
            return ExperimentResult(
                experiment_id=self.experiment_id,
                experiment_name=self.experiment_name,
                group=self.group,
                config=self.config,
                timestamp=self.timestamp
            )

        # 提取各指标
        evac_rates = [m.evacuation_rate for m in self.episode_metrics]
        evac_times = [m.avg_evacuation_time for m in self.episode_metrics]
        max_congs = [m.max_congestion for m in self.episode_metrics]
        balances = [m.exit_balance for m in self.episode_metrics]
        rewards = [m.cumulative_reward for m in self.episode_metrics]

        # 计算训练最终奖励
        final_reward = 0.0
        if self.training_log:
            final_reward = self.training_log[-1].get("mean_reward", 0.0)

        result = ExperimentResult(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            group=self.group,
            config=self.config,
            mean_evacuation_rate=float(np.mean(evac_rates)),
            std_evacuation_rate=float(np.std(evac_rates)),
            mean_evacuation_time=float(np.mean(evac_times)),
            std_evacuation_time=float(np.std(evac_times)),
            mean_max_congestion=float(np.mean(max_congs)),
            std_max_congestion=float(np.std(max_congs)),
            mean_exit_balance=float(np.mean(balances)),
            std_exit_balance=float(np.std(balances)),
            mean_cumulative_reward=float(np.mean(rewards)),
            std_cumulative_reward=float(np.std(rewards)),
            final_mean_reward=final_reward,
            timestamp=self.timestamp,
            episode_metrics=self.episode_metrics
        )

        return result

    def save_training_log(self):
        """保存训练日志到CSV"""
        if not self.training_log:
            print("[警告] 没有训练日志数据")
            return

        csv_path = self.output_dir / "training_log.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.training_log[0].keys())
            writer.writeheader()
            writer.writerows(self.training_log)

        print(f"[保存] 训练日志: {csv_path}")

    def save_evaluation_results(self) -> str:
        """
        保存评估结果到JSON

        Returns:
            str: JSON文件路径
        """
        result = self.compute_summary()
        json_path = self.output_dir / "eval_results.json"

        # 转换为可序列化的字典
        result_dict = {
            "experiment_id": result.experiment_id,
            "experiment_name": result.experiment_name,
            "group": result.group,
            "config": result.config,
            "metrics": {
                "evacuation_rate": {
                    "mean": result.mean_evacuation_rate,
                    "std": result.std_evacuation_rate
                },
                "evacuation_time": {
                    "mean": result.mean_evacuation_time,
                    "std": result.std_evacuation_time
                },
                "max_congestion": {
                    "mean": result.mean_max_congestion,
                    "std": result.std_max_congestion
                },
                "exit_balance": {
                    "mean": result.mean_exit_balance,
                    "std": result.std_exit_balance
                },
                "cumulative_reward": {
                    "mean": result.mean_cumulative_reward,
                    "std": result.std_cumulative_reward
                }
            },
            "training": {
                "final_mean_reward": result.final_mean_reward,
                "timesteps": result.training_timesteps,
                "time_seconds": result.training_time_seconds
            },
            "timestamp": result.timestamp,
            "episode_details": [asdict(m) for m in result.episode_metrics]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"[保存] 评估结果: {json_path}")
        return str(json_path)

    def save_config(self):
        """保存实验配置到YAML"""
        import yaml

        config_path = self.output_dir / "config.yaml"

        config_to_save = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "group": self.group,
            "timestamp": self.timestamp,
            **self.config
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, allow_unicode=True, default_flow_style=False)

        print(f"[保存] 实验配置: {config_path}")

    def plot_training_curve(self):
        """生成训练曲线图"""
        if not MATPLOTLIB_AVAILABLE:
            print("[警告] matplotlib不可用，跳过图表生成")
            return

        if not self.training_log:
            print("[警告] 没有训练日志数据，跳过图表生成")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        timesteps = [log["timestep"] for log in self.training_log]
        mean_rewards = [log["mean_reward"] for log in self.training_log]
        std_rewards = [log.get("std_reward", 0) for log in self.training_log]
        ep_lengths = [log.get("episode_length", 0) for log in self.training_log]

        # 奖励曲线
        ax1 = axes[0]
        ax1.plot(timesteps, mean_rewards, 'b-', label='Mean Reward')
        if any(std_rewards):
            ax1.fill_between(
                timesteps,
                np.array(mean_rewards) - np.array(std_rewards),
                np.array(mean_rewards) + np.array(std_rewards),
                alpha=0.2
            )
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title(f'{self.experiment_id}: Training Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode长度曲线
        ax2 = axes[1]
        if any(ep_lengths):
            ax2.plot(timesteps, ep_lengths, 'g-', label='Episode Length')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Episode Length')
        ax2.set_title(f'{self.experiment_id}: Episode Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = self.output_dir / "training_curve.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        print(f"[保存] 训练曲线: {fig_path}")

    def save_all(self):
        """保存所有结果"""
        self.save_config()
        self.save_training_log()
        self.save_evaluation_results()
        self.plot_training_curve()
        print(f"[完成] 实验 {self.experiment_id} 结果已保存到 {self.output_dir}")


class AblationSummaryGenerator:
    """消融实验汇总生成器"""

    def __init__(self, base_output_dir: str):
        """
        初始化汇总生成器

        Args:
            base_output_dir: 输出根目录 (如 outputs/ablation)
        """
        self.base_dir = Path(base_output_dir)
        self.summary_dir = self.base_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        """添加实验结果"""
        self.results.append(result)

    def load_results_from_dir(self):
        """从目录加载所有实验结果"""
        for group_dir in self.base_dir.iterdir():
            if group_dir.is_dir() and group_dir.name != "summary":
                json_path = group_dir / "eval_results.json"
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    result = ExperimentResult(
                        experiment_id=data["experiment_id"],
                        experiment_name=data["experiment_name"],
                        group=data["group"],
                        config=data.get("config", {}),
                        mean_evacuation_rate=data["metrics"]["evacuation_rate"]["mean"],
                        std_evacuation_rate=data["metrics"]["evacuation_rate"]["std"],
                        mean_evacuation_time=data["metrics"]["evacuation_time"]["mean"],
                        std_evacuation_time=data["metrics"]["evacuation_time"]["std"],
                        mean_max_congestion=data["metrics"]["max_congestion"]["mean"],
                        std_max_congestion=data["metrics"]["max_congestion"]["std"],
                        mean_exit_balance=data["metrics"]["exit_balance"]["mean"],
                        std_exit_balance=data["metrics"]["exit_balance"]["std"],
                        mean_cumulative_reward=data["metrics"]["cumulative_reward"]["mean"],
                        std_cumulative_reward=data["metrics"]["cumulative_reward"]["std"],
                        final_mean_reward=data.get("training", {}).get("final_mean_reward", 0),
                        timestamp=data.get("timestamp", "")
                    )
                    self.results.append(result)
                    print(f"[加载] {result.experiment_id}")

    def generate_summary_csv(self) -> str:
        """
        生成汇总CSV表格

        Returns:
            str: CSV文件路径
        """
        csv_path = self.summary_dir / "ablation_results.csv"

        fieldnames = [
            "experiment_id", "experiment_name", "group",
            "evacuation_rate_mean", "evacuation_rate_std",
            "evacuation_time_mean", "evacuation_time_std",
            "max_congestion_mean", "max_congestion_std",
            "exit_balance_mean", "exit_balance_std",
            "cumulative_reward_mean", "cumulative_reward_std",
            "final_mean_reward"
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in sorted(self.results, key=lambda x: (x.group, x.experiment_id)):
                writer.writerow({
                    "experiment_id": r.experiment_id,
                    "experiment_name": r.experiment_name,
                    "group": r.group,
                    "evacuation_rate_mean": f"{r.mean_evacuation_rate:.4f}",
                    "evacuation_rate_std": f"{r.std_evacuation_rate:.4f}",
                    "evacuation_time_mean": f"{r.mean_evacuation_time:.2f}",
                    "evacuation_time_std": f"{r.std_evacuation_time:.2f}",
                    "max_congestion_mean": f"{r.mean_max_congestion:.4f}",
                    "max_congestion_std": f"{r.std_max_congestion:.4f}",
                    "exit_balance_mean": f"{r.mean_exit_balance:.4f}",
                    "exit_balance_std": f"{r.std_exit_balance:.4f}",
                    "cumulative_reward_mean": f"{r.mean_cumulative_reward:.2f}",
                    "cumulative_reward_std": f"{r.std_cumulative_reward:.2f}",
                    "final_mean_reward": f"{r.final_mean_reward:.2f}"
                })

        print(f"[保存] 汇总表格: {csv_path}")
        return str(csv_path)

    def generate_comparison_chart(self):
        """生成对比图表"""
        if not MATPLOTLIB_AVAILABLE:
            print("[警告] matplotlib不可用，跳过图表生成")
            return

        if not self.results:
            print("[警告] 没有结果数据，跳过图表生成")
            return

        # 按组分组
        groups = {}
        for r in self.results:
            if r.group not in groups:
                groups[r.group] = []
            groups[r.group].append(r)

        # 为每个组生成对比图
        for group_name, group_results in groups.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            exp_ids = [r.experiment_id for r in group_results]

            # 疏散率对比
            ax1 = axes[0]
            evac_rates = [r.mean_evacuation_rate * 100 for r in group_results]
            evac_stds = [r.std_evacuation_rate * 100 for r in group_results]
            bars1 = ax1.bar(exp_ids, evac_rates, yerr=evac_stds, capsize=5, color='steelblue')
            ax1.set_ylabel('Evacuation Rate (%)')
            ax1.set_title(f'Group {group_name}: Evacuation Rate')
            ax1.set_ylim(0, 105)
            ax1.tick_params(axis='x', rotation=45)

            # 疏散时间对比
            ax2 = axes[1]
            evac_times = [r.mean_evacuation_time for r in group_results]
            time_stds = [r.std_evacuation_time for r in group_results]
            bars2 = ax2.bar(exp_ids, evac_times, yerr=time_stds, capsize=5, color='coral')
            ax2.set_ylabel('Evacuation Time (steps)')
            ax2.set_title(f'Group {group_name}: Evacuation Time')
            ax2.tick_params(axis='x', rotation=45)

            # 累计奖励对比
            ax3 = axes[2]
            rewards = [r.mean_cumulative_reward for r in group_results]
            reward_stds = [r.std_cumulative_reward for r in group_results]
            bars3 = ax3.bar(exp_ids, rewards, yerr=reward_stds, capsize=5, color='forestgreen')
            ax3.set_ylabel('Cumulative Reward')
            ax3.set_title(f'Group {group_name}: Cumulative Reward')
            ax3.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            fig_path = self.summary_dir / f"comparison_{group_name}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()

            print(f"[保存] 组{group_name}对比图: {fig_path}")

    def generate_report(self) -> str:
        """
        生成消融实验报告 (Markdown)

        Returns:
            str: 报告文件路径
        """
        report_path = self.summary_dir / "ablation_report.md"

        lines = [
            "# 消融实验报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 实验概述",
            "",
            f"总实验数: {len(self.results)}",
            "",
        ]

        # 按组生成结果表格
        groups = {}
        for r in self.results:
            if r.group not in groups:
                groups[r.group] = []
            groups[r.group].append(r)

        for group_name in sorted(groups.keys()):
            group_results = groups[group_name]

            lines.extend([
                f"## 组 {group_name}",
                "",
                "| 实验ID | 名称 | 疏散率 | 疏散时间 | 最大拥堵 | 出口均衡 | 累计奖励 |",
                "|--------|------|--------|----------|----------|----------|----------|",
            ])

            for r in sorted(group_results, key=lambda x: x.experiment_id):
                lines.append(
                    f"| {r.experiment_id} | {r.experiment_name} | "
                    f"{r.mean_evacuation_rate:.1%}±{r.std_evacuation_rate:.1%} | "
                    f"{r.mean_evacuation_time:.1f}±{r.std_evacuation_time:.1f} | "
                    f"{r.mean_max_congestion:.2f}±{r.std_max_congestion:.2f} | "
                    f"{r.mean_exit_balance:.2f}±{r.std_exit_balance:.2f} | "
                    f"{r.mean_cumulative_reward:.1f}±{r.std_cumulative_reward:.1f} |"
                )

            lines.append("")

        # 关键发现
        lines.extend([
            "## 关键发现",
            "",
            "TODO: 根据实验结果添加分析",
            "",
            "## 文件列表",
            "",
            "- `ablation_results.csv`: 汇总数据表格",
            "- `comparison_*.png`: 各组对比图表",
            "- `ablation_report.md`: 本报告",
            "",
        ])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"[保存] 消融报告: {report_path}")
        return str(report_path)

    def generate_all(self):
        """生成所有汇总文件"""
        self.generate_summary_csv()
        self.generate_comparison_chart()
        self.generate_report()
        print(f"[完成] 汇总文件已保存到 {self.summary_dir}")


if __name__ == "__main__":
    # 测试代码
    logger = ExperimentLogger(
        experiment_id="A1_16D",
        experiment_name="Full Observation (16D)",
        group="A",
        output_dir="outputs/ablation",
        config={"observation_dim": 16}
    )

    # 模拟训练日志
    for i in range(10):
        logger.log_training_step(
            timestep=i * 1000,
            mean_reward=50 + i * 5 + np.random.randn() * 2,
            std_reward=5,
            episode_length=300 - i * 10
        )

    # 模拟评估结果
    for ep in range(5):
        logger.log_evaluation_episode(
            episode_id=ep,
            evacuation_rate=0.9 + np.random.rand() * 0.1,
            avg_evacuation_time=250 + np.random.randn() * 20,
            max_congestion=1.5 + np.random.rand() * 0.5,
            exit_balance=10 + np.random.randn() * 3,
            cumulative_reward=150 + np.random.randn() * 20,
            episode_length=300,
            avg_speed=1.2 + np.random.rand() * 0.2,
            guidance_count=5 + int(np.random.rand() * 5)
        )

    # 保存所有结果
    logger.save_all()

    print("\n测试完成!")
