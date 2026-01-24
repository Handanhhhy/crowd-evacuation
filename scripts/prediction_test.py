#!/usr/bin/env python
"""
密度场预测对比实验

对比实验：
- SFM + 动态分流（当前密度）
- SFM + 动态分流（预测密度）

评估指标：
- 疏散时间
- 最大密度
- 出口均衡度
- 拥堵次数

参考文档: docs/new_station_plan.md 密度场预测模块 TODO
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.large_station_env import LargeStationEnv
from prediction import DensityFieldPredictor, GRID_SIZE, MAX_SAFE_DENSITY
from routing import DynamicRouter, ExitInfo


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        flow_level: str = "small",
        max_steps: int = 6000,
        dt: float = 0.1,
        prediction_model_path: str = None,
        output_dir: str = "outputs/experiments",
    ):
        """
        Args:
            flow_level: 人流量等级
            max_steps: 最大步数
            dt: 时间步长
            prediction_model_path: 预测模型路径
            output_dir: 输出目录
        """
        self.flow_level = flow_level
        self.max_steps = max_steps
        self.dt = dt
        self.prediction_model_path = prediction_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_env(self) -> LargeStationEnv:
        """创建环境"""
        return LargeStationEnv(
            flow_level=self.flow_level,
            max_steps=self.max_steps,
            dt=self.dt,
            emergency_mode=True,
        )
    
    def create_router(self, env: LargeStationEnv) -> DynamicRouter:
        """创建动态分流器"""
        exits = [
            ExitInfo(
                id=e.id,
                position=e.position.copy(),
                capacity=e.capacity,
            )
            for e in env.exits
        ]
        return DynamicRouter(exits=exits)
    
    def create_predictor(self, env: LargeStationEnv) -> DensityFieldPredictor:
        """创建密度预测器"""
        exits = [{'id': e.id, 'position': e.position.copy()} for e in env.exits]
        return DensityFieldPredictor(
            exits=exits,
            model_path=self.prediction_model_path,
        )
    
    def run_baseline(self, n_runs: int = 5) -> List[Dict]:
        """运行基线实验：SFM + 当前密度分流
        
        Args:
            n_runs: 运行次数
            
        Returns:
            实验结果列表
        """
        print("\n" + "=" * 60)
        print("实验1: SFM + 动态分流（当前密度）")
        print("=" * 60)
        
        results = []
        
        for run_idx in range(n_runs):
            print(f"\n[Run {run_idx + 1}/{n_runs}]")
            
            env = self.create_env()
            router = self.create_router(env)
            
            result = self._run_episode(env, router, use_prediction=False)
            results.append(result)
            
            print(f"  疏散时间: {result['evacuation_time']:.1f}s")
            print(f"  疏散率: {result['evacuation_rate']:.1%}")
            print(f"  最大密度: {result['max_density']:.2f} 人/m²")
            
            env.close()
        
        return results
    
    def run_with_prediction(self, n_runs: int = 5) -> List[Dict]:
        """运行预测实验：SFM + 预测密度分流
        
        Args:
            n_runs: 运行次数
            
        Returns:
            实验结果列表
        """
        print("\n" + "=" * 60)
        print("实验2: SFM + 动态分流（预测密度）")
        print("=" * 60)
        
        results = []
        
        for run_idx in range(n_runs):
            print(f"\n[Run {run_idx + 1}/{n_runs}]")
            
            env = self.create_env()
            router = self.create_router(env)
            predictor = self.create_predictor(env)
            
            result = self._run_episode(env, router, use_prediction=True, predictor=predictor)
            results.append(result)
            
            print(f"  疏散时间: {result['evacuation_time']:.1f}s")
            print(f"  疏散率: {result['evacuation_rate']:.1%}")
            print(f"  最大密度: {result['max_density']:.2f} 人/m²")
            
            env.close()
        
        return results
    
    def _run_episode(
        self,
        env: LargeStationEnv,
        router: DynamicRouter,
        use_prediction: bool = False,
        predictor: DensityFieldPredictor = None,
        routing_interval: int = 10,
        prediction_interval: int = 50,
    ) -> Dict:
        """运行单个episode
        
        Args:
            env: 环境
            router: 分流器
            use_prediction: 是否使用预测
            predictor: 预测器
            routing_interval: 分流间隔
            prediction_interval: 预测间隔
            
        Returns:
            实验结果
        """
        obs, _ = env.reset()
        
        # 记录指标
        max_density_history = []
        congestion_history = []
        evacuated_history = []
        exit_loads = {e.id: [] for e in env.exits}
        
        step = 0
        
        pbar = tqdm(total=self.max_steps, desc="Running")
        
        while step < self.max_steps:
            # 收集行人数据
            pedestrians = []
            for ped in env.sfm.pedestrians:
                pedestrians.append({
                    'id': ped.id,
                    'position': ped.position.copy(),
                    'velocity': ped.velocity.copy(),
                    'target_exit_id': self._find_target_exit_id(ped.target, env.exits),
                })
            
            # 更新预测器
            if use_prediction and predictor is not None:
                ped_data = [{'position': p['position'], 'velocity': p['velocity']} for p in pedestrians]
                field = predictor.compute_density_field(ped_data, timestamp=step * self.dt)
                predictor.add_frame(field)
                
                # 定期预测并更新分流器
                if step % prediction_interval == 0 and predictor.has_enough_frames():
                    predicted_density = predictor.predict()
                    predicted_densities = predictor.get_exit_predicted_densities(predicted_density)
                    router.set_predicted_densities(predicted_densities)
            
            # 更新出口状态
            exit_densities = {}
            exit_congestions = {}
            exit_load_counts = {}
            
            for exit_obj in env.exits:
                density, congestion = self._compute_exit_metrics(env, exit_obj)
                exit_densities[exit_obj.id] = density
                exit_congestions[exit_obj.id] = congestion
                
                load = sum(1 for p in pedestrians 
                          if p['target_exit_id'] == exit_obj.id)
                exit_load_counts[exit_obj.id] = load
                exit_loads[exit_obj.id].append(load)
            
            router.update_exit_states(exit_densities, exit_congestions, exit_load_counts)
            
            # 动态分流（每N步）
            if step % routing_interval == 0 and pedestrians:
                assignments = router.batch_assign(
                    pedestrians,
                    reassign_interval=routing_interval,
                    current_step=step,
                )
                
                # 应用分配结果
                for ped in env.sfm.pedestrians:
                    if ped.id in assignments:
                        _, new_target = assignments[ped.id]
                        ped.target = new_target
            
            # 执行一步
            action = env.action_space.sample()  # 动作由router决定，这里随机
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录指标
            max_density_history.append(info.get('max_density', 0))
            evacuated_history.append(env.evacuated_count)
            
            avg_congestion = np.mean(list(exit_congestions.values()))
            congestion_history.append(avg_congestion)
            
            step += 1
            pbar.update(1)
            
            if terminated:
                break
        
        pbar.close()
        
        # 计算结果
        evacuation_time = step * self.dt
        total_peds = env.n_pedestrians
        evacuation_rate = env.evacuated_count / total_peds if total_peds > 0 else 0
        
        # 计算出口均衡度
        final_loads = [env.evacuated_by_exit[e.id] for e in env.exits]
        if sum(final_loads) > 0:
            load_cv = np.std(final_loads) / np.mean(final_loads)
            balance_score = max(0, 1 - load_cv)
        else:
            balance_score = 0
        
        # 拥堵次数（密度超过阈值的步数）
        congestion_threshold = 0.75
        congestion_count = sum(1 for c in congestion_history if c > congestion_threshold)
        
        return {
            'evacuation_time': evacuation_time,
            'evacuation_rate': evacuation_rate,
            'total_evacuated': env.evacuated_count,
            'remaining': len(env.sfm.pedestrians),
            'max_density': max(max_density_history) if max_density_history else 0,
            'mean_density': np.mean(max_density_history) if max_density_history else 0,
            'balance_score': balance_score,
            'congestion_count': congestion_count,
            'evacuated_by_exit': env.evacuated_by_exit.copy(),
            'steps': step,
        }
    
    def _compute_exit_metrics(self, env: LargeStationEnv, exit_obj) -> Tuple[float, float]:
        """计算出口指标"""
        radius = 10.0
        
        nearby_peds = [
            ped for ped in env.sfm.pedestrians
            if np.linalg.norm(ped.position - exit_obj.position) < radius
        ]
        
        max_density_people = max(env.n_pedestrians / env.n_exits, 20.0)
        density = min(len(nearby_peds) / max_density_people, 1.0)
        
        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            congestion = max(0, 1 - avg_speed / 1.2)
        else:
            congestion = 0.0
        
        return density, congestion
    
    def _find_target_exit_id(self, target: np.ndarray, exits) -> str:
        """找到目标对应的出口ID"""
        min_dist = float('inf')
        nearest_id = None
        
        for exit_obj in exits:
            dist = np.linalg.norm(target - exit_obj.position)
            if dist < min_dist:
                min_dist = dist
                nearest_id = exit_obj.id
        
        return nearest_id
    
    def compare_results(
        self,
        baseline_results: List[Dict],
        prediction_results: List[Dict],
    ) -> Dict:
        """比较实验结果
        
        Args:
            baseline_results: 基线结果
            prediction_results: 预测结果
            
        Returns:
            比较结果
        """
        def compute_stats(results, key):
            values = [r[key] for r in results]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
        
        comparison = {
            'baseline': {
                'evacuation_time': compute_stats(baseline_results, 'evacuation_time'),
                'evacuation_rate': compute_stats(baseline_results, 'evacuation_rate'),
                'max_density': compute_stats(baseline_results, 'max_density'),
                'balance_score': compute_stats(baseline_results, 'balance_score'),
                'congestion_count': compute_stats(baseline_results, 'congestion_count'),
            },
            'prediction': {
                'evacuation_time': compute_stats(prediction_results, 'evacuation_time'),
                'evacuation_rate': compute_stats(prediction_results, 'evacuation_rate'),
                'max_density': compute_stats(prediction_results, 'max_density'),
                'balance_score': compute_stats(prediction_results, 'balance_score'),
                'congestion_count': compute_stats(prediction_results, 'congestion_count'),
            },
        }
        
        # 计算改进百分比
        improvements = {}
        for metric in ['evacuation_time', 'max_density', 'congestion_count']:
            baseline_mean = comparison['baseline'][metric]['mean']
            prediction_mean = comparison['prediction'][metric]['mean']
            if baseline_mean > 0:
                improvements[metric] = (baseline_mean - prediction_mean) / baseline_mean * 100
            else:
                improvements[metric] = 0
        
        for metric in ['evacuation_rate', 'balance_score']:
            baseline_mean = comparison['baseline'][metric]['mean']
            prediction_mean = comparison['prediction'][metric]['mean']
            if baseline_mean > 0:
                improvements[metric] = (prediction_mean - baseline_mean) / baseline_mean * 100
            else:
                improvements[metric] = 0
        
        comparison['improvements'] = improvements
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """打印比较结果"""
        print("\n" + "=" * 70)
        print("实验结果对比")
        print("=" * 70)
        
        metrics = [
            ('evacuation_time', '疏散时间 (s)', True),
            ('evacuation_rate', '疏散率', False),
            ('max_density', '最大密度 (人/m²)', True),
            ('balance_score', '出口均衡度', False),
            ('congestion_count', '拥堵次数', True),
        ]
        
        print(f"\n{'指标':<20} {'基线 (当前密度)':<25} {'预测密度':<25} {'改进':<10}")
        print("-" * 80)
        
        for metric_key, metric_name, lower_is_better in metrics:
            baseline = comparison['baseline'][metric_key]
            prediction = comparison['prediction'][metric_key]
            improvement = comparison['improvements'].get(metric_key, 0)
            
            baseline_str = f"{baseline['mean']:.2f} ± {baseline['std']:.2f}"
            prediction_str = f"{prediction['mean']:.2f} ± {prediction['std']:.2f}"
            
            if lower_is_better:
                if improvement > 0:
                    imp_str = f"↓{improvement:.1f}%"
                else:
                    imp_str = f"↑{-improvement:.1f}%"
            else:
                if improvement > 0:
                    imp_str = f"↑{improvement:.1f}%"
                else:
                    imp_str = f"↓{-improvement:.1f}%"
            
            print(f"{metric_name:<20} {baseline_str:<25} {prediction_str:<25} {imp_str:<10}")
        
        print("=" * 70)
    
    def save_results(
        self,
        baseline_results: List[Dict],
        prediction_results: List[Dict],
        comparison: Dict,
    ):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'config': {
                'flow_level': self.flow_level,
                'max_steps': self.max_steps,
                'dt': self.dt,
                'prediction_model': self.prediction_model_path,
            },
            'baseline_results': baseline_results,
            'prediction_results': prediction_results,
            'comparison': comparison,
            'timestamp': timestamp,
        }
        
        output_path = self.output_dir / f"prediction_test_{self.flow_level}_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n结果保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="密度场预测对比实验")
    
    parser.add_argument("--flow-level", type=str, default="small",
                        choices=["small", "medium", "large"], help="人流量等级")
    parser.add_argument("--n-runs", type=int, default=5, help="每种配置运行次数")
    parser.add_argument("--max-steps", type=int, default=6000, help="最大步数")
    parser.add_argument("--model-path", type=str, default=None,
                        help="预测模型路径（默认使用未训练模型）")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments",
                        help="输出目录")
    parser.add_argument("--baseline-only", action="store_true", help="仅运行基线实验")
    parser.add_argument("--prediction-only", action="store_true", help="仅运行预测实验")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        flow_level=args.flow_level,
        max_steps=args.max_steps,
        prediction_model_path=args.model_path,
        output_dir=args.output_dir,
    )
    
    baseline_results = []
    prediction_results = []
    
    if not args.prediction_only:
        baseline_results = runner.run_baseline(n_runs=args.n_runs)
    
    if not args.baseline_only:
        prediction_results = runner.run_with_prediction(n_runs=args.n_runs)
    
    if baseline_results and prediction_results:
        comparison = runner.compare_results(baseline_results, prediction_results)
        runner.print_comparison(comparison)
        runner.save_results(baseline_results, prediction_results, comparison)
    elif baseline_results:
        print("\n基线实验完成")
    elif prediction_results:
        print("\n预测实验完成")


if __name__ == "__main__":
    main()
