"""
AnyLogic 轨迹数据导出器

将 LargeStationEnv 仿真数据导出为 AnyLogic 可读格式。
支持 CSV 轨迹数据、JSON 元数据和场景布局导出。
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# 导入项目模块
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sfm.social_force import PedestrianType


@dataclass
class PedestrianSnapshot:
    """单个行人在某时刻的状态快照"""
    timestamp: float
    ped_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    ped_type: str
    target_exit: str
    radius: float


@dataclass
class EvacuationEvent:
    """疏散事件记录"""
    ped_id: int
    ped_type: str
    evacuation_time: float
    exit_used: str
    entry_time: float = 0.0


class TrajectoryExporter:
    """轨迹数据导出器
    
    用法:
        exporter = TrajectoryExporter(output_dir="anylogic/exported_data")
        
        # 在仿真循环中记录数据
        for step in range(max_steps):
            obs, reward, done, truncated, info = env.step(action)
            exporter.record_frame(env, step * dt)
            
        # 导出数据
        exporter.export_all()
    """
    
    def __init__(
        self,
        output_dir: str = "anylogic/exported_data",
        export_interval: int = 1,
    ):
        """初始化导出器
        
        Args:
            output_dir: 输出目录
            export_interval: 导出间隔 (每N步导出一次)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export_interval = export_interval
        
        # 数据缓存
        self.trajectory_data: List[PedestrianSnapshot] = []
        self.evacuation_events: List[EvacuationEvent] = []
        self.density_data: List[Dict] = []  # 可选的密度场数据
        
        # 元数据
        self.metadata: Dict[str, Any] = {}
        self.scene_layout: Dict[str, Any] = {}
        
        # 内部状态
        self._step_count = 0
        self._active_pedestrians: set = set()  # 跟踪活跃行人
        self._evacuated_pedestrians: Dict[int, float] = {}  # ped_id -> evacuation_time
        
        # 时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record_frame(
        self,
        env,
        sim_time: float,
        exit_mapping: Optional[Dict[int, str]] = None,
    ):
        """记录当前帧的行人数据
        
        Args:
            env: LargeStationEnv 环境实例
            sim_time: 当前仿真时间 (秒)
            exit_mapping: 出口ID映射 (可选)
        """
        self._step_count += 1
        
        # 检查导出间隔
        if self._step_count % self.export_interval != 0:
            return
        
        # 获取当前活跃行人集合
        current_pedestrians = set()
        
        # 根据 SFM 类型获取行人数据
        if hasattr(env.sfm, 'pedestrians'):
            # CPU 版本 SFM
            self._record_cpu_pedestrians(env, sim_time, exit_mapping, current_pedestrians)
        elif hasattr(env.sfm, '_positions_tensor'):
            # GPU 优化版本 SFM
            self._record_gpu_pedestrians(env, sim_time, exit_mapping, current_pedestrians)
        
        # 检测疏散事件
        self._detect_evacuation_events(current_pedestrians, sim_time, env)
        
        # 更新活跃行人集合
        self._active_pedestrians = current_pedestrians
    
    def _record_cpu_pedestrians(
        self,
        env,
        sim_time: float,
        exit_mapping: Optional[Dict[int, str]],
        current_pedestrians: set,
    ):
        """记录 CPU 版 SFM 行人数据"""
        for ped in env.sfm.pedestrians:
            current_pedestrians.add(ped.id)
            
            # 确定目标出口
            target_exit = self._get_target_exit(ped.target, env.exits, exit_mapping)
            
            snapshot = PedestrianSnapshot(
                timestamp=round(sim_time, 2),
                ped_id=ped.id,
                x=round(float(ped.position[0]), 3),
                y=round(float(ped.position[1]), 3),
                vx=round(float(ped.velocity[0]), 3),
                vy=round(float(ped.velocity[1]), 3),
                speed=round(float(np.linalg.norm(ped.velocity)), 3),
                ped_type=ped.ped_type.value if hasattr(ped.ped_type, 'value') else str(ped.ped_type),
                target_exit=target_exit,
                radius=round(float(ped.radius), 2),
            )
            self.trajectory_data.append(snapshot)
    
    def _record_gpu_pedestrians(
        self,
        env,
        sim_time: float,
        exit_mapping: Optional[Dict[int, str]],
        current_pedestrians: set,
    ):
        """记录 GPU 优化版 SFM 行人数据"""
        sfm = env.sfm
        
        # 获取 GPU 数据并转为 numpy
        positions = sfm._positions_tensor.cpu().numpy()
        velocities = sfm._velocities_tensor.cpu().numpy()
        targets = sfm._targets_tensor.cpu().numpy()
        radii = sfm._radii_tensor.cpu().numpy()
        active_mask = sfm._active_mask.cpu().numpy()
        
        # 类型信息 (如果有)
        ped_types = None
        if hasattr(sfm, '_types_tensor'):
            ped_types = sfm._types_tensor.cpu().numpy()
        
        for i in range(len(positions)):
            if not active_mask[i]:
                continue
            
            current_pedestrians.add(i)
            
            # 确定目标出口
            target_exit = self._get_target_exit(targets[i], env.exits, exit_mapping)
            
            # 行人类型
            ped_type_str = "NORMAL"
            if ped_types is not None:
                try:
                    ped_type_str = PedestrianType(ped_types[i]).value
                except (ValueError, TypeError):
                    ped_type_str = str(ped_types[i]) if ped_types[i] else "NORMAL"
            
            speed = float(np.linalg.norm(velocities[i]))
            
            snapshot = PedestrianSnapshot(
                timestamp=round(sim_time, 2),
                ped_id=i,
                x=round(float(positions[i][0]), 3),
                y=round(float(positions[i][1]), 3),
                vx=round(float(velocities[i][0]), 3),
                vy=round(float(velocities[i][1]), 3),
                speed=round(speed, 3),
                ped_type=ped_type_str,
                target_exit=target_exit,
                radius=round(float(radii[i]), 2),
            )
            self.trajectory_data.append(snapshot)
    
    def _get_target_exit(
        self,
        target_pos: np.ndarray,
        exits: List,
        exit_mapping: Optional[Dict[int, str]] = None,
    ) -> str:
        """根据目标位置确定目标出口名称"""
        if exit_mapping:
            # 使用提供的映射
            for exit_id, exit_name in exit_mapping.items():
                return exit_name
        
        # 根据位置距离匹配最近出口
        min_dist = float('inf')
        closest_exit = "unknown"
        
        for exit_obj in exits:
            exit_pos = np.array(exit_obj.position)
            dist = np.linalg.norm(target_pos - exit_pos)
            if dist < min_dist:
                min_dist = dist
                closest_exit = exit_obj.id
        
        return closest_exit
    
    def _detect_evacuation_events(
        self,
        current_pedestrians: set,
        sim_time: float,
        env,
    ):
        """检测疏散事件 (行人离开场景)"""
        # 找出已疏散的行人 (之前活跃，现在不活跃)
        evacuated = self._active_pedestrians - current_pedestrians
        
        for ped_id in evacuated:
            if ped_id in self._evacuated_pedestrians:
                continue  # 已记录
            
            self._evacuated_pedestrians[ped_id] = sim_time
            
            # 获取行人信息 (从最后记录的数据)
            ped_type = "NORMAL"
            exit_used = "unknown"
            
            # 查找该行人最后的记录
            for snapshot in reversed(self.trajectory_data):
                if snapshot.ped_id == ped_id:
                    ped_type = snapshot.ped_type
                    exit_used = snapshot.target_exit
                    break
            
            event = EvacuationEvent(
                ped_id=ped_id,
                ped_type=ped_type,
                evacuation_time=round(sim_time, 2),
                exit_used=exit_used,
                entry_time=0.0,  # 假设都从0开始
            )
            self.evacuation_events.append(event)
    
    def record_density(
        self,
        sim_time: float,
        density_grid: np.ndarray,
        grid_resolution: Tuple[int, int] = (30, 16),
    ):
        """记录密度场数据 (可选)
        
        Args:
            sim_time: 仿真时间
            density_grid: 密度网格 (shape: H x W)
            grid_resolution: 网格分辨率
        """
        for gy in range(density_grid.shape[0]):
            for gx in range(density_grid.shape[1]):
                if density_grid[gy, gx] > 0:
                    self.density_data.append({
                        'timestamp': round(sim_time, 2),
                        'grid_x': gx,
                        'grid_y': gy,
                        'density': round(float(density_grid[gy, gx]), 3),
                    })
    
    def set_metadata(
        self,
        flow_level: str,
        method: str,
        total_pedestrians: int,
        dt: float = 0.1,
        **kwargs,
    ):
        """设置元数据
        
        Args:
            flow_level: 人流等级
            method: 仿真方法
            total_pedestrians: 总行人数
            dt: 时间步长
            **kwargs: 其他元数据
        """
        self.metadata = {
            'export_time': datetime.now().isoformat(),
            'flow_level': flow_level,
            'method': method,
            'total_pedestrians': total_pedestrians,
            'dt': dt,
            **kwargs,
        }
    
    def set_scene_layout(self, env):
        """从环境提取场景布局
        
        Args:
            env: LargeStationEnv 环境实例
        """
        # 墙壁数据
        walls = []
        if hasattr(env.sfm, 'walls'):
            for wall in env.sfm.walls:
                if hasattr(wall, 'cpu'):
                    wall = wall.cpu().numpy()
                walls.append({
                    'start': [float(wall[0][0]), float(wall[0][1])],
                    'end': [float(wall[1][0]), float(wall[1][1])],
                })
        elif hasattr(env.sfm, 'obstacles'):
            for obs in env.sfm.obstacles:
                if hasattr(obs, 'cpu'):
                    obs = obs.cpu().numpy()
                walls.append({
                    'start': [float(obs[0][0]), float(obs[0][1])],
                    'end': [float(obs[1][0]), float(obs[1][1])],
                })
        
        # 出口数据
        exits = []
        for exit_obj in env.exits:
            exits.append({
                'id': exit_obj.id,
                'name': exit_obj.name,
                'position': [float(exit_obj.position[0]), float(exit_obj.position[1])],
                'width': float(exit_obj.width),
                'direction': exit_obj.direction,
            })
        
        # 扶梯数据
        escalators = []
        for esc in env.escalators:
            escalators.append({
                'id': esc.id,
                'position': [float(esc.position[0]), float(esc.position[1])],
                'size': [float(esc.size[0]), float(esc.size[1])],
            })
        
        self.scene_layout = {
            'scene_width': float(env.scene_width),
            'scene_height': float(env.scene_height),
            'walls': walls,
            'exits': exits,
            'escalators': escalators,
        }
        
        # 添加到元数据
        self.metadata['scene'] = {
            'width': float(env.scene_width),
            'height': float(env.scene_height),
            'exits': exits,
            'escalators': escalators,
        }
    
    def export_trajectory_csv(self, filename: Optional[str] = None) -> str:
        """导出轨迹数据为 CSV
        
        Args:
            filename: 文件名 (可选，默认使用时间戳)
            
        Returns:
            导出的文件路径
        """
        if filename is None:
            filename = f"trajectory_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                'timestamp', 'ped_id', 'x', 'y', 'vx', 'vy',
                'speed', 'ped_type', 'target_exit', 'radius'
            ])
            
            # 写入数据
            for snapshot in self.trajectory_data:
                writer.writerow([
                    snapshot.timestamp,
                    snapshot.ped_id,
                    snapshot.x,
                    snapshot.y,
                    snapshot.vx,
                    snapshot.vy,
                    snapshot.speed,
                    snapshot.ped_type,
                    snapshot.target_exit,
                    snapshot.radius,
                ])
        
        print(f"[TrajectoryExporter] 导出轨迹: {filepath}")
        print(f"  - 总记录数: {len(self.trajectory_data)}")
        return str(filepath)
    
    def export_metadata_json(self, filename: Optional[str] = None) -> str:
        """导出元数据为 JSON
        
        Args:
            filename: 文件名 (可选)
            
        Returns:
            导出的文件路径
        """
        if filename is None:
            filename = f"metadata_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # 添加统计信息
        self.metadata['total_frames'] = len(set(s.timestamp for s in self.trajectory_data))
        self.metadata['statistics'] = {
            'total_evacuated': len(self.evacuation_events),
            'evacuation_rate': len(self.evacuation_events) / self.metadata.get('total_pedestrians', 1),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[TrajectoryExporter] 导出元数据: {filepath}")
        return str(filepath)
    
    def export_layout_json(self, filename: str = "layout.json") -> str:
        """导出场景布局为 JSON
        
        Args:
            filename: 文件名
            
        Returns:
            导出的文件路径
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.scene_layout, f, indent=2, ensure_ascii=False)
        
        print(f"[TrajectoryExporter] 导出布局: {filepath}")
        return str(filepath)
    
    def export_evacuation_events(self, filename: Optional[str] = None) -> str:
        """导出疏散事件为 CSV
        
        Args:
            filename: 文件名 (可选)
            
        Returns:
            导出的文件路径
        """
        if filename is None:
            filename = f"evacuation_events_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ped_id', 'ped_type', 'evacuation_time', 'exit_used', 'entry_time'])
            
            for event in self.evacuation_events:
                writer.writerow([
                    event.ped_id,
                    event.ped_type,
                    event.evacuation_time,
                    event.exit_used,
                    event.entry_time,
                ])
        
        print(f"[TrajectoryExporter] 导出疏散事件: {filepath}")
        print(f"  - 总疏散人数: {len(self.evacuation_events)}")
        return str(filepath)
    
    def export_density_csv(self, filename: Optional[str] = None) -> str:
        """导出密度场数据为 CSV
        
        Args:
            filename: 文件名 (可选)
            
        Returns:
            导出的文件路径
        """
        if not self.density_data:
            print("[TrajectoryExporter] 无密度数据可导出")
            return ""
        
        if filename is None:
            filename = f"density_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'grid_x', 'grid_y', 'density'])
            
            for data in self.density_data:
                writer.writerow([
                    data['timestamp'],
                    data['grid_x'],
                    data['grid_y'],
                    data['density'],
                ])
        
        print(f"[TrajectoryExporter] 导出密度: {filepath}")
        return str(filepath)
    
    def export_all(self) -> Dict[str, str]:
        """导出所有数据
        
        Returns:
            导出的文件路径字典
        """
        paths = {}
        
        paths['trajectory'] = self.export_trajectory_csv()
        paths['metadata'] = self.export_metadata_json()
        paths['layout'] = self.export_layout_json()
        paths['evacuation_events'] = self.export_evacuation_events()
        
        if self.density_data:
            paths['density'] = self.export_density_csv()
        
        print(f"\n[TrajectoryExporter] 导出完成!")
        print(f"  输出目录: {self.output_dir}")
        return paths
    
    def get_summary(self) -> Dict[str, Any]:
        """获取导出数据摘要"""
        unique_peds = set(s.ped_id for s in self.trajectory_data)
        unique_times = set(s.timestamp for s in self.trajectory_data)
        
        return {
            'total_records': len(self.trajectory_data),
            'unique_pedestrians': len(unique_peds),
            'unique_timestamps': len(unique_times),
            'evacuation_events': len(self.evacuation_events),
            'time_range': (
                min(unique_times) if unique_times else 0,
                max(unique_times) if unique_times else 0,
            ),
        }


def export_from_env(
    env,
    output_dir: str = "anylogic/exported_data",
    max_steps: int = 3000,
    export_interval: int = 1,
    method: str = "baseline",
    model=None,
) -> Dict[str, str]:
    """便捷函数: 运行仿真并导出数据
    
    Args:
        env: LargeStationEnv 环境实例
        output_dir: 输出目录
        max_steps: 最大步数
        export_interval: 导出间隔
        method: 仿真方法
        model: PPO 模型 (可选)
        
    Returns:
        导出的文件路径字典
    """
    exporter = TrajectoryExporter(
        output_dir=output_dir,
        export_interval=export_interval,
    )
    
    # 设置元数据
    exporter.set_metadata(
        flow_level=env.flow_level,
        method=method,
        total_pedestrians=env.n_pedestrians,
        dt=env.dt,
    )
    
    # 设置场景布局
    exporter.set_scene_layout(env)
    
    # 重置环境
    obs, _ = env.reset()
    
    # 运行仿真
    print(f"\n[导出] 开始仿真... (max_steps={max_steps})")
    
    for step in range(max_steps):
        sim_time = step * env.dt
        
        # 记录当前帧
        exporter.record_frame(env, sim_time)
        
        # 选择动作
        if method == "ppo" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # 简单的负载均衡策略
            action = env.action_space.sample()
        
        # 执行步骤
        obs, reward, done, truncated, info = env.step(action)
        
        # 进度显示
        if step % 500 == 0:
            evacuated = info.get('evacuated', 0)
            remaining = info.get('remaining', env.n_pedestrians)
            print(f"  Step {step}: 疏散 {evacuated}/{env.n_pedestrians}, 剩余 {remaining}")
        
        if done or truncated:
            print(f"  仿真完成于步骤 {step}")
            break
    
    # 导出所有数据
    return exporter.export_all()


if __name__ == "__main__":
    # 简单测试
    print("TrajectoryExporter 模块加载成功")
    print("使用方法: python anylogic/run_and_export.py")
