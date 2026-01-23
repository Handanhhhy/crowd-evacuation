"""
动态分流决策模块

基于规则引擎的智能出口分配，核心思路：
1. 距离分数 - 越近越好
2. 拥堵分数 - 越空越好
3. 负载均衡 - 各出口均匀分配
4. 预测分数 - 预测未来拥堵（后续集成密度预测模块）

参考文档: docs/new_station_plan.md 6.4节
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExitInfo:
    """出口信息"""
    id: str
    position: np.ndarray
    capacity: int
    current_load: int = 0
    density: float = 0.0
    congestion: float = 0.0


class DynamicRouter:
    """动态分流决策引擎

    为每个行人计算最优出口，实现负载均衡和拥堵规避。
    """

    def __init__(
        self,
        exits: List[ExitInfo],
        weight_distance: float = 0.3,
        weight_congestion: float = 0.25,
        weight_predicted: float = 0.25,
        weight_balance: float = 0.2,
        reassign_threshold: float = 0.15,
        detection_radius: float = 10.0,
    ):
        """
        Args:
            exits: 出口列表
            weight_distance: 距离权重
            weight_congestion: 当前拥堵权重
            weight_predicted: 预测拥堵权重（暂用当前值代替）
            weight_balance: 负载均衡权重
            reassign_threshold: 重新分配阈值（拥堵差超过此值才切换）
            detection_radius: 拥堵检测半径
        """
        self.exits = exits
        self.n_exits = len(exits)

        # 权重配置
        self.weight_distance = weight_distance
        self.weight_congestion = weight_congestion
        self.weight_predicted = weight_predicted
        self.weight_balance = weight_balance

        self.reassign_threshold = reassign_threshold
        self.detection_radius = detection_radius

        # 统计信息
        self.assignment_counts = {e.id: 0 for e in exits}
        self.reassignment_count = 0

        # 预测密度（后续由密度预测模块更新）
        self.predicted_densities: Optional[Dict[str, float]] = None

    def update_exit_states(
        self,
        exit_densities: Dict[str, float],
        exit_congestions: Dict[str, float],
        exit_loads: Dict[str, int],
    ):
        """更新出口状态

        Args:
            exit_densities: 各出口密度 {exit_id: density}
            exit_congestions: 各出口拥堵度 {exit_id: congestion}
            exit_loads: 各出口当前负载 {exit_id: load}
        """
        for exit_info in self.exits:
            exit_info.density = exit_densities.get(exit_info.id, 0.0)
            exit_info.congestion = exit_congestions.get(exit_info.id, 0.0)
            exit_info.current_load = exit_loads.get(exit_info.id, 0)

    def set_predicted_densities(self, predicted: Dict[str, float]):
        """设置预测密度（由密度预测模块调用）"""
        self.predicted_densities = predicted

    def compute_exit_scores(
        self,
        pedestrian_pos: np.ndarray,
        current_target_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """计算各出口得分

        Args:
            pedestrian_pos: 行人位置
            current_target_id: 当前目标出口ID

        Returns:
            各出口得分 {exit_id: score}
        """
        scores = {}

        # 计算最大距离用于归一化
        max_dist = max(
            np.linalg.norm(pedestrian_pos - e.position)
            for e in self.exits
        ) + 1.0

        # 计算总负载用于均衡
        total_load = sum(e.current_load for e in self.exits) + 1
        avg_load = total_load / self.n_exits

        for exit_info in self.exits:
            # 1. 距离分数（越近越好）
            dist = np.linalg.norm(pedestrian_pos - exit_info.position)
            dist_score = 1.0 - (dist / max_dist)

            # 2. 当前拥堵分数（越空越好）
            congestion_score = 1.0 - exit_info.congestion

            # 3. 预测拥堵分数（暂用当前密度代替）
            if self.predicted_densities and exit_info.id in self.predicted_densities:
                predicted_density = self.predicted_densities[exit_info.id]
            else:
                predicted_density = exit_info.density
            predicted_score = 1.0 - min(predicted_density, 1.0)

            # 4. 负载均衡分数（负载低于平均的出口得分高）
            load_ratio = exit_info.current_load / max(avg_load, 1)
            balance_score = max(0.0, 1.0 - load_ratio * 0.5)

            # 综合评分
            score = (
                self.weight_distance * dist_score +
                self.weight_congestion * congestion_score +
                self.weight_predicted * predicted_score +
                self.weight_balance * balance_score
            )

            scores[exit_info.id] = score

        return scores

    def assign_exit(
        self,
        pedestrian_pos: np.ndarray,
        current_target_id: Optional[str] = None,
        force_reassign: bool = False,
    ) -> Tuple[str, np.ndarray]:
        """为行人分配最优出口

        Args:
            pedestrian_pos: 行人位置
            current_target_id: 当前目标出口ID
            force_reassign: 是否强制重新分配

        Returns:
            (出口ID, 出口位置)
        """
        scores = self.compute_exit_scores(pedestrian_pos, current_target_id)

        # 找到最优出口
        best_exit_id = max(scores, key=scores.get)
        best_score = scores[best_exit_id]

        # 判断是否需要切换出口
        should_switch = force_reassign

        if current_target_id and current_target_id in scores:
            current_score = scores[current_target_id]
            # 只有当新出口明显更优时才切换
            if best_score - current_score > self.reassign_threshold:
                should_switch = True
        else:
            should_switch = True

        if should_switch:
            self.assignment_counts[best_exit_id] += 1
            if current_target_id and current_target_id != best_exit_id:
                self.reassignment_count += 1

            # 返回最优出口
            for exit_info in self.exits:
                if exit_info.id == best_exit_id:
                    return best_exit_id, exit_info.position.copy()

        # 保持当前出口
        for exit_info in self.exits:
            if exit_info.id == current_target_id:
                return current_target_id, exit_info.position.copy()

        # fallback: 返回最优出口
        for exit_info in self.exits:
            if exit_info.id == best_exit_id:
                return best_exit_id, exit_info.position.copy()

        raise ValueError("No valid exit found")

    def batch_assign(
        self,
        pedestrians: List[dict],
        reassign_interval: int = 10,
        current_step: int = 0,
    ) -> Dict[int, Tuple[str, np.ndarray]]:
        """批量分配出口

        Args:
            pedestrians: 行人列表 [{'id': int, 'position': np.ndarray, 'target_exit_id': str}, ...]
            reassign_interval: 重新分配间隔（每N步重新评估）
            current_step: 当前步数

        Returns:
            {行人ID: (出口ID, 出口位置)}
        """
        assignments = {}

        for ped in pedestrians:
            # 每隔N步重新评估
            force_reassign = (current_step % reassign_interval == 0)

            exit_id, exit_pos = self.assign_exit(
                pedestrian_pos=ped['position'],
                current_target_id=ped.get('target_exit_id'),
                force_reassign=force_reassign,
            )

            assignments[ped['id']] = (exit_id, exit_pos)

        return assignments

    def get_statistics(self) -> Dict:
        """获取分配统计"""
        total_assignments = sum(self.assignment_counts.values())

        return {
            'assignment_counts': self.assignment_counts.copy(),
            'total_assignments': total_assignments,
            'reassignment_count': self.reassignment_count,
            'balance_ratio': self._compute_balance_ratio(),
        }

    def _compute_balance_ratio(self) -> float:
        """计算负载均衡比率（0-1，1表示完全均衡）"""
        counts = list(self.assignment_counts.values())
        if not counts or max(counts) == 0:
            return 1.0

        avg = sum(counts) / len(counts)
        if avg == 0:
            return 1.0

        # 变异系数的倒数
        std = np.std(counts)
        cv = std / avg
        return max(0.0, 1.0 - cv)

    def reset_statistics(self):
        """重置统计信息"""
        self.assignment_counts = {e.id: 0 for e in self.exits}
        self.reassignment_count = 0


# ============ 特殊规则 ============

class EvacuationRules:
    """疏散特殊规则"""

    @staticmethod
    def prioritize_near_escalator(
        pedestrian_pos: np.ndarray,
        escalator_positions: List[np.ndarray],
        threshold: float = 5.0,
    ) -> bool:
        """检查是否在扶梯口附近（需要优先疏散）"""
        for esc_pos in escalator_positions:
            if np.linalg.norm(pedestrian_pos - esc_pos) < threshold:
                return True
        return False

    @staticmethod
    def should_emergency_reroute(
        current_density: float,
        threshold: float = 3.5,
    ) -> bool:
        """检查是否需要紧急分流（当前密度过高）"""
        return current_density > threshold

    @staticmethod
    def should_avoid_congestion(
        predicted_density: float,
        threshold: float = 3.0,
    ) -> bool:
        """检查是否需要规避预测拥堵"""
        return predicted_density > threshold
