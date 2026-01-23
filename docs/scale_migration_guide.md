# 小规模训练到大规模应用迁移指南

## 目录

1. [概述](#概述)
2. [核心问题](#核心问题)
3. [归一化改造清单](#归一化改造清单)
4. [大规模安全机制](#大规模安全机制)
5. [阈值动态化](#阈值动态化)
6. [训练策略](#训练策略)
7. [验证方法](#验证方法)
8. [换场景检查清单](#换场景检查清单)

---

## 概述

### 背景

PPO强化学习模型在小规模（如80人）训练时速度快，但实际应用需要大规模（如1500人）。为了让模型能够泛化，需要对环境进行归一化改造。

### 核心原则

> **所有观测值和奖励信号都应该是相对值（比例），而非绝对值（人数）**

这样模型学到的策略才能跨规模迁移。

---

## 核心问题

### 问题1：观测空间不一致

| 场景 | 出口附近25人 | 模型理解 |
|------|-------------|----------|
| 50人训练 | density = 25/25 = 1.0 | "非常拥挤" |
| 1500人应用 | density = 25/25 = 1.0 | "非常拥挤" ❌ (实际只有1.7%) |

**解决**：密度基准随人数动态调整

### 问题2：奖励信号不一致

| 场景 | 疏散5人 | 奖励 |
|------|---------|------|
| 50人训练 | 5人 × 12 = 60 | 高奖励 |
| 1500人应用 | 5人 × 12 = 60 | 同样奖励 ❌ (实际只疏散0.3%) |

**解决**：按疏散比例计算奖励

### 问题3：固定阈值失效

| 参数 | 固定值 | 50人场景 | 1500人场景 |
|------|--------|----------|-----------|
| 出口过载阈值 | 15人 | 30%过载 | 1%过载 ❌ |
| 危险密度 | 20人 | 合理 | 太小 ❌ |

**解决**：阈值相对于总人数计算

### 问题4：大规模特有风险

| 风险 | 小规模 | 大规模 |
|------|--------|--------|
| 踩踏 | 几乎不可能 | 真实风险 |
| 恐慌传播 | 影响小 | 可能雪崩 |
| 瓶颈死锁 | 容易解决 | 可能完全堵死 |

**解决**：添加大规模安全检测和惩罚机制

---

## 归一化改造清单

### 1. 出口密度归一化

**文件**: `src/simulation/xxx_env.py`
**函数**: `_compute_exit_metrics()`

```python
# ❌ 改造前：固定基准
density = min(len(nearby_peds) / 25.0, 1.0)

# ✅ 改造后：动态基准
max_density_people = max(self.n_pedestrians / n_exits, 10.0)
density = min(len(nearby_peds) / max_density_people, 1.0)
```

**原理**：假设人均分到各出口，每个出口最多承载 `总人数/出口数`

---

### 2. 瓶颈密度归一化

**文件**: `src/simulation/xxx_env.py`
**函数**: `_compute_bottleneck_densities()`

```python
# ❌ 改造前：固定基准
gate_density = min(gate_count / 20.0, 1.0)

# ✅ 改造后：动态基准（按区域容量比例）
max_gate_people = max(self.n_pedestrians * 0.25, 5.0)  # 瓶颈区最多25%的人
gate_density = min(gate_count / max_gate_people, 1.0)
```

---

### 3. 疏散奖励归一化

**文件**: `src/simulation/xxx_env.py`
**函数**: `_compute_reward()`

```python
# ❌ 改造前：按绝对人数
reward += new_evacuated * evac_per_person

# ✅ 改造后：按疏散比例
evacuation_ratio = new_evacuated / max(self.n_pedestrians, 1)
reward += evacuation_ratio * evac_per_person * 100
```

**原理**：50人疏散5人(10%) 和 1500人疏散150人(10%) 应得相同奖励

---

### 4. 均衡惩罚归一化

**文件**: `src/simulation/xxx_env.py`
**函数**: `_compute_reward()`

```python
# ❌ 改造前：用方差（与人数平方相关）
variance = sum((c - mean_count) ** 2 for c in counts) / n_exits
balance_penalty = min(variance / 100.0, 1.0)

# ✅ 改造后：用变异系数（自动归一化）
std_count = np.sqrt(sum((c - mean_count) ** 2 for c in counts) / n_exits)
cv = std_count / mean_count if mean_count > 0 else 0  # 变异系数 (0-1)
balance_penalty = min(cv, 1.0)
```

---

### 5. 疏散速率归一化

**文件**: `src/simulation/xxx_env.py`
**函数**: `_get_observation()` 和 `_compute_reward()`

```python
# 存储比例而非绝对数
evacuation_ratio = new_evacuated / max(self.n_pedestrians, 1)
self.evacuation_rate_buffer.append(evacuation_ratio)

# 观测时映射到0-1
evacuation_rates = [min(r * 10.0, 1.0) for r in self.evacuation_rate_buffer]
```

---

## 大规模安全机制

### 1. 安全阈值配置

```python
LARGE_SCALE_SAFETY = {
    "critical_density": 4.0,       # 危险密度 (人/m²) - 超过可能踩踏
    "warning_density": 2.5,        # 警告密度 (人/m²)
    "min_safe_distance": 0.5,      # 最小安全距离 (米)
    "panic_spread_radius": 5.0,    # 恐慌传播半径 (米)
    "panic_spread_rate": 0.1,      # 恐慌传播速率
}
```

### 2. 踩踏风险检测

```python
def _detect_crush_risk(self) -> Tuple[float, int]:
    """检测踩踏风险

    Returns:
        (最大局部密度, 危险区域行人数)
    """
    detection_radius = 2.0  # 检测半径

    for ped in self.sfm.pedestrians:
        # 计算局部密度
        nearby_count = sum(1 for other in self.sfm.pedestrians
                         if np.linalg.norm(ped.position - other.position) < detection_radius
                         and other.id != ped.id)
        local_density = nearby_count / (np.pi * detection_radius ** 2)

        if local_density > critical_density:
            danger_count += 1

    return max_density, danger_count
```

### 3. 恐慌传播机制

```python
def _spread_panic(self) -> int:
    """恐慌传播

    规则:
    - 恐慌因子 > 0.3 的行人会传播恐慌
    - 传播半径内的行人恐慌因子增加
    - 传播强度随距离衰减
    """
    for source in panicked_peds:
        for target in self.sfm.pedestrians:
            dist = np.linalg.norm(target.position - source.position)
            if dist < spread_radius:
                spread_strength = spread_rate * (1 - dist / spread_radius)
                target.panic_factor += spread_strength * source.panic_factor
```

### 4. 踩踏惩罚

```python
# 在奖励函数中添加
max_density, danger_count = self._detect_crush_risk()
if danger_count > 0:
    crush_penalty = (danger_count / self.n_pedestrians) * crush_penalty_weight
    reward -= crush_penalty
```

---

## 阈值动态化

### 需要动态化的阈值

| 参数 | 固定值问题 | 动态化公式 |
|------|-----------|-----------|
| 出口过载阈值 | 人少时太大，人多时太小 | `n_pedestrians / n_exits * 1.5` |
| 负载均衡阈值 | 同上 | `n_pedestrians / n_exits * 1.2` |
| 分流距离阈值 | 场景相关 | 保持固定或按场景尺寸调整 |

### 修改示例

```python
def rebalance_exit_assignments(self, threshold=None, ...):
    # 动态阈值
    if threshold is None:
        threshold = max(int(self.n_pedestrians / self.n_exits * 1.5), 5)
    ...
```

---

## 训练策略

### 推荐流程

```
┌─────────────────────────────────────────────────────────┐
│  阶段1: 小规模快速训练 (80-100人)                         │
│  - timesteps: 200,000-500,000                           │
│  - 时间: 1-2小时 (CPU)                                   │
│  - 目的: 学习基本策略                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2: 中规模验证 (300-500人)                           │
│  - timesteps: 50,000-100,000                            │
│  - 目的: 验证泛化性，微调                                 │
│  - 可选: 加载阶段1模型继续训练                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  阶段3: 大规模应用 (1000-1500人)                         │
│  - 直接使用训练好的模型                                   │
│  - 监控安全指标                                          │
│  - 如效果不佳，用大规模少量训练微调                        │
└─────────────────────────────────────────────────────────┘
```

### 命令示例

```bash
# 阶段1: 小规模训练
python train_ppo.py --n_pedestrians 80 --timesteps 300000

# 阶段2: 中规模验证 (可选微调)
python train_ppo.py --n_pedestrians 400 --timesteps 50000 --load_model model.zip

# 阶段3: 大规模应用
python run_simulation.py --n_pedestrians 1500 --model model.zip
```

---

## 验证方法

### 1. 观测值一致性验证

```python
# 检查不同人数下观测值范围是否一致
for n_peds in [50, 200, 800, 1500]:
    env = YourEnv(n_pedestrians=n_peds)
    obs, _ = env.reset()
    print(f"n={n_peds}: obs range = [{obs.min():.3f}, {obs.max():.3f}]")
    # 期望: 所有规模的观测值都在 [0, 1] 范围内
```

### 2. 奖励一致性验证

```python
# 检查相同疏散比例的奖励是否一致
# 50人疏散5人 vs 1500人疏散150人 应该得到相近的奖励
```

### 3. 策略泛化性验证

```python
# 用小规模训练的模型在大规模上评估
model = PPO.load("small_scale_model.zip")
env = YourEnv(n_pedestrians=1500)

# 评估指标
evacuation_rate = []
avg_time = []
max_density = []

for _ in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    evacuation_rate.append(info['evacuated'] / env.n_pedestrians)
    max_density.append(info.get('max_local_density', 0))

print(f"疏散率: {np.mean(evacuation_rate):.1%}")
print(f"最大密度: {np.max(max_density):.2f} 人/m²")
```

---

## 换场景检查清单

当更换到新场景时，按以下清单检查：

### 环境配置

- [ ] 场景尺寸 (`scene_size`)
- [ ] 出口数量和位置 (`exits`)
- [ ] 障碍物位置 (`obstacles`, `pillars`)
- [ ] 瓶颈区域定义 (`bottleneck zones`)
- [ ] 行人生成区域 (`spawn area`)

### 归一化参数

- [ ] 出口密度基准：`n_pedestrians / n_exits`
- [ ] 瓶颈密度基准：根据瓶颈区域面积调整
- [ ] 检测半径：根据场景尺寸调整

### 安全阈值

- [ ] 危险密度阈值：通常 4.0 人/m² 不变
- [ ] 恐慌传播半径：根据场景尺寸可调整
- [ ] 分流距离阈值：根据场景尺寸调整

### 奖励权重

- [ ] 疏散奖励权重
- [ ] 拥堵惩罚权重
- [ ] 踩踏惩罚权重
- [ ] 均衡惩罚权重

### 验证测试

- [ ] 小规模 (50-100人) 观测值范围 [0,1]
- [ ] 大规模 (1000+人) 观测值范围 [0,1]
- [ ] 奖励信号一致性
- [ ] 安全机制触发正常

---

## 附录：关键代码位置

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| 出口密度计算 | `xxx_env.py` | `_compute_exit_metrics()` |
| 瓶颈密度计算 | `xxx_env.py` | `_compute_bottleneck_densities()` |
| 观测空间构建 | `xxx_env.py` | `_get_observation()` |
| 奖励计算 | `xxx_env.py` | `_compute_reward()` |
| 踩踏检测 | `xxx_env.py` | `_detect_crush_risk()` |
| 恐慌传播 | `xxx_env.py` | `_spread_panic()` |
| 社会力模型 | `social_force.py` | `SocialForceModel` |
| 恐慌因子更新 | `social_force.py` | `update_panic_factor()` |

---

## 参考文献

1. Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
2. Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic.
3. Fruin, J.J. (1993). The causes and prevention of crowd disasters. (踩踏密度阈值)
