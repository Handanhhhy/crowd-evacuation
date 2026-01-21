# 分层预测式动态引导系统 - 改动说明文档

## 版本信息

- **实现日期**: 2026-01-21
- **版本**: v2.0
- **作者**: Claude Code Assistant

---

## 一、改动背景

### 1.1 原有PPO引导的问题

| 问题 | 原有实现 | 后果 |
|-----|---------|------|
| **随机概率** | 每步8-20%概率改道 | 不符合现实，人不会每秒重新考虑 |
| **无记忆** | 不记录谁被引导过 | 同一人被反复"骚扰" |
| **全体广播** | 对所有人应用同样逻辑 | 浪费引导资源，效率低 |
| **无冷却** | 可以连续改道 | 行人路线混乱 |

### 1.2 改进目标

实现更符合现实的动态引导系统：
- **定向引导**：只引导需要的人
- **一次决策**：引导后给时间执行
- **有冷却期**：不会立即再次改变指令
- **基于预测**：根据预测的问题主动干预

---

## 二、系统架构

### 2.1 三层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    分层预测式引导系统                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第1层：全局决策 (PPO)                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  观察当前状态 → PPO模型 → 推荐最优出口               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  第2层：预测筛选 (Social-LSTM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  预测所有人轨迹 → 识别"将遇到问题"的行人              │   │
│  │  问题类型：走向拥堵出口 / 走向角落 / 走向人群          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  第3层：个体决策 (引导管理器)                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  检查引导条件：                                       │   │
│  │  - 是否在引导区域内？                                 │   │
│  │  - 引导次数 < 最大次数？                              │   │
│  │  - 冷却时间已过？                                     │   │
│  │  - 距离目标足够远？                                   │   │
│  │  → 满足条件则引导，记录状态                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 配置参数

```python
GUIDANCE_CONFIG = {
    'max_guidance_count': 2,           # 每人最多被引导2次
    'cooldown_time': 5.0,              # 冷却时间5秒
    'min_distance_to_target': 8.0,     # 距目标>8米才可引导
    'guidance_zone_x': 22.0,           # x>22进入引导区（闸机位置后）
    'problem_prediction_horizon': 12,  # 预测12步(4.8秒)
    'congestion_threshold': 0.5,       # 拥堵度阈值
    'corner_trap_radius': 3.0,         # 角落陷阱检测半径
}
```

---

## 三、修改文件清单

### 3.1 `src/sfm/social_force.py`

#### 修改内容：扩展 Pedestrian 类

**新增属性** (第67-91行):

```python
@dataclass
class Pedestrian:
    # ... 原有属性 ...

    # 新增：引导状态 (分层预测式引导系统)
    guidance_count: int = 0              # 已被引导次数
    last_guidance_time: float = -999.0   # 上次引导时间
    original_target: Optional[np.ndarray] = field(default=None)  # 原始目标
```

**修改工厂方法** `create_with_type()` (第126-143行):

```python
return cls(
    # ... 原有参数 ...
    # 引导状态初始化
    guidance_count=0,
    last_guidance_time=-999.0,
    original_target=target.copy()  # 保存原始目标
)
```

---

### 3.2 `src/simulation/metro_evacuation_env.py`

#### 3.2.1 新增配置常量 (第36-46行)

```python
GUIDANCE_CONFIG = {
    'max_guidance_count': 2,
    'cooldown_time': 5.0,
    'min_distance_to_target': 8.0,
    'guidance_zone_x': 22.0,
    'problem_prediction_horizon': 12,
    'congestion_threshold': 0.5,
    'corner_trap_radius': 3.0,
}
```

#### 3.2.2 新增实例属性 (第189-190行)

```python
# 分层预测式引导系统状态
self.last_action = 0  # 上一次PPO动作（推荐出口）
```

#### 3.2.3 新增方法

| 方法名 | 功能 | 位置 |
|-------|------|------|
| `predictive_guidance_system()` | 三层引导主逻辑 | 第753-789行 |
| `_identify_problem_pedestrians()` | 识别问题行人 | 第791-825行 |
| `_will_reach_congested_exit()` | 检查是否走向拥堵出口 | 第827-851行 |
| `_can_be_guided()` | 检查引导条件 | 第853-882行 |
| `_find_best_alternative_exit()` | 找最佳替代出口 | 第884-928行 |
| `_apply_guidance()` | 应用引导更新状态 | 第930-952行 |

#### 3.2.4 修改 `step()` 方法 (第954-985行)

**原有代码**:
```python
def step(self, action: int):
    # ...
    self._apply_action(action)  # 随机概率引导
```

**新代码**:
```python
def step(self, action: int):
    # 保存PPO决策
    self.last_action = action

    # 使用分层预测式引导系统
    if self.trajectory_predictor is not None:
        guided_count = self.predictive_guidance_system()
        corner_avoided = self.proactive_corner_avoidance()
    else:
        # 回退到旧版负载均衡
        guided_count = self.rebalance_exit_assignments(...)
```

#### 3.2.5 修改 `reset()` 方法

新增重置引导状态：
```python
self.last_action = 0
```

---

### 3.3 `examples/pygame_metro_with_ppo.py`

#### 3.3.1 新增配置常量 (第46-56行)

与 `metro_evacuation_env.py` 相同的 `GUIDANCE_CONFIG`

#### 3.3.2 新增统计属性 (第182-189行)

```python
# 分层预测式引导系统统计
self.total_guided_count = 0
self.current_step_guided = 0
self.guidance_stats = {
    'by_reason': {'congestion': 0, 'corner_trap': 0},
    'by_exit': {'A': 0, 'B': 0, 'C': 0}
}
```

#### 3.3.3 新增方法

| 方法名 | 功能 |
|-------|------|
| `predictive_guidance_system()` | 三层引导主逻辑 |
| `_identify_problem_pedestrians()` | 识别问题行人 |
| `_will_reach_congested_exit()` | 检查是否走向拥堵出口 |
| `_can_be_guided()` | 检查引导条件 |
| `_find_best_alternative_exit()` | 找最佳替代出口 |
| `_apply_guidance()` | 应用引导更新状态 |
| `_avoid_corner_traps()` | 角落陷阱避免 |

#### 3.3.4 修改 `update()` 方法

**原有代码**:
```python
def update(self):
    # PPO引导（随机概率）
    if self.ppo_model and self.use_ppo:
        self.apply_ppo_guidance()
```

**新代码**:
```python
def update(self):
    # 分层预测式引导系统
    if self.trajectory_predictor is not None:
        self.predictive_guidance_system()
        self._avoid_corner_traps()
    else:
        self.predict_and_rebalance()
```

#### 3.3.5 更新信息面板

新增显示引导统计：
- `Guided Total: X` - 总引导次数
- 疏散完成时打印详细引导统计

#### 3.3.6 更新启动信息

```python
print("== Hierarchical Predictive Guidance System ==")
print("  - Layer 1: PPO Global Decision")
print("  - Layer 2: Social-LSTM Filtering")
print("  - Layer 3: Individual Decision")
print(f"  - Max guidance per person: {GUIDANCE_CONFIG['max_guidance_count']}")
print(f"  - Cooldown time: {GUIDANCE_CONFIG['cooldown_time']}s")
```

---

## 四、核心算法说明

### 4.1 问题行人识别

```python
def _identify_problem_pedestrians(predictions):
    problem_peds = []
    for ped in pedestrians:
        pred_traj = predictions[ped.id]

        # 检查1：走向拥堵出口
        if will_reach_congested_exit(ped, pred_traj):
            problem_peds.append(ped)
            continue

        # 检查2：走向角落陷阱
        if detect_corner_trap(pred_traj):
            problem_peds.append(ped)
            continue

    return problem_peds
```

### 4.2 引导条件检查

```python
def _can_be_guided(ped, current_time):
    # 条件1：在引导区域内（x > 22，已过闸机）
    if ped.position[0] <= 22.0:
        return False

    # 条件2：引导次数未超限（最多2次）
    if ped.guidance_count >= 2:
        return False

    # 条件3：冷却时间已过（5秒）
    if current_time - ped.last_guidance_time < 5.0:
        return False

    # 条件4：距离目标足够远（>8米）
    if distance(ped.position, ped.target) < 8.0:
        return False

    return True
```

### 4.3 最佳出口选择

```python
def _find_best_alternative_exit(ped, recommended_exit):
    # 优先使用PPO推荐的出口（如果不拥堵）
    if congestion(recommended_exit) < 0.5:
        return recommended_exit

    # 否则选择评分最高的出口
    # 评分 = (1 - 拥堵度) * 10 - 距离 * 0.2
    best_exit = max(exits, key=lambda e: score(e, ped))
    return best_exit
```

---

## 五、改进对比

| 特性 | 改进前 | 改进后 |
|-----|-------|-------|
| 引导触发 | 每步随机8-20% | 预测到问题才引导 |
| 引导对象 | 所有人 | 只引导问题行人 |
| 引导次数 | 无限制 | 最多2次 |
| 冷却时间 | 无 | 5秒 |
| 状态记录 | 无 | 记录次数、时间、原始目标 |
| 符合现实 | ❌ | ✅ |

---

## 六、使用方法

### 6.1 运行可视化

```bash
cd /Users/denghandan/Documents/Demo/crowd-evacuation
.venv/bin/python examples/pygame_metro_with_ppo.py
```

### 6.2 观察指标

1. **引导次数分布**：每个行人被引导0-2次，不应超过
2. **引导间隔**：同一行人两次引导间隔 ≥ 5秒
3. **引导效果**：被引导的行人确实是"将遇到问题"的
4. **疏散效率**：对比改进前后的疏散时间

### 6.3 调试日志

如需启用调试日志，取消注释 `_apply_guidance()` 中的打印语句：

```python
def _apply_guidance(self, ped, new_exit, current_time):
    # ...
    # 调试日志（取消注释启用）
    print(f"引导行人 {ped.id}: 第{ped.guidance_count}次, 目标变更为出口{new_exit.name}")
```

---

## 七、可配置参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|-----|-------|------|---------|
| `max_guidance_count` | 2 | 每人最大引导次数 | 1-3 |
| `cooldown_time` | 5.0 | 冷却时间（秒） | 3-10 |
| `min_distance_to_target` | 8.0 | 最小引导距离（米） | 5-15 |
| `guidance_zone_x` | 22.0 | 引导区域起始x坐标 | 固定（闸机位置） |
| `congestion_threshold` | 0.5 | 拥堵判定阈值 | 0.3-0.7 |
| `corner_trap_radius` | 3.0 | 角落陷阱检测半径（米） | 2-5 |

---

## 八、后续优化建议

1. **引导效果评估**：添加统计分析模块，对比引导前后的疏散效率
2. **动态参数调整**：根据人群密度自动调整配置参数
3. **多模态引导**：结合视觉/语音引导的模拟
4. **历史数据学习**：基于历史疏散数据优化引导策略

---

## 九、相关文献

- Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
- Alahi, A., et al. (2016). Social LSTM: Human trajectory prediction in crowded spaces.
- Weidmann, U. (1993). Transporttechnik der Fußgänger.
- Fruin, J. J. (1971). Pedestrian planning and design.
