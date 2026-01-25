# AnyLogic 可视化集成指南

本模块提供将人群疏散仿真数据导出到 AnyLogic 进行可视化的工具。

---

## 1. 概述

### 1.1 集成方案

采用 **数据导出 + AnyLogic 回放** 方案：

```
Python 仿真                    AnyLogic 可视化
┌─────────────────┐           ┌─────────────────┐
│ LargeStationEnv │  导出CSV  │ Pedestrian Lib  │
│ 社会力模型(SFM) │ ────────► │ 2D/3D 场景      │
│ 密度预测/PPO    │           │ 动画回放        │
└─────────────────┘           └─────────────────┘
```

### 1.2 场景规格

成都东站大型地铁站 T 形布局：

| 参数 | 值 |
|------|-----|
| 场景尺寸 | 150m × 80m |
| 布局类型 | T 形 (左走廊 + 上下走廊 + 中间连通区) |
| 出口数量 | 8 个闘機 (a-g + 子) |
| 扶梯涌入 | 3 个扶梯点 |
| 人流等级 | small(1000人) / medium(2000人) / large(3000人) |

---

## 2. 数据格式

### 2.1 轨迹数据 (CSV)

导出文件: `exported_data/trajectory_<timestamp>.csv`

```csv
timestamp,ped_id,x,y,vx,vy,speed,ped_type,target_exit,radius
0.00,0,45.23,30.51,0.82,-0.21,0.85,NORMAL,exit_b,0.30
0.10,0,45.31,30.29,0.85,-0.25,0.89,NORMAL,exit_b,0.30
0.10,1,78.45,62.10,0.10,0.95,0.96,ELDERLY,exit_d,0.30
...
```

**字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| `timestamp` | float | 仿真时间 (秒) |
| `ped_id` | int | 行人唯一 ID |
| `x`, `y` | float | 位置坐标 (米) |
| `vx`, `vy` | float | 速度分量 (米/秒) |
| `speed` | float | 速度标量 (米/秒) |
| `ped_type` | string | 行人类型 |
| `target_exit` | string | 目标出口 ID |
| `radius` | float | 行人半径 (米) |

### 2.2 行人类型

| 类型 | 描述 | 期望速度 | 半径 |
|------|------|----------|------|
| `NORMAL` | 普通成年人 | 1.34 m/s | 0.30 m |
| `ELDERLY` | 老年人 | 0.90 m/s | 0.30 m |
| `CHILD` | 儿童 | 0.70 m/s | 0.25 m |
| `IMPATIENT` | 急躁型 | 1.60 m/s | 0.30 m |
| `WITH_SMALL_BAG` | 携带小包 | 1.20 m/s | 0.35 m |
| `WITH_LUGGAGE` | 携带拉杆箱 | 0.90 m/s | 0.50 m |
| `WITH_LARGE_LUGGAGE` | 携带大行李 | 0.70 m/s | 0.60 m |

### 2.3 元数据文件 (JSON)

导出文件: `exported_data/metadata_<timestamp>.json`

```json
{
  "export_time": "2026-01-25T12:00:00",
  "flow_level": "medium",
  "method": "full",
  "total_pedestrians": 2000,
  "total_frames": 3000,
  "dt": 0.1,
  "scene": {
    "width": 150.0,
    "height": 80.0,
    "exits": [
      {"id": "exit_a", "position": [0, 40], "width": 20, "direction": "left"},
      {"id": "exit_b", "position": [52.5, 70], "width": 15, "direction": "up"},
      ...
    ],
    "escalators": [
      {"id": "escalator_A", "position": [35, 40], "size": [8, 20]},
      ...
    ]
  },
  "statistics": {
    "evacuation_rate": 0.95,
    "total_evacuated": 1900,
    "max_density": 3.2,
    "avg_evacuation_time": 180.5
  }
}
```

### 2.4 场景布局文件 (JSON)

导出文件: `exported_data/layout.json`

用于在 AnyLogic 中重建场景几何：

```json
{
  "walls": [
    {"start": [0, 0], "end": [0, 30]},
    {"start": [0, 50], "end": [0, 80]},
    ...
  ],
  "exits": [...],
  "obstacles": [...]
}
```

---

## 3. 快速开始

### 3.1 导出仿真数据

```bash
# 运行仿真并导出轨迹 (默认 medium 流量)
python anylogic/run_and_export.py

# 指定流量等级和方法
python anylogic/run_and_export.py --flow large --method full

# 指定输出目录
python anylogic/run_and_export.py --output anylogic/exported_data/my_sim

# 使用 PPO 模型
python anylogic/run_and_export.py --method ppo --model outputs/models/ppo_large_station_small.zip
```

### 3.2 命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--flow` | medium | 人流等级: small/medium/large |
| `--method` | full | 仿真方法: baseline/prediction/routing/full/ppo |
| `--model` | - | PPO 模型路径 (method=ppo 时使用) |
| `--output` | anylogic/exported_data | 输出目录 |
| `--max-steps` | 3000 | 最大仿真步数 |
| `--export-interval` | 1 | 每 N 步导出一次 (1=每步都导出) |

### 3.3 输出文件

```
anylogic/exported_data/
├── trajectory_20260125_120000.csv    # 轨迹数据
├── metadata_20260125_120000.json     # 元数据
├── layout.json                        # 场景布局
└── evacuation_events.csv              # 疏散事件 (可选)
```

---

## 4. AnyLogic 导入指南

### 4.1 创建新项目

1. 打开 AnyLogic，创建新项目
2. 选择 **Pedestrian Library** 模板
3. 设置场景尺寸: 150m × 80m

### 4.2 导入场景布局

1. 在 **Presentation** 面板中创建 T 形区域
2. 根据 `layout.json` 添加墙壁和障碍物
3. 添加 8 个 **PedService** 作为出口闸机

### 4.3 导入轨迹数据

**方法 1: Database Table**

```java
// 在 Main 的 Startup 中
DatabaseReference db = new DatabaseReference(this, 
    "trajectory_20260125_120000.csv", DatabaseType.CSV);
```

**方法 2: Excel File**

1. 将 CSV 转换为 Excel
2. 使用 **Excel File** 连接器导入

### 4.4 创建行人代理

```java
// Agent: Pedestrian
// Parameters:
double x, y;           // 位置
double vx, vy;         // 速度
String pedType;        // 类型
String targetExit;     // 目标出口

// 在 Main 中根据数据创建代理
for (int i = 0; i < trajectoryData.size(); i++) {
    Row row = trajectoryData.get(i);
    Pedestrian ped = add_pedestrians();
    ped.x = row.getDouble("x");
    ped.y = row.getDouble("y");
    // ...
}
```

### 4.5 动画回放

在 Main 的 **On Timer** 事件中：

```java
// 每 0.1 秒更新一次
currentTime += 0.1;

// 获取当前时刻的所有行人位置
List<Row> currentPositions = trajectoryData.where(
    row -> Math.abs(row.getDouble("timestamp") - currentTime) < 0.05
);

// 更新行人位置
for (Row row : currentPositions) {
    int pedId = row.getInt("ped_id");
    Pedestrian ped = findPedestrian(pedId);
    if (ped != null) {
        ped.setXY(row.getDouble("x"), row.getDouble("y"));
    }
}
```

---

## 5. 高级功能

### 5.1 密度热力图

可以同时导出密度场数据用于热力图可视化：

```bash
python anylogic/run_and_export.py --export-density
```

这会生成 `density_<timestamp>.csv`：

```csv
timestamp,grid_x,grid_y,density
0.0,0,0,0.5
0.0,0,1,1.2
...
```

### 5.2 疏散事件

导出每个行人的疏散时刻：

```csv
ped_id,ped_type,evacuation_time,exit_used,entry_time
0,NORMAL,125.3,exit_b,0.0
1,ELDERLY,180.5,exit_d,0.0
...
```

### 5.3 实时可视化对比

可以同时运行 Python 和 AnyLogic 可视化进行对比：

```bash
# 终端 1: Python 可视化
python scripts/visualize_pedestrian_flow.py --method full

# 终端 2: 导出数据
python anylogic/run_and_export.py --method full
```

---

## 6. 文件结构

```
anylogic/
├── README.md                    # 本文档
├── trajectory_exporter.py       # 轨迹导出器核心模块
├── run_and_export.py           # 运行仿真并导出
└── exported_data/              # 导出的数据 (gitignore)
    ├── trajectory_*.csv
    ├── metadata_*.json
    └── layout.json
```

---

## 7. 注意事项

1. **坐标系统**: Python 仿真使用标准笛卡尔坐标系 (Y 向上)，AnyLogic 默认 Y 向下，导入时需要翻转 Y 坐标

2. **时间单位**: 仿真时间步长 dt=0.1 秒，AnyLogic 中相应设置定时器

3. **行人 ID**: 行人可能在仿真过程中被移除（已疏散），AnyLogic 中需要处理行人消失

4. **性能**: 大流量 (3000 人) 的轨迹数据可能很大，建议使用 `--export-interval 5` 降低数据量

---

## 8. 常见问题

### Q: CSV 文件太大怎么办？

使用导出间隔：
```bash
python anylogic/run_and_export.py --export-interval 5  # 每5步导出一次
```

### Q: 如何只导出特定时间段？

```python
from trajectory_exporter import TrajectoryExporter

exporter = TrajectoryExporter()
exporter.export_time_range(start=50.0, end=100.0)  # 只导出50-100秒
```

### Q: 如何在 AnyLogic 中显示不同类型行人？

根据 `ped_type` 字段设置不同颜色：

```java
// 在 Pedestrian agent 的 On Create 中
switch (pedType) {
    case "NORMAL": setFillColor(blue); break;
    case "ELDERLY": setFillColor(purple); break;
    case "CHILD": setFillColor(orange); break;
    // ...
}
```
