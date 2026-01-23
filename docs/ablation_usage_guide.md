# 消融实验使用指南

---

## ⚠️ 核心原则

> **所有消融实验必须通过跨流量泛化性验证：小流量训练的模型必须能用于大流量场景。**

---

## 目录

1. [快速开始](#快速开始)
2. [实验组说明](#实验组说明)
3. [运行实验](#运行实验)
4. [查看结果](#查看结果)
5. [自定义实验](#自定义实验)
6. [常见问题](#常见问题)

---

## 快速开始

### 前置条件

确保已安装所有依赖：

```bash
cd /Users/denghandan/Documents/Demo/crowd-evacuation
pip install -r requirements.txt
```

### 最简单的使用方式

```bash
# 运行一个快速测试（10000步，约5分钟）
python examples/run_ablation.py --groups A --timesteps 10000
```

### 完整运行所有实验

```bash
# 运行全部17组实验（预计17小时）
python examples/run_ablation.py
```

---

## 实验组说明

| 组 | 名称 | 实验内容 | 实验数量 |
|----|------|----------|----------|
| A | 观测空间 | 测试不同维度的观测向量 | 3 |
| B | 奖励函数 | 逐个移除奖励组件 | 6 |
| C | 轨迹预测 | 神经网络 vs 线性外推 | 2 |
| D | 行人仿真 | SFM参数和行人类型 | 6 |
| E | 引导策略 | 有无PPO引导对比 | 2 |
| **F** | **泛化性测试** | **跨流量等级验证** | **3** |

### A组：观测空间消融

测试PPO智能体需要多少信息才能做出好决策。

| 实验ID | 观测维度 | 包含信息 |
|--------|----------|----------|
| A1_41D | 41维 | 全部信息（基准）：出口(30) + 全局(6) + 瓶颈(5) |
| A2_20D | 20维 | 出口密度(10) + 出口拥堵(10) |
| A3_12D | 12维 | 出口密度(10) + 剩余比例 + 时间比例 |

**观测空间构成（41维基准）**：
```
出口相关 (30维):
  - exit_densities[10]: 各出口密度 (归一化)
  - exit_congestions[10]: 各出口拥堵度 (归一化)
  - exit_flow_rates[10]: 各出口流量 (归一化)

全局状态 (6维):
  - remaining_ratio: 剩余人数比例
  - time_ratio: 时间比例
  - avg_speed_ratio: 平均速度比例
  - panic_level: 恐慌水平
  - crush_risk: 踩踏风险
  - spawn_pressure: 涌入压力

瓶颈状态 (5维):
  - bottleneck_densities[5]: 各瓶颈区密度
```

### B组：奖励函数消融

测试每个奖励组件的重要性。

| 实验ID | 移除的奖励 |
|--------|-----------|
| B1_full | 无（基准） |
| B2_no_evac | 移除疏散奖励 |
| B3_no_congestion | 移除拥堵惩罚 |
| B4_no_balance | 移除均衡惩罚 |
| B5_no_flow_efficiency | 移除人流效率奖励 |
| B6_no_safety | 移除安全间距奖励 |

### C组：轨迹预测消融

| 实验ID | 预测方法 |
|--------|----------|
| C1_neural | Social-LSTM神经网络 |
| C2_linear | 简单线性外推 |

### D组：行人仿真消融

| 实验ID | 配置 | 说明 |
|--------|------|------|
| D1_full | 完整配置（基准） | 7种行人类型 + 完整SFM |
| D2_no_luggage | 无行李行人 | 只有 NORMAL/ELDERLY/CHILD/IMPATIENT |
| D3_single_type | 只用NORMAL类型 | 测试行人多样性影响 |
| D4_weak_social_force | 减弱社会力强度 | A=1000, B=0.04 |
| D5_no_gbm | 禁用GBM行为修正 | 测试GBM预测器影响 |
| D6_high_luggage | 高行李比例 | 行李行人占比 60% |

**行人类型分布对比**：

| 类型 | D1_full | D2_no_luggage | D3_single | D6_high_luggage |
|------|---------|---------------|-----------|-----------------|
| NORMAL | 40% | 57% | 100% | 20% |
| ELDERLY | 10% | 14% | - | 5% |
| CHILD | 5% | 7% | - | 5% |
| IMPATIENT | 5% | 7% | - | 5% |
| WITH_SMALL_BAG | 25% | - | - | 30% |
| WITH_LUGGAGE | 12% | - | - | 25% |
| WITH_LARGE_LUGGAGE | 3% | - | - | 10% |

### E组：引导策略消融

| 实验ID | 引导方式 |
|--------|----------|
| E1_ppo_guidance | PPO智能引导 |
| E2_no_guidance | 无引导（自由选择） |

### F组：跨流量泛化性测试 ⭐

> **这是最重要的实验组，验证模型是否满足核心设计原则。**

| 实验ID | 训练流量 | 测试流量 | 说明 |
|--------|----------|----------|------|
| F1_small_to_all | 小(1000人) | 小/中/大 | **核心验证**：小流量训练→全流量测试 |
| F2_medium_to_all | 中(2000人) | 小/中/大 | 对比：中流量训练效果 |
| F3_curriculum | 渐进式 | 小/中/大 | 课程学习：小→中→大 |

**F组验证标准**：

| 测试流量 | 疏散率要求 | 最大密度要求 | 状态 |
|----------|-----------|-------------|------|
| 小(1000人) | ≥ 95% | < 4.0 人/m² | 必须通过 |
| 中(2000人) | ≥ 90% | < 4.0 人/m² | 必须通过 |
| 大(3000人) | ≥ 85% | < 4.5 人/m² | 必须通过 |

**F组运行示例**：

```bash
# 运行核心泛化性验证
python examples/run_ablation.py -e F1_small_to_all

# 输出示例
┌─────────────────────────────────────────────────────────┐
│ F1_small_to_all 泛化性测试结果                            │
├──────────┬────────────┬────────────┬───────────────────┤
│ 测试流量  │ 疏散率      │ 最大密度    │ 状态              │
├──────────┼────────────┼────────────┼───────────────────┤
│ small    │ 97.2%      │ 2.8 人/m²  │ ✅ PASS           │
│ medium   │ 93.5%      │ 3.4 人/m²  │ ✅ PASS           │
│ large    │ 88.1%      │ 4.2 人/m²  │ ✅ PASS           │
├──────────┴────────────┴────────────┴───────────────────┤
│ 总体结果: ✅ 泛化性验证通过                               │
└─────────────────────────────────────────────────────────┘
```

---

## 运行实验

### 命令行参数

```bash
python examples/run_ablation.py [选项]
```

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--groups` | `-g` | 指定实验组 | `-g A B` |
| `--experiments` | `-e` | 指定实验ID | `-e A1_16D A2_8D` |
| `--timesteps` | `-t` | 训练步数 | `-t 50000` |
| `--device` | `-d` | 计算设备 | `-d mps` |
| `--output-dir` | `-o` | 输出目录 | `-o outputs/my_ablation` |
| `--summary-only` | | 只生成报告 | `--summary-only` |
| `--config` | `-c` | 配置文件 | `-c my_config.yaml` |

### 使用示例

#### 1. 快速测试单个实验

```bash
# 只运行A1实验，1万步（约3分钟）
python examples/run_ablation.py -e A1_41D -t 10000
```

#### 2. 运行一个完整的实验组

```bash
# 运行A组所有实验（3个实验 × 3个种子 = 9次）
python examples/run_ablation.py -g A
```

#### 3. 运行多个实验组

```bash
# 运行A组和B组
python examples/run_ablation.py -g A B
```

#### 4. 使用GPU加速

```bash
# Apple Silicon Mac
python examples/run_ablation.py -d mps -g A

# NVIDIA GPU
python examples/run_ablation.py -d cuda -g A
```

#### 5. 减少训练步数（快速验证）

```bash
# 用1万步快速测试所有组
python examples/run_ablation.py -t 10000
```

#### 6. 从已有结果生成报告

```bash
# 不训练，只汇总已有结果
python examples/run_ablation.py --summary-only
```

### 推荐的运行顺序

如果时间有限，建议按以下顺序运行：

```bash
# 第1步：快速验证流程是否正常（5分钟）
python examples/run_ablation.py -e A1_41D -t 5000

# 第2步：⭐ 运行泛化性验证（最重要，必须通过）
python examples/run_ablation.py -g F

# 第3步：运行引导策略对比实验（1小时）
python examples/run_ablation.py -g E -t 50000

# 第4步：运行观测空间消融（3小时）
python examples/run_ablation.py -g A

# 第5步：运行奖励函数消融（6小时）
python examples/run_ablation.py -g B

# 第6步：运行行人仿真消融（含行李类型）
python examples/run_ablation.py -g D

# 第7步：运行轨迹预测消融
python examples/run_ablation.py -g C
```

> ⚠️ **重要**：F组泛化性测试必须通过，否则其他消融实验结果无意义。

---

## 查看结果

### 输出目录结构

运行实验后，结果保存在 `outputs/ablation/` 目录：

```
outputs/ablation/
├── A_A1_16D_seed42/          # 单个实验结果
│   ├── model.zip             # 训练好的PPO模型
│   ├── training_log.csv      # 训练过程日志
│   ├── eval_results.json     # 评估结果
│   ├── training_curve.png    # 训练曲线图
│   └── config.yaml           # 实验配置
├── A_A2_8D_seed42/
├── ...
└── summary/                  # 汇总结果
    ├── ablation_results.csv  # 所有实验结果表格
    ├── comparison_A.png      # A组对比图
    ├── comparison_B.png      # B组对比图
    └── ablation_report.md    # 完整报告
```

### 查看汇总表格

```bash
# 用命令行查看
cat outputs/ablation/summary/ablation_results.csv

# 或用Excel/Numbers打开
open outputs/ablation/summary/ablation_results.csv
```

### 查看对比图

```bash
# 打开所有图片
open outputs/ablation/summary/*.png
```

### 查看完整报告

```bash
# 用Markdown预览器或编辑器打开
open outputs/ablation/summary/ablation_report.md
```

### 评估指标说明

| 指标 | 含义 | 理想方向 |
|------|------|----------|
| evacuation_rate | 疏散完成率 | 越高越好 |
| evacuation_time | 平均疏散时间（步数） | 越低越好 |
| max_congestion | 最大拥堵度 | 越低越好 |
| max_local_density | 最大局部密度（人/m²） | < 4.5 |
| exit_balance | 出口负载均衡度（变异系数） | 越低越好 |
| cumulative_reward | 累计奖励 | 越高越好 |
| **generalization_score** | **泛化性得分** | **越高越好** |

**泛化性得分计算**：
```
generalization_score = (small_rate + medium_rate + large_rate) / 3
                     × (1 - density_penalty)

其中:
- small_rate: 小流量疏散率
- medium_rate: 中流量疏散率
- large_rate: 大流量疏散率
- density_penalty: 超过安全密度的惩罚
```

---

## 自定义实验

### 修改训练参数

编辑 `configs/ablation_configs.yaml` 中的 `global.training` 部分：

```yaml
global:
  training:
    total_timesteps: 100000    # 训练步数
    n_envs: 4                  # 并行环境数（越多越快）
    learning_rate: 0.0003      # 学习率
    batch_size: 128            # 批大小
```

### 修改评估参数

```yaml
global:
  evaluation:
    n_eval_episodes: 10        # 评估episode数
    random_seeds: [42, 123, 456]  # 随机种子（重复次数）
```

### 添加新实验

在 `configs/ablation_configs.yaml` 中添加：

```yaml
group_B:
  experiments:
    # 添加新实验
    B7_custom:
      name: "Custom Reward Config"
      reward_weights:
        evac_per_person: 20.0   # 修改为你想测试的值
        congestion_penalty: 5.0
        # ... 其他参数
      baseline: false
```

然后运行：

```bash
python examples/run_ablation.py -e B7_custom
```

---

## 常见问题

### Q: 运行太慢怎么办？

**A:** 有几种方法加速：

1. 减少训练步数：
   ```bash
   python examples/run_ablation.py -t 10000
   ```

2. 使用GPU：
   ```bash
   python examples/run_ablation.py -d mps  # Mac
   python examples/run_ablation.py -d cuda # NVIDIA
   ```

3. 减少重复次数（编辑配置文件）：
   ```yaml
   evaluation:
     random_seeds: [42]  # 只用1个种子
   ```

### Q: 内存不足怎么办？

**A:** 减少并行环境数：

编辑 `configs/ablation_configs.yaml`：
```yaml
training:
  n_envs: 2  # 从4改为2
```

### Q: 如何只重新生成报告？

**A:** 使用 `--summary-only` 参数：

```bash
python examples/run_ablation.py --summary-only
```

### Q: 实验中断了怎么办？

**A:** 已完成的实验结果会保存。可以只运行未完成的实验：

```bash
# 假设A组已完成，只运行B组
python examples/run_ablation.py -g B C D E
```

### Q: 如何查看训练进度？

**A:** 训练时会实时打印进度：

```
[训练] 开始训练 A1_16D_seed42, 总步数: 100000
| rollout/ep_rew_mean | 45.2 |
| rollout/ep_len_mean | 320  |
| time/total_timesteps| 10000|
```

### Q: 如何对比不同实验的结果？

**A:** 查看汇总CSV或报告：

```bash
# 查看表格
cat outputs/ablation/summary/ablation_results.csv | column -t -s,

# 或打开报告
open outputs/ablation/summary/ablation_report.md
```

---

## 完整示例流程

```bash
# 1. 进入项目目录
cd /Users/denghandan/Documents/Demo/crowd-evacuation

# 2. 快速测试（确保一切正常）
python examples/run_ablation.py -e A1_41D -t 5000

# 3. 查看测试结果
ls outputs/ablation/
cat outputs/ablation/A_A1_41D_seed42/eval_results.json

# 4. ⭐ 运行泛化性验证（最重要）
python examples/run_ablation.py -g F

# 5. 确认泛化性通过后，运行其他实验
python examples/run_ablation.py -g A B D E

# 6. 生成汇总报告
python examples/run_ablation.py --summary-only

# 7. 查看报告
open outputs/ablation/summary/ablation_report.md
open outputs/ablation/summary/comparison_F.png  # 泛化性对比图
```

> ⚠️ **注意**：如果 F 组泛化性验证未通过，需要先排查归一化问题，再进行其他实验。

---

## 联系与反馈

如有问题，请检查：
1. 依赖是否安装完整：`pip install -r requirements.txt`
2. Python版本是否正确：需要 Python 3.8+
3. 查看错误日志，定位问题

如需帮助，请提供：
- 运行的完整命令
- 错误信息
- Python版本：`python --version`
