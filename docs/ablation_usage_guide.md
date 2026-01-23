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
| **A** | **SFM改进对比** | **核心实验：验证改进方案效果** | **5** |
| B | 密度预测消融 | 预测模块的影响 | 3 |
| C | 动态分流消融 | 决策模块的影响 | 4 |
| D | 行人类型影响 | 行李等类型的影响 | 4 |
| E | 泛化性测试 | 跨流量等级验证 | 3 |
| F | PPO对比 | 证明RL方法的缺点 | 2 |

---

### A组：SFM 改进对比 ⭐⭐（核心实验）

> **这是最重要的实验组，验证改进方案的效果。**

| 实验ID | 方案 | 说明 |
|--------|------|------|
| A1_baseline | 原始 SFM | 基线模型，无任何改进 |
| A2_prediction | SFM + 密度预测 | 仅添加密度场预测模块 |
| A3_routing | SFM + 动态分流 | 仅添加动态分流决策模块 |
| A4_full | SFM + 预测 + 分流 | **完整方案（最终模型）** |
| A5_ppo | SFM + PPO引导 | 对比：强化学习方法 |

**预期结果**：
```
疏散效率: A4_full > A2/A3 > A1_baseline
稳定性:   A4_full > A2/A3 > A1_baseline
```

---

### B组：密度预测消融

测试密度预测模块的各项设计选择。

| 实验ID | 配置 | 说明 |
|--------|------|------|
| B1_convlstm | ConvLSTM预测 | 推荐方案 |
| B2_cnn | 简单CNN预测 | 无时序建模 |
| B3_physics | 物理模型预测 | 基于连续性方程 |

**对比维度**：预测精度、计算时间、疏散效果

---

### C组：动态分流消融

测试决策模块的各项规则。

| 实验ID | 配置 | 说明 |
|--------|------|------|
| C1_full_rules | 完整规则 | 距离+拥堵+预测+均衡 |
| C2_no_prediction | 无预测权重 | 移除预测拥堵分数 |
| C3_no_balance | 无均衡权重 | 移除负载均衡分数 |
| C4_distance_only | 仅距离 | 最近出口策略（基线） |

**预期结果**：C1 > C2 > C3 > C4

### D组：行人类型影响

| 实验ID | 配置 | 说明 |
|--------|------|------|
| D1_full | 完整配置（基准） | 7种行人类型 |
| D2_no_luggage | 无行李行人 | 只有基础4种类型 |
| D3_single_type | 只用NORMAL类型 | 测试多样性影响 |
| D4_high_luggage | 高行李比例 | 行李行人占比 60% |

**行人类型分布对比**：

| 类型 | D1_full | D2_no_luggage | D3_single | D4_high_luggage |
|------|---------|---------------|-----------|-----------------|
| NORMAL | 40% | 57% | 100% | 20% |
| ELDERLY | 10% | 14% | - | 5% |
| CHILD | 5% | 7% | - | 5% |
| IMPATIENT | 5% | 7% | - | 5% |
| WITH_SMALL_BAG | 25% | - | - | 30% |
| WITH_LUGGAGE | 12% | - | - | 25% |
| WITH_LARGE_LUGGAGE | 3% | - | - | 10% |

---

### E组：跨流量泛化性测试 ⭐

> **验证模型是否满足核心设计原则：小流量训练→大流量应用。**

| 实验ID | 训练流量 | 测试流量 | 说明 |
|--------|----------|----------|------|
| E1_small_to_all | 小(1000人) | 小/中/大 | **核心验证** |
| E2_medium_to_all | 中(2000人) | 小/中/大 | 对比 |
| E3_large_only | 大(3000人) | 大 | 对比 |

**验证标准**：

| 测试流量 | 疏散时间要求 | 最大密度要求 |
|----------|-------------|-------------|
| 小(1000人) | < 5 分钟 | < 4.0 人/m² |
| 中(2000人) | < 8 分钟 | < 4.0 人/m² |
| 大(3000人) | < 10 分钟 | < 4.5 人/m² |

---

### F组：PPO 对比实验

> **证明强化学习方法的缺点，说明为什么选择密度预测+分流方案。**

| 实验ID | 方案 | 说明 |
|--------|------|------|
| F1_ppo_small | PPO (小流量训练) | 训练 20万步 |
| F2_ppo_full | PPO (完整训练) | 训练 50万步 |

**对比维度**：

| 对比项 | 密度预测+分流 | PPO引导 |
|--------|--------------|---------|
| 训练时间 | 较短 | 数小时 |
| 泛化性 | 天然泛化 | 需验证 |
| 可解释性 | 规则清晰 | 黑盒 |
| 计算资源 | CPU即可 | 需GPU训练 |

**预期结论**：
- PPO 需要大量训练时间
- PPO 泛化性不如密度预测方案
- 密度预测+分流更适合实际部署

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
# 第1步：⭐⭐ 运行核心对比实验（最重要）
python examples/run_ablation.py -g A

# 第2步：运行泛化性验证
python examples/run_ablation.py -g E

# 第3步：运行密度预测消融
python examples/run_ablation.py -g B

# 第4步：运行动态分流消融
python examples/run_ablation.py -g C

# 第5步：运行行人类型影响实验
python examples/run_ablation.py -g D

# 第6步：运行PPO对比实验（证明其缺点）
python examples/run_ablation.py -g F
```

> ⚠️ **重要**：A组是核心实验，必须证明 A4_full > A1_baseline。

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
