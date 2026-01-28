# 消融实验报告

生成时间: 2026-01-28 16:12:15

## 实验概述

总实验数: 8

## 组 B

| 实验ID | 名称 | 疏散率 | 疏散时间 | 最大拥堵 | 出口均衡 | 累计奖励 |
|--------|------|--------|----------|----------|----------|----------|
| B3_no_congestion_seed123 | No Congestion Penalty | 98.8%±0.0% | 600.0±0.0 | 1.02±0.31 | 9.42±3.13 | 1287.2±59.1 |
| B3_no_congestion_seed42 | No Congestion Penalty | 99.0%±0.5% | 516.2±167.6 | 0.96±0.13 | 10.36±3.39 | 1315.6±114.9 |
| B4_no_balance_seed123 | No Balance Penalty | 99.0%±0.5% | 516.8±166.4 | 1.22±0.25 | 8.29±1.86 | 1354.6±66.7 |
| B4_no_balance_seed42 | No Balance Penalty | 99.5%±0.6% | 360.8±196.8 | 1.03±0.22 | 8.80±3.55 | 1378.5±79.6 |
| B5_no_flow_efficiency_seed123 | No Flow Efficiency Bonus | 99.0%±0.5% | 514.2±171.6 | 0.92±0.25 | 9.92±4.82 | 1194.9±145.6 |
| B5_no_flow_efficiency_seed42 | No Flow Efficiency Bonus | 98.5%±0.5% | 600.0±0.0 | 1.04±0.28 | 9.30±2.85 | 1109.7±37.6 |
| B6_no_safety_seed123 | No Safety Distance Bonus | 99.2%±0.6% | 457.4±174.7 | 1.07±0.14 | 9.21±0.91 | 991.7±189.7 |
| B6_no_safety_seed42 | No Safety Distance Bonus | 99.0%±0.5% | 532.2±135.6 | 0.88±0.15 | 8.48±2.37 | 931.4±153.4 |

## 关键发现

TODO: 根据实验结果添加分析

## 文件列表

- `ablation_results.csv`: 汇总数据表格
- `comparison_*.png`: 各组对比图表
- `ablation_report.md`: 本报告
