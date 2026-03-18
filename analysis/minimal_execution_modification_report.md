# Minimal Execution Modification Report

## 1) 修改内容详细说明

本次改动聚焦于“最小执行建议”的可落地版本：**先监控、再抑制冲突、再稳定蒸馏强度**。

### A. 在检测器训练损失聚合中加入蒸馏 warm-up 与冲突抑制
- 文件：`pcdet/models/detectors/mambafusion.py`
- 关键改动：
  1. 新增蒸馏调度/冲突配置读取（来自 `FUSER` 配置）：
     - `DISTILL_WARMUP_STEPS`
     - `ENABLE_DISTILL_GRAD_MONITOR`
     - `DISTILL_CONFLICT_THRESHOLD`
     - `DISTILL_CONFLICT_MODE`
     - `DISTILL_CONFLICT_DROP_SCALE`
  2. 对 `loss_fusion_distill` 应用 step-based warm-up：
     \[
     L_{dist}^{eff}(t)=\min\left(1,\frac{t+1}{T_w}\right)L_{dist}
     \]
  3. 新增梯度统计（仅针对 `fuser` 参数集合）并记录：
     - `grad_cos_distill_vs_det`
     - `grad_ratio_distill_vs_det`
  4. 当检测到冲突（余弦小于阈值）时执行动作：
     - `drop`: 将蒸馏项清零
     - `downweight`: 将蒸馏项乘以 `DISTILL_CONFLICT_DROP_SCALE`

### B. 在配置中开启最小执行策略参数
- 文件：`tools/cfgs/mambafusion_models/mamba_fusion.yaml`
- 新增参数：
  - `DISTILL_WARMUP_STEPS: 2000`
  - `ENABLE_DISTILL_GRAD_MONITOR: True`
  - `DISTILL_CONFLICT_THRESHOLD: 0.0`
  - `DISTILL_CONFLICT_MODE: "downweight"`
  - `DISTILL_CONFLICT_DROP_SCALE: 0.5`

## 2) 数学依据与论文依据

### A. 多损失训练的梯度分解
总目标：
\[
L = L_{det} + \sum_i \lambda_i L_i
\]
总梯度：
\[
\nabla L = \nabla L_{det} + \sum_i \lambda_i \nabla L_i
\]
当辅助项与主任务方向冲突（内积为负）时，会降低主任务有效下降速度。

### B. 监控指标
- 梯度幅值比：
\[
r_i=\frac{\|\lambda_i g_i\|_2}{\|g_{det}\|_2+\epsilon}
\]
- 方向余弦：
\[
\cos(g_{det},\lambda_i g_i)=\frac{\langle g_{det},\lambda_i g_i\rangle}{\|g_{det}\|_2\|\lambda_i g_i\|_2+\epsilon}
\]

### C. 本次策略与论文对应关系
1. **Warm-up 蒸馏权重**：降低训练早期蒸馏干扰，属于多任务动态加权思想。
2. **冲突检测 + downweight/drop**：当余弦为负时抑制冲突项，属于冲突感知优化思想的轻量实现。

### D. 论文真实链接
- GradNorm (ICML 2018): https://arxiv.org/abs/1711.02257
- PCGrad (NeurIPS 2020): https://arxiv.org/abs/2001.06782
- Multi-Task Learning Using Uncertainty (CVPR 2018): https://arxiv.org/abs/1705.07115
- Knowledge Distillation (NIPS Workshop 2015): https://arxiv.org/abs/1503.02531
- Multi-Objective Optimization / MGDA in deep MTL: https://arxiv.org/abs/1810.04650

## 3) 说明
本次改动不改变数据集/epoch/主网络结构，仅在损失聚合阶段增加蒸馏调度与冲突抑制，并补充可观测统计量，属于“最小侵入式”实现。
