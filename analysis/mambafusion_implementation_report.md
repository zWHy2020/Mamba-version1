# 10.3 最小实现顺序落地后的修改方案报告

## 1) 模块级修改清单：改了什么、对应问题、程序位置、数学与论文依据

### 1.1 `pcdet/models/backbones_2d/fuser/gated_fusion.py`

**修改内容**
- 在 `SparseMoESpatialGate` 中新增/暴露路由统计信息：`router_logits`、`keep_score`、`keep_score_2d`，用于后续计算 `L_z`、预算约束与软门控训练。  

**针对问题**
- 解决“MoE 缺少可监督路由统计”的实现瓶颈（之前只有 gate 结果，缺少 logits 与连续保留分数）。

**程序与数学对应**
- 路由概率：`p = softmax(z)`；
- z-loss 依赖 `z`：
  \[
  L_z = \frac{1}{N}\sum_t\Big(\log\sum_i e^{z_{t,i}}\Big)^2
  \]
- 预算损失依赖连续保留分数 `m`（由 `keep_score` 提供）：
  \[
  L_{budget}=\|\mathbb{E}[m]-\rho\|_2^2
  \]

**论文依据**
- ST-MoE, NeurIPS 2022: https://arxiv.org/abs/2202.08906  
- DynamicViT, NeurIPS 2021: https://arxiv.org/abs/2106.02034  
- A-ViT, ICLR 2022: https://arxiv.org/abs/2112.07658

---

### 1.2 `pcdet/models/backbones_2d/fuser/convfuser.py`

**修改内容（严格按 10.3 顺序落地）**
1. 新增 MoE 辅助损失配置与实现：`L_lb`、`L_z`、`L_cap`、`L_budget`、`L_null`。  
2. 训练期把硬掩码乘法改为 soft mask 路径（推理保持硬门控逻辑）。  
3. 路由 `topk` 训练/推理分离：`TOPK_TRAIN` 与 `TOPK_EVAL`。  
4. 新增稀疏蒸馏（teacher=融合前 dense 路径，student=稀疏路径）`loss_fusion_distill`。  

**针对问题**
- MoE 路由塌缩/不均衡（缺负载均衡与稳定化）；
- `cat_bev * spatial_keep_mask` 硬置零导致信息丢失；
- top1+null 过于激进；
- 稀疏路径表达能力下降。

**程序与数学对应**
- 总体辅助损失：
  \[
  L = L_{det} + \lambda_{lb}L_{lb}+\lambda_zL_z+\lambda_{cap}L_{cap}+\lambda_bL_{budget}+\lambda_nL_{null}+\lambda_dL_{distill}
  \]
- 负载均衡：
  \[
  L_{lb}=E\sum_i f_iP_i
  \]
- 容量约束：
  \[
  C_i=\left\lceil cf\cdot\frac{N}{E}\right\rceil,\quad
  L_{cap}=\frac{1}{N}\sum_i\max(0,n_i-C_i)
  \]
- 软门控：
  \[
  \tilde h = m\odot h,\quad m\in[0,1]
  \]
- null 占比预算：
  \[
  L_{null}=\max(0,\mathbb{E}[p_{null}] - \rho_{null})
  \]
- 蒸馏：
  \[
  L_{distill}=\|F_s-F_t\|_2^2
  \]

**论文依据**
- GShard, ICLR 2021: https://arxiv.org/abs/2006.16668  
- Switch Transformers, JMLR 2022: https://arxiv.org/abs/2101.03961  
- ST-MoE, NeurIPS 2022: https://arxiv.org/abs/2202.08906  
- DynamicViT, NeurIPS 2021: https://arxiv.org/abs/2106.02034  
- V-MoE, ICLR 2022: https://arxiv.org/abs/2106.05974

---

### 1.3 `pcdet/models/detectors/mambafusion.py`

**修改内容**
- 训练总损失处加入辅助项聚合：`loss_moe_lb`、`loss_moe_z`、`loss_moe_cap`、`loss_mask_budget`、`loss_null_budget`、`loss_fusion_distill`，并记录 `loss_total`。  

**针对问题**
- 之前辅助损失没有进入优化目标，导致“定义了策略但不生效”。

**程序与数学对应**
- 实现 `L_total = L_det + Σ λ_i L_i` 的显式优化目标。

**论文依据**
- 多目标联合训练是上述 MoE/稀疏化论文的标准做法（见 GShard、Switch、ST-MoE、DynamicViT）。

---

### 1.4 `tools/cfgs/mambafusion_models/mamba_fusion.yaml`

**修改内容**
- 新增可控参数：`TOPK_TRAIN/EVAL`、`SPARSE_GATE_INCLUDE_NULL_EXPERT`、`USE_SOFT_MASK_TRAIN`、`MASK_BUDGET_TARGET`、`LOSS_WEIGHT_*`、`CAPACITY_FACTOR`、`NULL_BUDGET_TARGET`、`USE_SPARSE_DISTILL` 等。  

**针对问题**
- 让 10.3 路线可配置、可消融、可复现实验。

---

## 2) 修改后模型整体架构与“实际发挥作用”的损失函数

### 2.1 架构流程（训练时）
1. 图像/点云 BEV 特征进入 `ConvFuser`；
2. `SparseMoESpatialGate` 输出稀疏路由结果 + 路由统计（logits/prob/keep_score）；
3. 融合主干（含 vmamba 分支）得到 `cat_bev`；
4. 训练期使用 soft mask 对 `cat_bev` 连续门控；
5. 计算主检测损失 `L_det`（原有 dense head）；
6. 同时计算并回传辅助损失：`L_lb, L_z, L_cap, L_budget, L_null, L_distill`；
7. 在 detector 层统一求和得到 `loss_total`。

### 2.2 实际生效损失函数（当前代码）
- 主损失：`loss_trans`（来自检测头，原有）。
- 新增生效项：
  - `loss_moe_lb`
  - `loss_moe_z`
  - `loss_moe_cap`
  - `loss_mask_budget`
  - `loss_null_budget`
  - `loss_fusion_distill`
- 汇总逻辑：`mambafusion.py::get_training_loss` 中显式相加并记录。

### 2.3 论文与数学依据（对应上述损失）
- `L_lb / capacity`: GShard, Switch Transformer。  
- `L_z`: ST-MoE。  
- `L_budget / soft mask`: DynamicViT, A-ViT。  
- `L_distill`: DynamicViT 的稀疏蒸馏范式。  

参考链接：
- https://arxiv.org/abs/2006.16668
- https://arxiv.org/abs/2101.03961
- https://arxiv.org/abs/2202.08906
- https://arxiv.org/abs/2106.02034
- https://arxiv.org/abs/2112.07658

---

## 3) 修改后模型仍可能导致“训练与测试事实核对问题”的风险点

> 下述风险是对“你本地日志中已暴露的问题”进行的修改后复盘，不是额外假设。

1. **训练轮次仍可能不足**  
   日志显示之前 10 epoch 末段 `matched_ious` 约 0.51、`loss_bbox` 约 1.2，说明可能尚未充分收敛。即便加了辅助损失，如果训练预算不提升，最终 mAP/NDS 仍可能偏低。  

2. **多辅助损失权重耦合敏感**  
   现在新增了 6 个辅助项，若 `λ` 设定不当，可能出现“辅助项压过检测主损失”或“辅助项太弱无效”。这会导致训练前期 loss 波动增大。  

3. **蒸馏额外前向带来计算/显存压力**  
   训练期 distill 需要 teacher 路径，可能拉长 step time；若 batch 受限，会反向影响统计稳定性。  

4. **top-k / null 约束虽已引入，但仍需与数据分布匹配**  
   若 `NULL_BUDGET_TARGET` 过大，仍可能过度稀疏；若过小则节省计算不明显。需要结合日志中的 keep-ratio 与验证指标联调。  

5. **NDS 是耦合指标，误差项会共同放大**  
   即使 mAP 有回升，如果 mATE/mAOE/mAVE 改善不足，NDS 提升也可能受限（nuScenes 官方定义）。  

**对应日志证据与评估依据**
- 训练设置与优化器事实：`BATCH_SIZE_PER_GPU:3`, `ACCUMULATION_STEPS:2`, `NUM_EPOCHS:10`。  
- 末段收敛状态：`loss_bbox≈1.21`, `matched_ious≈0.51`。  
- NDS 定义与耦合误差项：nuScenes detection eval 文档。  

参考链接：
- nuScenes detection eval: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
