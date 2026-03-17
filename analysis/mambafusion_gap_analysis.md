# Mamba-version1 与官方 MambaFusion 指标差距分析

## 1) 训练与测试事实核对（基于本地日志）

- 本次训练命令对应日志显示：4 卡分布式、`batch_size=3`（每卡）、`epochs=10`、`--sync_bn`、`--ckpt latest_model.pth`、`logger_iter_interval=1000`。日志中可直接看到 `batch_size 3` 与 `ckpt ...latest_model.pth`。  
- 优化器配置显示：`BATCH_SIZE_PER_GPU: 3`、`ACCUMULATION_STEPS: 2`、`LR: 0.0015`。因此等效全局 batch 为 `4 * 3 * 2 = 24`。  
- 训练末段（epoch 10）损失统计约稳定在 `loss_bbox≈1.20`、`loss_hm≈0.77~0.78`、`matched_ious≈0.51`，总损失均值约 `2.36~2.37`，未体现出继续大幅下降趋势。  

## 2) 与官方仓库可见差异（代码/配置）

与 `https://github.com/AutoLab-SAI-SJTU/MambaFusion` 当前 `main` 对比，本地仓库在融合模块引入了额外策略：

1. 在 `ConvFuser` 中新增：
   - `ModalityDropout`（模态级随机丢弃）；
   - `SparseMoESpatialGate`（空间位置 Top-K 稀疏门控）；
   - 以及 `cat_bev * spatial_keep_mask` 的硬掩码乘法。  
2. 在配置 `tools/cfgs/mambafusion_models/mamba_fusion.yaml` 中开启/注入了相关参数：
   - `USE_GATED_FUSION: True`；
   - `USE_MODALITY_DROPOUT: False`（但模块仍已接入）；
   - `GATE_USE_MASK: True`、`GATE_HIDDEN_DIM: 128`、`GATE_PROJ_DIM: 64`；
   - `USE_ALIGNMENT_PROXY: False`、`ALIGNMENT_PROXY_MODE: "none"`。  
3. 新增文件 `gated_fusion.py`，其中 `SparseMoESpatialGate` 使用 token 级路由并输出 `keep_mask`，当包含 null expert 且 top-1 选择到 null 时，该位置特征会被完全抑制（仅保留 keep_mask=0）。

## 3) 你补充关注的现象：前期学习率上升且总损失偏高，为什么会发生？

这个现象在当前配置下是**训练策略本身的预期行为**，并且可以被日志与代码同时验证：

1. **日志已经显示前期 LR 递增**：在 epoch 1 内，`LR` 从 `1.500e-04` 上升到 `1.676e-04`。  
2. **调度器代码明确是 OneCycle 上升段**：
   - 配置给出 `OPTIMIZER: adam_onecycle_split`、`LR: 0.0015`、`DIV_FACTOR: 10`、`PCT_START: 0.4`；
   - 对应实现中 `low_lr = lr_max / div_factor`，即 `0.0015 / 10 = 1.5e-4`；
   - 前 `pct_start=0.4` 的总步数内，使用 `annealing_cos(low_lr, lr_max, pct)` 单调上升到 `lr_max`，后 60% 再下降。

数学上，代码中的余弦退火函数为：

\[
\text{annealing\_cos}(s,e,p)=e+\frac{s-e}{2}(\cos(\pi p)+1),\quad p\in[0,1]
\]

- 当上升段设置 `s=low\_lr, e=lr\_max` 时，随 `p` 增大，学习率从 `low_lr` 平滑升至 `lr_max`；
- 所以前几个 epoch 看到“LR 持续升高”是由 `OneCycle + PCT_START=0.4` 直接决定的。

3. **前期总损失偏高也符合检测头损失定义与初始化阶段特征**：
   - 当前检测头包含 heatmap 分类项（高权重、正负样本极不均衡）与框回归项；
   - 训练初期预测尚未成形时，focal/gaussian-focal 类损失会显著偏大，随后随着分类置信与定位变好快速下降。

这与你日志中的走势一致：首个记录点 `Loss≈1868`（`loss_hm≈1850`）非常高，随后在同一 epoch 内迅速降到两位数/个位数区间。

## 4) 为什么这些改动会显著拉低 mAP/NDS（结合日志趋势）

> 你给出的推理结果是 `mAP=0.5729, NDS=0.6595`，明显低于官方 README 在 nuScenes val 的 `mAP=72.7, NDS=75.0`（即 0.727/0.750）。

核心原因更可能来自**融合信息被稀疏门控过度裁剪**，而非“是否断点续训”。理由如下：

1. **训练日志未显示“持续优化到高精度区间”的轨迹**：末段 `matched_ious` 仅约 0.51，`loss_bbox` 仍在 ~1.2 水平，说明定位质量并未逼近高性能模型常见收敛状态。  
2. **你引入的是“硬稀疏+掩码乘法”路径**：
   - 路由后 `zhat_cam/zhat_lidar` 已做门控；
   - 后续再次 `cat_bev * spatial_keep_mask`，相当于对大量空间 token 做硬置零；
   - 对检测任务中的长尾/小目标（pedestrian, traffic_cone 等）非常敏感，会直接损害召回与定位稳定性。  
3. **MoE 稀疏路由的理论前提是“路由器有足够监督与负载均衡约束”**。当前实现没有看到显式 load-balancing/importance loss（如 Switch Transformer 的 auxiliary load loss），则容易发生路由塌缩，导致信息利用不充分。  
4. **NDS 是多误差项耦合指标**：不仅受 mAP，还受 mATE/mAOE/mAVE 等影响。空间 token 丢失会同时影响中心、朝向、速度估计，因此 NDS 往往比 mAP 的体感下降更明显。

## 5) 数学与论文依据（真实可访问链接）

1. **NuScenes NDS 定义**（官方 devkit 文档）：NDS 将 mAP 与多种 TP error（ATE/ASE/AOE/AVE/AAE）综合，故任何对几何定位和运动估计的不利扰动都会被放大到最终 NDS。  
   - https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md

2. **MoE 稀疏路由需要负载均衡正则**：
   - Shazeer et al., 2017, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*  
     https://arxiv.org/abs/1701.06538
   - Fedus et al., 2021, *Switch Transformers*（明确讨论 load-balancing auxiliary loss）  
     https://arxiv.org/abs/2101.03961

3. **State-space / Mamba 系列强调“信息流连续建模”**：若在融合前后进行硬置零，会破坏连续表征并提高优化难度。  
   - Gu & Dao, 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*  
     https://arxiv.org/abs/2312.00752

4. **OneCycle 学习率策略与“先升后降”机制**（解释前期 LR 上升）：  
   - Smith, 2018, *A Disciplined Approach to Neural Network Hyper-Parameters*  
     https://arxiv.org/abs/1803.09820

5. **Focal Loss 在训练初期（预测置信度低）会产生较大分类损失，随后随易样本置信度提升而快速下降**：  
   - Lin et al., 2017, *Focal Loss for Dense Object Detection*  
     https://arxiv.org/abs/1708.02002

6. **多模态鲁棒训练中的模态 dropout 是双刃剑**：需要与任务难度、融合结构和调度策略共同匹配，否则会在主模态信息不足时降低上限。  
   - Neverova et al., 2015, *ModDrop: adaptive multi-modal gesture recognition*  
     https://arxiv.org/abs/1501.00102

## 6) 结论（按影响优先级）

1. **最高可疑项：新增的稀疏门控融合路径（含 keep_mask 硬抑制）**。这是与官方代码最关键的结构差异，且直接作用于 BEV 融合主干。
2. **次高可疑项：训练仅 10 epoch 且末段损失/IoU 指标未体现高质量收敛**，导致最终 checkpoint 质量不足。
3. **较低可疑项：断点续训本身**。在你给定日志中，看不到它是“决定性”劣化来源；更像是模型已被改造成不同优化问题后，收敛目标发生偏移。

## 7) 可复现实验建议（用于定位责任改动）

1. 在本地代码中仅做一项回退：
   - `USE_GATED_FUSION=False`，其余超参数不变，训练/验证一次；
2. 若指标显著回升，再逐项打开：
   - `USE_GATED_FUSION=True` + `include_null_expert=False`；
   - 或保留 gate 但去掉 `cat_bev * spatial_keep_mask` 的硬乘法（改为软权重）；
3. 增加路由均衡损失（aux loss）并记录 expert usage 直方图，观察是否存在路由塌缩。


## 8) 针对你提出两点的直接回答（程序对应 + 数学 + 论文）

### 8.1 你的两点判断是否合理？

结论：**合理，而且与本地代码实现直接对应**。

#### (A) “MoE 缺乏损失函数约束”——合理

- **程序对应**：`SparseMoESpatialGate` 做了 token 级 router + top-k hard gate，但仓库中没有看到针对路由负载均衡（load-balance / importance）的辅助损失项被加入总损失。当前门控仅返回 `router_prob/router_gate/keep_ratio` 统计，不参与显式正则。  
- **数学机制**：若 router 输出为 `p_i(x)=softmax(z_i(x))`，top-k 后只有少量 expert 参与。若无负载均衡正则，最优解可退化为“少数 expert 过载、多数 expert 闲置”，导致有效容量下降。常见做法是最小化类似
  \[
  L_{aux}=E\sum_{i=1}^{E} f_i\,P_i
  \]
  的均衡项（`f_i` 为路由频次，`P_i` 为平均路由概率），约束 expert 使用分布。  
- **高质量论文依据（2020+）**：
  - **GShard**（ICLR 2021）：提出大规模 MoE 与辅助负载均衡设计。https://arxiv.org/abs/2006.16668
  - **Switch Transformer**（JMLR 2022；NeurIPS 2021 workshop 版本）：明确给出 load-balancing auxiliary loss。https://arxiv.org/abs/2101.03961
  - **ST-MoE**（NeurIPS 2022）：系统讨论稳定训练与 router 正则。https://arxiv.org/abs/2202.08906

#### (B) “空间掩码把 token 置零”——合理

- **程序对应**：`ConvFuser.forward` 中门控后存在 `cat_bev = cat_bev * spatial_keep_mask`；当某位置被路由到 null expert（或 cam/lidar 全被抑制）时，该位置特征被硬置零。  
- **数学机制**：设融合特征为 `h_t`，硬掩码为 `m_t∈{0,1}`，则
  \[
  \tilde h_t = m_t\cdot h_t
  \]
  若 `m_t=0`，该 token 信息在该层被完全丢弃；在检测任务中，这会直接影响召回（尤其是稀疏小目标）与后续定位误差项，从而同时拉低 mAP 与 NDS。  
- **高质量论文依据（2020+）**：
  - **DynamicViT**（NeurIPS 2021）：token 稀疏/裁剪会影响信息保留，需谨慎设计策略与训练约束。https://arxiv.org/abs/2106.02034
  - **EViT**（ICLR 2022）：讨论 token reorganizing/pruning 的精度-效率权衡。https://arxiv.org/abs/2202.07800
  - **nuScenes Detection Eval**（官方）说明 NDS 同时耦合 mAP 与多 TP error，解释为何 token 级信息丢失会被 NDS 放大。https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md

### 8.2 你的分析是否有漏缺？

有，至少还应补 3 个“从本地仓库可直接观察到”的关键点：

1. **训练轮次与收敛阶段不匹配风险**  
   - 本地配置 `NUM_EPOCHS: 10`，日志末段 `matched_ious≈0.51` 且 `loss_bbox≈1.2`，显示尚未进入更高质量收敛区间。即使结构完全正确，训练不足也会显著影响最终 mAP/NDS。  
   - 相关优化理论可参考 OneCycle 训练动态与超参数耦合分析。https://arxiv.org/abs/1803.09820

2. **门控是“硬路由 + top1 + null expert”组合，天然更激进**  
   - 本地 `SPARSE_GATE_TOPK` 默认 1，且 `include_null_expert=True`（代码默认），这比软融合更容易产生大面积信息截断。  
   - MoE 文献普遍强调 hard routing 需要额外稳定化（aux loss、capacity、z-loss 等）。参考 ST-MoE。https://arxiv.org/abs/2202.08906

3. **与 3D 检测融合范式的偏离**  
   - 主流多模态 3D 检测（如 BEVFusion、TransFusion）通常强调“信息增强式融合”而非大面积 token 丢弃；你当前策略更偏“稀疏计算优先”，若无充分正则与蒸馏/约束，容易牺牲检测上限。  
   - 参考：
     - **BEVFusion**（ICRA 2023 扩展；CVPR 2022 路线）https://arxiv.org/abs/2205.13542
     - **TransFusion**（CVPR 2022）https://arxiv.org/abs/2203.11496

### 8.3 归纳

- 你给出的两点是**主因级别**，且“程序—数学—论文”链条完整。  
- 你的分析还可补上“训练轮次/收敛程度”与“hard routing 组合激进度”这两类因素，二者与前两点共同作用，能够更完整解释本地结果与官方结果的差距。

## 9) 你这次问题的直接答案（仅针对 2020+ 顶会/顶刊方法）

### 9.1 对“MoE 缺乏损失函数约束”的回答：有，且应优先加 **负载均衡 + 路由稳定** 辅助损失

结合本地实现（`router -> softmax -> topk hard gate`，且无显式 aux loss），最匹配的辅助损失有三类：

1. **Load-balancing auxiliary loss（首选）**
   - 目标：让 expert 的“被选频率”与“平均路由概率”更均衡，避免少数 expert 过载。
   - 常见形式（Switch/GShard 系）：
     \[
     L_{lb}=E\sum_{i=1}^{E} f_i P_i
     \]
     其中 `f_i` 为 expert `i` 的路由频次占比，`P_i` 为其平均路由概率。
   - 论文依据：
     - GShard (ICLR 2021): https://arxiv.org/abs/2006.16668
     - Switch Transformers (JMLR 2022): https://arxiv.org/abs/2101.03961

2. **Router z-loss / logit regularization（强烈建议与 load-balance 联用）**
   - 目标：抑制 router logits 过大导致的极端 one-hot 化与训练不稳定。
   - 一般思想：对 router logits 的幅值做惩罚（如平方项），降低数值爆炸与过饱和。
   - 论文依据：
     - ST-MoE (NeurIPS 2022): https://arxiv.org/abs/2202.08906

3. **Capacity-aware routing regularization（容量约束）**
   - 目标：限制单 expert 可接收 token 数，防止拥塞与丢 token。
   - 数学上相当于给每个 expert 加容量上限 `C_i`，并对溢出 token 做重分配/丢弃惩罚。
   - 论文依据：
     - GShard (ICLR 2021): https://arxiv.org/abs/2006.16668
     - V-MoE (ICLR 2022): https://arxiv.org/abs/2106.05974

> 对本地仓库的直接对应：`SparseMoESpatialGate` 当前只产出 `router_prob/router_gate/keep_ratio`，没有把上面三类约束显式加入总损失，因此你的判断是成立的。

### 9.2 对“ConFuser.forward 中 cat_bev 硬置零”的回答：有，建议把“硬裁剪”改为“可学习软保留”

本地代码中：
\[
\tilde h = m\odot h,\; m\in\{0,1\}
\]
这会在 `m=0` 时完全删除 token 信息。针对该问题，2020+ 顶会方法给出可执行优化路径：

1. **Soft mask / differentiable sparsification（替代硬 0/1 乘法）**
   - 思路：用连续门控 `m\in[0,1]`（sigmoid/Gumbel-sigmoid/softmax 温度）替代硬门控，并用预算正则控制平均保留率。
   - 常见目标：
     \[
     L = L_{det} + \lambda\,\|\mathbb{E}[m]-\rho\|_2^2
     \]
     其中 `\rho` 是目标保留率。
   - 论文依据：
     - DynamicViT (NeurIPS 2021): https://arxiv.org/abs/2106.02034
     - A-ViT (ICLR 2022): https://arxiv.org/abs/2112.07658

2. **Token reorganization/merging（不直接丢弃，先聚合再降算）**
   - 思路：把低重要性 token 合并到高重要性 token（而非直接清零），减少信息损失。
   - 数学上是学习映射 `T: \mathbb{R}^{N\times d}\to\mathbb{R}^{N'\times d}`，使 `N' < N` 同时最小化任务损失退化。
   - 论文依据：
     - EViT (ICLR 2022): https://arxiv.org/abs/2202.07800
     - ToMe (ICLR 2023): https://arxiv.org/abs/2210.09461

3. **Teacher-student distillation for sparse tokens（稀疏分支蒸馏）**
   - 思路：用 dense teacher 约束 sparse student 的中间特征或 logits，补偿稀疏带来的表达丢失。
   - 常见形式：
     \[
     L = L_{det}^{student}+\alpha\,\|F_s-F_t\|_2^2+\beta\,KL(p_s\|p_t)
     \]
   - 论文依据：
     - DynamicViT (NeurIPS 2021, 含 distillation setting): https://arxiv.org/abs/2106.02034

### 9.3 结合你当前仓库，最小改动优先级（从高到低）

1. 给 `SparseMoESpatialGate` 增加 `L_lb + L_z`（至少先加 `L_lb`）。
2. 将 `cat_bev * spatial_keep_mask` 替换为连续软门控（训练期），推理期再阈值化。
3. 若仍需激进稀疏，优先做 token merge（EViT/ToMe 路线）而非直接置零。

以上三项都能直接对应你指出的两个问题，并且都有 2020 年以来顶会/顶刊文献支撑。

## 10) 对你给定“联合优化方案”的直接评估

你提出的联合方案是：

- MoE 约束：`Load-balancing auxiliary loss + Router z-loss/logit regularization`；
- 硬置零优化：`Soft mask/differentiable sparsification + Token reorganization/merging + Teacher-student distillation`。

### 10.1 问题1：该联合方案是否有数学与论文依据？

结论：**有，且每一项都有可复现的数学目标与高质量文献支撑；联用在方法论上是自洽的。**

1. **`L_lb + L_z` 联用（MoE）有充分依据**
   - `L_lb` 控制 expert 使用均衡，防止路由塌缩：
     \[
     L_{lb}=E\sum_{i=1}^{E} f_iP_i
     \]
     其中 `f_i` 为 expert 选择频率，`P_i` 为平均路由概率。
   - `L_z`（或 logit 正则）抑制 router logits 过大，缓解过饱和与训练不稳定。
   - 论文依据：
     - GShard (ICLR 2021): https://arxiv.org/abs/2006.16668
     - Switch Transformers (JMLR 2022): https://arxiv.org/abs/2101.03961
     - ST-MoE (NeurIPS 2022): https://arxiv.org/abs/2202.08906

2. **`Soft mask` 替代硬 0/1 乘法有依据**
   - 把离散门控改为连续门控 `m\in[0,1]`，并配预算约束：
     \[
     \tilde h = m\odot h,\quad
     L = L_{det}+\lambda\|\mathbb{E}[m]-\rho\|_2^2
     \]
   - 能在保留可微训练信号的同时控制稀疏率。
   - 论文依据：
     - DynamicViT (NeurIPS 2021): https://arxiv.org/abs/2106.02034
     - A-ViT (ICLR 2022): https://arxiv.org/abs/2112.07658

3. **`Token merging/reorganization` 有依据**
   - 思路：低重要 token 不直接丢弃，而是合并到高重要 token，降低信息损失。
   - 数学上学习压缩映射 `T: \mathbb{R}^{N\times d}\to\mathbb{R}^{N'\times d}`，优化 `L_{task}` 同时约束 `N' < N`。
   - 论文依据：
     - EViT (ICLR 2022): https://arxiv.org/abs/2202.07800
     - ToMe (ICLR 2023): https://arxiv.org/abs/2210.09461

4. **`Sparse distillation` 有依据**
   - 用 dense teacher 约束 sparse student，补偿稀疏化信息损失：
     \[
     L=L_{det}^{s}+\alpha\|F_s-F_t\|_2^2+\beta KL(p_s\|p_t)
     \]
   - 论文依据：
     - DynamicViT (NeurIPS 2021, distillation setting): https://arxiv.org/abs/2106.02034

> 小结：你提出的“MoE 约束 + 稀疏信息保留”联合方案在理论与文献上是完整闭环，可作为本地仓库的优先改造路线。

### 10.2 问题2：采用该方案后，“硬路由 + top1 + null expert”还需优化吗？

结论：**仍需优化**。即使加入上述联合方案，这个门控组合仍存在结构性风险，建议至少再做以下 4 点。

#### (A) top-1 改为可控 top-k（训练期）

- **程序落点**：`SparseMoESpatialGate.__init__` 中 `topk` 与 `_topk_hard_gate`。当前默认 `topk=1`。  
- **原因**：top-1 离散度太高，路由方差大；训练期适度 `k>1` 可增加梯度覆盖与稳定性。
- **数学**：
  \[
  g = \text{TopKMask}(p, k),\;\tilde p = p\odot g
  \]
  增大 `k` 可减少单 expert 决策抖动的影响。
- **论文依据**：
  - GShard (ICLR 2021): https://arxiv.org/abs/2006.16668
  - V-MoE (ICLR 2022): https://arxiv.org/abs/2106.05974

#### (B) null expert 采用“延迟启用/占比约束”

- **程序落点**：`include_null_expert` 与 `keep_mask=((gate_cam+gate_lidar)>0)` 逻辑。  
- **原因**：null expert 直接对应 token 丢弃，早期训练易过度选择 null。
- **数学**：给 null 路由概率增加预算约束：
  \[
  L_{null}=\gamma\,\max(0,\mathbb{E}[p_{null}]-\rho_{null})
  \]
  或前若干 epoch 禁用 null（课程学习）。
- **论文依据**：
  - ST-MoE (NeurIPS 2022, router 稳定化): https://arxiv.org/abs/2202.08906
  - DynamicViT (NeurIPS 2021, 渐进稀疏思想): https://arxiv.org/abs/2106.02034

#### (C) 加入容量因子（capacity factor）与溢出处理

- **程序落点**：当前 `SparseMoESpatialGate` 未见 per-expert capacity 限制。  
- **原因**：无容量控制时会出现 expert 拥塞与不均衡，反向加剧 top1 + null 的不稳定。
- **数学**：设置 `C_i = \lceil cf\cdot N/E \rceil`，对超容量 token 重路由/延迟处理。
- **论文依据**：
  - GShard (ICLR 2021): https://arxiv.org/abs/2006.16668
  - Switch Transformers (JMLR 2022): https://arxiv.org/abs/2101.03961

#### (D) 训练-推理稀疏策略解耦（train soft, infer hard）

- **程序落点**：`cat_bev = cat_bev * spatial_keep_mask` 可在训练期替换为 soft mask，推理期阈值化。  
- **原因**：训练期保留连续梯度，推理期再做硬化可兼顾稳定性与效率。
- **数学**：
  \[
  m_{train}=\sigma((a-\tau)/T),\quad m_{infer}=\mathbf{1}[a>\tau]
  \]
- **论文依据**：
  - A-ViT (ICLR 2022): https://arxiv.org/abs/2112.07658
  - EViT (ICLR 2022): https://arxiv.org/abs/2202.07800

### 10.3 对本地代码的最小实现顺序（建议）

1. 在 `SparseMoESpatialGate` 对接 `L_lb + L_z (+ capacity)`；
2. 训练期把 `spatial_keep_mask` 改 soft（保留预算损失），推理期再 hard；
3. 将 `topk` 训练期设为 2（或可调），并限制 `p_null` 占比；
4. 在融合输出上加 teacher-student 蒸馏项，最后再评估是否仍需 token merge。

以上顺序能最大化利用你已提出的联合方案，同时专门补齐“硬路由 + top1 + null expert”残余风险。

## 11) 将 10.3 落地到本地代码的“最小实现顺序”——具体改法（含数学与论文依据）

下面给出可直接映射到你当前仓库文件的最小改动路线，按“先稳定训练，再逐步恢复稀疏效率”的顺序执行。

### Step 1：在 `SparseMoESpatialGate` 增加 `L_lb + L_z (+ capacity)`（先做）

**代码落点**
- `pcdet/models/backbones_2d/fuser/gated_fusion.py`
  - `SparseMoESpatialGate.forward` 已返回 `router_prob`、`router_gate`、`keep_ratio`，可继续返回 `aux_loss_dict`。
- `pcdet/models/backbones_2d/fuser/convfuser.py`
  - `forward` 中把 `aux_loss_dict` 写入 `batch_dict`（如 `batch_dict['loss_moe_lb']` 等）。
- `pcdet/models/detectors/mambafusion.py`（或 loss 汇总位置）
  - 在总损失中加入：
  \[
  L = L_{det} + \lambda_{lb}L_{lb} + \lambda_{z}L_{z} + \lambda_{cap}L_{cap}
  \]

**数学定义（可直接实现）**
1. 负载均衡（Switch/GShard 系）：
   \[
   L_{lb}=E\sum_{i=1}^{E} f_iP_i,
   \quad
   f_i=\frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\arg\max p_t=i],
   \quad
   P_i=\frac{1}{N}\sum_{t=1}^{N}p_{t,i}
   \]
2. Router z-loss（ST-MoE）：
   \[
   L_{z}=\frac{1}{N}\sum_{t=1}^{N}\left(\log\sum_{i=1}^{E}e^{z_{t,i}}\right)^2
   \]
3. Capacity 约束（可选但推荐）：
   \[
   C_i=\left\lceil cf\cdot\frac{N}{E}\right\rceil,
   \quad
   L_{cap}=\frac{1}{N}\sum_{i=1}^{E}\max(0,n_i-C_i)
   \]
   其中 `n_i` 为分配到 expert `i` 的 token 数。

**论文依据**
- GShard, ICLR 2021: https://arxiv.org/abs/2006.16668  
- Switch Transformers, JMLR 2022: https://arxiv.org/abs/2101.03961  
- ST-MoE, NeurIPS 2022: https://arxiv.org/abs/2202.08906

---

### Step 2：把训练期 `cat_bev * spatial_keep_mask` 改为 soft mask（推理期再 hard）

**代码落点**
- `pcdet/models/backbones_2d/fuser/convfuser.py` 第 449/454 行附近：
  - 当前：`cat_bev = cat_bev * spatial_keep_mask`
  - 建议：训练期使用连续门控 `m_train`；推理期保留阈值化门控。

**最小实现形式**
\[
m_{train}=\sigma\left(\frac{a-\tau}{T}\right),\quad
m_{infer}=\mathbf{1}[a>\tau],\quad
\tilde h=m\odot h
\]
并加入预算项（控制平均保留率）：
\[
L_{budget}=\|\mathbb{E}[m_{train}]-\rho\|_2^2
\]
总损失可扩展为：
\[
L \leftarrow L + \lambda_bL_{budget}
\]

**论文依据**
- A-ViT, ICLR 2022: https://arxiv.org/abs/2112.07658  
- DynamicViT, NeurIPS 2021: https://arxiv.org/abs/2106.02034

---

### Step 3：训练期 `topk` 从 1 调到 2，并限制 `null expert` 占比

**代码落点**
- `pcdet/models/backbones_2d/fuser/gated_fusion.py`
  - `SparseMoESpatialGate.__init__`：`topk` 改为可调调度（例如前半程 2，后半程 1）。
  - `include_null_expert=True` 时对 `p_null` 添加预算约束。

**数学形式**
1. 训练期 top-k 路由：
   \[
   g_t=\text{TopKMask}(p_t,k),\quad \tilde p_t=p_t\odot g_t
   \]
2. null 占比约束：
   \[
   L_{null}=\gamma\max\left(0,\mathbb{E}[p_{null}]-\rho_{null}\right)
   \]
3. 可做课程式调度：`k:2\to1`、`\rho_{null}:0\to目标值`。

**论文依据**
- V-MoE, ICLR 2022: https://arxiv.org/abs/2106.05974  
- ST-MoE, NeurIPS 2022: https://arxiv.org/abs/2202.08906  
- DynamicViT, NeurIPS 2021（渐进稀疏思想）: https://arxiv.org/abs/2106.02034

---

### Step 4：加 teacher-student distillation，再决定是否引入 token merge

**代码落点**
- `pcdet/models/detectors/mambafusion.py` 或训练脚本 loss 汇总位置：
  - 增加 teacher 前向（冻结参数），对齐 BEV 融合特征/检测 logits。

**数学形式**
\[
L_{distill}=\alpha\|F_s-F_t\|_2^2+\beta\,KL(p_s\|p_t)
\]
\[
L = L_{det}+\lambda_{lb}L_{lb}+\lambda_{z}L_z+\lambda_bL_{budget}+\lambda_nL_{null}+\lambda_dL_{distill}
\]

若蒸馏后仍有明显掉点，再引入 token merge（EViT/ToMe）：
\[
T: \mathbb{R}^{N\times d}\rightarrow\mathbb{R}^{N'\times d},\;N'<N
\]
并最小化 `L_{task}` 退化。

**论文依据**
- DynamicViT, NeurIPS 2021（含 distillation）: https://arxiv.org/abs/2106.02034  
- EViT, ICLR 2022: https://arxiv.org/abs/2202.07800  
- ToMe, ICLR 2023: https://arxiv.org/abs/2210.09461

---

### 11.1 一句话执行清单（按风险最小化）

1. 先加 `L_lb + L_z (+capacity)`，不改推理图；
2. 再把训练期 hard mask 改 soft mask + `L_budget`；
3. 再做 `topk` 与 `null` 约束调度；
4. 最后加蒸馏，必要时再上 token merge。

该顺序的核心是：先修复路由统计稳定性，再降低硬裁剪信息损失，最后再追求更激进稀疏效率。
