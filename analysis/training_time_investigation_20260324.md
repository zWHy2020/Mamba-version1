# Training time investigation (2026-03-24)

## Key observations from `train_20260323-111212.log`

- Effective batch settings during that run:
  - `total_batch_size: 4`
  - `batch_size: 1` (per GPU)
  - `cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 1`
  - `cfg.OPTIMIZATION.ACCUMULATION_STEPS: 6`
- Dataset expansion:
  - raw train samples: `28130`
  - after balanced resampling: `123580`
- Throughput:
  - at iter 1000, average batch time is `6.78 s`
  - projected total runtime shown by logger is around `580 h`

The logger ETA is computed as:

`remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)`

This is implemented in `tools/train_utils/train_utils.py`.

## Comparison against upstream

Compared local repo to upstream clone (`https://github.com/AutoLab-SAI-SJTU/MambaFusion.git`):

1. Local config `tools/cfgs/mambafusion_models/mamba_fusion.yaml` includes an extended gated fusion / sparse MoE / distillation setup under `MODEL.FUSER`.
2. Local `pcdet/models/backbones_2d/fuser/convfuser.py` adds:
   - sparse spatial gating,
   - soft mask logic,
   - multiple auxiliary MoE losses,
   - optional sparse distillation path that computes an extra teacher forward (`teacher_cat_bev = self.mamba_forward(...)`).

These additions are absent in upstream `convfuser.py`, and can increase per-iteration compute significantly.

## Mathematical sanity check

Using the observed values:

- `iters_per_epoch = 30895`
- `epochs = 10`
- `avg_batch_time ~= 6.78 s`

Total time estimate:

`T ~= 30895 * 10 * 6.78 / 3600 = 581.86 h`

So a 500+ hour estimate is consistent with observed throughput and iteration count.

## Related references

- MambaFusion repo: https://github.com/AutoLab-SAI-SJTU/MambaFusion
- Mamba paper: https://arxiv.org/abs/2312.00752
- Switch Transformer (load balancing MoE objective): https://arxiv.org/abs/2101.03961
- ST-MoE (router z-loss): https://arxiv.org/abs/2202.08906
- Amdahl’s Law (parallel speedup bound): https://dl.acm.org/doi/10.1145/1465482.1465560

## Q&A: upstream是否也开启 `BALANCED_RESAMPLING`

是。当前本地仓库的 `tools/cfgs/dataset_configs/nuscenes_dataset.yaml` 中该项为 `True`，并且与上游仓库同路径文件逐行对比没有差异（`diff -u` 无输出），说明上游基础模型配置里该项也是开启状态。

这意味着：
- 仅凭 `BALANCED_RESAMPLING=True` 这一项，不能解释“本地比上游慢很多”；
- 更可能的差异来自本地新增的 FUSER 稀疏门控/蒸馏等计算路径。

## 10天时长约束（<=240小时）的硬性预算

你给出的可接受上限是 10 天，即 `240 h = 864000 s`。

### 基于当前日志的硬约束

- 当前规模：`total_iters = 30895 * 10 = 308950`
- 若保持这个迭代总数不变，则必须满足：

`sec_per_iter <= 864000 / 308950 = 2.80 s`

而日志中的稳定区间约为 `6.7 s/iter`，明显超预算。

### 达到10天的可执行路径（不依赖假设）

1. **先把全局 batch 恢复到 12（4卡 x 3）**：
   - 目标：`BATCH_SIZE_PER_GPU=3`（4卡时 global=12）
   - 这会把每个 epoch 的 iter 从 `30895` 降到约 `10298`。
   - 在 `~6.7 s/iter` 下，总时长约 `10298*10*6.7/3600 = 191.6 h`（<240h）。

2. **如果仍超时，再关闭本地新增高开销路径**：
   - `MODEL.FUSER.USE_GATED_FUSION: False`
   - `MODEL.FUSER.USE_SPARSE_DISTILL: False`
   - 保持其余设置不变做 A/B 对照，直接比较 `b_time`。

3. **若你把“时长优先级”放在第一位，可关闭重采样**：
   - `DATA_CONFIG.BALANCED_RESAMPLING: False`
   - 你的日志里样本数会从 `123580` 回到 `28130` 量级，iter 数近似按比例下降。
   - 该选项会改变训练分布，应单独评估精度影响。

### 建议的最小改动启动命令（先测吞吐）

```bash
bash tools/scripts/dist_train.sh 4 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --sync_bn \
  --pretrained_model ckpts/pretrained.pth \
  --logger_iter_interval 200 \
  --extra_tag train-10day-check \
  --set OPTIMIZATION.BATCH_SIZE_PER_GPU 3 OPTIMIZATION.ACCUMULATION_STEPS 1
```

先跑到 1000 iter，读取日志里的 `b_time(avg)`；若 `b_time(avg) <= 8.39s`，在 10 epoch / 4卡 / global batch=12 条件下可满足 10 天预算。

## 4x4090 且 `BATCH_SIZE_PER_GPU=3` OOM 的约束下（你当前真实约束）

你补充的信息是：
- 服务器是 4 张 4090；
- 按链接基础模型设置（`BATCH_SIZE_PER_GPU=3`, `ACCUMULATION_STEPS=2`）会 OOM；
- 因此你当前改成了 `1` 和 `6`；
- 当前日志里 `use_amp=False`。

在该约束下，要满足 10 天预算，需要改的是“总迭代数”或“单迭代耗时”，而不仅是梯度累积。

### 预算公式（固定4卡）

`total_hours = (iters_per_epoch * epochs * sec_per_iter) / 3600`

下面给出几组可直接判定是否满足 240h 的组合（用你日志的 `sec_per_iter≈6.7s` 先做粗估）：

1. `batch_per_gpu=1`, `BALANCED_RESAMPLING=True`:
   - `iters/epoch≈30895`，10 epoch 约 `576h`（不满足）

2. `batch_per_gpu=1`, `BALANCED_RESAMPLING=False`:
   - `iters/epoch≈7033`，10 epoch 约 `131h`（满足）

3. `batch_per_gpu=2`, `BALANCED_RESAMPLING=True`:
   - global batch=8，`iters/epoch≈15448`
   - 10天预算下允许的最大 `sec_per_iter` 是 `5.59s`
   - 需要通过 AMP/关闭高开销模块把 `b_time(avg)` 压到该阈值以下

### 针对 4090 OOM 的执行优先级（按风险从低到高）

1. **先开 AMP 再测是否能上 `batch_per_gpu=2`**（当前日志是 `use_amp=False`）
   - 命令增加 `--use_amp`
   - 若不 OOM，先跑 1000 iter，看 `b_time(avg)` 是否 <= `5.59s`

2. **若仍 OOM/超时，先关本地新增开销**
   - `MODEL.FUSER.USE_GATED_FUSION=False`
   - `MODEL.FUSER.USE_SPARSE_DISTILL=False`
   - 目的：降低 `sec_per_iter`

3. **若时长是硬指标（必须 <=10天），直接关重采样**
   - `DATA_CONFIG.BALANCED_RESAMPLING=False`
   - 在 `batch_per_gpu=1` 下也可把估算降到 10 天以内（代价是类分布变化，需要看精度）

### 建议先跑的两条命令（A/B）

A. 保持重采样，尝试 AMP + batch=2（优先保留数据分布）：

```bash
bash tools/scripts/dist_train.sh 4 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --sync_bn \
  --pretrained_model ckpts/pretrained.pth \
  --logger_iter_interval 200 \
  --extra_tag train-4090-amp-b2 \
  --use_amp \
  --set OPTIMIZATION.BATCH_SIZE_PER_GPU 2 OPTIMIZATION.ACCUMULATION_STEPS 3
```

B. 时长优先，直接关重采样（最容易满足 10 天）：

```bash
bash tools/scripts/dist_train.sh 4 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --sync_bn \
  --pretrained_model ckpts/pretrained.pth \
  --logger_iter_interval 200 \
  --extra_tag train-4090-noresample \
  --set OPTIMIZATION.BATCH_SIZE_PER_GPU 1 OPTIMIZATION.ACCUMULATION_STEPS 6 DATA_CONFIG.BALANCED_RESAMPLING False
```

## 为什么说开启 AMP 可能降低训练时间（而不是只省显存）

结论先说：AMP 不是“必然提速”，但在支持 Tensor Core 的 GPU 上，常见卷积/矩阵乘会因为使用 FP16/BF16 路径而提升吞吐，因此经常同时带来**更低显存占用**和**更短 step 时间**。

### 机制层面的可验证原因

1. **算子吞吐更高**
   - AMP 通过 `autocast` 让适合的算子以低精度执行（典型是 GEMM/Conv）。
   - 这些算子在 Tensor Core 路径上通常能获得更高 FLOPS。

2. **显存带宽与访存压力下降**
   - 半精度激活/中间张量占用更少字节数，读写压力下降，很多模型会因此减少内存瓶颈。

3. **能在同卡显存下提升 batch 的可行性**
   - 你的核心痛点是 OOM；AMP 常见的第一收益是“让 batch 从 1 提到 2 可行”。
   - 在固定 epoch 下，batch 变大可直接减少 `iters_per_epoch`，从而缩短总时间。

### 为什么我在你的场景里把 AMP 放第一优先级

- 你的日志明确是 `use_amp=False`；
- 你又受到 4x4090 下 OOM 限制；
- 因此 AMP 是“先尝试、低侵入、可立即验证”的第一步。

### 但必须强调的边界（避免误解）

- AMP **不保证**总是更快：某些操作仍在 FP32，或数据加载/通信成为瓶颈时，收益会变小；
- 所以建议按 1000 iter 的 `b_time(avg)` 做 A/B 实测，而不是只看理论。

### 参考（官方/原始）

- PyTorch AMP examples: https://pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch Automatic Mixed Precision recipe: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- NVIDIA Tensor Core mixed precision overview: https://developer.nvidia.com/tensor-cores
- Mixed Precision Training（经典论文）: https://arxiv.org/abs/1710.03740

## 本地 vs 链接模型：对“训练时长”影响最大的因素（基于可证据项）

### 结论

在你给出的运行条件下，**影响训练总时长最大的可证据因素是“有效全局 batch 从 12 降到 4”带来的迭代数增加（精确 3 倍）**。

### 数学依据（确定性）

固定 `samples_per_epoch` 与 `epochs` 时：

`total_iters = ceil(samples_per_epoch / global_batch) * epochs`

- 你日志中的有效全局 batch 是 `4`（`batch_size=1` 且 4 卡），并记录了 `iters_per_epoch=30895`。
- 若用链接模型常见设置 `global_batch=12`（4 卡 × 3），则
  `iters_per_epoch ≈ ceil(123580/12) = 10299`。
- 因此仅迭代数比值就是：

`30895 / 10299 ≈ 3.00`

即在每 iter 耗时同量级时，总时长先天就是约 3 倍差距。

### 代码差异项为何排在其后（同样给出可证据结论）

本地相对上游在 FUSER 增加了门控/辅助损失/蒸馏相关逻辑，这些都会增加每 iter 计算量；但“增幅具体多少”需要 profiler 才能定量，不能像 batch 引起的迭代数变化那样给出确定 3x。故按“可严格量化”的标准，batch 导致的迭代数放大是最大项。

### 与论文的一致性（不做超出证据推断）

- MoE/路由会引入额外路由网络与辅助目标计算（见 Switch Transformer, ST-MoE）；这支持“本地新增 FUSER 逻辑会增算力开销”的方向性判断，但不直接给出你项目中的精确倍数。
- 训练总时长分解为 `iters × sec_per_iter` 的做法是标准工程计时分解，与任何特定模型无关。

## 2023年以来加速相关顶会论文清单（面向本仓库可对接模块）

> 说明：以下均给出可访问的真实论文/会议信息链接；优先选取 CVPR/ICCV/ICLR/ICML/NeurIPS 主会论文。

### A. 注意力/Token 计算加速（可映射到图像分支与跨模态分支）

1. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, ICLR 2024.  
   https://iclr.cc/virtual/2024/poster/17889
2. A General and Efficient Training for Transformer via Token Expansion, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Huang_A_General_and_Efficient_Training_for_Transformer_via_Token_Expansion_CVPR_2024_paper.html
3. Zero-TPrune: Zero-Shot Token Pruning through Leveraging of the Attention Graph, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-TPrune_Zero-Shot_Token_Pruning_through_Leveraging_of_the_Attention_Graph_CVPR_2024_paper.html
4. Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Multi-criteria_Token_Fusion_with_One-step-ahead_Attention_for_Efficient_Vision_Transformers_CVPR_2024_paper.html
5. MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Cao_MADTP_Multimodal_Alignment-Guided_Dynamic_Token_Pruning_for_Accelerating_Vision-Language_Transformer_CVPR_2024_paper.html
6. ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation with Plain Vision Transformers, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Norouzi_ALGM_Adaptive_Local-then-Global_Token_Merging_for_Efficient_Semantic_Segmentation_with_CVPR_2024_paper.html

### B. 高吞吐视觉骨干/模块设计（可映射到 MM_BACKBONE/FUSER 前后）

7. EfficientViT: Memory Efficient Vision Transformer With Cascaded Group Attention, CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html
8. EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction, ICCV 2023.  
   https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html
9. FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization, ICCV 2023.  
   https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html
10. SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications, ICCV 2023.  
    https://openaccess.thecvf.com/content/ICCV2023/html/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.html
11. Rethinking Vision Transformers for MobileNet Size and Speed, ICCV 2023.  
    https://openaccess.thecvf.com/content/ICCV2023/html/Li_Rethinking_Vision_Transformers_for_MobileNet_Size_and_Speed_ICCV_2023_paper.html
12. FasterViT: Fast Vision Transformers with Hierarchical Attention, ICLR 2024.  
    https://proceedings.iclr.cc/paper_files/paper/2024/file/7e49642375e3315467fa33120813547f-Paper-Conference.pdf
13. VMamba: Visual State Space Model, NeurIPS 2024.  
    https://proceedings.neurips.cc/paper_files/paper/2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf
14. MambaVision: A Hybrid Mamba-Transformer Vision Backbone, CVPR 2025.  
    https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html

### C. 训练内存/优化器效率（直接对应 4090 OOM 约束）

15. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, ICML 2024.  
    https://proceedings.mlr.press/v235/zhao24s.html

## 结合本文训练报告的“可落地”降时方案（按优先级）

### 数学约束先行

- 当前日志：`global_batch=4`, `iters_per_epoch=30895`, `b_time(avg)=6.78s`。
- 总时长公式：`T_hours = iters_per_epoch * epochs * sec_per_iter / 3600`。
- 10天预算（240h）下，需要满足：`iters_per_epoch * sec_per_iter <= 86400`。

### 方案 P0（不改模型结构，先拿到可运行收益）

1. 开启 AMP（现状是 `use_amp=False`），先测 1000 iter 的 `b_time(avg)`。
2. 若显存允许，把 `batch_per_gpu` 从 1 升到 2（global 8）。
3. 预算判定线：`global 8` 时 `iters_per_epoch≈15448`，需 `sec_per_iter<=5.59s` 才能进 10 天。

### 方案 P1（针对本地分支新增开销，做严格 A/B）

1. 关闭 `MODEL.FUSER.USE_GATED_FUSION`。
2. 关闭 `MODEL.FUSER.USE_SPARSE_DISTILL`。
3. 保持其余不变，比较 `b_time(avg)` 与 NDS/mAP。

### 方案 P2（硬性时长优先）

1. `DATA_CONFIG.BALANCED_RESAMPLING=False`。
2. 在 `batch_per_gpu=1` 下，`iters_per_epoch` 从约 `30895` 降到约 `7033`，按当前 `6.7s` 粗估约 `131h`。
3. 该方案会改变类别分布，需单独报告精度回归。

### 为什么以上方案与论文是一致的

- P0 对应混合精度/硬件友好算子路径（FlashAttention-2、AMP 文档）；
- P1 对应“减少路由/辅助损失/冗余分支”以降低每 iter 计算；
- P2 对应训练总时长分解 `T = iters × sec_per_iter` 的直接数学控制。

## 关于日志中反复出现“Save latest model”是否正常

是正常行为。原因是训练循环里按“时间间隔”保存中间 checkpoint：

- 参数 `ckpt_save_time_interval` 默认/当前是 `300` 秒（5 分钟）；
- 代码逻辑是当 `elapsed // ckpt_save_time_interval` 增长时就保存一次 `latest_model`；
- 因此在 2 小时左右会出现约 20+ 次“Save latest model”日志。

你的日志与该逻辑一致：在 `11:18` 到 `13:03` 之间基本每 5 分钟保存一次。

## 如果要开启 AMP，本地仓库应修改/使用的位置

优先推荐：**不改代码，直接在训练命令里加 `--use_amp`**。

原因：
- `tools/train.py` 已定义 `--use_amp` 参数；
- 并在解析后把该值传入训练主循环；
- 训练循环里已用 `autocast` 与 `GradScaler(enabled=use_amp)` 接入 AMP。

可选的“配置文件方式”：在 YAML 的 `OPTIMIZATION` 下增加 `USE_AMP: True`，因为 `train.py` 会读取 `cfg.OPTIMIZATION.USE_AMP` 作为后备开关。

最小命令示例：

```bash
bash tools/scripts/dist_train.sh 4 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --sync_bn \
  --pretrained_model ckpts/pretrained.pth \
  --use_amp \
  --extra_tag train-amp
```

## 关于 `BATCH_SIZE_PER_GPU:1->2` 与 `ACCUMULATION_STEPS:6->3` 是否对应

在“保持每次优化更新的有效样本数”这个目标下，这个对应关系是对的：

- 现在：`global_batch = 4*1 = 4`，`effective_batch_per_update = 4 * 6 = 24`
- 调整后：`global_batch = 4*2 = 8`，`effective_batch_per_update = 8 * 3 = 24`

但要注意本仓库实现细节：学习率调度在每个 iteration 都 `step`，而不是只在 `optimizer.step()` 时调度；因此 `6->3` 不保证与原配置“完全等价”，需要做短跑验证（loss 曲线 + mAP/NDS）。

## 新报错：`ValueError: matrix contains invalid numeric entries`（Hungarian matching）

该错误来自 `linear_sum_assignment(cost)`，其输入代价矩阵 `cost` 必须是有限数值（不能含 NaN/Inf）。

本地修复策略：
- 在 `HungarianAssigner3D.assign` 内对 `cost` 做 `torch.nan_to_num`；
- 转到 CPU/Numpy 后再做一次 `np.nan_to_num` 兜底；
- 再调用 `linear_sum_assignment`。

这能防止训练因非法数值直接中断，并保留匹配流程可执行。

## 为何会出现非有限值（NaN/Inf）

在当前检测头路径里，Hungarian 代价由三部分相加：

`cost = cls_cost + reg_cost + iou_cost`

对应实现中，非有限值常见来源有：
1. `dim.exp()`（解码尺寸）在半精度下可能溢出为 `Inf`；
2. 上游预测里若已有 NaN，会通过 `sigmoid/log/cdist` 继续传播；
3. 一旦任一项出现 NaN/Inf，`cost` 就会包含非法值，SciPy 的 `linear_sum_assignment` 会抛出 `ValueError`。

本地新增的修复分两层：
- 在 `TransFusionHead.get_targets_single` 中，将用于匹配的预测张量强制到 fp32，并对解码后的 `pred_boxes` 做 `nan_to_num`；
- 在 `HungarianAssigner3D.assign` 中再次对 `cost` 做 finite 兜底，确保匹配器输入合法。
