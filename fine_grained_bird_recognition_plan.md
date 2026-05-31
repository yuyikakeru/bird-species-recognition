# CUB_200_2011 细粒度鸟类识别完整方案

## 1. 项目目标

本项目面向 CUB_200_2011 鸟类细粒度识别任务，目标是在统一训练框架下完成两阶段实验：

1. 第一阶段使用 ResNet50 作为 baseline，跑通数据读取、训练、验证、日志、checkpoint 和多随机种子重复实验。
2. 第二阶段实现效果更强的细粒度模型，围绕多尺度特征、背景抑制、关键部件、区域关系和 CLIP 外部知识逐步增强，并通过消融实验验证每个模块的作用。

最终输出包括 Top-1/Top-5 accuracy、训练曲线、混淆矩阵、注意力或部件可视化、失败样本分析、多 seed 均值/标准差和完整消融表。

## 2. 总体路线

项目按模块分块执行，避免一次性堆叠复杂模型导致难以定位问题。

### Block A：数据与训练骨架

- 实现 CUB_200_2011 Dataset，读取 `images.txt`、`image_class_labels.txt`、`train_test_split.txt`。
- 可选读取 `bounding_boxes.txt` 和 `parts/part_locs.txt`，为后续前景和部件模块预留接口。
- 实现 train/val transforms。
- 实现统一配置、训练器、accuracy、checkpoint、随机种子和 smoke test。
- smoke test 验证 DataLoader 能输出 image、label、image_id、path、bbox、parts，并验证模型前向传播输出 `(batch_size, 200)`。

### Block B：ResNet50 Baseline

- 使用 ImageNet 预训练 ResNet50，替换最后分类层为 200 类。
- 默认训练配置：
  - 输入尺寸：`448x448`。
  - DataLoader：`num_workers=4`
  - 优化器：`SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)`。
  - Scheduler：`CosineAnnealingLR`。
  - Loss：`CrossEntropyLoss(label_smoothing=0.1)`。
  - AMP：默认开启，CUDA 可用时自动使用 GPU。
- 支持调试与正式训练两种模式：
  - 调试：使用 `--max-train-batches` 和 `--max-val-batches` 快速验证流程。
  - 正式：完整训练 50 epoch 或更长。
- 支持单 seed 和多 seed：
  - 单次实验：`--seed 42`。
  - 重复实验：`--seeds 42,2024,3407`。
- 支持 `--run-name` 区分实验目录，避免覆盖历史结果。
- 每个 seed 独立保存：
  - `log/<model>/<run_name>/seed_<seed>/<model>_history.csv`
  - `log/<model>/<run_name>/seed_<seed>/<model>_history.json`
  - `ckpt/<model>/<run_name>/seed_<seed>/<model>_best.pt`
- 多 seed 汇总保存：
  - `log/<model>/<run_name>/repeat_summary.csv`
  - `log/<model>/<run_name>/repeat_summary.json`
- 汇总指标包括每次运行的 train loss、val loss、val top1、val top5、best top1，以及 best top1 的 mean/std。

推荐正式命令：

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 16 --image-size 448 --resize-size 512 --epochs 50 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --seeds 42,2024,3407 --run-name resnet50_baseline_3seeds
```

推荐调试命令：

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 2 --image-size 224 --resize-size 256 --epochs 1 --num-workers 4 --seeds 1,2 --run-name debug_resnet --max-train-batches 1 --max-val-batches 1 --no-pretrained
```

### Block C：背景抑制基础实验

- 使用 CUB 官方 bbox 做前景裁剪训练。
- 比较 ResNet50 原图训练与 bbox crop 训练。
- 两组实验均采用相同 seed 列表和日志结构，便于公平对比。
- 判断背景区域对识别性能的影响。

### Block D：高性能 Backbone

- 第二阶段默认采用 Swin-T 或 ViT-B/16。
- 推荐默认 Swin-T，因为其层级结构天然适合多尺度特征融合。
- 先实现 backbone-only 版本，作为强 backbone baseline。
- 训练与日志接口沿用 Block B 的多 seed 框架。

### Block E：多尺度特征模块

- 从 backbone 多个 stage 提取特征。
- 使用 FPN 或 Pyramid Convolution 统一通道并融合。
- 捕获鸟类整体形态、局部纹理和细粒度颜色差异。

### Block F：前景与关键部件注意力

- 基于 attention/CAM 生成前景响应，降低背景影响。
- 选取 top-k 高响应区域作为候选部件。
- 可使用 CUB part locations 做训练辅助和可视化检查，默认推理阶段不依赖部件标注。

### Block G：区域关系建模

- 将关键区域作为节点，构建局部区域图。
- 使用轻量 GCN 或 Transformer relation head 建模不同部件间的共现关系。
- 可加入 compact bilinear pooling 或低秩 bilinear pooling，增强局部交互表达。

### Block H：CLIP 外部知识辅助

- 使用类别名称构造文本 prompt，例如 `a photo of a Black footed Albatross bird`。
- 使用 CLIP 文本特征作为类别语义原型。
- 将视觉分类 logits 与 CLIP 语义 logits 融合。
- 可加入视觉-文本对齐损失，帮助相似类别区分。

### Block I：完整实验与分析

- 训练完整模型。
- 汇总 baseline、强 backbone 和所有模块消融结果。
- 输出注意力可视化、混淆矩阵和错误案例分析。
- 对关键结果报告 mean/std，确保分析不依赖单次随机结果。

## 3. 两个模型设计

### 模型一：ResNet50 Baseline

结构：

- Backbone：ResNet50 ImageNet pretrained。
- Pooling：Global Average Pooling。
- Classifier：Linear(2048, 200)。
- Loss：CrossEntropyLoss(label_smoothing=0.1)。
- Optimizer：SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)，保留 AdamW 选项。
- Scheduler：CosineAnnealingLR。

实验：

- ResNet50 原图输入。
- ResNet50 + bbox crop。
- 输入尺寸 224 与 448 对比。
- 每个正式配置至少跑 3 个 seed，并报告平均值和标准差。

### 模型二：FGViTNet / Fine-Grained Swin Model

结构：

- Backbone：Swin-T 或 ViT-B/16 pretrained。
- Multi-scale：FPN/Pyramid Conv 融合多层特征。
- Foreground attention：抑制背景，强化鸟体区域。
- Part attention：提取头、翅膀、尾部、腹部等关键区域候选。
- Relation module：GNN/Transformer 建模部件间空间与语义关系。
- Co-feature / bilinear module：增强区域间细粒度交互。
- CLIP knowledge：使用类别文本语义辅助分类。
- Classifier：融合全局特征、多尺度特征、关系特征和文本语义特征。

总损失：

`L = L_ce + lambda_fg * L_fg + lambda_part * L_part + lambda_rel * L_rel + lambda_clip * L_clip`

默认先只启用 `L_ce`，后续模块逐步加入辅助损失。

## 4. 消融实验

| 编号 | 配置 | 验证目标 | 重复次数 |
| --- | --- | --- | --- |
| A0 | ResNet50 | 基础 CNN baseline | 3 seeds |
| A1 | ResNet50 + bbox crop | 验证背景抑制收益 | 3 seeds |
| B0 | Swin-T backbone-only | 强 backbone baseline | 3 seeds |
| B1 | B0 + multi-scale | 验证多尺度特征 | 至少 1 seed，关键结果 3 seeds |
| B2 | B1 + foreground attention | 验证背景抑制 | 至少 1 seed，关键结果 3 seeds |
| B3 | B2 + part attention | 验证关键部件 | 至少 1 seed，关键结果 3 seeds |
| B4 | B3 + relation/GNN | 验证区域关系 | 至少 1 seed，关键结果 3 seeds |
| B5 | B4 + bilinear/co-feature | 验证部件交互 | 至少 1 seed，关键结果 3 seeds |
| B6 | B5 + CLIP knowledge | 验证外部知识 | 3 seeds |
| B7 | 完整模型逐个去模块 | 验证每个模块独立贡献 | 视训练成本决定 |

## 5. 日志与结果分析

- 每个 epoch 记录 train loss、train top1、val loss、val top1、val top5。
- 每个 seed 保存独立 CSV/JSON 日志和 best checkpoint。
- 多 seed 实验保存 repeat summary，直接用于报告表格。
- 分析时优先使用 `best_top1_mean ± best_top1_std`，同时报告 top5。
- 对 baseline 和最终模型额外输出混淆矩阵、错误样本和可视化结果。

## 6. 评价指标

- Top-1 accuracy。
- Top-5 accuracy。
- Per-class accuracy。
- Confusion matrix。
- 参数量与训练时间。
- 注意力热力图是否覆盖鸟体和关键部件。
- 相似类别失败案例分析。
- 多 seed mean/std。

## 7. 默认实现假设

- 使用 PyTorch、torchvision、timm、open_clip。
- 数据路径为 `D:\pattern recognition47\bird-species-recognition\datasets\CUB_200_2011\CUB_200_2011`。
- 训练时可以使用 bbox 和 part locations；推理时默认只输入图像。
- 第二阶段优先追求稳妥复现，而不是极限刷榜。
- 日志和 checkpoint 默认写入 `log/` 与 `ckpt/`，这两个目录已被 `.gitignore` 忽略。
- 不进行批量删除；如需清理大量日志或 checkpoint，应由用户手动处理。
