# CUB_200_2011 细粒度鸟类识别实验计划

## 1. 目标

在统一数据划分、训练轮数、优化器设置、EMA/TTA 和三 seed 协议下，对比不同 backbone 和轻量增强模块在 CUB_200_2011 官方测试集上的表现。

最终报告以官方测试集结果为准，验证集只用于搜索阶段或调参判断。

## 2. 统一协议

1. 数据集使用 CUB_200_2011 官方划分。
2. 官方训练集 5994 张用于训练；官方测试集 5794 张只做最终评估。
3. 所有正式实验使用 `pipeline` 模式：先把官方训练集划分为 train/val 搜索超参，再合并为 `train_full` 做全量训练，最后在官方测试集评估。
4. 每个正式实验由一条命令完整完成搜索、全量训练和测试。
5. 默认三 seed：

```text
42,2024,3407
```

6. 默认训练 50 epoch。
7. 默认启用 EMA 和水平翻转 TTA。
8. 同一实验使用相同输入分辨率、学习率、batch size 和数据增强。

## 3. 当前正式实验

### 3.1 ResNet 对照线

| 实验 | 模型名 | 目的 |
| --- | --- | --- |
| ResNet50 | `resnet50_baseline` | CNN baseline |
| ResNet50 + bbox | `resnet50_baseline --use-bbox-crop` | 分析官方 bbox 裁剪收益 |

### 3.2 SwinV2 主线

| 实验 | 模型名 | 目的 |
| --- | --- | --- |
| SwinV2-Tiny | `swinv2_tiny` | Transformer baseline |
| SwinV2-Tiny + FPN | `swinv2_tiny_fpn` | 融合 stage3/stage4 特征 |
| SwinV2-Tiny + FPN + Parts | `swinv2_tiny_fpn_parts` | 在 FPN 特征上做 CAM-guided Top-K parts |

### 3.3 ConvNeXtV2 主线

| 实验 | 模型名 | 目的 |
| --- | --- | --- |
| ConvNeXtV2-Tiny | `convnextv2_tiny` | ConvNeXt baseline |
| ConvNeXtV2-Tiny + DCA | `convnextv2_tiny_dca` | 判别上下文注意力 residual 分支 |
| ConvNeXtV2-Tiny + DCA-Region | `convnextv2_tiny_dca_region` | DCA 引导的 4 个 soft region token 分支 |

## 4. 模型结构设计

### 4.1 ResNet50 对照线

ResNet50 作为传统 CNN baseline，用于提供基础参照。该模型不引入额外注意力、部件选择或区域建模，输入图像经过 ImageNet 预训练 ResNet50 后直接输出 200 类分类 logits。

ResNet50 + bbox 使用 CUB 官方 bounding box 做前景裁剪，目的是观察人工前景裁剪对 CNN baseline 的影响。该实验使用额外 bbox 标注，因此只作为对照，不与无额外标注模型做完全公平的主结果比较。

### 4.2 SwinV2-Tiny Baseline

`swinv2_tiny` 使用 ImageNet 预训练的 SwinV2-Tiny 作为 backbone。输入图像经过 SwinV2 的 patch embedding 和 4 个 stage，最后由原始分类头输出 200 类 logits。

该模型作为 Transformer baseline，不使用 CAM、bbox 或额外部件信息。

### 4.3 SwinV2-Tiny FPN

`swinv2_tiny_fpn` 在 SwinV2-Tiny 的 stage3 和 stage4 特征上增加轻量 FPN 分支。

结构流程：

```text
image
-> SwinV2 stage3 feature
-> SwinV2 stage4 feature
-> stage3/stage4 lateral projection
-> 上采样 stage4 到 stage3 分辨率
-> fused FPN feature map
-> attention pooling
-> fpn_logits
-> logits = baseline_logits + fpn_logits
```

stage4 语义强、分辨率低；stage3 保留更多局部纹理和形状。FPN 分支融合两层特征，用于补充羽毛边缘、身体局部形态和细粒度纹理信息。

### 4.4 SwinV2-Tiny FPN + Parts

`swinv2_tiny_fpn_parts` 在 FPN 特征上做 CAM-guided Top-K parts。

训练时输入：

```text
image
cam_mask
```

测试时输入：

```text
image
cam_mask
```

结构流程：

```text
image
-> SwinV2 stage3/stage4
-> FPN fused feature map
-> baseline_logits + fpn_logits
-> CAM mask resize 到 FPN 分辨率
-> part_score map
-> score_map * CAM prior
-> Top-K 选择 num_parts 个位置
-> part_features
-> learnable weighted sum
-> part_logits
-> logits = baseline_logits + fpn_logits + part_logits
```

其中 CAM mask 在训练、验证和测试阶段都使用，用于约束 part selection 更倾向鸟体区域。训练前若干 epoch 使用 soft attention pooling warmup，之后切换到 Top-K parts。

该分支显式聚合局部判别区域，例如头部、翅膀、胸部、背部或尾部附近的细粒度特征。

### 4.5 ConvNeXtV2-Tiny Baseline

`convnextv2_tiny` 使用 ImageNet 预训练 ConvNeXtV2-Tiny。该模型是 ConvNeXt 主线的基础版本，直接通过原始 ConvNeXtV2-Tiny 分类头输出 200 类 logits。

ConvNeXtV2 保留卷积结构对纹理、边缘和局部模式的归纳偏置，计算量也比更大的 backbone 更可控，因此作为第二条主线。

### 4.6 ConvNeXtV2-Tiny + DCA

`convnextv2_tiny_dca` 在 ConvNeXtV2-Tiny 的最后 stage 特征上增加 DCA residual 分支。

DCA 指 discriminative context attention。它不替代 baseline，而是作为额外 residual logits：

```text
image
-> ConvNeXtV2 stage4 feature
-> baseline global pooling
-> baseline_logits

stage4 feature
-> depthwise local context
-> channel gate
-> spatial score map
-> attention pooling
-> dca_feature
-> dca_logits

logits = baseline_logits + dca_logits
```

ConvNeXt baseline 的全局平均池化会混合判别区域和非判别区域。DCA 分支通过空间打分图聚合更有分类价值的上下文区域，并以 residual logits 补充 baseline。DCA 分类头使用零初始化，初始输出接近 baseline，训练中逐步学习增量信息。

### 4.7 ConvNeXtV2-Tiny + DCA-Region

`convnextv2_tiny_dca_region` 在 DCA 基础上增加 soft region token 分支，用于建模多个局部区域之间的互补信息。

结构流程：

```text
image
-> ConvNeXtV2 stage3 feature
-> ConvNeXtV2 stage4 feature
-> baseline_logits
-> dca_logits

stage3 feature + upsample(stage4 feature)
-> region feature map
-> region_score maps, K=4
-> DCA spatial score 作为轻量引导
-> softmax 得到 4 个 region attention
-> attention pooling 得到 4 个 region tokens
-> concat global DCA token
-> 1-layer lightweight Transformer token mixer
-> region_feature
-> region_logits

logits = baseline_logits + dca_logits + region_logits
```

该分支不使用外部 CAM 或 bbox。它利用 DCA 的空间响应作为弱引导，region token 端到端学习。相比全局二阶纹理统计，region 分支更直接对应 CUB 的局部判别信息：头部颜色、翅膀花纹、胸部颜色、背部纹理和尾部形态。

Region 分支包含以下正则项：

- region token dropout
- token mixer dropout
- region feature dropout
- DropPath
- region attention diversity loss

diversity loss 约束 4 个 region attention map 的相似度，减少多个 token 关注同一区域的情况。该损失只在训练时加入，不改变测试阶段输入和推理流程。

## 5. 关键消融问题

1. **Backbone 收益**
   比较 ResNet50、SwinV2-Tiny、ConvNeXtV2-Tiny 的 baseline。

2. **多层特征收益**
   比较 `swinv2_tiny` 与 `swinv2_tiny_fpn`，判断 stage3/stage4 融合是否有效。

3. **Parts 收益**
   比较 `swinv2_tiny_fpn` 与 `swinv2_tiny_fpn_parts`，判断 CAM-guided Top-K parts 是否提升细粒度识别。

4. **DCA 收益**
   比较 `convnextv2_tiny` 与 `convnextv2_tiny_dca`，判断判别上下文 residual 分支是否稳定提升。

5. **Region 收益**
   比较 `convnextv2_tiny_dca` 与 `convnextv2_tiny_dca_region`，判断 region token 是否比全局上下文更有泛化价值。

6. **输入分辨率收益**
   只对 `swinv2_tiny_fpn_parts` 和 `convnextv2_tiny_dca_region` 复跑 448 输入，比较最终增强模型在 224 与 448 下的均值收益和计算代价。

## 6. 推荐正式命令

所有命令都使用 `pipeline`。ResNet 使用 `search_space_resnet.json` 和 SGD；SwinV2、ConvNeXtV2 使用 `search_space_adamw.json` 和 AdamW。输出目录命名沿用当前项目已有风格：`resnet_final`、`resnet_bbox_final`、`swin_final`、`fpn_final`、`parts_final`，ConvNeXt 主线使用同样简洁的 `convnext_*_final` 命名。

### 6.1 ResNet50 baseline

```bash
python main.py --mode pipeline --model resnet50_baseline --search-space search_space_resnet.json --optimizer sgd --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name resnet_final
```

### 6.2 ResNet50 + bbox

```bash
python main.py --mode pipeline --model resnet50_baseline --use-bbox-crop --search-space search_space_resnet.json --optimizer sgd --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name resnet_bbox_final
```

### 6.3 SwinV2-Tiny baseline

```bash
python main.py --mode pipeline --model swinv2_tiny --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name swin_final
```

### 6.4 SwinV2-Tiny FPN

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name fpn_final
```

### 6.5 SwinV2-Tiny FPN + Parts

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn_parts --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --num-parts 6 --part-window-size 3 --cam-root datasets/CUB_200_2011/cam_masks --run-name parts_final
```

### 6.6 ConvNeXtV2-Tiny baseline

```bash
python main.py --mode pipeline --model convnextv2_tiny --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name convnext_final
```

### 6.7 ConvNeXtV2-Tiny + DCA

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name convnext_dca_final
```

### 6.8 ConvNeXtV2-Tiny + DCA-Region

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca_region --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name convnext_region_final
```

### 6.9 SwinV2-Tiny FPN + Parts 448

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn_parts --search-space search_space_adamw.json --optimizer adamw --epochs 50 --image-size 448 --resize-size 512 --batch-size 16 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --num-parts 6 --part-window-size 3 --cam-root datasets/CUB_200_2011/cam_masks --run-name parts_448_final
```

### 6.10 ConvNeXtV2-Tiny + DCA-Region 448

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca_region --search-space search_space_adamw.json --optimizer adamw --epochs 50 --image-size 448 --resize-size 512 --batch-size 16 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name convnext_region_448_final
```

448 输入只用于两条主线的最终增强模型：`swinv2_tiny_fpn_parts` 和 `convnextv2_tiny_dca_region`。

## 7. 结果表模板

| 实验 | 输入 | Top-1 mean | Top-1 std | Top-5 mean | Top-5 std |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 224 |  |  |  |  |
| SwinV2-Tiny | 224 |  |  |  |  |
| SwinV2-Tiny + FPN | 224 |  |  |  |  |
| SwinV2-Tiny + FPN + Parts | 224 |  |  |  |  |
| SwinV2-Tiny + FPN + Parts | 448 |  |  |  |  |
| ConvNeXtV2-Tiny | 224 |  |  |  |  |
| ConvNeXtV2-Tiny + DCA | 224 |  |  |  |  |
| ConvNeXtV2-Tiny + DCA-Region | 224 |  |  |  |  |
| ConvNeXtV2-Tiny + DCA-Region | 448 |  |  |  |  |

## 8. Web 演示系统

在完成最终模型选择后，新增 `bird_web_app` 作为鸟类识别 Web 演示模块，用于答辩和功能展示。

该模块使用 `convnext_region_448_final` 中测试精度最高的一份模型权重进行本地推理。网页支持上传任意尺寸、任意比例的鸟类图片，前端展示原始输入图片，后端完成图像预处理、模型推理和结果返回。

Web 演示系统的主要功能包括：

1. 上传鸟类图片并在页面中预览。
2. 输出模型预测的英文鸟类类别。
3. 输出模型置信度和后续候选类别。
4. 根据预测英文名获取英文 Wikipedia 描述，用于辅助展示识别结果。
5. 通过临时公网隧道提供外部访问，满足答辩现场演示需求。

代码目录：

```text
bird_web_app/
```

当前状态：

- 已完成本地 Web 推理服务。
- 已完成网页上传、展示和预测结果返回。
- 已支持临时公网访问。
- 尚未进行长期稳定的正式云端上线部署。
