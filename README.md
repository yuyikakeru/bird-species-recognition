# CUB_200_2011 Bird Species Recognition

这个仓库是一套面向 CUB_200_2011 的细粒度鸟类识别实验框架，而不是单个模型脚本。整体目标是用同一套数据读取、训练、验证、日志、checkpoint 和多随机种子汇总流程，逐步比较 ResNet50 baseline、bbox 前景裁剪、SwinV2-Tiny 强 backbone、SwinV2-Tiny + FPN 多尺度融合等实验块。

当前统一入口是 `main.py`，支持 `smoke`、`train`、`summarize` 三种模式。实验计划见 `fine_grained_bird_recognition_plan.md`。

## 全局实验约定

- 数据集：CUB_200_2011，默认路径为 `datasets\CUB_200_2011\CUB_200_2011`。
- 任务：200 类鸟种分类，主要报告 Top-1、Top-5，以及多 seed 的均值和标准差。
- 默认输入：正式对比使用 `--image-size 224 --resize-size 256`；更高分辨率实验需要在 `--run-name` 中写清尺寸。
- 重复实验：关键结论默认使用 `--seeds 42,2024,3407`。
- 早停规则：正式实验统一监控 `val_top1`，使用 `--early-stop-patience 10 --early-stop-min-delta 0.01`。也就是验证 Top-1 连续 10 轮没有提升至少 0.01 个百分点时早停；任意更高的验证 Top-1 仍保存为 best checkpoint。
- 对比口径：不同实验块可能在不同 epoch 早停，因此比较 `best_top1_mean ± best_top1_std`，不比较最后一个 epoch 的准确率。
- 产物位置：日志写入 `log\<model>\<run-name>\`，checkpoint 写入 `ckpt\<model>\<run-name>\`。

## 已实现实验块

| Block | 模型或策略 | 作用 | 推荐重复 |
| --- | --- | --- | --- |
| A | Dataset、DataLoader、Trainer、smoke test | 跑通数据和训练骨架 | smoke |
| B | ResNet50 baseline | CNN 原图基线 | 3 seeds |
| C | ResNet50 + bbox crop | 验证官方标注框前景裁剪收益 | 3 seeds |
| D | SwinV2-Tiny | 更强 backbone baseline | 3 seeds |
| E | SwinV2-Tiny + FPN | 验证 C2/C3/C4 多尺度融合 | 3 seeds |

后续计划中的 foreground attention、part attention、relation module、CLIP knowledge 等模块会继续沿用同一套训练与汇总接口，保证新增模块可以和已完成实验公平比较。

## 环境准备

建议在项目根目录创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果已经有可用环境，可以直接激活后执行 `pip install -r requirements.txt`。

## 数据准备

默认数据目录应为：

```text
datasets\CUB_200_2011\CUB_200_2011
```

该目录下至少包含：

```text
images\
parts\
images.txt
classes.txt
image_class_labels.txt
train_test_split.txt
bounding_boxes.txt
```

如果数据集放在其他路径，运行时通过 `--data-root <路径>` 指定。

## 推荐运行顺序

### 1. Smoke 检查

先确认数据能读取、batch 字段完整、模型能完成一次前向传播：

```powershell
python main.py --mode smoke --num-workers 0
```

正常情况下会输出训练集和验证集大小、batch 摘要，以及 `smoke_logits_shape`。

### 2. 短训练调试

再跑一个每轮只取少量 batch 的调试实验，确认训练、验证、日志和 checkpoint 流程都可用：

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 2 --image-size 224 --resize-size 256 --epochs 80 --num-workers 4 --seeds 1,2 --run-name debug_resnet50_baseline_224 --max-train-batches 1 --max-val-batches 1 --no-pretrained --no-parts
```

这里使用 `--no-pretrained` 是为了避免调试阶段因为下载 ImageNet 权重而中断。

### 3. 正式训练

Block B：ResNet50 原图基线

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 16 --image-size 224 --resize-size 256 --epochs 80 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --early-stop-patience 10 --early-stop-min-delta 0.01 --seeds 42,2024,3407 --run-name resnet50_baseline_224_3seeds --no-parts
```

Block C：ResNet50 + 官方 bbox 前景裁剪

```powershell
python main.py --mode train --model resnet50_baseline --use-bbox-crop --batch-size 16 --image-size 224 --resize-size 256 --epochs 80 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --early-stop-patience 10 --early-stop-min-delta 0.01 --seeds 42,2024,3407 --run-name blockC_bbox_crop_224_3seeds --no-parts
```

Block D：SwinV2-Tiny backbone

```powershell
python main.py --mode train --model swinv2_tiny --batch-size 16 --image-size 224 --resize-size 256 --epochs 80 --optimizer adamw --lr 5e-5 --weight-decay 0.05 --num-workers 4 --early-stop-patience 10 --early-stop-min-delta 0.01 --seeds 42,2024,3407 --run-name blockD_swinv2_tiny_224_3seeds --no-parts
```

Block E：SwinV2-Tiny + FPN 多尺度融合

```powershell
python main.py --mode train --model swinv2_tiny_fpn --batch-size 16 --image-size 224 --resize-size 256 --epochs 80 --optimizer adamw --lr 5e-5 --weight-decay 0.05 --num-workers 4 --early-stop-patience 10 --early-stop-min-delta 0.01 --seeds 42,2024,3407 --run-name blockE_swinv2_tiny_fpn_224_3seeds --no-parts
```

SwinV2 相关实验默认使用 ImageNet-1K 预训练权重。若显存不足，优先把 `--batch-size 16` 调整为 `8`，仍不足再调整为 `4`。

### 4. 汇总已有日志

如果 3 个 seed 是分开运行完成的，可以用 `summarize` 只读取历史日志并重新生成汇总结果，不会重新训练：

```powershell
python main.py --mode summarize --model resnet50_baseline --seeds 42,2024,3407 --run-name resnet50_baseline_224_3seeds
```

```powershell
python main.py --mode summarize --model resnet50_baseline --seeds 42,2024,3407 --run-name blockC_bbox_crop_224_3seeds
```

```powershell
python main.py --mode summarize --model swinv2_tiny --seeds 42,2024,3407 --run-name blockD_swinv2_tiny_224_3seeds
```

```powershell
python main.py --mode summarize --model swinv2_tiny_fpn --seeds 42,2024,3407 --run-name blockE_swinv2_tiny_fpn_224_3seeds
```

汇总文件会生成在对应 `run-name` 目录下：

```text
log\<model>\<run-name>\repeat_summary.json
log\<model>\<run-name>\repeat_summary.csv
```

## 结果文件结构

单个 seed 的历史记录：

```text
log\<model>\<run-name>\seed_<seed>\<model>_history.csv
log\<model>\<run-name>\seed_<seed>\<model>_history.json
```

单个 seed 的最佳模型：

```text
ckpt\<model>\<run-name>\seed_<seed>\<model>_best.pt
```

多 seed 汇总：

```text
log\<model>\<run-name>\repeat_summary.csv
log\<model>\<run-name>\repeat_summary.json
```

训练启动时会在 `log\<model>\<run-name>\.run.lock` 写入锁文件，防止多个同名实验同时写入同一目录。如果异常退出留下旧锁，只在确认没有对应训练进程后，手动删除这一单个 `.run.lock` 文件。

## 常用参数

- `--mode smoke|train|summarize`：运行模式。
- `--model smoke|resnet50_baseline|swinv2_tiny|swinv2_tiny_fpn`：模型名称。
- `--data-root <路径>`：指定 CUB_200_2011 数据集根目录。
- `--seeds 42,2024,3407`：按顺序运行多个随机种子并汇总。
- `--run-name <名称>`：同一组实验的输出目录名，建议包含模型、尺寸和 seed 数。
- `--epochs <数字>`：最大训练轮数；正式实验常用 `80`。
- `--early-stop-patience 10`：验证 Top-1 连续 10 轮没有达到最小提升时早停。
- `--early-stop-min-delta 0.01`：早停所需的最小 Top-1 提升。
- `--batch-size <数字>`、`--image-size <数字>`、`--resize-size <数字>`：控制输入和批大小。
- `--optimizer sgd|adamw`、`--lr <数字>`、`--weight-decay <数字>`：优化器配置。
- `--scheduler cosine|none`：学习率调度器。
- `--device auto|cpu|cuda`：运行设备。
- `--use-bbox-crop`：使用 CUB 官方 bbox 裁剪鸟体区域。
- `--no-parts`：不返回 part locations，适合当前不使用部件监督的实验。
- `--no-pretrained`：不加载 ImageNet 预训练权重，通常只用于离线调试。
- `--fpn-channels <数字>`：Block E 中 FPN 的统一通道数。
- `--max-train-batches <数字>` / `--max-val-batches <数字>`：限制每轮 batch 数，用于快速调试。

## 阅读结果

正式比较优先看 `repeat_summary.json` 中的 `best_top1_mean`、`best_top1_std`、`last_val_top5_mean`。如果某个实验触发早停较早，这是预期行为；只要它使用了相同 seed、相同输入设置和统一的 `10` 轮、`0.01` 早停规则，就可以和其他实验按最佳验证 Top-1 公平比较。
