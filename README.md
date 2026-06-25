# CUB_200_2011 Bird Species Recognition

本项目用于 CUB_200_2011 细粒度鸟类分类，提供统一的数据划分、训练、验证、超参数选择、全量重训和官方测试流程。F/G 模型从前景响应图自动选取候选区域，其余模型只使用图像级类别标签。

## 项目结构

```text
.
|-- main.py                         # train / pipeline / test / summarize 入口
|-- config.py                       # 数据、训练和模型配置及校验
|-- trainer.py                      # AMP 训练、验证和评估
|-- utils.py                        # 指标、随机种子、日志和运行锁
|-- data_utils/
|   |-- data_loader.py              # CUB 数据集及分层划分
|   `-- transform.py                # 训练与评估预处理
|-- model/
|   |-- resnet_baseline.py
|   |-- swinv2_baseline.py
|   |-- swinv2_fpn.py
|   `-- swinv2_parts.py             # 候选区域采样与关系模型
|-- search_space_swin.json          # Swin 系列训练超参数候选
|-- search_space_resnet.json
`-- fine_grained_bird_recognition_plan.md
```

## 实验协议

1. 官方训练集 5994 张图像按类别分层划分为 `80% train + 20% val`，划分由 `--split-seed` 固定。
2. 验证阶段用 `val_top1` 保存最佳 checkpoint 和选择训练超参数。
3. 模型和优化器由命令行固定，不属于搜索空间。
4. 选定配置后，将 train 与 val 合并成完整的 5994 张官方训练图像。
5. 选定配置后重新初始化并固定训练 50 个 epoch。
6. 三个最终 seed 的模型分别在 5794 张官方测试图像上评估一次，并汇总 Top-1/Top-5 均值和样本标准差。

默认重复实验使用 `--seeds 42,2024,3407`。所有训练流程固定运行 50 个 epoch。

## 环境与数据

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

默认数据目录：

```text
datasets\CUB_200_2011\CUB_200_2011\
|-- images\
|-- images.txt
|-- image_class_labels.txt
|-- train_test_split.txt
`-- bounding_boxes.txt              # 仅 bbox 实验需要
```

数据在其他位置时使用 `--data-root <路径>`。

## 支持模型

| 命令行名称 | 结构 | 额外标注 |
| --- | --- | --- |
| `resnet50_baseline` | ResNet50 | 无 |
| `resnet50_baseline --use-bbox-crop` | ResNet50 + 官方 bbox 裁剪 | bbox |
| `swinv2_tiny` | SwinV2-Tiny | 无 |
| `swinv2_tiny_fpn` | SwinV2-Tiny + FPN | 无 |
| `swinv2_tiny_fpn_parts` | FPN + 响应候选区域采样 | 无 |
| `swinv2_tiny_fpn_relation` | 候选区域关系 Transformer + 低秩双线性头 | 无 |

## 运行流程

正式课程实验直接运行下面六条 `pipeline` 命令。每条命令都会先用 train/val 在搜索文件中选择学习率，再合并为 5994 张官方训练图像做 50 epoch 全量重训，最后在 5794 张官方测试图像上评估三 seed。

### ResNet50 原图

```powershell
python main.py --mode pipeline --model resnet50_baseline `
  --search-space search_space_resnet.json --optimizer sgd `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --run-name resnet_final
```

### ResNet50 bbox

```powershell
python main.py --mode pipeline --model resnet50_baseline --use-bbox-crop `
  --search-space search_space_resnet.json --optimizer sgd `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --run-name resnet_bbox_final
```

### SwinV2-Tiny

```powershell
python main.py --mode pipeline --model swinv2_tiny `
  --search-space search_space_swin.json --optimizer adamw `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --run-name swin_final
```

### SwinV2-Tiny + FPN

```powershell
python main.py --mode pipeline --model swinv2_tiny_fpn `
  --search-space search_space_swin.json --optimizer adamw `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --fpn-channels 256 `
  --run-name fpn_final
```

### FPN + 候选区域采样

```powershell
python main.py --mode pipeline --model swinv2_tiny_fpn_parts `
  --search-space search_space_swin.json --optimizer adamw `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --fpn-channels 256 --num-parts 6 --part-window-size 3 `
  --run-name parts_final
```

### FPN + 部件关系

```powershell
python main.py --mode pipeline --model swinv2_tiny_fpn_relation `
  --search-space search_space_swin.json --optimizer adamw `
  --epochs 50 --batch-size 32 --seeds 42,2024,3407 `
  --fpn-channels 256 --num-parts 6 --part-window-size 3 `
  --relation-heads 4 --bilinear-dim 256 `
  --run-name relation_final
```

ResNet 使用 `search_space_resnet.json`，Swin 系列使用 `search_space_swin.json`。当前搜索文件只改变 `lr`，其余训练参数保持固定。

```json
{
  "candidates": [
    {
      "id": "lr_...",
      "lr": "..."
    }
  ]
}
```

ResNet 学习率候选为 `0.005`、`0.01`、`0.02`；Swin 系列学习率候选为 `0.00003`、`0.00005`、`0.0001`。`model`、`optimizer` 和结构参数不写入搜索 JSON。

## 输出文件

完整 pipeline 写入：

```text
log\pipeline\<run-name>\
|-- search_summary.csv              # 每个候选、每个 seed 的验证结果
|-- selection.json                  # 选择规则、超参数和固定 epoch
|-- final_test_summary.json
|-- final_test_summary.csv
`-- final\seed_<seed>\test_metrics.json

ckpt\pipeline\<run-name>\final\seed_<seed>\<model>_final.pt
```

最终测试结果写入 `final_test_summary.json` 和 `final_test_summary.csv`。

## 稳定性与排错

- 训练与评估检测非有限 logits/loss，发现 NaN 或 Inf 时立即报告模型、epoch 和 step。
- AMP 下先反缩放梯度，再执行 `--grad-clip-norm` 指定的梯度裁剪。
- 关系模型的双线性 signed-sqrt 与归一化在 FP32 中执行，再转回原精度。
- 同一实验目录使用 `.run.lock` 防止两个训练进程同时写入；确认对应进程已结束后，才能手动删除这一明确的锁文件。
- 显存不足时优先减小 `--batch-size`，结构参数不应为迁就显存而混入超参数搜索。
