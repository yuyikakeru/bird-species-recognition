# Bird Species Recognition

基于 CUB_200_2011 数据集的鸟类细粒度识别实验代码。当前入口文件是 `main.py`，支持快速冒烟检查和训练两种运行模式。

## 环境准备

建议在项目根目录运行以下命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果已经有可用虚拟环境，可以直接激活现有环境后安装依赖。

## 数据路径

默认数据目录为：

```text
datasets\CUB_200_2011\CUB_200_2011
```

该目录下应包含：

```text
images\
parts\
images.txt
classes.txt
image_class_labels.txt
train_test_split.txt
bounding_boxes.txt
```

如果数据放在其他位置，运行时使用 `--data-root` 指定。

## 快速检查

先运行 smoke 模式，确认数据能被读取、batch 形状正确、模型能完成一次前向传播：

```powershell
python main.py --mode smoke --num-workers 0
```

默认 smoke 模型是一个很小的卷积网络，适合检查流程是否可用。正常情况下会输出训练集/验证集大小、batch 信息和 `smoke_logits_shape`。

## 开始训练

### 训练 smoke 模型

```powershell
python main.py --mode train --model smoke --epochs 1 --batch-size 16 --num-workers 0
```

### 推荐正式训练命令

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 16 --image-size 448 --resize-size 512 --epochs 50 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --early-stop-patience 5 --early-stop-min-delta 0.1 --seeds 42,2024,3407 --run-name resnet50_baseline_3seeds
```

该命令会使用 3 个随机种子重复训练，并把结果保存到 `log\resnet50_baseline\resnet50_baseline_3seeds\` 和 `ckpt\resnet50_baseline\resnet50_baseline_3seeds\`。

ResNet50 默认使用 ImageNet 预训练权重。如果当前环境不能联网下载权重，可以先加 `--no-pretrained` 跑通流程：

```powershell
python main.py --mode train --model resnet50_baseline --no-pretrained --batch-size 16 --image-size 448 --resize-size 512 --epochs 50 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --early-stop-patience 5 --early-stop-min-delta 0.1 --seeds 42,2024,3407 --run-name resnet50_baseline_3seeds
```

## 常用参数

- `--mode smoke|train|summarize`：运行模式，默认 `smoke`。
- `--model smoke|resnet50_baseline`：模型名称，默认 `smoke`。
- `--data-root <路径>`：指定 CUB_200_2011 数据集根目录。
- `--epochs <数字>`：训练轮数。
- `--batch-size <数字>`：batch size。
- `--image-size <数字>`：输入裁剪尺寸，默认 `448`。
- `--resize-size <数字>`：resize 尺寸，默认 `512`。
- `--lr <数字>`：学习率。
- `--optimizer sgd|adamw`：优化器，默认 `sgd`。
- `--scheduler cosine|none`：学习率调度器，默认 `cosine`。
- `--device auto|cpu|cuda`：运行设备，默认 `auto`。
- `--use-bbox-crop`：使用标注框裁剪鸟体区域。
- `--no-parts`：不返回 parts 标注。
- `--no-pretrained`：ResNet50 不加载 ImageNet 预训练权重。
- `--max-train-batches <数字>` / `--max-val-batches <数字>`：限制每轮训练/验证 batch 数，适合快速调试。

## 多随机种子实验

可以用逗号传入多个 seed，程序会依次训练并汇总结果。正式训练推荐使用：

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 16 --image-size 448 --resize-size 512 --epochs 50 --optimizer sgd --lr 0.01 --weight-decay 1e-4 --num-workers 4 --early-stop-patience 5 --early-stop-min-delta 0.1 --seeds 42,2024,3407 --run-name resnet50_baseline_3seeds
```

### Block D：SwinV2-Tiny 训练命令

```powershell
python main.py --mode train --model swinv2_tiny --batch-size 8 --image-size 448 --resize-size 512 --epochs 50 --optimizer adamw --lr 2e-5 --weight-decay 0.05 --num-workers 4 --early-stop-patience 5 --early-stop-min-delta 0.1 --seeds 42,2024,3407 --run-name blockD_swinv2_tiny_1k_448 --no-parts
```

该命令使用 ImageNet-1K 预训练的 SwinV2-Tiny 作为 Block D，对 CUB-200-2011 进行 448x448 高分辨率微调。代码会先加载 `swinv2_tiny_window16_256` 预训练权重，再通过 timm 的 `set_input_size` 适配到 448x448 输入。若显存不足，优先把 `--batch-size 8` 改为 `--batch-size 4`，仍不足再改为 `--batch-size 2`。

## 输出位置

训练日志和指标默认保存到：

```text
log\<model>\
```

模型 checkpoint 默认保存到：

```text
ckpt\<model>\
```

多 seed 运行时会在 `log\<model>\<run-name>\seed_<seed>\` 和 `ckpt\<model>\<run-name>\seed_<seed>\` 下分别保存每次实验结果。

如果三个 seed 是单独运行完成的，可以只读取已有历史日志并生成综合结果，不会重新训练：

Block B 原图基线：

```powershell
python main.py --mode summarize --model resnet50_baseline --seeds 42,2024,3407 --run-name resnet50_baseline_3seeds
```

Block C bbox 裁剪：

```powershell
python main.py --mode summarize --model resnet50_baseline --seeds 42,2024,3407 --run-name blockC_bbox_crop
```

命令会在对应的 `run-name` 目录中生成：

```text
log\resnet50_baseline\<run-name>\repeat_summary.json
log\resnet50_baseline\<run-name>\repeat_summary.csv
```

## 当前推荐调试命令

第一次运行建议先执行 smoke 检查：

```powershell
python main.py --mode smoke --num-workers 0
```

然后运行一个很短的 ResNet50 调试实验：

```powershell
python main.py --mode train --model resnet50_baseline --batch-size 2 --image-size 224 --resize-size 256 --epochs 1 --num-workers 4 --seeds 1,2 --run-name debug_resnet --max-train-batches 1 --max-val-batches 1 --no-pretrained
```

确认流程正常后，再使用正式命令训练。调试命令里加了 `--no-pretrained`，可以避免联网下载权重导致调试中断。
