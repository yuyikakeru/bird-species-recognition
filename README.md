# CUB_200_2011 Bird Species Recognition

本项目用于 CUB_200_2011 细粒度鸟类分类实验，统一管理数据划分、训练、全量训练、官方测试、日志和 checkpoint。当前模型注册保留两条主线：SwinV2-Tiny 系列和 ConvNeXtV2-Tiny 系列；ResNet50 作为基础对照线。

## 项目结构

```text
.
|-- config.py                       # 数据、训练、模型配置
|-- main.py                         # train / pipeline / test / summarize 入口
|-- trainer.py                      # AMP、EMA、TTA、训练、验证和测试
|-- utils.py                        # 指标、随机种子、日志、checkpoint
|-- generate_cam_masks.py           # 基于 SwinV2-Tiny baseline 生成离线 CAM mask
|-- search_space_resnet.json        # ResNet 学习率搜索
|-- search_space_adamw.json         # AdamW 学习率搜索
|-- data_utils/
|   |-- data_loader.py              # CUB 数据集、官方 train/test、train/val 划分
|   |-- transform.py                # 训练和测试预处理
|   `-- __init__.py
|-- model/
|   |-- resnet_baseline.py
|   |-- swinv2_baseline.py
|   |-- swinv2_fpn.py
|   |-- swinv2_fpn_cam_parts.py
|   |-- convnextv2_tiny.py
|   `-- __init__.py
|-- bird_web_app/                  # 鸟类识别 Web 演示系统源码
|-- fine_grained_bird_recognition_plan.md
`-- requirements.txt
```

`analysis/`、`log/`、`ckpt/`、`datasets/`、`references/` 是本地实验产物或数据目录，不属于核心训练代码。

## 环境

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
`-- bounding_boxes.txt
```

数据不在默认位置时加：

```bash
--data-root /path/to/CUB_200_2011
```

## 当前模型

| 命令名 | 结构 | 额外输入 |
| --- | --- | --- |
| `resnet50_baseline` | ResNet50 baseline | 无 |
| `resnet50_baseline --use-bbox-crop` | ResNet50 + 官方 bbox 裁剪 | bbox |
| `swinv2_tiny` | SwinV2-Tiny baseline | 无 |
| `swinv2_tiny_fpn` | SwinV2-Tiny + stage3/stage4 FPN | 无 |
| `swinv2_tiny_fpn_parts` | FPN + CAM-guided Top-K parts | CAM mask |
| `convnextv2_tiny` | ConvNeXtV2-Tiny baseline | 无 |
| `convnextv2_tiny_dca` | ConvNeXtV2-Tiny + DCA residual branch | 无 |
| `convnextv2_tiny_dca_region` | DCA + 4 个 soft region token residual branch | 无 |

所有训练流程默认启用：

- AMP mixed precision
- EMA checkpoint
- 测试时水平翻转 TTA
- 非有限 logits/loss 检查

## CAM Mask

`swinv2_tiny_fpn_parts` 训练、验证和测试都使用外部 CAM mask。生成训练集 CAM：

```bash
python generate_cam_masks.py --model swinv2_tiny --checkpoint ckpt/pipeline/swin_final/final/seed_42/swinv2_tiny_final.pt --output-root datasets/CUB_200_2011/cam_masks --split train_full
```

生成测试集 CAM：

```bash
python generate_cam_masks.py --model swinv2_tiny --checkpoint ckpt/pipeline/swin_final/final/seed_42/swinv2_tiny_final.pt --output-root datasets/CUB_200_2011/cam_masks --split test
```

训练和测试 parts 时传入：

```bash
--cam-root datasets/CUB_200_2011/cam_masks
```

## 正式 Pipeline 命令

以下命令均使用 `pipeline` 模式。一个命令完成一个实验：先在官方训练集上划分 train/val 搜索学习率，再用选中的配置合并为 train_full 全量训练，最后在官方测试集评估。ResNet 使用 `search_space_resnet.json`，SwinV2 和 ConvNeXtV2 使用 `search_space_adamw.json`。

### ResNet50 baseline

```bash
python main.py --mode pipeline --model resnet50_baseline --search-space search_space_resnet.json --optimizer sgd --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name resnet_final
```

### ResNet50 + bbox

```bash
python main.py --mode pipeline --model resnet50_baseline --use-bbox-crop --search-space search_space_resnet.json --optimizer sgd --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name resnet_bbox_final
```

### SwinV2-Tiny baseline

```bash
python main.py --mode pipeline --model swinv2_tiny --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name swin_final
```

### SwinV2-Tiny FPN

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name fpn_final
```

### SwinV2-Tiny FPN + Parts

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn_parts --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --num-parts 6 --part-window-size 3 --cam-root datasets/CUB_200_2011/cam_masks --run-name parts_final
```

### ConvNeXtV2-Tiny baseline

```bash
python main.py --mode pipeline --model convnextv2_tiny --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name convnext_final
```

### ConvNeXtV2-Tiny + DCA

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --run-name convnext_dca_final
```

### ConvNeXtV2-Tiny + DCA-Region

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca_region --search-space search_space_adamw.json --optimizer adamw --epochs 50 --batch-size 32 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name convnext_region_final
```

### SwinV2-Tiny FPN + Parts 448

```bash
python main.py --mode pipeline --model swinv2_tiny_fpn_parts --search-space search_space_adamw.json --optimizer adamw --epochs 50 --image-size 448 --resize-size 512 --batch-size 16 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --num-parts 6 --part-window-size 3 --cam-root datasets/CUB_200_2011/cam_masks --run-name parts_448_final
```

### ConvNeXtV2-Tiny + DCA-Region 448

```bash
python main.py --mode pipeline --model convnextv2_tiny_dca_region --search-space search_space_adamw.json --optimizer adamw --epochs 50 --image-size 448 --resize-size 512 --batch-size 16 --num-workers 8 --seeds 42,2024,3407 --fpn-channels 256 --run-name convnext_region_448_final
```

448 只作为 SwinV2 FPN + Parts 和 ConvNeXtV2 DCA-Region 的高分辨率复跑。

## Web 演示系统

`bird_web_app/` 是基于最终高分辨率 ConvNeXtV2 DCA-Region 模型的鸟类识别 Web 演示系统，用于本地演示和答辩展示。

从项目根目录启动本地服务：

```powershell
python .\bird_web_app\server.py --host 0.0.0.0 --port 8000 --warmup
```

本机访问：

```text
http://127.0.0.1:8000
```

生成临时公网 HTTPS 地址：

```powershell
.\bird_web_app\start_public.ps1
```

网页支持上传任意尺寸、任意比例的鸟类图片，展示输入图片、模型预测的英文鸟类类别、置信度、候选类别，并在联网可用时根据英文类别获取 English Wikipedia 描述。当前网站属于本地部署演示系统，可通过临时公网隧道访问，尚未进行长期稳定的正式云端上线部署。

## 输出

`pipeline` 输出结构与当前项目日志目录保持一致：

```text
log\pipeline\<run-name>\
|-- search\
|   `-- <candidate>\seed_<seed>\
|       |-- <model>_history.csv
|       `-- <model>_history.json
|-- final\
|   `-- seed_<seed>\
|       |-- <model>_full_history.csv
|       |-- <model>_full_history.json
|       `-- test_metrics.json
|-- final_test_summary.json
|-- final_test_summary.csv
|-- search_summary.csv
`-- selection.json

ckpt\pipeline\<run-name>\
|-- search\
|   `-- <candidate>\seed_<seed>\
|       |-- <model>_best.pt
|       `-- <model>_last.pt
`-- final\
    `-- seed_<seed>\<model>_final.pt
```

重点查看：

```text
final_test_summary.csv
final_test_summary.json
```
