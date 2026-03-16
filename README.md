# autoresearch — 垃圾目标检测 (Garbage Object Detection)

基于多种创新模型架构的垃圾目标检测系统——识别一张图片中的**所有垃圾**并逐个分类，支持国内四分类标准（可回收物、有害垃圾、厨余垃圾、其他垃圾）。

An autonomous computer vision research system for garbage **object detection**. Supports **multiple innovative model architectures** (CNN-based YOLO variants, Transformer-based RT-DETR, ensemble methods) to detect and classify ALL garbage objects in a single image. An AI agent explores different architectures and training strategies, trains for 5 minutes, checks if mAP improved, keeps or discards, and repeats.

## 与图像分类的区别 (Difference from Classification)

| | 图像分类 (Classification) | 目标检测 (Object Detection) |
|---|---|---|
| 输入 | 一张只含一个物体的图片 | 一张包含多个垃圾的照片 |
| 输出 | 一个类别标签 | 每个垃圾的位置(边界框)+类别 |
| 模型 | MobileNet, ResNet, etc. | **YOLOv8, RT-DETR, YOLOv9**, etc. |
| 数据集 | TrashNet (单物体) | **TACO** (多物体+标注框) |
| 评价指标 | Accuracy | **mAP (Mean Average Precision)** |

## 垃圾分类四类标准 (4-Category Standard)

| 类别 | Category | 示例 Examples |
|------|----------|--------------|
| 可回收物 | Recyclable | 塑料瓶、玻璃瓶、金属罐、纸板、纸杯 |
| 有害垃圾 | Hazardous | 电池、气雾罐、药品泡罩包装 |
| 厨余垃圾 | Kitchen waste | 食物残渣、果皮 |
| 其他垃圾 | Other waste | 烟头、破碎玻璃、一次性塑料制品、垃圾袋 |

## 数据集 (Dataset)

使用 **TACO (Trash Annotations in Context)** 数据集：
- ~1500张真实场景图片，~5000个标注框
- 60种细分垃圾类别 → 映射为4种国内分类
- COCO格式标注 → 自动转换为YOLO格式
- 每张图片可能包含多个垃圾目标

## How it works

The repo has four files that matter:

- **`prepare.py`** — TACO data download + YOLO format conversion, category mapping, and evaluation harness. **Agent-editable** — the agent can innovate on data pipeline, category mapping refinements, and evaluation (must keep 4 classes and object detection).
- **`train.py`** — Multi-architecture model config (YOLOv8, RT-DETR, YOLOv5, YOLO11, YOLOv9, YOLOv10), hyperparameters, augmentation, multi-phase training, ensemble evaluation. **Agent-editable** — the agent edits and iterates on this file.
- **`predict.py`** — single-image inference script. Detects all garbage and shows results.
- **`program.md`** — baseline instructions for the agent. **This file is edited and iterated on by the human**.

Training runs for a **fixed 5-minute time budget** (wall clock). The primary metric is **val_mAP50** (mean Average Precision at IoU=0.5) — higher is better.

## Quick start

**Requirements:** Python 3.10+, conda environment with PyTorch

```bash
# 1. Activate conda environment
conda activate Pytorch

# 2. Install additional dependencies
pip install ultralytics

# 3. Download TACO dataset and prepare YOLO format data (~5-15 min)
python prepare.py

# 4. Train the object detection model (~5 min)
python train.py

# 5. Detect garbage in a photo
python predict.py your_photo.jpg --save result.jpg
```

## Environment setup

If creating a new conda environment from scratch:

```bash
conda env create -f environment.yml
conda activate Pytorch
```

Or install into existing "Pytorch" environment:

```bash
conda activate Pytorch
pip install ultralytics
```

## 拍照识别 (Photo Detection)

Train the model, then detect garbage in any photo:

```bash
# Detect garbage in a photo and save annotated result
python predict.py photo.jpg --save result.jpg

# Adjust confidence threshold (lower = more detections)
python predict.py photo.jpg --conf 0.15 --save result.jpg

# Use a specific model checkpoint
python predict.py photo.jpg --model runs/train/weights/best.pt --save result.jpg
```

The output shows:
- Bounding boxes around each detected garbage item
- Category labels (可回收物/有害垃圾/厨余垃圾/其他垃圾)
- Confidence scores
- Category summary (count per category)

## Running the agent

Spin up your Claude/Codex agent in this repo, then prompt:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will autonomously:
1. Establish a baseline with YOLOv8s
2. Explore different model architectures (RT-DETR Transformer, YOLOv9, YOLOv10, YOLO11, YOLOv5)
3. Try different model sizes within each architecture family
4. Experiment with innovative training strategies (multi-phase freeze/unfreeze, AdamW)
5. Tune hyperparameters, augmentation, and loss weights
6. Compare CNN vs. Transformer approaches
7. Keep improvements, discard regressions
8. Log all results to `results.tsv`

## Project structure

```
prepare.py       — TACO data download, YOLO conversion, evaluation (agent-editable, keep 4 classes)
train.py         — Multi-architecture training config and hyperparameters (agent-editable)
predict.py       — single-image inference (detect all garbage in a photo)
program.md       — agent instructions
environment.yml  — conda environment specification
pyproject.toml   — Python project metadata
```

## Design choices

- **Multiple architecture exploration.** CNN (YOLOv8/v5/v9/v10/v11) and Transformer (RT-DETR) models.
- **Object detection, not classification.** Detects ALL garbage objects in one image, not just one label.
- **TACO dataset.** Real-world images with bounding-box annotations, mapped to Chinese 4-category standard.
- **Architecture-aware defaults.** Training config auto-adjusts for CNN vs Transformer models.
- **Fixed time budget.** Training always runs for exactly 5 minutes. ~12 experiments/hour, ~100 overnight.
- **Two files to modify.** The agent can edit both `train.py` (training) and `prepare.py` (data/eval).
- **Paper-driven innovation.** The agent can search arxiv/Google Scholar for state-of-the-art techniques.
- **conda + PyTorch.** Uses local conda environment named "Pytorch".

## Baseline model

The default `train.py` uses:
- **Model**: YOLOv8s (small) pretrained on COCO
- **Image size**: 640×640
- **Augmentation**: Mosaic, HSV, flip, scale
- **Optimizer**: SGD with cosine schedule (auto-switches to AdamW for Transformer models)
- **Time budget**: 5 minutes (using YOLO's built-in time parameter)

## Supported model architectures

| Model | Type | Key Innovation | Recommended Sizes |
|-------|------|---------------|-------------------|
| YOLOv8 | CNN | Anchor-free, C2f modules | n, s, m |
| YOLOv5 | CNN | CSPDarknet + PAN neck | nu, su, mu |
| YOLOv9 | CNN | PGI + GELAN blocks | c, e |
| YOLOv10 | CNN | NMS-free end-to-end | n, s, m, b |
| YOLO11 | CNN | C3k2 + SPPF modules | n, s, m |
| RT-DETR | Transformer | Hybrid CNN+Transformer encoder | l, x |

## License

MIT
