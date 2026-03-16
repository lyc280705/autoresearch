# autoresearch — 垃圾目标检测 (Garbage Object Detection)

基于YOLOv8的垃圾目标检测系统——识别一张图片中的**所有垃圾**并逐个分类，支持国内四分类标准（可回收物、有害垃圾、厨余垃圾、其他垃圾）。

An autonomous computer vision research system for garbage **object detection**. Detects and classifies ALL garbage objects in a single image. An AI agent modifies the training code, trains for 5 minutes, checks if mAP improved, keeps or discards, and repeats.

## 与图像分类的区别 (Difference from Classification)

| | 图像分类 (Classification) | 目标检测 (Object Detection) |
|---|---|---|
| 输入 | 一张只含一个物体的图片 | 一张包含多个垃圾的照片 |
| 输出 | 一个类别标签 | 每个垃圾的位置(边界框)+类别 |
| 模型 | MobileNet, ResNet, etc. | **YOLOv8**, Faster R-CNN, etc. |
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

- **`prepare.py`** — fixed constants, TACO data download + YOLO format conversion, and evaluation harness. Not modified by agent.
- **`train.py`** — the single file the agent edits. YOLOv8 model config, hyperparameters, augmentation. **This file is edited and iterated on by the agent**.
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
2. Try different model sizes (YOLOv8n/s/m)
3. Experiment with hyperparameters, augmentation, loss weights
4. Keep improvements, discard regressions
5. Log all results to `results.tsv`

## Project structure

```
prepare.py       — TACO data download, YOLO conversion, evaluation (do not modify)
train.py         — YOLOv8 training config and hyperparameters (agent modifies this)
predict.py       — single-image inference (detect all garbage in a photo)
program.md       — agent instructions
environment.yml  — conda environment specification
pyproject.toml   — Python project metadata
```

## Design choices

- **Object detection, not classification.** Detects ALL garbage objects in one image, not just one label.
- **TACO dataset.** Real-world images with bounding-box annotations, mapped to Chinese 4-category standard.
- **YOLOv8 baseline.** State-of-the-art real-time object detection, supports MPS (Apple Silicon).
- **Fixed time budget.** Training always runs for exactly 5 minutes. ~12 experiments/hour, ~100 overnight.
- **Single file to modify.** The agent only touches `train.py`. Diffs are reviewable.
- **conda + PyTorch.** Uses local conda environment named "Pytorch".

## Baseline model

The default `train.py` uses:
- **Model**: YOLOv8s (small) pretrained on COCO
- **Image size**: 640×640
- **Augmentation**: Mosaic, HSV, flip, scale
- **Optimizer**: SGD with cosine schedule
- **Time budget**: 5 minutes (using YOLO's built-in time parameter)

## License

MIT
