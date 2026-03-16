# autoresearch — 垃圾分类 (Garbage Classification)

基于计算机视觉的垃圾自动分类系统，支持国内四分类标准（可回收物、有害垃圾、厨余垃圾、其他垃圾）。

An autonomous computer vision research system for garbage classification. An AI agent modifies the training code, trains for 5 minutes, checks if accuracy improved, keeps or discards, and repeats — optimizing the model while you sleep.

## 垃圾分类四类标准 (4-Category Standard)

| 类别 | Category | 示例 Examples |
|------|----------|--------------|
| 可回收物 | Recyclable | 纸板、玻璃、金属、纸张、塑料 |
| 有害垃圾 | Hazardous | 电池、灯泡、药品、油漆 |
| 厨余垃圾 | Kitchen waste | 食物残渣、果皮、骨头 |
| 其他垃圾 | Other waste | 纸巾、陶瓷、不可回收物 |

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads TrashNet dataset, organizes into categories), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the CNN model (MobileNetV2 baseline), optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for the agent. **This file is edited and iterated on by the human**.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup). The metric is **val_acc** (validation accuracy) — higher is better.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and one of:
- NVIDIA GPU (CUDA)
- Apple Silicon Mac (MPS) — tested on M4 Pro with 24GB
- CPU (slower but works)

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and organize into categories (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

## Using your own data

The system uses an ImageFolder structure. To use your own garbage photos:

```
~/.cache/autoresearch/data/
  train/
    recyclable/     ← 可回收物 photos
    hazardous/      ← 有害垃圾 photos
    kitchen/        ← 厨余垃圾 photos
    other/          ← 其他垃圾 photos
  val/
    recyclable/
    hazardous/
    kitchen/
    other/
```

Simply place your photos (`.jpg`, `.png`, etc.) in the appropriate directories. The system auto-detects classes from subdirectory names.

**Tips for collecting data:**
- Use your phone camera to take photos of garbage items
- Aim for 50-100+ images per category for good results
- Include variety: different lighting, angles, backgrounds
- The default TrashNet dataset provides ~2500 images for recyclable and other categories

## Running the agent

Spin up your Claude/Codex agent in this repo, then prompt:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will autonomously:
1. Establish a baseline with the default MobileNetV2 model
2. Try different architectures (ResNet, EfficientNet, ViT, etc.)
3. Experiment with hyperparameters, data augmentation, training strategies
4. Keep improvements, discard regressions
5. Log all results to `results.tsv`

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs are reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes. ~12 experiments/hour, ~100 overnight.
- **Cross-platform.** Supports CUDA, MPS (Apple Silicon), and CPU.
- **Transfer learning baseline.** MobileNetV2 pretrained on ImageNet provides a strong starting point.
- **Class imbalance handling.** Weighted sampling and class-weighted loss handle imbalanced datasets.

## Baseline model

The default `train.py` uses:
- **Backbone**: MobileNetV2 (pretrained on ImageNet, partially frozen)
- **Head**: Dropout → FC(256) → ReLU → Dropout → FC(num_classes)
- **Optimizer**: AdamW with different LRs for backbone and head
- **Schedule**: Warmup + cosine annealing
- **Loss**: Cross-entropy with class weights and label smoothing
- **Data augmentation**: Random crop, flip, color jitter, rotation

## License

MIT
