# autoresearch — 垃圾目标检测 (Garbage Object Detection)

This is an experiment to have an AI agent autonomously optimize a garbage **object detection** model.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — data prep, TACO conversion, category mapping, evaluation. **Editable** — innovate on data pipeline (keep 4 classes, keep object detection).
   - `train.py` — model architecture, hyperparameters, training strategy. **Editable** — innovate on training.
   - `predict.py` — single-image inference script. Usually no need to modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/data/train/images/` contains images. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task: Garbage Object Detection (垃圾目标检测)

The goal is to build a computer vision model that **detects and classifies ALL garbage objects in a single image**. This is an object detection task — each image may contain multiple garbage items, and the model must:
1. Locate each garbage object with a bounding box
2. Classify each object into one of 4 Chinese waste categories

The 4 categories follow the Chinese national waste sorting standard:
- **可回收物 (Recyclable)**: paper, clean plastic bottles, glass, metal, cardboard
- **有害垃圾 (Hazardous)**: batteries, aerosols, medicine packaging
- **厨余垃圾 (Kitchen waste)**: food scraps, fruit peels
- **其他垃圾 (Other waste)**: contaminated items, cigarettes, mixed waste

The dataset is TACO (Trash Annotations in Context) — real-world images with bounding-box annotations for 60 garbage categories, mapped to the 4 Chinese categories.

## Environment

Use the local conda environment named **Pytorch**:
```bash
conda activate Pytorch
```

## Experimentation

Each experiment runs on a single GPU (CUDA or MPS). The training script runs for a **fixed time budget of 5 minutes** (wall clock). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — model architecture, hyperparameters, training strategy:
  - **Model architecture** — switch between completely different model families:
    - CNN-based: `yolov8n/s/m.pt`, `yolov5nu/su/mu.pt`, `yolo11n/s/m.pt`, `yolov9c/e.pt`, `yolov10n/s/m.pt`
    - Transformer-based: `rtdetr-l.pt`, `rtdetr-x.pt` (hybrid CNN+Transformer encoder)
    - Custom: write a `.yaml` config to define a custom YOLO architecture
  - **Optimizer selection**: `OPTIMIZER = "auto"/"SGD"/"Adam"/"AdamW"/"RMSProp"`
  - Learning rate, momentum, weight decay, warmup settings
  - Augmentation parameters: mosaic, mixup, copy-paste, HSV, rotation, scale, etc.
  - Loss weights: box, cls, dfl gains
  - Batch size (trade off VRAM vs. training speed)
  - Freeze backbone layers (transfer learning strategy)
  - **Multi-phase training**: `MULTI_PHASE_TRAINING = True` freezes backbone first, then fine-tunes
  - **Ensemble evaluation**: `ENSEMBLE_EVAL = True` to evaluate multiple models together
  - Confidence and IoU thresholds for inference
  - Custom YOLO configuration (modify model architecture via YAML)

- Modify `prepare.py` — data pipeline, category mapping, evaluation:
  - **Category mapping** (`TACO_NAME_TO_CHINESE`): reclassify ambiguous items between the 4 categories, or set a value to `None` to exclude noisy categories
  - **Data splits** (`VAL_RATIO`, `TEST_RATIO`): adjust train/val/test proportions
  - **Image size** (`IMAGE_SIZE`): try 480, 640, 800, 1024
  - **Time budget** (`TIME_BUDGET`): adjust training time
  - **Class weights** (`compute_class_weights()`): use for class-imbalanced training
  - **Evaluation** (`evaluate()`): add TTA (`augment=True`), custom NMS thresholds
  - **Data rebuild** (`rebuild_data()`): call after changing category mapping or splits
  - **Dataset stats** (`get_dataset_stats()`): analyze class distribution before experiments

**What you CANNOT do:**
- Change `NUM_CLASSES` from 4 or rename the 4 Chinese waste categories.
- Change the task from object detection to classification (must keep bounding boxes).
- Install new packages or add dependencies beyond what's already available.

**The goal is simple: get the highest val_mAP50 (mean Average Precision at IoU=0.5).** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything else is fair game.

**VRAM** is a soft constraint. The target platform is Apple M4 Pro with 24GB unified memory. Larger models (yolov8l, yolov8x) may not fit. Use `yolov8s` or `yolov8m` as practical defaults.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement with ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mAP50:        0.450000
val_mAP50_95:     0.280000
val_precision:    0.650000
val_recall:       0.420000
training_seconds: 305.1
total_seconds:    325.9
peak_vram_mb:     2048.0
num_classes:      4
batch_size:       16
model:            yolov8s.pt
model_family:     yolov8
image_size:       640
```

You can extract the key metric from the log file:

```
grep "^val_mAP50:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_mAP50	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_mAP50 achieved (e.g. 0.450000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 2.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_mAP50	memory_gb	status	description
a1b2c3d	0.450000	2.0	keep	baseline YOLOv8s (CNN)
b2c3d4e	0.485000	2.1	keep	increase LR to 0.02
c3d4e5f	0.440000	2.0	discard	switch to yolov8n (worse mAP)
d4e5f6g	0.000000	0.0	crash	yolov8x (OOM on MPS)
e5f6g7h	0.510000	3.5	keep	RT-DETR-l (Transformer) AdamW lr=0.0001
f6g7h8i	0.495000	2.3	keep	YOLOv9c (PGI architecture)
g7h8i9j	0.475000	2.0	discard	YOLO11s (latest arch — slightly worse)
h8i9j0k	0.520000	2.5	keep	YOLOv8m + multi-phase freeze training
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_mAP50:\|^val_precision:\|^val_recall:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_mAP50 improved (higher), you "advance" the branch, keeping the git commit
9. If val_mAP50 is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Ideas to try** (non-exhaustive):

**Architecture exploration (high impact):**
- **RT-DETR** (Transformer-based): Set `MODEL_SIZE = "rtdetr-l.pt"` — hybrid CNN+Transformer, uses cross-attention, often higher mAP but slower
- **YOLOv9** (PGI architecture): Set `MODEL_SIZE = "yolov9c.pt"` — Programmable Gradient Information + GELAN blocks
- **YOLOv10** (NMS-free): Set `MODEL_SIZE = "yolov10s.pt"` — end-to-end detection without NMS post-processing
- **YOLO11** (latest): Set `MODEL_SIZE = "yolo11s.pt"` — newest architecture with C3k2 blocks
- **YOLOv5** (proven): Set `MODEL_SIZE = "yolov5su.pt"` — classic and well-tested baseline

**Model size sweep (within each family):**
- YOLOv8: n → s → m (test each for best mAP/speed tradeoff within time budget)
- RT-DETR: l → x (larger Transformer for potentially higher accuracy)
- YOLO11: n → s → m (latest architecture at different capacities)

**Innovative training strategies:**
- **Multi-phase training**: Set `MULTI_PHASE_TRAINING = True` — freeze backbone first, then unfreeze and fine-tune (transfer learning innovation)
- **AdamW optimizer**: Set `OPTIMIZER = "AdamW"` with lower LR — especially beneficial for Transformer-based models
- **Progressive fine-tuning**: Freeze 10 layers → train → unfreeze → train with lower LR

**Hyperparameter tuning:**
- Learning rate: LR0 from 0.001 to 0.1 (CNN) or 0.00001 to 0.001 (Transformer)
- Batch size: 8, 16, 32 (trade off noise vs. updates per time budget)
- Augmentation: increase/decrease mosaic, try mixup, copy-paste, different HSV ranges
- Loss weights: tune box/cls/dfl loss gains
- Image size: try 640, 800, or 480 (tradeoff between detail and speed)
- Multi-scale training

**Ensemble methods:**
- Train multiple architectures (e.g., YOLOv8 + RT-DETR), then set `ENSEMBLE_EVAL = True` with `ENSEMBLE_MODELS = ["path/to/model1.pt"]`
- Compare CNN vs. Transformer performance on garbage detection

**Transfer learning strategies:**
- Freeze first N layers for better transfer from COCO pretrained weights
- Different freeze depths for different model families
- Multi-phase: heavy freeze → partial freeze → full fine-tune

**Data-level innovations (modify prepare.py):**
- **Category mapping refinement**: reclassify ambiguous TACO items between the 4 Chinese categories (e.g., move "Broken glass" from 其他 to 有害)
- **Noise filtering**: set ambiguous TACO categories to `None` in `TACO_NAME_TO_CHINESE` to exclude them
- **Class balance analysis**: call `get_dataset_stats()` to understand distribution, then use `compute_class_weights()` for class-weighted training
- **Data split tuning**: try different `VAL_RATIO` (0.1 vs 0.2) to get more or less training data
- **Higher resolution**: set `IMAGE_SIZE = 800` or `1024` for detecting small garbage objects
- **Evaluation with TTA**: call `evaluate(..., augment=True)` for test-time augmentation
- **Evaluation thresholds**: tune `conf` and `iou` in `evaluate()` to find better operating points
- **Data rebuilding**: after changing `TACO_NAME_TO_CHINESE` mapping, call `rebuild_data()` to regenerate data

## Searching papers for innovation

When you hit a plateau or want to try a fundamentally new approach, **search for relevant research papers** to find state-of-the-art techniques. This gives you more diverse innovation ideas.

**When to search papers:**
- After 3–5 experiments with diminishing returns on the current approach
- When a specific class has persistently low mAP (find targeted solutions)
- When switching to a new model family (understand its strengths)
- Before trying a novel technique (verify it's well-founded)

**How to search:**
- Use `web_fetch` to search arxiv, Google Scholar, or Papers With Code
- Example queries: "garbage detection YOLO 2024", "waste classification deep learning", "small object detection augmentation", "class imbalanced object detection"
- Look for: key hyperparameter settings, augmentation strategies, loss function innovations, architectural modifications

**What to look for in papers:**
- **Data augmentation innovations**: CutMix, GridMask, AutoAugment for detection tasks
- **Loss function improvements**: Focal loss tuning, SIoU, WIoU, MPDIoU for better bbox regression
- **Architecture modifications**: attention mechanisms (CBAM, SE, ECA), FPN variants
- **Training strategies**: knowledge distillation, progressive resizing, SWA
- **Class imbalance solutions**: oversampling, class-aware sampling, balanced GroupSoftmax
- **Post-processing**: Soft-NMS, Weighted Boxes Fusion, learned NMS

**How to apply findings:**
- Modify `train.py` for model/training innovations
- Modify `prepare.py` for data pipeline/evaluation innovations
- Always compare against the current best result
- Document the paper reference in the experiment description

Example experiment descriptions referencing papers:
```
"Apply CopyPaste augmentation (Ghiasi et al., 2021) for rare classes"
"Use AdamW + cosine decay (Loshchilov & Hutter, 2019) for Transformer model"
"Add CBAM attention to YOLOv8 neck (Woo et al., 2018)"
```

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. The loop runs until the human interrupts you, period.
