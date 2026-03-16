"""
Autoresearch garbage OBJECT DETECTION training script.
Uses YOLOv8 to detect and classify multiple garbage objects in each image.

Supports: CUDA (NVIDIA GPU), MPS (Apple Silicon M4 Pro), CPU.
Usage: python train.py

The agent edits THIS file to optimize detection performance.
"""

import os
import time

import torch
from ultralytics import YOLO

from prepare import (
    IMAGE_SIZE, TIME_BUDGET, NUM_CLASSES, CLASS_NAMES, CLASS_NAMES_EN,
    DATA_DIR, evaluate, get_data_yaml_path,
)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# ---------------------------------------------------------------------------
# Hyperparameters (edit these to optimize — this is the main lever)
# ---------------------------------------------------------------------------

# Model
MODEL_SIZE = "yolov8s.pt"     # Options: yolov8n/s/m/l/x (.pt)
FREEZE_LAYERS = 0             # Number of backbone layers to freeze (0 = none)

# Training
BATCH_SIZE = 16               # batch size (adjust for VRAM)
LR0 = 0.01                   # initial learning rate
LRF = 0.01                   # final LR as fraction of LR0
MOMENTUM = 0.937              # SGD momentum / Adam beta1
WEIGHT_DECAY = 0.0005         # L2 regularization
WARMUP_EPOCHS = 3.0           # warmup epochs
WARMUP_MOMENTUM = 0.8         # warmup initial momentum

# Loss weights
BOX_LOSS_GAIN = 7.5           # box loss weight
CLS_LOSS_GAIN = 0.5           # classification loss weight
DFL_LOSS_GAIN = 1.5           # distribution focal loss weight

# Data augmentation
HSV_H = 0.015                 # HSV-Hue augmentation range
HSV_S = 0.7                   # HSV-Saturation augmentation range
HSV_V = 0.4                   # HSV-Value augmentation range
DEGREES = 0.0                 # rotation augmentation (degrees)
TRANSLATE = 0.1               # translation augmentation (fraction)
SCALE = 0.5                   # scale augmentation (fraction)
SHEAR = 0.0                   # shear augmentation (degrees)
PERSPECTIVE = 0.0             # perspective augmentation
FLIPUD = 0.0                  # vertical flip probability
FLIPLR = 0.5                  # horizontal flip probability
MOSAIC = 1.0                  # mosaic augmentation probability
MIXUP = 0.0                   # mixup augmentation probability
COPY_PASTE = 0.0              # copy-paste augmentation probability

# Inference
CONF_THRESHOLD = 0.25         # confidence threshold for predictions
IOU_THRESHOLD = 0.7           # NMS IoU threshold

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = get_device()
print(f"Device: {device}")
print(f"Model:  {MODEL_SIZE}")

# Load data config
data_yaml = get_data_yaml_path()
assert os.path.exists(data_yaml), (
    f"Data config not found: {data_yaml}. Run `python prepare.py` first."
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Load pretrained YOLO model
model = YOLO(MODEL_SIZE)

# Train with time budget (YOLO 'time' parameter uses hours)
results = model.train(
    data=data_yaml,
    epochs=200,                        # max epochs (time budget will stop earlier)
    time=TIME_BUDGET / 3600,           # time limit in hours (5 min = 0.0833h)
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,

    # Optimizer
    lr0=LR0,
    lrf=LRF,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    warmup_epochs=WARMUP_EPOCHS,
    warmup_momentum=WARMUP_MOMENTUM,

    # Loss
    box=BOX_LOSS_GAIN,
    cls=CLS_LOSS_GAIN,
    dfl=DFL_LOSS_GAIN,

    # Augmentation
    hsv_h=HSV_H,
    hsv_s=HSV_S,
    hsv_v=HSV_V,
    degrees=DEGREES,
    translate=TRANSLATE,
    scale=SCALE,
    shear=SHEAR,
    perspective=PERSPECTIVE,
    flipud=FLIPUD,
    fliplr=FLIPLR,
    mosaic=MOSAIC,
    mixup=MIXUP,
    copy_paste=COPY_PASTE,

    # Freeze backbone layers
    freeze=FREEZE_LAYERS if FREEZE_LAYERS > 0 else None,

    # Output
    patience=0,                        # disable early stopping (use time budget)
    save=True,
    save_period=-1,                    # save only best and last
    project="runs",
    name="train",
    exist_ok=True,
    verbose=True,
    seed=42,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

t_train_end = time.time()
training_seconds = t_train_end - t_start

# Evaluate best model
best_weights = os.path.join("runs", "train", "weights", "best.pt")
if not os.path.exists(best_weights):
    best_weights = os.path.join("runs", "train", "weights", "last.pt")

print(f"\nEvaluating: {best_weights}")
eval_results = evaluate(best_weights, data_yaml, device=device)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

t_end = time.time()

if device == "cuda" and torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print()
print("---")
print(f"val_mAP50:        {eval_results['val_mAP50']:.6f}")
print(f"val_mAP50_95:     {eval_results['val_mAP50_95']:.6f}")
print(f"val_precision:    {eval_results['val_precision']:.6f}")
print(f"val_recall:       {eval_results['val_recall']:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_classes:      {NUM_CLASSES}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"model:            {MODEL_SIZE}")
print(f"image_size:       {IMAGE_SIZE}")

if eval_results.get("per_class_mAP50"):
    print()
    print("Per-class mAP50:")
    for name, ap in eval_results["per_class_mAP50"].items():
        print(f"  {name}: {ap:.4f}")
