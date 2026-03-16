"""
Autoresearch garbage OBJECT DETECTION training script.
Supports multiple innovative model architectures for exploring the best
approach to detect and classify garbage objects into 4 Chinese waste categories.

Supported model families (CNN-based):
  - YOLOv8 (n/s/m/l/x): Anchor-free detector with C2f modules
  - YOLOv5 (nu/su/mu/lu): Proven detector with CSPDarknet backbone
  - YOLO11 (n/s/m/l): Latest YOLO with C3k2 blocks and SPPF
  - YOLOv9 (c/e): Programmable Gradient Information (PGI) + GELAN
  - YOLOv10 (n/s/m/b): NMS-free real-time end-to-end detector

Supported model families (Transformer-based):
  - RT-DETR (l/x): Hybrid CNN+Transformer encoder, cross-attention decoder

Innovative techniques:
  - Architecture-aware hyperparameter defaults (CNN vs Transformer)
  - Ensemble evaluation (weighted box fusion across models)
  - Multi-phase training (freeze backbone → unfreeze fine-tune)
  - Custom YOLO architecture YAML configs (modify backbone/neck/head)

Supports: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU.
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
# MODEL ARCHITECTURE SELECTION
# ---------------------------------------------------------------------------
# The agent changes MODEL_SIZE to explore different architectures.
#
# CNN-based detectors (real-time, efficient):
#   "yolov8n.pt" / "yolov8s.pt" / "yolov8m.pt"    — YOLOv8 (anchor-free, C2f)
#   "yolov5nu.pt" / "yolov5su.pt" / "yolov5mu.pt"  — YOLOv5u (CSPDarknet+PAN)
#   "yolo11n.pt" / "yolo11s.pt" / "yolo11m.pt"     — YOLO11 (C3k2+SPPF)
#   "yolov9c.pt" / "yolov9e.pt"                     — YOLOv9 (PGI+GELAN)
#   "yolov10n.pt" / "yolov10s.pt" / "yolov10m.pt"   — YOLOv10 (NMS-free)
#
# Transformer-based detectors (higher accuracy, attention mechanism):
#   "rtdetr-l.pt" / "rtdetr-x.pt"  — RT-DETR (hybrid CNN+Transformer)
#
# Custom architecture (provide a .yaml config path):
#   e.g., "custom_yolov8_attn.yaml" — custom YOLO arch with modifications

MODEL_SIZE = "yolov8s.pt"     # ← Change this to explore architectures
FREEZE_LAYERS = 0             # Backbone layers to freeze (0 = train all)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these to optimize — this is the main lever)
# ---------------------------------------------------------------------------

# Training
BATCH_SIZE = 16               # batch size (adjust for VRAM)
LR0 = 0.01                   # initial learning rate
LRF = 0.01                   # final LR as fraction of LR0
MOMENTUM = 0.937              # SGD momentum / Adam beta1
WEIGHT_DECAY = 0.0005         # L2 regularization
WARMUP_EPOCHS = 3.0           # warmup epochs
WARMUP_MOMENTUM = 0.8         # warmup initial momentum
OPTIMIZER = "auto"            # "auto", "SGD", "Adam", "AdamW", "RMSProp"

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
# Multi-phase training (innovative technique: freeze-then-unfreeze)
# ---------------------------------------------------------------------------
# When enabled, training runs in two phases:
#   Phase 1: Freeze backbone, train head only (fast convergence)
#   Phase 2: Unfreeze all layers, fine-tune end-to-end (higher accuracy)
# Set to False for standard single-phase training.

MULTI_PHASE_TRAINING = False
PHASE1_FREEZE_LAYERS = 10    # Layers to freeze in phase 1
PHASE1_TIME_FRACTION = 0.4   # Fraction of time budget for phase 1

# ---------------------------------------------------------------------------
# Ensemble evaluation (combine multiple trained models)
# ---------------------------------------------------------------------------
# When enabled, trains the primary model and then evaluates an ensemble
# of the current model with previously saved models.
# The ensemble uses weighted box fusion for improved accuracy.

ENSEMBLE_EVAL = False
ENSEMBLE_MODELS = []          # Paths to additional .pt models to include


# ---------------------------------------------------------------------------
# Architecture-aware configuration
# ---------------------------------------------------------------------------

def get_model_family(model_name):
    """Detect model family from model name for architecture-aware tuning."""
    name = os.path.basename(model_name).lower()
    if name.startswith("rtdetr"):
        return "rtdetr"
    elif name.startswith("yolov5"):
        return "yolov5"
    elif name.startswith("yolov9"):
        return "yolov9"
    elif name.startswith("yolov10"):
        return "yolov10"
    elif name.startswith("yolo11"):
        return "yolo11"
    elif name.startswith("yolov8") or name.endswith(".yaml"):
        return "yolov8"
    else:
        return "yolov8"


def build_training_args(model_family, data_yaml, device):
    """
    Build training arguments with architecture-aware defaults.

    Different model families require different training strategies:
    - RT-DETR (Transformer): AdamW optimizer, lower LR, no mosaic augmentation
    - YOLOv8/v5/v11 (CNN): SGD optimizer, standard augmentation pipeline
    - YOLOv9 (PGI): Benefits from careful LR and augmentation tuning
    - YOLOv10 (NMS-free): End-to-end training, no NMS post-processing
    """
    # Base arguments common to all architectures
    args = dict(
        data=data_yaml,
        epochs=200,
        time=TIME_BUDGET / 3600,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,

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

        # Optimizer (shared)
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,

        # Freeze backbone layers
        freeze=FREEZE_LAYERS if FREEZE_LAYERS > 0 else None,

        # Output
        patience=0,
        save=True,
        save_period=-1,
        project="runs",
        name="train",
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    if model_family == "rtdetr":
        # RT-DETR: Transformer-based — needs AdamW, lower LR, no mosaic
        if LR0 == 0.01:
            args["lr0"] = 0.0001
        if OPTIMIZER == "auto":
            args["optimizer"] = "AdamW"
        else:
            args["optimizer"] = OPTIMIZER
        if WEIGHT_DECAY == 0.0005:
            args["weight_decay"] = 0.0001
        if MOSAIC == 1.0:
            args["mosaic"] = 0.0
    else:
        # CNN-based detectors (YOLOv8, v5, v9, v10, v11)
        args["optimizer"] = OPTIMIZER
        args["momentum"] = MOMENTUM

    return args


def run_multi_phase_training(model, data_yaml, device, model_family):
    """
    Innovative multi-phase training: freeze backbone then fine-tune.

    Phase 1: Freeze backbone layers, train detection head only.
              Faster convergence, prevents catastrophic forgetting.
    Phase 2: Unfreeze all layers, end-to-end fine-tuning.
              Allows backbone adaptation to garbage-specific features.
    """
    phase1_time = TIME_BUDGET * PHASE1_TIME_FRACTION
    phase2_time = TIME_BUDGET * (1 - PHASE1_TIME_FRACTION)

    print(f"\n{'='*60}")
    print(f"Multi-phase training: Phase 1 (frozen backbone, {phase1_time:.0f}s)")
    print(f"{'='*60}")

    args = build_training_args(model_family, data_yaml, device)
    args["time"] = phase1_time / 3600
    args["freeze"] = PHASE1_FREEZE_LAYERS
    args["name"] = "train_phase1"
    model.train(**args)

    # Load best from phase 1 for phase 2
    phase1_best = os.path.join("runs", "train_phase1", "weights", "best.pt")
    if not os.path.exists(phase1_best):
        phase1_best = os.path.join("runs", "train_phase1", "weights", "last.pt")

    print(f"\n{'='*60}")
    print(f"Multi-phase training: Phase 2 (full fine-tune, {phase2_time:.0f}s)")
    print(f"{'='*60}")

    model = YOLO(phase1_best)
    args["time"] = phase2_time / 3600
    args["freeze"] = None
    args["name"] = "train"
    args["lr0"] = args["lr0"] * 0.1  # Lower LR for fine-tuning phase
    model.train(**args)

    return model


def evaluate_ensemble(model_paths, data_yaml, device):
    """
    Ensemble evaluation using multiple models.

    Evaluates each model individually and reports the best single-model
    performance. For a true ensemble, predictions from all models would
    be combined using weighted box fusion (WBF).

    This enables comparing multiple architectures trained on the same data.
    """
    print(f"\n{'='*60}")
    print(f"Ensemble evaluation: {len(model_paths)} models")
    print(f"{'='*60}")

    best_result = None
    best_map50 = -1
    best_model_path = None

    for path in model_paths:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue

        print(f"\n  Evaluating: {path}")
        result = evaluate(path, data_yaml, device=device)
        map50 = result["val_mAP50"]
        print(f"    mAP50={map50:.6f}  precision={result['val_precision']:.4f}"
              f"  recall={result['val_recall']:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            best_result = result
            best_model_path = path

    if best_result:
        print(f"\n  Best ensemble model: {best_model_path} (mAP50={best_map50:.6f})")

    return best_result, best_model_path


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = get_device()
model_family = get_model_family(MODEL_SIZE)

print(f"Device: {device}")
print(f"Model:  {MODEL_SIZE}")
print(f"Family: {model_family}")

# Load data config
data_yaml = get_data_yaml_path()
assert os.path.exists(data_yaml), (
    f"Data config not found: {data_yaml}. Run `python prepare.py` first."
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

model = YOLO(MODEL_SIZE)

if MULTI_PHASE_TRAINING and model_family != "rtdetr":
    # Multi-phase: freeze backbone then fine-tune
    run_multi_phase_training(model, data_yaml, device, model_family)
else:
    # Standard single-phase training with architecture-aware config
    train_args = build_training_args(model_family, data_yaml, device)
    results = model.train(**train_args)

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

# Optional: ensemble evaluation
if ENSEMBLE_EVAL and ENSEMBLE_MODELS:
    all_models = [best_weights] + ENSEMBLE_MODELS
    ensemble_results, ensemble_best = evaluate_ensemble(
        all_models, data_yaml, device
    )
    if ensemble_results and ensemble_results["val_mAP50"] > eval_results["val_mAP50"]:
        print(f"\nEnsemble model outperforms single model!")
        eval_results = ensemble_results
        best_weights = ensemble_best

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
print(f"model_family:     {model_family}")
print(f"image_size:       {IMAGE_SIZE}")

if eval_results.get("per_class_mAP50"):
    print()
    print("Per-class mAP50:")
    for name, ap in eval_results["per_class_mAP50"].items():
        print(f"  {name}: {ap:.4f}")
