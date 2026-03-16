"""
Autoresearch garbage classification training script.
Single-GPU (CUDA or MPS), single-file.

Baseline: MobileNetV2 with transfer learning for 4-category
Chinese garbage classification (可回收物/有害垃圾/厨余垃圾/其他垃圾).

Usage: uv run train.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from prepare import (
    IMAGE_SIZE, TIME_BUDGET, NUM_CLASSES, CLASS_NAMES, CLASS_NAMES_EN,
    make_dataloader, evaluate,
)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GarbageClassifier(nn.Module):
    """
    MobileNetV2-based garbage classifier with custom head.

    Architecture:
      - MobileNetV2 backbone (pretrained on ImageNet)
      - Global Average Pooling
      - Dropout → FC → ReLU → Dropout → FC (num_classes)

    The backbone is partially frozen: only the last few layers are fine-tuned
    along with the custom classification head.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, freeze_backbone_ratio=0.7):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        backbone_out_features = self.backbone.classifier[1].in_features

        # Freeze early layers of backbone
        all_params = list(self.backbone.features.parameters())
        n_freeze = int(len(all_params) * freeze_backbone_ratio)
        for param in all_params[:n_freeze]:
            param.requires_grad = False

        # Replace classifier head
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_param_groups(self, backbone_lr, head_lr, weight_decay=0.01):
        """Return parameter groups with different learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay * 0.1},
        ]


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model
DROPOUT = 0.3                  # dropout rate in classifier head
FREEZE_BACKBONE_RATIO = 0.7    # fraction of backbone layers to freeze

# Optimization
BACKBONE_LR = 1e-4             # learning rate for backbone (fine-tuning)
HEAD_LR = 1e-3                 # learning rate for classifier head
WEIGHT_DECAY = 0.01            # L2 regularization
BATCH_SIZE = 32                # training batch size
LABEL_SMOOTHING = 0.1          # label smoothing for cross-entropy

# Schedule
WARMUP_RATIO = 0.1             # fraction of steps for LR warmup
USE_COSINE_SCHEDULE = True     # use cosine annealing (vs. step decay)

# Data
IMAGE_INPUT_SIZE = IMAGE_SIZE  # input image resolution (from prepare.py)
NUM_WORKERS = 4                # data loading workers
BALANCED_SAMPLING = True       # use weighted sampling for class imbalance

# ---------------------------------------------------------------------------
# Setup: device, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = get_device()
print(f"Device: {device}")

if device.type == "cuda":
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

# Determine number of classes from actual data
train_loader, train_dataset = make_dataloader(
    "train", BATCH_SIZE, IMAGE_INPUT_SIZE,
    num_workers=NUM_WORKERS, balanced_sampling=BALANCED_SAMPLING,
)
val_loader, val_dataset = make_dataloader(
    "val", BATCH_SIZE, IMAGE_INPUT_SIZE,
    num_workers=NUM_WORKERS, balanced_sampling=False,
)

actual_num_classes = len(train_dataset.class_to_idx)
print(f"Number of classes: {actual_num_classes}")
print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Class distribution
class_counts = {}
for cls_name, cls_idx in train_dataset.class_to_idx.items():
    count = (train_dataset.targets == cls_idx).sum()
    class_counts[cls_name] = int(count)
    print(f"  {cls_name}: {count} images")

# Build model
model = GarbageClassifier(
    num_classes=actual_num_classes,
    dropout=DROPOUT,
    freeze_backbone_ratio=FREEZE_BACKBONE_RATIO,
)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {num_params:,}")
print(f"Trainable parameters: {num_trainable:,}")

# Optimizer
param_groups = model.get_param_groups(BACKBONE_LR, HEAD_LR, WEIGHT_DECAY)
optimizer = torch.optim.AdamW(param_groups)

# Loss function with class weights and label smoothing
class_weights = train_dataset.get_class_weights().to(device)
print(f"Class weights: {class_weights.tolist()}")
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# Mixed precision
use_amp = device.type in ("cuda", "mps")
if device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}")
print(f"Mixed precision: {use_amp}")

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress, warmup_ratio=WARMUP_RATIO):
    """Warmup + cosine/linear decay schedule."""
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    if USE_COSINE_SCHEDULE:
        # Cosine annealing from 1.0 to 0.01
        cos_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * cos_progress))
    else:
        return 1.0

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
best_val_acc = 0.0
smooth_train_loss = 0.0

model.train()

while True:
    epoch += 1
    for images, labels in train_loader:
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with mixed precision
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        elif use_amp and device.type == "mps":
            with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        train_loss_f = loss.item()

        # Fast fail: abort if loss is NaN or exploding
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("\nFAIL: loss exploded")
            exit(1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Update learning rate
        progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
            group["lr"] = group["initial_lr"] * lrm

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        # Compute batch accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_acc = (preds == labels).float().mean().item()

        print(f"\rstep {step:05d} ep{epoch} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.4f} | "
              f"acc: {batch_acc:.3f} | lrm: {lrm:.3f} | dt: {dt*1000:.0f}ms | "
              f"remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()
            if device.type == "cuda":
                gc.freeze()
                gc.disable()

        step += 1

        # Time's up — stop after warmup steps
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log
print(f"Training complete: {step} steps, {epoch} epochs, {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
results = evaluate(model, val_loader, device, num_classes=actual_num_classes,
                   class_names=train_dataset.classes)

# Print summary
t_end = time.time()
startup_time = t_start_training - t_start

if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device.type == "mps":
    peak_vram_mb = 0.0  # MPS doesn't provide memory tracking like CUDA
else:
    peak_vram_mb = 0.0

print()
print("Classification Report:")
print(results["report"])

print("Per-class accuracy:")
for name, acc in results["per_class_acc"].items():
    print(f"  {name}: {acc:.4f}")

print()
print("---")
print(f"val_acc:          {results['val_acc']:.6f}")
print(f"val_f1:           {results['val_f1']:.6f}")
print(f"val_loss:         {results['val_loss']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"num_trainable_M:  {num_trainable / 1e6:.1f}")
print(f"num_classes:      {actual_num_classes}")
print(f"batch_size:       {BATCH_SIZE}")
