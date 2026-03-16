"""
One-time data preparation for garbage classification experiments.
Downloads the TrashNet dataset and organizes it into 4 Chinese waste categories.

Usage:
    python prepare.py                  # full prep (download + organize)
    python prepare.py --data-dir PATH  # use custom data directory

Data is stored in ~/.cache/autoresearch/.

The 4 Chinese waste categories (国内四分类):
  0: 可回收物 (Recyclable)   — cardboard, paper, plastic, glass, metal
  1: 有害垃圾 (Hazardous)    — (user-supplied, e.g. batteries, bulbs)
  2: 厨余垃圾 (Kitchen waste) — (user-supplied, e.g. food scraps)
  3: 其他垃圾 (Other waste)   — trash, non-recyclable items

When using the default TrashNet dataset, only Recyclable and Other have data.
Users are encouraged to add their own photos for all 4 categories.
"""

import os
import sys
import time
import math
import shutil
import zipfile
import random
import argparse
from pathlib import Path

import requests
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224          # input image resolution
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
NUM_CLASSES = 4           # 4 Chinese waste categories
VAL_RATIO = 0.15          # fraction of data for validation
TEST_RATIO = 0.10         # fraction of data for testing
RANDOM_SEED = 42          # reproducible splits

# Class names: Chinese 4-category garbage classification
CLASS_NAMES = ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"]
CLASS_NAMES_EN = ["recyclable", "hazardous", "kitchen", "other"]

# TrashNet category → Chinese category mapping
TRASHNET_MAPPING = {
    "cardboard": 0,  # 可回收物
    "glass":     0,  # 可回收物
    "metal":     0,  # 可回收物
    "paper":     0,  # 可回收物
    "plastic":   0,  # 可回收物
    "trash":     3,  # 其他垃圾
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

# ---------------------------------------------------------------------------
# Data download and organization
# ---------------------------------------------------------------------------

def download_file(url, filepath, max_attempts=5):
    """Download a file with retries. Returns True on success."""
    if os.path.exists(filepath):
        return True
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  Downloading (attempt {attempt}/{max_attempts})...")
            response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 100 * downloaded / total
                            print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)
            print()
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            print(f"\n  Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def organize_trashnet(zip_path, data_dir):
    """Extract TrashNet zip and organize into 4 Chinese categories."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Check if already organized
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        n_train = sum(len(os.listdir(os.path.join(train_dir, d)))
                      for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        if n_train > 0:
            print(f"Data: already organized at {data_dir} ({n_train} training images)")
            return

    print("Data: extracting and organizing TrashNet dataset...")

    # Extract zip
    extract_dir = os.path.join(data_dir, "_extract")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Find the actual image directories
    # TrashNet zip structure: dataset-resized/{cardboard,glass,metal,paper,plastic,trash}/
    img_root = None
    for root, dirs, files in os.walk(extract_dir):
        if "cardboard" in dirs and "glass" in dirs:
            img_root = root
            break

    if img_root is None:
        print("Error: Could not find TrashNet image directories in zip")
        sys.exit(1)

    # Collect all images by Chinese category
    images_by_class = {i: [] for i in range(NUM_CLASSES)}
    for trashnet_class, chinese_class in TRASHNET_MAPPING.items():
        class_dir = os.path.join(img_root, trashnet_class)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                images_by_class[chinese_class].append(os.path.join(class_dir, fname))

    # Create split directories
    random.seed(RANDOM_SEED)
    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for i in range(NUM_CLASSES):
            os.makedirs(os.path.join(split_dir, CLASS_NAMES_EN[i]), exist_ok=True)

    # Split and copy images
    total_images = 0
    for class_idx, paths in images_by_class.items():
        if not paths:
            print(f"  Warning: No images for class {CLASS_NAMES[class_idx]} ({CLASS_NAMES_EN[class_idx]})")
            continue

        random.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_train = n - n_val - n_test

        splits = {
            "train": (train_dir, paths[:n_train]),
            "val": (val_dir, paths[n_train:n_train + n_val]),
            "test": (test_dir, paths[n_train + n_val:]),
        }

        for split_name, (split_dir, split_paths) in splits.items():
            class_dir = os.path.join(split_dir, CLASS_NAMES_EN[class_idx])
            for src in split_paths:
                dst = os.path.join(class_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                total_images += 1

        print(f"  {CLASS_NAMES[class_idx]:8s} ({CLASS_NAMES_EN[class_idx]:12s}): "
              f"{n_train} train, {n_val} val, {n_test} test ({n} total)")

    # Clean up extracted files
    shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"Data: organized {total_images} images into {data_dir}")


def download_and_prepare(data_dir=None):
    """Download TrashNet and organize into 4-category structure."""
    if data_dir is None:
        data_dir = DATA_DIR

    # Check for existing organized data
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir):
        class_dirs = [d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))]
        n_images = sum(len(os.listdir(os.path.join(train_dir, d))) for d in class_dirs)
        if n_images > 0:
            print(f"Data: found {n_images} training images in {len(class_dirs)} classes at {data_dir}")
            return

    # Download TrashNet
    zip_path = os.path.join(CACHE_DIR, "trashnet.zip")
    print(f"Data: downloading TrashNet dataset...")
    if not download_file(TRASHNET_URL, zip_path):
        print("\nError: Failed to download TrashNet dataset.")
        print("You can manually download it from:")
        print(f"  {TRASHNET_URL}")
        print(f"And place it at: {zip_path}")
        print("\nAlternatively, organize your own images into:")
        print(f"  {data_dir}/train/{{recyclable,hazardous,kitchen,other}}/")
        print(f"  {data_dir}/val/{{recyclable,hazardous,kitchen,other}}/")
        sys.exit(1)

    # Organize into categories
    organize_trashnet(zip_path, data_dir)


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class GarbageDataset(Dataset):
    """Image classification dataset using directory structure.

    Expected layout:
        root/
          class_name_1/
            img001.jpg
            img002.jpg
          class_name_2/
            img003.jpg
            ...
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            # Auto-detect classes from directory names, sorted for reproducibility
            classes = sorted(d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d)))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.classes = list(self.class_to_idx.keys())
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.targets.append(class_idx)

        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency class weights for balanced training."""
        counts = np.bincount(self.targets, minlength=len(self.class_to_idx))
        # Avoid division by zero for classes with no data
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(self.class_to_idx)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self):
        """Compute per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return [class_weights[t] for t in self.targets]


def get_train_transform(image_size=IMAGE_SIZE):
    """Training data augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=IMAGE_SIZE):
    """Validation/test transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_dataloader(split, batch_size, image_size=IMAGE_SIZE, data_dir=None,
                    num_workers=4, balanced_sampling=True):
    """
    Create a DataLoader for the specified split.

    Args:
        split: "train", "val", or "test"
        batch_size: batch size
        image_size: input image resolution
        data_dir: data directory (default: DATA_DIR)
        num_workers: number of data loading workers
        balanced_sampling: use weighted sampling for training (handles class imbalance)

    Returns:
        DataLoader, dataset
    """
    if data_dir is None:
        data_dir = DATA_DIR

    split_dir = os.path.join(data_dir, split)
    assert os.path.isdir(split_dir), f"Split directory not found: {split_dir}. Run prepare.py first."

    if split == "train":
        transform = get_train_transform(image_size)
    else:
        transform = get_val_transform(image_size)

    dataset = GarbageDataset(split_dir, transform=transform)
    assert len(dataset) > 0, f"No images found in {split_dir}"

    sampler = None
    shuffle = False
    if split == "train":
        if balanced_sampling and len(set(dataset.targets.tolist())) > 1:
            sample_weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
        else:
            shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, device, num_classes=None):
    """
    Evaluate model on a data loader.

    Returns dict with:
        val_acc:  overall accuracy (primary metric, higher is better)
        val_f1:   macro-averaged F1 score
        val_loss: average cross-entropy loss
        per_class_acc: per-class accuracy dict
        report: full sklearn classification report string
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    if num_classes is None:
        num_classes = NUM_CLASSES

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type in ("cuda", "mps"))):
            logits = model(images)

        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_samples += labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Determine active classes (those present in the dataset)
    active_classes = sorted(set(all_labels.tolist()))
    active_names = []
    for idx in active_classes:
        if idx < len(CLASS_NAMES):
            active_names.append(f"{CLASS_NAMES[idx]}({CLASS_NAMES_EN[idx]})")
        else:
            active_names.append(f"class_{idx}")

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(total_samples, 1)

    # Per-class accuracy
    per_class_acc = {}
    for idx in active_classes:
        mask = all_labels == idx
        if mask.sum() > 0:
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            per_class_acc[name] = (all_preds[mask] == all_labels[mask]).mean()

    report = classification_report(
        all_labels, all_preds,
        labels=active_classes,
        target_names=active_names,
        zero_division=0,
    )

    return {
        "val_acc": float(accuracy),
        "val_f1": float(macro_f1),
        "val_loss": float(avg_loss),
        "per_class_acc": per_class_acc,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for garbage classification")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory (default: ~/.cache/autoresearch/data)")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else DATA_DIR

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Data directory:  {data_dir}")
    print()

    # Step 1: Download and organize data
    download_and_prepare(data_dir)
    print()

    # Step 2: Print dataset statistics
    print("Dataset statistics:")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        total = 0
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                n = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
                total += n
                print(f"  {split:5s}/{class_name:12s}: {n:4d} images")
        print(f"  {split:5s} total: {total} images")
        print()

    print("Done! Ready to train.")
    print()
    print("To add your own data, place images in:")
    for en_name, cn_name in zip(CLASS_NAMES_EN, CLASS_NAMES):
        print(f"  {data_dir}/train/{en_name}/  — {cn_name}")
    print()
    print("Then re-run prepare.py to update splits, or directly add to train/val/test folders.")
