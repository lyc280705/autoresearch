"""
One-time data preparation for garbage OBJECT DETECTION experiments.
Downloads the TACO (Trash Annotations in Context) dataset and converts it
to YOLO format with 4 Chinese waste categories.

Usage:
    python prepare.py                  # full prep (download + organize)
    python prepare.py --data-dir PATH  # use custom data directory

Data is stored in ~/.cache/autoresearch/.

The 4 Chinese waste categories (国内四分类):
  0: 可回收物 (Recyclable)   — paper, clean plastics, glass, metal
  1: 有害垃圾 (Hazardous)    — batteries, aerosols, medicine packaging
  2: 厨余垃圾 (Kitchen waste) — food scraps, fruit peels
  3: 其他垃圾 (Other waste)   — contaminated items, cigarettes, mixed waste

Dataset: TACO (Trash Annotations in Context)
  - ~1500 images with ~5000 bounding-box annotations
  - 60 fine-grained categories mapped to 4 Chinese categories
  - Real-world scenes with MULTIPLE garbage items per image
  - COCO-format annotations converted to YOLO format
"""

import os
import sys
import json
import time
import shutil
import zipfile
import random
import argparse
import concurrent.futures
from collections import defaultdict

import requests

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640          # YOLO input image resolution
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
NUM_CLASSES = 4           # 4 Chinese waste categories
VAL_RATIO = 0.15          # fraction of data for validation
TEST_RATIO = 0.10         # fraction of data for testing
RANDOM_SEED = 42          # reproducible splits

# Class names: Chinese 4-category garbage classification
CLASS_NAMES = ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"]
CLASS_NAMES_EN = ["recyclable", "hazardous", "kitchen", "other"]

# ---------------------------------------------------------------------------
# TACO category name → Chinese 4-category ID mapping
#
# Based on Chinese national waste sorting standard (GB/T 19095-2019):
#   0 可回收物: clean paper, plastic, glass, metal, fabric
#   1 有害垃圾: batteries, aerosols, medicine, chemicals
#   2 厨余垃圾: food waste, organic matter
#   3 其他垃圾: contaminated items, mixed waste, non-recyclable
# ---------------------------------------------------------------------------

TACO_NAME_TO_CHINESE = {
    # 0: 可回收物 (Recyclable) — clean recyclable materials
    "Aluminium foil": 0,
    "Other plastic bottle": 0,
    "Clear plastic bottle": 0,
    "Glass bottle": 0,
    "Plastic bottle cap": 0,
    "Metal bottle cap": 0,
    "Food Can": 0,
    "Drink can": 0,
    "Toilet tube": 0,
    "Other carton": 0,
    "Egg carton": 0,
    "Drink carton": 0,
    "Corrugated carton": 0,
    "Meal carton": 0,
    "Pizza box": 0,
    "Paper cup": 0,
    "Glass cup": 0,
    "Glass jar": 0,
    "Plastic lid": 0,
    "Metal lid": 0,
    "Magazine paper": 0,
    "Wrapping paper": 0,
    "Normal paper": 0,
    "Paper bag": 0,
    "Spread tub": 0,
    "Tupperware": 0,
    "Pop tab": 0,
    "Scrap metal": 0,
    "Paper straw": 0,

    # 1: 有害垃圾 (Hazardous) — dangerous / chemical waste
    "Battery": 1,
    "Aluminium blister pack": 1,
    "Carded blister pack": 1,
    "Aerosol": 1,

    # 2: 厨余垃圾 (Kitchen waste) — food / organic waste
    "Food waste": 2,

    # 3: 其他垃圾 (Other waste) — contaminated / non-recyclable
    "Broken glass": 3,
    "Disposable plastic cup": 3,
    "Foam cup": 3,
    "Other plastic cup": 3,
    "Other plastic": 3,
    "Tissues": 3,
    "Plastified paper bag": 3,
    "Plastic Film": 3,
    "Six pack rings": 3,
    "Garbage bag": 3,
    "Other plastic wrapper": 3,
    "Single-use carrier bag": 3,
    "Polypropylene bag": 3,
    "Crisp packet": 3,
    "Disposable food container": 3,
    "Foam food container": 3,
    "Other plastic container": 3,
    "Plastic glooves": 3,  # [sic] typo in TACO dataset, must match exactly
    "Plastic utensils": 3,
    "Rope & strings": 3,
    "Shoe": 3,
    "Squeezable tube": 3,
    "Plastic straw": 3,
    "Styrofoam piece": 3,
    "Unlabeled litter": 3,
    "Cigarette": 3,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# TACO repository zip (contains annotations.json)
TACO_ZIP_URL = "https://github.com/pedropro/TACO/archive/refs/heads/master.zip"

# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def download_file(url, filepath, max_attempts=5, timeout=120):
    """Download a file with retries. Returns True on success."""
    if os.path.exists(filepath):
        return True
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  Downloading (attempt {attempt}/{max_attempts})...")
            response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
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
                            print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)",
                                  end="", flush=True)
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


def _download_single_image(args):
    """Download one image. Returns (image_id, filepath, success)."""
    image_id, url, filepath = args
    if os.path.exists(filepath):
        return image_id, filepath, True
    try:
        resp = requests.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return image_id, filepath, True
    except (requests.RequestException, IOError):
        return image_id, filepath, False


# ---------------------------------------------------------------------------
# TACO dataset download and conversion
# ---------------------------------------------------------------------------

def download_taco_annotations(cache_dir):
    """Download TACO repo zip and extract annotations.json."""
    ann_path = os.path.join(cache_dir, "taco_annotations.json")
    if os.path.exists(ann_path):
        print("TACO annotations: already downloaded")
        return ann_path

    # Download the TACO repo zip
    zip_path = os.path.join(cache_dir, "taco-master.zip")
    print("Downloading TACO repository...")
    if not download_file(TACO_ZIP_URL, zip_path, max_attempts=3, timeout=120):
        print("\nError: Failed to download TACO repository.")
        print("You can manually download from:")
        print(f"  {TACO_ZIP_URL}")
        print(f"And place the zip at: {zip_path}")
        sys.exit(1)

    # Extract annotations.json from the zip
    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find annotations.json in the zip
        ann_names = [n for n in z.namelist() if n.endswith("annotations.json")]
        if not ann_names:
            print("Error: annotations.json not found in TACO zip")
            sys.exit(1)
        with z.open(ann_names[0]) as src, open(ann_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"Extracted annotations to: {ann_path}")
    return ann_path


def download_taco_images(coco_data, images_dir, max_workers=8):
    """Download TACO images from Flickr URLs in parallel."""
    images = {img["id"]: img for img in coco_data["images"]}

    # Build download task list
    tasks = []
    for img_id, img_info in images.items():
        url = img_info.get("flickr_url") or img_info.get("coco_url", "")
        if not url:
            continue
        ext = os.path.splitext(img_info.get("file_name", ""))[1] or ".jpg"
        filepath = os.path.join(images_dir, f"{img_id}{ext}")
        tasks.append((img_id, url, filepath))

    # Count already downloaded
    already = sum(1 for _, _, fp in tasks if os.path.exists(fp))
    remaining = len(tasks) - already
    if remaining == 0:
        print(f"TACO images: all {already} images already downloaded")
        return

    print(f"Downloading TACO images: {remaining} remaining ({already} already cached)...")

    success = already
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_single_image, t) for t in tasks
                   if not os.path.exists(t[2])]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            _, _, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            done = i + 1
            if done % 100 == 0 or done == len(futures):
                print(f"  Progress: {done}/{len(futures)} "
                      f"(total success: {success}, failed: {failed})")

    print(f"Download complete: {success} images available, {failed} failed")
    if success < 200:
        print("Warning: fewer than 200 images available. Model quality may be limited.")
        print("Consider manually adding garbage images to the dataset.")


def convert_taco_to_yolo(coco_data, images_dir, data_dir):
    """Convert TACO COCO annotations to YOLO format with 4-category mapping."""
    images = {img["id"]: img for img in coco_data["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Build TACO category ID → Chinese 4-category mapping
    taco_id_to_chinese = {}
    unmapped = []
    for cat_id, cat_name in categories.items():
        if cat_name in TACO_NAME_TO_CHINESE:
            taco_id_to_chinese[cat_id] = TACO_NAME_TO_CHINESE[cat_name]
        else:
            unmapped.append(cat_name)
            # Default unmapped categories to "其他垃圾" (Other waste)
            taco_id_to_chinese[cat_id] = 3

    if unmapped:
        print(f"  Note: {len(unmapped)} categories not in mapping (defaulted to 其他垃圾):")
        for name in unmapped:
            print(f"    - {name}")

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Filter to images that exist on disk and have annotations
    valid_ids = []
    for img_id, img_info in images.items():
        if img_id not in anns_by_image:
            continue
        ext = os.path.splitext(img_info.get("file_name", ""))[1] or ".jpg"
        filepath = os.path.join(images_dir, f"{img_id}{ext}")
        if os.path.exists(filepath):
            valid_ids.append(img_id)

    print(f"  Valid images with annotations: {len(valid_ids)}")

    # Split into train/val/test
    random.seed(RANDOM_SEED)
    random.shuffle(valid_ids)
    n = len(valid_ids)
    n_test = max(1, int(n * TEST_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_train = n - n_val - n_test

    splits = {
        "train": valid_ids[:n_train],
        "val": valid_ids[n_train:n_train + n_val],
        "test": valid_ids[n_train + n_val:],
    }

    # Convert and write YOLO format labels
    class_counts = defaultdict(lambda: defaultdict(int))

    for split_name, split_ids in splits.items():
        split_img_dir = os.path.join(data_dir, split_name, "images")
        split_lbl_dir = os.path.join(data_dir, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_id in split_ids:
            img_info = images[img_id]
            ext = os.path.splitext(img_info.get("file_name", ""))[1] or ".jpg"
            src_path = os.path.join(images_dir, f"{img_id}{ext}")

            # Copy image to split directory
            dst_img = os.path.join(split_img_dir, f"{img_id}{ext}")
            if not os.path.exists(dst_img):
                shutil.copy2(src_path, dst_img)

            # Convert annotations to YOLO format
            img_w = img_info["width"]
            img_h = img_info["height"]
            label_lines = []

            for ann in anns_by_image[img_id]:
                cat_id = ann["category_id"]
                chinese_id = taco_id_to_chinese.get(cat_id, 3)

                # COCO bbox: [x_min, y_min, width, height] in pixels
                bx, by, bw, bh = ann["bbox"]
                # YOLO bbox: [x_center, y_center, width, height] normalized to 0-1
                x_center = min(max((bx + bw / 2) / img_w, 0.0), 1.0)
                y_center = min(max((by + bh / 2) / img_h, 0.0), 1.0)
                w_norm = min(max(bw / img_w, 0.001), 1.0)
                h_norm = min(max(bh / img_h, 0.001), 1.0)

                label_lines.append(
                    f"{chinese_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )
                class_counts[split_name][chinese_id] += 1

            stem = os.path.splitext(f"{img_id}{ext}")[0]
            label_path = os.path.join(split_lbl_dir, f"{stem}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n" if label_lines else "")

    # Print statistics
    for split_name in ["train", "val", "test"]:
        total = sum(class_counts[split_name].values())
        print(f"  {split_name}: {len(splits[split_name])} images, {total} annotations")
        for cls_id in range(NUM_CLASSES):
            count = class_counts[split_name].get(cls_id, 0)
            print(f"    {CLASS_NAMES[cls_id]} ({CLASS_NAMES_EN[cls_id]}): {count}")

    return splits


def create_data_yaml(data_dir):
    """Create YOLO data.yaml configuration file."""
    yaml_path = os.path.join(data_dir, "data.yaml")
    content = (
        f"# YOLO data config — garbage detection (4-category Chinese waste sorting)\n"
        f"path: {data_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {NUM_CLASSES}\n"
        f"names:\n"
        f"  0: recyclable\n"
        f"  1: hazardous\n"
        f"  2: kitchen\n"
        f"  3: other\n"
        f"\n"
        f"# Chinese names:\n"
        f"# 0: 可回收物  1: 有害垃圾  2: 厨余垃圾  3: 其他垃圾\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Created data config: {yaml_path}")
    return yaml_path


def get_data_yaml_path(data_dir=None):
    """Return the path to data.yaml (convenience helper for train.py)."""
    if data_dir is None:
        data_dir = DATA_DIR
    return os.path.join(data_dir, "data.yaml")


# ---------------------------------------------------------------------------
# Main download-and-prepare pipeline
# ---------------------------------------------------------------------------

def download_and_prepare(data_dir=None):
    """Download TACO dataset and prepare YOLO-format data."""
    if data_dir is None:
        data_dir = DATA_DIR

    yaml_path = os.path.join(data_dir, "data.yaml")

    # Skip if already prepared
    train_imgs = os.path.join(data_dir, "train", "images")
    if os.path.exists(yaml_path) and os.path.isdir(train_imgs):
        n = len([f for f in os.listdir(train_imgs)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if n > 0:
            print(f"Data: already prepared at {data_dir} ({n} training images)")
            return yaml_path

    # Step 1: Download annotations
    ann_path = download_taco_annotations(CACHE_DIR)

    # Step 2: Parse annotations
    with open(ann_path) as f:
        coco_data = json.load(f)
    print(f"TACO: {len(coco_data['images'])} images, "
          f"{len(coco_data['annotations'])} annotations, "
          f"{len(coco_data['categories'])} categories")

    # Step 3: Download images
    images_dir = os.path.join(CACHE_DIR, "taco_images")
    os.makedirs(images_dir, exist_ok=True)
    download_taco_images(coco_data, images_dir)

    # Step 4: Convert to YOLO format
    print("Converting to YOLO format...")
    convert_taco_to_yolo(coco_data, images_dir, data_dir)

    # Step 5: Create data.yaml
    yaml_path = create_data_yaml(data_dir)

    return yaml_path


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric for object detection)
# ---------------------------------------------------------------------------

def evaluate(model, data_yaml, device=None):
    """
    Evaluate object detection model using ultralytics YOLO val().

    Args:
        model: ultralytics YOLO model (or path to .pt weights)
        data_yaml: path to data.yaml
        device: device string ("cuda", "mps", "cpu", or None for auto)

    Returns dict with:
        val_mAP50:     mAP at IoU=0.5 (primary metric, higher is better)
        val_mAP50_95:  mAP at IoU=0.5:0.95
        val_precision:  overall precision
        val_recall:     overall recall
        per_class_mAP50: dict of per-class mAP50
    """
    from ultralytics import YOLO

    if isinstance(model, str):
        model = YOLO(model)

    kwargs = {"data": data_yaml, "verbose": False}
    if device is not None:
        kwargs["device"] = device

    results = model.val(**kwargs)

    metrics = {
        "val_mAP50": float(results.box.map50),
        "val_mAP50_95": float(results.box.map),
        "val_precision": float(results.box.mp),
        "val_recall": float(results.box.mr),
    }

    # Per-class mAP50
    per_class = {}
    if hasattr(results.box, "ap50") and results.box.ap50 is not None:
        for i, ap in enumerate(results.box.ap50):
            if i < NUM_CLASSES:
                per_class[f"{CLASS_NAMES[i]}({CLASS_NAMES_EN[i]})"] = float(ap)
    metrics["per_class_mAP50"] = per_class

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare TACO dataset for garbage object detection"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory (default: ~/.cache/autoresearch/data)")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else DATA_DIR

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Data directory:  {data_dir}")
    print()

    # Download and prepare
    yaml_path = download_and_prepare(data_dir)
    print()

    # Print dataset statistics
    print("Dataset statistics:")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(data_dir, split, "images")
        lbl_dir = os.path.join(data_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        n_imgs = len([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        n_lbls = len([f for f in os.listdir(lbl_dir)
                      if f.endswith(".txt")]) if os.path.isdir(lbl_dir) else 0
        print(f"  {split:5s}: {n_imgs} images, {n_lbls} label files")
    print()

    print(f"Data config: {yaml_path}")
    print("Done! Ready to train with: python train.py")
