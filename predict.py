"""
Single-image garbage detection inference.

Detects ALL garbage objects in an image and classifies each one into
4 Chinese waste categories (可回收物/有害垃圾/厨余垃圾/其他垃圾).

Usage:
    python predict.py photo.jpg                         # detect and print
    python predict.py photo.jpg --save result.jpg       # save annotated image
    python predict.py photo.jpg --model runs/train/weights/best.pt
"""

import argparse
import os

from ultralytics import YOLO

from prepare import CLASS_NAMES, CLASS_NAMES_EN

# Chinese display names for each class (used for annotated images)
DISPLAY_NAMES = {i: f"{CLASS_NAMES[i]}({CLASS_NAMES_EN[i]})" for i in range(len(CLASS_NAMES))}


def predict(image_path, model_path="runs/train/weights/best.pt",
            conf=0.25, iou=0.7, save_path=None, show=False):
    """
    Detect garbage objects in an image.

    Args:
        image_path: path to input image
        model_path: path to YOLO weights (.pt)
        conf: minimum confidence threshold
        iou: NMS IoU threshold
        save_path: if set, save annotated image to this path
        show: if True, display the result (requires display)

    Returns:
        list of dicts with keys: class_id, class_cn, class_en, confidence, bbox
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=iou)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            cn = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            en = CLASS_NAMES_EN[cls_id] if cls_id < len(CLASS_NAMES_EN) else f"class_{cls_id}"

            detections.append({
                "class_id": cls_id,
                "class_cn": cn,
                "class_en": en,
                "confidence": confidence,
                "bbox": xyxy,
            })

    # Print summary
    print(f"\nDetected {len(detections)} garbage objects in: {image_path}")
    for det in detections:
        bbox = det["bbox"]
        print(f"  [{det['class_cn']}] {det['class_en']} "
              f"(conf={det['confidence']:.2f}) "
              f"at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    # Category summary
    if detections:
        from collections import Counter
        counts = Counter(d["class_cn"] for d in detections)
        print(f"\nCategory summary:")
        for cn, count in counts.most_common():
            print(f"  {cn}: {count} objects")

    # Save annotated image
    if save_path:
        result_img = results[0].plot()
        from PIL import Image
        import numpy as np
        img = Image.fromarray(result_img[..., ::-1])
        img.save(save_path)
        print(f"\nAnnotated image saved to: {save_path}")

    # Show result
    if show:
        results[0].show()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and classify garbage objects in an image"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="runs/train/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--save", default=None,
                        help="Save annotated image to this path")
    parser.add_argument("--show", action="store_true",
                        help="Display result (requires display)")
    args = parser.parse_args()

    predict(args.image, args.model, args.conf, args.iou, args.save, args.show)
