import json
import os
from typing import Dict, List, DefaultDict
from collections import defaultdict

import fire
import numpy as np

from cranet_wrapper import cranet_segment


def _load_seg_gt(jsonl_path: str) -> List[Dict]:
    gt: List[Dict] = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Segmentation ground-truth file {jsonl_path} not found."
        )
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("mask_path"):
                gt.append(obj)
    return gt


def _iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return inter / union


def eval_cracks_segmentation(
    gt_jsonl: str = os.path.join(
        "data", "cracks_bench_v1", "cracks_bench_v1_segmentation.jsonl"
    ),
    images_root: str = os.path.join("data", "cracks_bench_v1", "images"),
    masks_root: str = os.path.join("data", "cracks_bench_v1", "masks"),
):
    """
    Evaluate CraNET's weakly supervised segmentation using Grad-CAM masks
    against pixel-level ground-truth masks for the segmentation subset.

    This script runs CraNET directly rather than reading model_answers,
    because segmentation outputs are masks rather than pure text.
    """
    from PIL import Image

    gt_list = _load_seg_gt(gt_jsonl)

    ious = []
    by_material: DefaultDict[str, List[float]] = defaultdict(list)
    by_category: DefaultDict[str, List[float]] = defaultdict(list)
    by_difficulty: DefaultDict[str, List[float]] = defaultdict(list)
    by_domain: DefaultDict[str, List[float]] = defaultdict(list)

    for meta in gt_list:
        image_rel = meta.get("image_path")
        mask_rel = meta.get("mask_path")
        if not image_rel or not mask_rel:
            continue
        image_path = os.path.join(images_root, image_rel)
        mask_path = os.path.join(masks_root, mask_rel)
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        image = Image.open(image_path)
        gt_mask = np.array(Image.open(mask_path).convert("L")) > 0

        seg = cranet_segment(image)
        pred_mask = seg["mask"]

        # Resize pred_mask to gt dimensions if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
            pred_mask_img = pred_mask_img.resize(gt_mask.shape[::-1], resample=Image.NEAREST)
            pred_mask = np.array(pred_mask_img) > 0

        iou_val = _iou(pred_mask, gt_mask)
        ious.append(iou_val)
        
        # Taxonomy breakdown
        material = meta.get("material", "unknown")
        category = meta.get("category", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        domain = meta.get("domain", "unknown")
        
        by_material[material].append(iou_val)
        by_category[category].append(iou_val)
        by_difficulty[difficulty].append(iou_val)
        by_domain[domain].append(iou_val)

    if not ious:
        print("No valid segmentation pairs found. Please ensure gt_jsonl, images_root, and masks_root are set correctly.")
        return

    print("=" * 40)
    print("=== SEGMENTATION METRICS (CraNET) ===")
    print("=" * 40)
    print(f"Overall Mean IoU over {len(ious)} samples: {float(np.mean(ious)):.4f}")

    for taxonomy_name, taxonomy_dict in [
        ("Material Breakdown", by_material),
        ("Category Breakdown", by_category),
        ("Difficulty Breakdown", by_difficulty),
        ("Domain Breakdown", by_domain),
    ]:
        print(f"\n--- {taxonomy_name} ---")
        for key, vals in taxonomy_dict.items():
            print(f"{key:15s}: mean_iou={float(np.mean(vals)):.4f} (n={len(vals)})")


if __name__ == "__main__":
    fire.Fire(eval_cracks_segmentation)

