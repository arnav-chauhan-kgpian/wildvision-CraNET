import json
import os
from collections import Counter, defaultdict
from typing import Dict, Tuple, DefaultDict

import fire

from bench_utils import load_model_answers


def _normalize_label(text: str) -> str:
    t = text.lower()
    if "noncrack" in t or "no crack" in t or "no visible crack" in t:
        return "noncrack"
    if "crack" in t:
        return "crack"
    return "unknown"


def _load_ground_truth(jsonl_path: str) -> Dict[str, Dict]:
    gt: Dict[str, Dict] = {}
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Ground-truth file {jsonl_path} not found. "
            "Please export the cracks benchmark with `answer` and taxonomy fields."
        )
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gt[obj["question_id"]] = obj
    return gt


def _compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def eval_cracks_detection(
    bench_name: str = "cracks_bench_v1",
    gt_jsonl: str = os.path.join(
        "data", "cracks_bench_v1", "cracks_bench_v1_detection.jsonl"
    ),
):
    """
    Evaluate detection performance (crack vs noncrack) for all models
    that have answers under data/{bench_name}/model_answers/.
    """
    answers = load_model_answers(os.path.join("data", bench_name, "model_answers"))
    gt = _load_ground_truth(gt_jsonl)

    results: Dict[str, Dict[str, float]] = {}
    by_material: DefaultDict[str, Counter] = defaultdict(Counter)
    by_category: DefaultDict[str, Counter] = defaultdict(Counter)
    by_difficulty: DefaultDict[str, Counter] = defaultdict(Counter)
    by_domain: DefaultDict[str, Counter] = defaultdict(Counter)

    for model_id, per_q in answers.items():
        counts = Counter()
        for qid, record in per_q.items():
            if qid not in gt:
                continue
            g = gt[qid]
            # Categories: detection_analytical, specific_detail
            if g.get("category") not in ("detection_analytical", "specific_detail"):
                continue
                
            true_label = _normalize_label(g.get("answer", ""))
            if true_label == "unknown":
                continue
            pred_label = _normalize_label(record.get("output", ""))

            material = g.get("material", "unknown")
            category = g.get("category", "unknown")
            difficulty = g.get("difficulty", "unknown")
            domain = g.get("domain", "unknown")

            # Determine match
            match_type = "unknown"
            if true_label == "crack":
                match_type = "tp" if pred_label == "crack" else "fn"
            else:
                match_type = "fp" if pred_label == "crack" else "tn"
            
            counts[match_type] += 1
            by_material[material][match_type] += 1
            by_category[category][match_type] += 1
            by_difficulty[difficulty][match_type] += 1
            by_domain[domain][match_type] += 1

        metrics = _compute_metrics(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        )
        results[model_id] = metrics

    print("=" * 40)
    print("=== OVERALL DETECTION METRICS ===")
    print("=" * 40)
    for mid, m in results.items():
        print(
            f"{mid:20s}: "
            f"acc={m['accuracy']:.3f}, "
            f"prec={m['precision']:.3f}, "
            f"rec={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}"
        )

    for taxonomy_name, taxonomy_dict in [
        ("Material Breakdown", by_material),
        ("Category Breakdown", by_category),
        ("Difficulty Breakdown", by_difficulty),
        ("Domain Breakdown", by_domain),
    ]:
        print(f"\n--- {taxonomy_name} ---")
        for key, c in taxonomy_dict.items():
            m = _compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"])
            print(
                f"{key:15s}: "
                f"acc={m['accuracy']:.3f}, "
                f"prec={m['precision']:.3f}, "
                f"rec={m['recall']:.3f}, "
                f"f1={m['f1']:.3f}"
            )


if __name__ == "__main__":
    fire.Fire(eval_cracks_detection)

