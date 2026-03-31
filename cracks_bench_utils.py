import json
import os
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset, Dataset


@dataclass
class CracksExample:
    question_id: str
    instruction: str
    answer: str
    category: str
    material: str
    difficulty: str
    domain: Optional[str] = None
    mask_path: Optional[str] = None


def load_cracks_bench_hf(
    path: str = "WildVision/cracks-bench",
    name: str = "cracks_bench_v1_detection",
    split: str = "test",
) -> Dataset:
    """
    Load the cracks benchmark from HuggingFace Hub, following the
    WildVision-style dataset API.
    """
    return load_dataset(path, name=name, split=split)


def load_cracks_bench_local(
    jsonl_path: str = os.path.join(
        "data", "cracks_bench_v1", "cracks_bench_v1_detection.jsonl"
    )
) -> List[CracksExample]:
    """
    Load a local cracks benchmark JSONL file into structured examples.
    This is mainly for inspection or custom scripts; `gen_answers.py`
    should prefer the HuggingFace-style `load_dataset` interface.
    """
    examples: List[CracksExample] = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Cracks benchmark JSONL not found at {jsonl_path}. "
            f"Please create it following data/cracks_bench_v1/README.md."
        )
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            examples.append(
                CracksExample(
                    question_id=obj["question_id"],
                    instruction=obj["instruction"],
                    answer=obj.get("answer", ""),
                    category=obj.get("category", "detection_analytical"),
                    material=obj.get("material", "unknown"),
                    difficulty=obj.get("difficulty", "medium"),
                    domain=obj.get("domain"),
                    mask_path=obj.get("mask_path"),
                )
            )
    return examples


