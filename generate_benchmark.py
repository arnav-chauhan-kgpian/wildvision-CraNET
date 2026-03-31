import json
import os
import random
from typing import List, Dict

# Configuration
IMAGE_DIR = "data/cracks_bench_v1/images"
MASK_DIR = "data/cracks_bench_v1/masks"
OUTPUT_DIR = "data/cracks_bench_v1"
DOMAIN = "bridge"
MATERIALS = ["concrete", "steel", "asphalt"]
DIFFICULTIES = ["easy", "medium", "hard"]

PROMPT_TEMPLATES = {
    "detection_analytical": [
        "I'm inspecting an older bridge. Can you analyze this section and point out any structural cracks?",
        "Can you look at this surface and tell me if there are any fractures that might indicate structural fatigue?",
        "Look closely at this image. Is there any evidence of cracking on the surface?",
        "As a structural engineer, I need to know if this area has any visible damage. Do you see any cracks?",
        "I'm checking for signs of decay in this bridge component. Are there any surface fractures visible?"
    ],
    "segmentation_datagen": [
        "Please process this image and output the exact pixel-level mask of the damage.",
        "Provide a detailed segmentation mask for all cracks found in this section.",
        "Extract a binary mask showing the exact extent of any surface cracking.",
        "Can you map out the full surface area of the cracks in this visual data?",
        "Output the pixel-level outline of any structural fractures in this photo."
    ],
    "specific_detail": [
        "Ignore any surface stains or shadows; focus only on the material and tell me if there is a fine hairline fracture.",
        "Isolate the tiny, faint fractures near the edge of the frame.",
        "Focus on the area near the junction; are those actual cracks or just surface dirt?",
        "Check the lower-left corner carefully. Is there a small crack starting to form there?",
        "Excluding the obvious large cracks, are there any smaller branching fractures visible?"
    ]
}

def generate_benchmarks():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    images.sort()
    
    detection_records = []
    segmentation_records = []
    
    for i, img_name in enumerate(images):
        question_id = f"crack_{i:04d}"
        
        # All these images are from the bridge crack dataset, so answer is "crack"
        # In a real scenario, we would have noncrack as well.
        answer = "crack"
        
        material = random.choice(MATERIALS)
        difficulty = random.choice(DIFFICULTIES)
        
        # Categorize images into different prompt types to diversify the benchmark
        # We can create multiple records per image to test different prompt styles
        
        # 1. Detection Analytical
        prompt_analytical = random.choice(PROMPT_TEMPLATES["detection_analytical"])
        detection_records.append({
            "question_id": f"{question_id}_analytical",
            "image_path": f"images/{img_name}",
            "instruction": prompt_analytical,
            "answer": answer,
            "category": "detection_analytical",
            "material": material,
            "difficulty": difficulty,
            "domain": DOMAIN
        })
        
        # 2. Specific Detail
        prompt_detail = random.choice(PROMPT_TEMPLATES["specific_detail"])
        detection_records.append({
            "question_id": f"{question_id}_detail",
            "image_path": f"images/{img_name}",
            "instruction": prompt_detail,
            "answer": answer,
            "category": "specific_detail",
            "material": material,
            "difficulty": difficulty,
            "domain": DOMAIN
        })
        
        # 3. Segmentation Data Generation
        # Only if mask exists
        mask_path = img_name # Masks have same name in this dataset
        if os.path.exists(os.path.join(MASK_DIR, mask_path)):
            prompt_seg = random.choice(PROMPT_TEMPLATES["segmentation_datagen"])
            segmentation_records.append({
                "question_id": f"{question_id}_seg",
                "image_path": f"images/{img_name}",
                "mask_path": mask_path,
                "instruction": prompt_seg,
                "answer": "segmentation_mask",
                "category": "segmentation_datagen",
                "material": material,
                "difficulty": difficulty,
                "domain": DOMAIN
            })

    # Save to JSONL
    det_file = os.path.join(OUTPUT_DIR, "cracks_bench_v1_detection.jsonl")
    with open(det_file, "w") as f:
        for rec in detection_records:
            f.write(json.dumps(rec) + "\n")
    
    seg_file = os.path.join(OUTPUT_DIR, "cracks_bench_v1_segmentation.jsonl")
    with open(seg_file, "w") as f:
        for rec in segmentation_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Generated {len(detection_records)} detection records in {det_file}")
    print(f"Generated {len(segmentation_records)} segmentation records in {seg_file}")

if __name__ == "__main__":
    generate_benchmarks()
