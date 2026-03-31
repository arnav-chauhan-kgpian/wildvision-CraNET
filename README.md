# WildVision x CraNET: Structural Crack Detection Benchmark

This repository adapt the **WildVision** evaluation framework for the **CraNET** crack detection and segmentation model. It provides a structured evaluation pipeline to assess model performance in "in-the-wild" bridge inspection scenarios, moving beyond simple binary classification to granular, context-heavy analysis.

## 🏗️ Architecture Overview

The integration bridges the gap between state-of-the-art vision models and domain-specific structural health monitoring:

1.  **CraNET Wrapper:** A lightweight TensorFlow implementation of CraNET with Convolutional Block Attention Modules (CBAM), integrated into the WildVision `gen_answers` pipeline.
2.  **Benchmark Generator:** A tool to create WildVision-style JSONL datasets from structural inspection images, tagged with metadata for material, difficulty, and domain.
3.  **Granular Taxonomy:** Performance analysis broken down by:
    *   **Materials:** Concrete, Steel, Asphalt.
    *   **Prompts:** Analytical, Specific Detail, Data Generation.
    *   **Difficulty:** Easy, Medium, Hard.

## 🚀 Quick Start

### 1. Installation
Install the project in editable mode:
```bash
pip install -e .
```

### 2. Generate the Benchmark
Create the evaluation records from your image and mask directories:
```bash
python generate_benchmark.py
```
*Creates `cracks_bench_v1_detection.jsonl` and `cracks_bench_v1_segmentation.jsonl` in `data/cracks_bench_v1/`.*

### 3. Generate Predictions
Run the CraNET model on the generated benchmark:
```bash
python gen_answers.py --model_name CraNET --bench_name cracks_bench_v1 --dataset_name cracks_bench_v1_detection --dataset_path data/cracks_bench_v1/ --num_proc 1
```

### 4. Run Evaluation
Get detailed metrics across the WildVision taxonomy:

**Detection Metrics:**
```bash
python eval_cracks_detection.py --bench_name cracks_bench_v1
```

**Segmentation Metrics (mIoU):**
```bash
python eval_cracks_segmentation.py --gt_jsonl data/cracks_bench_v1/cracks_bench_v1_segmentation.jsonl --images_root data/cracks_bench_v1/ --masks_root data/cracks_bench_v1/masks/
```

## 📋 Repository Structure

*   `cranet_wrapper.py`: Main interface for model inference (Detection/Grad-CAM Segmentation).
*   `self_supervised_learning_CraNET.py`: Core architecture with CBAM blocks.
*   `gen_answers.py`: WildVision prediction engine, adapted for local model calls.
*   `eval_cracks_*.py`: Metric computation scripts with taxonomy breakdown.
*   `data/`: contains the generated benchmark and model results.
*   `docs/`: contains the original research paper and raw data source.
*   `notebooks/`: original research notebooks for reference.

## 📄 License
This project is licensed under the MIT License.
