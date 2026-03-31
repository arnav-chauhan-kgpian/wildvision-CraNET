## Cracks-Bench v1 (CraNET-focused)

This folder defines a WildVision-style benchmark specialized for structural crack detection and weakly supervised segmentation.

### Schema

Each example in the benchmark follows this JSON schema:

- `question_id` (string): Unique identifier for the prompt-image pair.
- `image` (image): The crack-related inspection image.
- `instruction` (string): Natural, in-the-wild style prompt from the perspective of an inspector/engineer.
- `answer` (string): Ground-truth textual answer appropriate for the prompt, typically:
  - For detection prompts: `"crack"` or `"noncrack"`.
  - For segmentation prompts: a description or reference to the mask (used mainly for judging / sanity checks).
- `category` (string): Prompt category, one of:
  - `detection_analytical`
  - `segmentation_datagen`
  - `specific_detail`
- `material` (string): Surface material, e.g. `concrete`, `asphalt`, `brick`, `metal`.
- `difficulty` (string): One of `easy`, `medium`, `hard`, based on visual conditions (lighting, clutter, crack subtlety).
- `domain` (string, optional): Broader domain tag, e.g. `bridge`, `building`, `pavement`.
- `mask_path` (string, optional): Relative path to a pixel-level crack mask (if available) under `data/cracks_bench_v1/masks/`.

### Splits

We use the following splits for the cracks benchmark:

- `cracks_bench_v1_detection`: Detection-focused prompts (with or without masks).
- `cracks_bench_v1_segmentation`: Subset of images with pixel-level masks for segmentation/IoU evaluation.

These can be hosted as a HuggingFace dataset (`WildVision/cracks-bench`) or loaded locally via the `load_cracks_bench` utility.

### Prompt philosophy

Prompts are written to mimic realistic requests from inspectors and engineers, rather than synthetic yes/no questions. For example:

- `detection_analytical`:
  - _"I'm inspecting an older retaining wall. Can you look carefully and tell me if you see any structural cracks starting to form?"_
- `segmentation_datagen`:
  - _"Please go over this entire slab and produce an exact pixel-level outline of any surface cracking you find."_
- `specific_detail`:
  - _"Ignore the shadows from the railing; focus only on the concrete and tell me whether there is a fine hairline fracture near the lower-left corner."_

These instructions are shared across CraNET and general vision-language models; detection labels and masks are used to score CraNET quantitatively, while general models are mainly used as qualitative or baseline comparisons.

