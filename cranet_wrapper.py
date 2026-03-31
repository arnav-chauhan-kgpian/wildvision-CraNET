import tensorflow as tf
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple


def load_cranet_model(model_path: Optional[str] = "Nov_22_cbam.h5") -> tf.keras.Model:
    """
    Build a lightweight CraNET model from the notebook definition and
    optionally load trained weights. Default weights are Nov_22_cbam.h5.
    """
    from self_supervised_learning_CraNET import build_cranet_light
    import os

    model = build_cranet_light()
    if model_path and os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Loaded CraNET weights from {model_path}")
    elif model_path:
        print(f"Weights file not found at {model_path}. Initialized with random weights.")
    return model


def preprocess_image(image) -> np.ndarray:
    """
    Resize and normalize an input image to the CraNET expected format.
    """
    if isinstance(image, str):
        image = Image.open(image)
    image = image.resize((224, 224))
    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr / 255.0
    arr = arr.astype(np.float32)
    return arr


def cranet_detect(image, model_path: Optional[str] = None) -> Dict[str, float]:
    """
    Run CraNET in detection mode and return a structured prediction.

    Returns a dict with `label` in {\"crack\", \"noncrack\"} and a
    confidence score for the predicted label.
    """
    model = load_cranet_model(model_path)
    arr = preprocess_image(image)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    if preds.shape[-1] == 1:
        prob_crack = float(preds[0, 0])
        label = "crack" if prob_crack >= 0.5 else "noncrack"
        score = prob_crack if label == "crack" else 1.0 - prob_crack
    else:
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
        label_idx = int(np.argmax(probs))
        label = "crack" if label_idx == 0 else "noncrack"
        score = float(probs[label_idx])
    return {"label": label, "score": score}


def _get_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Heuristic to retrieve the last convolutional layer used for Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            return layer
    raise ValueError("No convolutional layer found in CraNET model for Grad-CAM.")


def cranet_segment(
    image, model_path: Optional[str] = None, threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Run CraNET in weakly supervised segmentation mode using Grad-CAM.

    Returns a dict with:
      - `heatmap`: float32 array in [0, 1] (H, W)
      - `mask`: uint8 binary mask (H, W), where 1 denotes crack regions
    """
    model = load_cranet_model(model_path)
    last_conv = _get_last_conv_layer(model)

    arr = preprocess_image(image)
    arr_batch = np.expand_dims(arr, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(arr_batch, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if predictions.shape[-1] == 1:
            crack_logit = predictions[:, 0]
        else:
            crack_logit = predictions[:, 0]
    grads = tape.gradient(crack_logit, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (arr.shape[0], arr.shape[1])
    )[..., 0]
    heatmap_np = heatmap.numpy().astype(np.float32)

    mask = (heatmap_np >= threshold).astype(np.uint8)
    return {"heatmap": heatmap_np, "mask": mask}


def cranet_answer(
    image, instruction: Optional[str] = None, model_path: Optional[str] = None
) -> str:
    """
    Default CraNET entry point used by WildVision `gen_answers.py`.

    For now this returns a textual crack / noncrack decision so it can be
    compared against general models on crack-detection prompts.
    """
    result = cranet_detect(image, model_path=model_path)
    return result["label"]

