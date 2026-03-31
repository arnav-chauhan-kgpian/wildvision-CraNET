import tensorflow as tf


def _conv_block(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _cbam_block(x, reduction_ratio=8):
    """
    Lightweight CBAM-style attention block.

    This is a simplified implementation intended to mirror the paper's use
    of CBAM without exactly reproducing the research code.
    """
    channel_axis = -1
    channels = x.shape[channel_axis]

    # Channel attention
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    shared_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(channels // reduction_ratio, activation="relu"),
            tf.keras.layers.Dense(channels),
        ]
    )
    avg_out = shared_mlp(avg_pool)
    max_out = shared_mlp(max_pool)
    channel_attn = tf.keras.layers.Activation("sigmoid")(avg_out + max_out)
    channel_attn = tf.keras.layers.Reshape((1, 1, channels))(channel_attn)
    x = tf.keras.layers.Multiply()([x, channel_attn])

    # Spatial attention
    avg_pool_spatial = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=channel_axis, keepdims=True))(x)
    max_pool_spatial = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=channel_axis, keepdims=True))(x)
    concat = tf.keras.layers.Concatenate(axis=channel_axis)([avg_pool_spatial, max_pool_spatial])
    spatial_attn = tf.keras.layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    x = tf.keras.layers.Multiply()([x, spatial_attn])

    return x


def build_cranet_light(input_shape=(224, 224, 3), num_classes=2) -> tf.keras.Model:
    """
    Build a lightweight CraNET-style architecture:
    - 6 convolutional blocks
    - CBAM modules on later blocks, as in the paper's high-level description

    This model is designed for:
      - Binary crack detection (crack vs noncrack)
      - Grad-CAM-based weakly supervised segmentation from the last conv layer
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = _conv_block(inputs, 32)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = _conv_block(x, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = _conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = _conv_block(x, 128)
    x = _cbam_block(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = _conv_block(x, 256)
    x = _cbam_block(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = _conv_block(x, 256, kernel_size=3)
    x = _cbam_block(x)
    last_conv = x  # last conv feature map for Grad-CAM

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(last_conv)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cranet_light")
    return model


# For backward compatibility with existing wrapper import style:
cranet_light = build_cranet_light()

