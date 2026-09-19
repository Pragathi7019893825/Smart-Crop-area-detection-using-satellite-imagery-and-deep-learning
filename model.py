import tensorflow as tf


def conv_block(inputs, filters):
    """A small convolution block with two Conv2D layers and batch normalization."""
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def encoder_block(inputs, filters):
    """Downsampling block for the U-Net encoder."""
    x = conv_block(inputs, filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, filters):
    """Upsampling block for the U-Net decoder."""
    x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.concatenate([x, skip_features])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=(256, 256, 3), num_classes=7):
    """Build and return a U-Net model for semantic segmentation."""
    inputs = tf.keras.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = tf.keras.Model(inputs, outputs, name="unet")
    return model
