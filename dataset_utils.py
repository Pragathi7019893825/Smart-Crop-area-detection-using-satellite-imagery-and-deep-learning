from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
CLASS_CSV = BASE_DIR / "class_dict.csv"


def load_class_colors():
    """Read the class color definitions from class_dict.csv."""
    df = pd.read_csv(CLASS_CSV)
    names = df["name"].tolist()
    colors = [tuple(row) for row in df[["r", "g", "b"]].values]
    return names, colors


CLASS_NAMES, CLASS_COLORS = load_class_colors()
NUM_CLASSES = len(CLASS_COLORS)


def get_image_mask_paths(folder):
    """Return sorted pairs of satellite images and their matching mask paths."""
    folder = Path(folder)
    all_images = sorted(folder.glob("*_sat.jpg"))
    pairs = []
    for image_path in all_images:
        mask_path = folder / image_path.name.replace("_sat.jpg", "_mask.png")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def rgb_to_mask(rgb, colors):
    """Convert a color mask image into a 2D array of class indices."""
    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    for index, color in enumerate(colors):
        matches = np.all(rgb == color, axis=-1)
        mask[matches] = index
    return mask


def mask_to_rgb(mask, colors):
    """Convert a numeric class mask back to a color image for display."""
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        rgb[mask == index] = color
    return rgb


def load_image(image_path, mask_path=None, image_size=(256, 256)):
    """Load and resize a satellite image and optionally the associated mask."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, resample=Image.BILINEAR)
    image = np.array(image, dtype=np.float32) / 255.0

    if mask_path is None:
        return image

    mask_image = Image.open(mask_path).convert("RGB")
    mask_image = mask_image.resize(image_size, resample=Image.NEAREST)
    mask_array = np.array(mask_image, dtype=np.uint8)
    mask = rgb_to_mask(mask_array, CLASS_COLORS)
    return image, mask


def _load_sample(image_path, mask_path, image_size):
    """Helper for tf.data to read one image-mask pair from disk."""
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")
    image, mask = load_image(image_path, mask_path, image_size)
    return image.astype(np.float32), mask.astype(np.uint8)


def preprocess_sample(image_path, mask_path, image_size):
    """Create a TensorFlow-friendly sample from file paths."""
    image, mask = tf.py_function(
        _load_sample,
        [image_path, mask_path, image_size],
        [tf.float32, tf.uint8],
    )
    image.set_shape([image_size[0], image_size[1], 3])
    mask.set_shape([image_size[0], image_size[1]])
    return image, mask


def create_dataset(folder, image_size=(256, 256), batch_size=8, shuffle=True):
    """Build a train or validation dataset from a folder of image-mask pairs."""
    pairs = get_image_mask_paths(folder)
    image_paths = [str(pair[0]) for pair in pairs]
    mask_paths = [str(pair[1]) for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda img, mask: preprocess_sample(img, mask, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def highlight_crop_area(original_image, mask, crop_index, alpha=0.4):
    """Overlay the predicted crop pixels on the original image."""
    original = np.array(original_image.convert("RGB"), dtype=np.uint8)
    overlay = original.copy()
    highlight = np.zeros_like(original)
    highlight[mask == crop_index] = [0, 255, 0]
    result = np.uint8(original * (1 - alpha) + highlight * alpha)
    return Image.fromarray(result)
