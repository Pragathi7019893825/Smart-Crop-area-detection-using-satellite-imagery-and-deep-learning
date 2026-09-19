from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from dataset_utils import load_image, mask_to_rgb, CLASS_NAMES, CLASS_COLORS, highlight_crop_area

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_area_unet.keras"


def load_segmentation_model():
    """Load the trained U-Net model from disk."""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_crop_mask(image_path, model, image_size=(256, 256)):
    """Predict the class mask for a new satellite image."""
    image = load_image(image_path, image_size=image_size)
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    # The saved model ends in softmax; avoid applying softmax twice.
    probabilities = prediction if np.allclose(np.sum(prediction, axis=-1), 1.0, atol=1e-3) else tf.nn.softmax(prediction, axis=-1).numpy()
    mask = np.argmax(probabilities, axis=-1).astype(np.uint8)
    return mask, image, probabilities


def format_class_name(class_name):
    """Convert a model class name into a readable display label."""
    return class_name.replace("_", " ").title()


def get_dominant_class_name(mask):
    """Return the most common predicted class in the mask."""
    if mask.size == 0:
        return "unknown"
    class_counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
    dominant_index = int(np.argmax(class_counts))
    return CLASS_NAMES[dominant_index]


def get_detected_crop_label(mask):
    """Return a user-friendly crop or land-use label for the dominant class."""
    dominant_class_name = get_dominant_class_name(mask)
    crop_labels = {
        "agriculture_land": "Agricultural crop field",
        "forest_land": "Forest land",
        "rangeland": "Pasture / grazing land",
        "urban_land": "Urban / built-up area",
        "water": "Water body",
        "barren_land": "Barren land",
        "unknown": "Unknown area",
    }
    return crop_labels.get(dominant_class_name, format_class_name(dominant_class_name))


def get_crop_boundary(mask, crop_class_name="agriculture_land"):
    """Return the bounding box of the detected agricultural area in image coordinates."""
    if crop_class_name not in CLASS_NAMES:
        raise ValueError(f"Class '{crop_class_name}' not found in class list.")

    crop_index = CLASS_NAMES.index(crop_class_name)
    crop_pixels = np.argwhere(mask == crop_index)
    if crop_pixels.size == 0:
        return None

    rows = crop_pixels[:, 0]
    cols = crop_pixels[:, 1]
    return {
        "x_min": int(cols.min()),
        "x_max": int(cols.max()),
        "y_min": int(rows.min()),
        "y_max": int(rows.max()),
        "width": int(cols.max() - cols.min() + 1),
        "height": int(rows.max() - rows.min() + 1),
    }


def get_health_score(crop_percentage, confidence_score, image=None, mask=None):
    """Produce a field-health score using crop coverage, vegetation intensity, and image texture."""
    if image is not None and mask is not None:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim == 3 and image_array.shape[-1] >= 3:
            red = image_array[..., 0]
            green = image_array[..., 1]
            grayscale = np.mean(image_array[..., :3], axis=-1)
            vegetation_index = np.mean(np.clip((green - red) / (green + red + 1e-6), 0.0, 1.0))
            texture_index = np.clip(np.std(grayscale) / 60.0, 0.0, 1.0)
            crop_mask_index = CLASS_NAMES.index("agriculture_land") if "agriculture_land" in CLASS_NAMES else 0
            crop_ratio = float(np.mean(mask == crop_mask_index)) if mask.size else 0.0
            score = (crop_ratio * 0.55) + (vegetation_index * 0.25) + (texture_index * 0.20)
            score = (score * 0.8) + ((confidence_score / 100.0) * 0.2)
            return max(0.0, min(100.0, float(score * 100.0)))

    score = (crop_percentage * 0.7) + (confidence_score * 0.3)
    return max(0.0, min(100.0, float(score)))


def get_health_status(health_score):
    """Map a numeric health score to a user-friendly field status."""
    if health_score >= 75:
        return "Excellent"
    if health_score >= 55:
        return "Good"
    if health_score >= 35:
        return "Fair"
    return "Needs attention"


def get_health_recommendation(health_score, dominant_class_name):
    """Return an operational recommendation based on the field condition."""
    if dominant_class_name != "agriculture_land":
        return "Land type is not agricultural. Review the field classification before planning crop operations."
    if health_score >= 75:
        return "Healthy crop coverage detected. Continue regular irrigation, fertiliser checks, and crop monitoring."
    if health_score >= 55:
        return "Moderate crop health. Review irrigation timing and inspect for stressed vegetation or nutrient imbalance."
    return "Field conditions need attention. Check irrigation, soil moisture, and plant stress before continuing operations."


def get_field_recommendations(crop_percentage, confidence_score, dominant_class_name, health_score):
    """Generate concise actionable recommendations for the detected field."""
    recommendations = []

    if dominant_class_name == "agriculture_land":
        if crop_percentage >= 60:
            recommendations.append("Field expansion possible")
        else:
            recommendations.append("Monitor field boundary for growth potential")

        if health_score < 55:
            recommendations.append("Irrigation recommended")
            recommendations.append("Soil moisture appears low")
        elif health_score >= 75:
            recommendations.append("Field remains in a healthy growth state")

        if confidence_score < 80:
            recommendations.append("Monitor for disease risk")
        elif health_score < 65:
            recommendations.append("Check plant stress and nutrient deficiency")

        if not recommendations:
            recommendations.append("Continue routine crop monitoring")
    else:
        recommendations.append("This area is not classified as agricultural land")
        recommendations.append("Review classification confidence before field planning")

    return recommendations[:4]


def calculate_crop_percentage(mask, crop_class_name="agriculture_land"):
    """Estimate the percent of pixels predicted as the crop class."""
    if crop_class_name not in CLASS_NAMES:
        raise ValueError(f"Class '{crop_class_name}' not found in class list.")
    crop_index = CLASS_NAMES.index(crop_class_name)
    crop_pixels = np.sum(mask == crop_index)
    total_pixels = mask.size
    return crop_pixels / total_pixels * 100.0


def calculate_confidence_score(probabilities, crop_class_name="agriculture_land"):
    """Estimate a confidence score for the crop class predictions."""
    if crop_class_name not in CLASS_NAMES:
        raise ValueError(f"Class '{crop_class_name}' not found in class list.")
    return float(np.mean(np.max(probabilities, axis=-1)) * 100.0)


def calculate_crop_area_m2(mask, meters_per_pixel, crop_class_name="agriculture_land"):
    """Calculate crop area in square meters from mask and scale."""
    if crop_class_name not in CLASS_NAMES:
        raise ValueError(f"Class '{crop_class_name}' not found in class list.")
    if meters_per_pixel is None or meters_per_pixel <= 0:
        return None
    
    crop_index = CLASS_NAMES.index(crop_class_name)
    crop_pixels = np.sum(mask == crop_index)
    area_per_pixel_m2 = meters_per_pixel ** 2
    return float(crop_pixels * area_per_pixel_m2)


def save_prediction_results(image_path, model, output_dir, image_size=(256, 256), meters_per_pixel=None):
    """Create and save visual prediction outputs to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask, image, probabilities = predict_crop_mask(image_path, model, image_size=image_size)
    crop_percentage = calculate_crop_percentage(mask)
    confidence_score = calculate_confidence_score(probabilities)
    dominant_class_name = get_dominant_class_name(mask)
    detected_crop_name = format_class_name(dominant_class_name)
    detected_crop_label = get_detected_crop_label(mask)
    health_score = get_health_score(crop_percentage, confidence_score, image, mask)
    health_status = get_health_status(health_score)
    health_recommendation = get_health_recommendation(health_score, dominant_class_name)
    field_recommendations = get_field_recommendations(crop_percentage, confidence_score, dominant_class_name, health_score)
    field_boundary = get_crop_boundary(mask)

    original = Image.open(image_path).convert("RGB").resize(image_size, Image.BILINEAR)

    mask_rgb = mask_to_rgb(mask, CLASS_COLORS)
    mask_image = Image.fromarray(mask_rgb)
    highlighted = highlight_crop_area(original, mask, CLASS_NAMES.index("agriculture_land"))

    base_name = Path(image_path).stem
    original_path = output_dir / f"{base_name}_original.jpg"
    mask_path = output_dir / f"{base_name}_mask.png"
    overlay_path = output_dir / f"{base_name}_overlay.jpg"

    original.save(original_path)
    mask_image.save(mask_path)
    highlighted.save(overlay_path)

    # Calculate crop area in square meters and hectares if scale is provided
    crop_area_m2 = calculate_crop_area_m2(mask, meters_per_pixel)
    crop_area_hectares = None
    crop_area_acres = None
    if crop_area_m2 is not None:
        crop_area_hectares = crop_area_m2 / 10000.0
        crop_area_acres = crop_area_m2 / 4046.8564224

    return {
        "original": str(original_path.name),
        "mask": str(mask_path.name),
        "overlay": str(overlay_path.name),
        "crop_percentage": float(crop_percentage),
        "confidence_score": float(confidence_score),
        "dominant_class_name": dominant_class_name,
        "detected_crop_name": detected_crop_name,
        "detected_crop_label": detected_crop_label,
        "health_score": float(health_score),
        "health_status": health_status,
        "health_recommendation": health_recommendation,
        "field_recommendations": field_recommendations,
        "field_boundary": field_boundary,
        "crop_class_index": CLASS_NAMES.index("agriculture_land"),
        "prediction_mask": mask,
        "crop_area_m2": crop_area_m2,
        "crop_area_hectares": crop_area_hectares,
        "crop_area_acres": crop_area_acres,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict crop area from a satellite image.")
    parser.add_argument("image_path", help="Path to the input satellite image.")
    parser.add_argument("--output", default="static/results", help="Folder to save prediction images.")
    args = parser.parse_args()

    model = load_segmentation_model()
    result = save_prediction_results(args.image_path, model, args.output)
    print(f"Crop area: {result['crop_percentage']:.2f}%")
    print(f"Detected crop / land type: {result['detected_crop_label']}")
    print(f"Confidence: {result['confidence_score']:.2f}%")
    print("Saved:")
    print(f"  Original image: {result['original']}")
    print(f"  Predicted mask: {result['mask']}")
    print(f"  Overlay image: {result['overlay']}")


if __name__ == "__main__":
    main()
