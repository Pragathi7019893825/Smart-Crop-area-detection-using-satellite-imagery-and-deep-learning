import math
import os
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from predict import load_segmentation_model, save_prediction_results

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)

model = load_segmentation_model()


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def meters_per_pixel(latitude, zoom):
    """Convert zoom and latitude into meters per pixel."""
    return 156543.03392 * math.cos(math.radians(latitude)) / (2 ** zoom)


def image_area_m2(latitude, zoom, width, height):
    """Estimate the ground area covered by a satellite image."""
    mpp = meters_per_pixel(latitude, zoom)
    return (width * mpp) * (height * mpp)


def _mercator_y(latitude):
    """Return normalized Web Mercator Y for a latitude."""
    clipped = max(-85.05112878, min(85.05112878, latitude))
    return (1.0 - math.asinh(math.tan(math.radians(clipped))) / math.pi) / 2.0


def georeferenced_crop_area_m2(mask, north, south, west, east, crop_class_index):
    """Calculate crop area from a mask aligned with a Web Mercator image."""
    if mask.ndim != 2 or mask.size == 0:
        return 0.0
    if not (-90.0 <= south <= north <= 90.0):
        raise ValueError("Invalid image latitude bounds")
    if not (-180.0 <= west <= east <= 180.0):
        raise ValueError("Invalid image longitude bounds")

    height, width = mask.shape
    earth_radius_m = 6378137.0
    longitude_width = math.radians(east - west)
    top_y = _mercator_y(north)
    bottom_y = _mercator_y(south)
    crop_area = 0.0
    for row in range(height):
        y0 = top_y + (bottom_y - top_y) * row / height
        y1 = top_y + (bottom_y - top_y) * (row + 1) / height
        lat0 = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y0)))
        lat1 = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y1)))
        row_area = earth_radius_m**2 * longitude_width * (math.sin(lat0) - math.sin(lat1))
        crop_area += int(np.count_nonzero(mask[row] == crop_class_index)) * row_area / width
    return crop_area


def build_crop_forecast(crop_percentage, years, annual_growth_rate):
    """Create a transparent scenario estimate from one image measurement."""
    projected_percentage = min(100.0, crop_percentage * ((1 + annual_growth_rate / 100.0) ** years))
    return {
        "years": years,
        "annual_growth_rate": annual_growth_rate,
        "current_percentage": crop_percentage,
        "projected_percentage": projected_percentage,
        "increase": projected_percentage - crop_percentage,
    }


@app.route("/", methods=["GET"])
def index():
    """Render the homepage with upload form and the latest prediction results."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle direct upload and location-based satellite image prediction."""
    source = request.form.get("source", "upload")
    result = None
    location_info = None
    original_url = None
    mask_url = None
    overlay_url = None
    if source in {"map", "location"}:
        try:
            latf = float(request.form.get("latitude", ""))
            lonf = float(request.form.get("longitude", ""))
            zoom = int(request.form.get("zoom", 18))
            width = int(request.form.get("width", 0))
            height = int(request.form.get("height", 0))
            north = float(request.form.get("north", ""))
            south = float(request.form.get("south", ""))
            west = float(request.form.get("west", ""))
            east = float(request.form.get("east", ""))
            if not (-90 <= latf <= 90 and -180 <= lonf <= 180):
                raise ValueError("Coordinates are outside valid ranges")
        except (TypeError, ValueError):
            return render_template("index.html", error="Please select a valid latitude and longitude on the map.")
        location_name = request.form.get("location_name", "").strip()

        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="No map image received. Click on the map and capture first.")

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)

        result = save_prediction_results(str(upload_path), model, RESULT_FOLDER)
        result["is_location"] = True

        try:
            if width > 0 and height > 0:
                crop_area_m2 = georeferenced_crop_area_m2(
                    result.pop("prediction_mask"), north, south, west, east,
                    result["crop_class_index"],
                )
                result["crop_area_hectares"] = crop_area_m2 / 10000.0
                result["crop_area_acres"] = crop_area_m2 / 4046.8564224
            result["location_info"] = location_name or f"{latf:.6f},{lonf:.6f}"
            result["coordinates"] = [lonf, latf]
            result["location_geojson"] = {
                "type": "Point",
                "coordinates": [lonf, latf],
            }
            result["zoom"] = zoom
        except Exception:
            pass
    else:
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please upload a JPG or PNG image.")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Please upload a JPG or PNG image.")

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)
        
        # Get meters per pixel for area calculation if provided
        meters_per_pixel = None
        try:
            mpp = request.form.get("meters_per_pixel", "").strip()
            if mpp:
                meters_per_pixel = float(mpp)
        except (TypeError, ValueError):
            pass
        
        result = save_prediction_results(str(upload_path), model, RESULT_FOLDER, meters_per_pixel=meters_per_pixel)
        result["is_location"] = False

    try:
        forecast_years = min(max(int(request.form.get("forecast_years", 1)), 1), 10)
    except (TypeError, ValueError):
        forecast_years = 1
    annual_growth_rate = 10.0
    result.pop("prediction_mask", None)
    result.pop("crop_class_index", None)
    result["forecast"] = build_crop_forecast(
        result["crop_percentage"], forecast_years, annual_growth_rate
    )

    original_url = url_for("static", filename=f"uploads/{upload_path.name}")
    mask_url = url_for("static", filename=f"results/{result['mask']}")
    overlay_url = url_for("static", filename=f"results/{result['overlay']}")

    return render_template(
        "index.html",
        result=result,
        original_url=original_url,
        mask_url=mask_url,
        overlay_url=overlay_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
