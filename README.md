# Crop Area Detection from Satellite Imagery

This project detects agriculture land in satellite images using the DeepGlobe Land Cover Classification dataset. It trains a U-Net semantic segmentation model with TensorFlow/Keras and provides a Flask web app to upload images, show crop area masks, and estimate crop percentage.

## Project structure

- `app.py` - Flask web application for uploading images and showing predictions.
- `dataset_utils.py` - Dataset loading, preprocessing, and utility functions.
- `model.py` - U-Net model architecture definition.
- `train.py` - Training script for the segmentation model.
- `predict.py` - Prediction script to run inference on new images.
- `requirements.txt` - Required Python packages.
- `README.md` - Project overview and instructions.
- `class_dict.csv` - Class color palette for the DeepGlobe masks.
- `models/` - Saved model files after training.
- `static/` - Static assets, upload, and results folders.
- `templates/` - HTML template for the Flask app.

## Dataset layout

The dataset folders should contain paired satellite and mask images in this format:

- `train/` - training images and masks
- `valid/` - validation images and masks
- `test/` - optional test images and masks

Files are expected with names like `100694_sat.jpg` for satellite and `100694_mask.png` for mask images.

## How to run in Visual Studio Code

1. Open the workspace folder in VS Code.
2. Create and activate a Python virtual environment:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Run the Flask app:
   ```bash
   python app.py
   ```
6. Open your browser to `http://127.0.0.1:5000`.

## Training and prediction workflow

- `train.py` loads images and masks from `train/` and `valid/`, builds a U-Net, trains it, and saves the model to `models/`.
- `predict.py` loads the saved model and makes predictions for a new image.
- `app.py` starts a web app where users can upload a satellite image and view the detected crop area.

## How the code works

### `dataset_utils.py`
- Reads the class palette from `class_dict.csv`.
- Matches image files ending in `_sat.jpg` with masks ending in `_mask.png`.
- Converts color masks into class index masks and back.
- Loads satellite images and resizes them for training.
- Creates TensorFlow `tf.data.Dataset` pipelines.
- Adds a helper to highlight predicted crop pixels.

### `model.py`
- Defines a U-Net model using convolution, pooling, and transpose convolution layers.
- The model outputs a pixel-wise softmax prediction across classes.

### `train.py`
- Loads datasets, builds the model, compiles it with Adam and sparse categorical crossentropy.
- Uses model checkpointing and early stopping.
- Saves both `.keras` and `.h5` model files.

### `predict.py`
- Loads the saved model.
- Predicts a segmentation mask for a new input image.
- Computes the percentage of pixels classified as agriculture land.
- Saves visual outputs to `static/results/`.

### `app.py`
- Starts a Flask application.
- Accepts image uploads from a web form.
- Runs prediction and returns images for original, mask, and overlay.

## Notes

- The model is trained on 256x256 images.
- The crop area percentage is estimated from mask pixels labeled as `agriculture_land`.
- To improve results, add more training data, augment images, or use a larger model.

## Feature roadmap

### Phase 1: Smarter farm insights
- Add a farm health score based on crop coverage, confidence, and field condition.
- Show a dashboard of area coverage, model confidence, and recommended field actions.
- Improve upload results with better visualization and easier interpretation for farmers.

### Phase 2: Seasonal monitoring and trend analysis
- Allow users to compare field images across weeks or months.
- Show growth trends, crop area changes, and seasonal performance indicators.
- Add a simple chart to visualize how the field evolves over time.

### Phase 3: Decision-support features
- Recommend irrigation, fertiliser, or crop stress checks based on the detected field state.
- Add risk alerts for poor vegetation health, water stress, or unusual land patterns.
- Generate downloadable summary reports for farm managers and agronomists.

### Phase 4: Commercial-ready product features
- Add farmer login and saved field history.
- Support multiple farms, regions, and historical image uploads.
- Include map-based boundary overlays and geo-tagged crop monitoring.
- Integrate weather data, soil moisture estimation, and yield forecasting models.

### Phase 5: Advanced AI improvements
- Train a crop-species classification model instead of only land-cover segmentation.
- Combine remote sensing data with weather and soil features.
- Use larger backbone models and satellite time-series data for better predictions.

This roadmap turns the current project from a simple segmentation demo into a practical agriculture decision-support system.
