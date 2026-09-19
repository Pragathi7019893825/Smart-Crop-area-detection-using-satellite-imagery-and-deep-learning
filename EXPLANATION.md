# Project Explanation

This file explains every project file in simple, beginner-friendly language.

## app.py
- Imports the Flask framework and tools to save uploaded files.
- Defines where uploads and prediction results are stored.
- Loads the trained model once so predictions are fast.
- Defines a helper function to check allowed file types.
- Shows the homepage when users visit `/`.
- Accepts file uploads at `/predict`, saves the file, runs prediction, and returns the images and crop percentage.
- Starts the app when you run `python app.py`.

## dataset_utils.py
- Reads class colors from `class_dict.csv` so the code knows which color means agriculture land.
- Finds matched satellite and mask files in a given folder.
- Converts color masks to numeric labels that the model can learn.
- Converts numeric labels back to a color image for display.
- Loads and resizes satellite images and masks.
- Builds a TensorFlow dataset pipeline for training and validation.
- Adds an overlay helper to highlight predicted crop areas.

## model.py
- Defines a small U-Net model, a standard neural network for segmentation.
- Uses convolution blocks and pooling to learn from images.
- Uses upsampling blocks to create a mask output the same size as the input.
- The final layer predicts one class per pixel with softmax.

## train.py
- Loads training and validation data.
- Builds and compiles the U-Net model.
- Trains with early stopping and saves the best model.
- Saves the model in both `.keras` and `.h5` formats.

## predict.py
- Loads the saved model and prepares a new image.
- Runs inference and converts the model output into a mask.
- Calculates the crop area percentage from the predicted mask.
- Saves the original image, the predicted mask, and the overlay image.
- Allows command line usage with `python predict.py <image_path>`.

## templates/index.html
- Provides the user interface for uploading images.
- Displays the original satellite image, predicted mask, overlay, and crop percentage.
- Uses Flask template tags to fill in results after prediction.

## static/styles.css
- Styles the webpage and controls with modern borders and spacing.
- Makes the upload form and results look clean.
- Highlights messages and result cards for readability.

## static/scripts.js
- Updates the upload label when the user chooses a file.
- Makes the page easier to use by showing the selected file name.

## requirements.txt
- Lists the Python packages needed to run the project.
- Includes Flask, TensorFlow, NumPy, Pandas, Pillow, and Werkzeug.

## class_dict.csv
- Defines the colors used in each mask class.
- The project uses these colors to read and display label images.

## .vscode/launch.json
- Lets VS Code launch the Flask app directly in the debugger.
- Uses `app.py` as the startup file.

## .vscode/tasks.json
- Defines tasks for installing packages, training the model, and running the app.
- Makes it easy to run common commands from VS Code.

## README.md
- Provides project overview and instructions.
- Explains the folder structure and how to run the code.
- Includes notes about model training and the dataset format.
