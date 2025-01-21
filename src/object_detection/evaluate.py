import os
"""
This script performs object detection inference on a set of test images using a pre-trained YOLO model from the Ultralytics library.

Modules:
    os: Provides a way of using operating system dependent functionality.
    pathlib.Path: Offers classes representing filesystem paths with semantics appropriate for different operating systems.
    ultralytics.YOLO: Ultralytics library for YOLO model loading and inference.
    logging: Provides a way to configure and use loggers.
    typing.List: Provides support for type hints.

Logging:
    Configures logging to display information and error messages.

Paths:
    data_dir (Path): Path to the directory containing test images.
    output_dir (Path): Path to save inference results.
    model_dir (Path): Path to the model directory.
    model_path (Path): Path to the trained model weights.

Functionality:
    - Ensures the output directory exists.
    - Loads the trained YOLO model and sets it to evaluation mode.
    - Iterates over all .jpg images in the test directory.
    - Performs inference on each image and saves the results (image with bounding boxes) to the output directory.
    - Optionally displays the result.
    - Logs the progress and any errors encountered during processing.
"""
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths (adjust as needed)
data_dir: Path = Path(
    "C:/Users/jdiaz/Desktop/DTU_MLOpsProject/data/processed/test/images"
)  # Path to the directory containing test images
output_dir: Path = Path("C:/Users/jdiaz/Desktop/DTU_MLOpsProject/results/inference")  # Path to save inference results
model_dir: Path = Path("C:/Users/jdiaz/Desktop/DTU_MLOpsProject/models/yolov8_voc_test_new2")  # Path to the model directory
model_path: Path = model_dir / "weights/best.pt"  # Path to the trained model weights

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Load the trained YOLO model
logger.info("Loading model...")
model: YOLO = YOLO(str(model_path))  # Load model using the Ultralytics library
model.eval()  # Set to evaluation mode

# Perform inference on all images in the test set
logger.info(f"Performing inference on images in {data_dir}...")
for image_path in data_dir.glob("*.jpg"):  # Iterate over all .jpg images in the test directory
    try:
        logger.info(f"Processing {image_path.name}...")

        # Perform inference on the image
        results: List = model(image_path)

        # Extract the first result (as it's a list)
        result = results[0]

        # Save the results (image with bounding boxes)
        output_image_path: Path = output_dir / image_path.name
        result.save(str(output_image_path))  # Save the image with bounding boxes

        # Optionally, display the result
        result.show()  # Will display the image with bounding boxes

        logger.info(f"Saved output to {output_image_path}")

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")

logger.info("Inference completed.")
