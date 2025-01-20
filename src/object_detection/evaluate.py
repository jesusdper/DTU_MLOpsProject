import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths (adjust as needed)
data_dir = Path(
    "C:/Users/jdiaz/Desktop/DTU_MLOpsProject/data/processed/test/images"
)  # Assuming images are inside 'images' subfolder
output_dir = Path("C:/Users/jdiaz/Desktop/DTU_MLOpsProject/results/inference")
model_dir = Path("C:/Users/jdiaz/Desktop/DTU_MLOpsProject/models/yolov8_voc_test_new2")
model_path = model_dir / "weights/best.pt"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Load the trained YOLO model
logger.info("Loading model...")
model = YOLO(str(model_path))  # Load model using the Ultralytics library
model.eval()  # Set to evaluation mode

# Perform inference on all images in the test set
logger.info(f"Performing inference on images in {data_dir}...")
for image_path in data_dir.glob("*.jpg"):  # You can adjust the file extension if needed
    try:
        logger.info(f"Processing {image_path.name}...")

        # Perform inference on the image
        results = model(image_path)

        # Extract the first result (as it's a list)
        result = results[0]

        # Save the results (image with bounding boxes)
        output_image_path = output_dir / image_path.name
        result.save(str(output_image_path))  # Save the image with bounding boxes

        # Optionally, display the result
        result.show()  # Will display the image with bounding boxes

        logger.info(f"Saved output to {output_image_path}")

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")

logger.info("Inference completed.")
