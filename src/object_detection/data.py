import xml.etree.ElementTree as ET
from pathlib import Path
import typer
from PIL import Image
import numpy as np
import os
import logging
from random import shuffle
import urllib.request
import tarfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VOC 2012 class labels (20 classes)
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "diningtable",
]

# Create a mapping of class names to class indices
CLASS_TO_ID = {class_name: idx for idx, class_name in enumerate(VOC_CLASSES)}


def download_voc_dataset(save_dir: Path):
    """Download and extract the PASCAL VOC 2012 dataset."""
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    save_dir = save_dir.resolve()
    os.makedirs(save_dir, exist_ok=True)
    dataset_tar_path = save_dir / "VOC2012.tar"

    if not dataset_tar_path.exists():
        logger.info("Downloading PASCAL VOC 2012 dataset...")
        urllib.request.urlretrieve(url, dataset_tar_path)
        logger.info("Download completed.")
    else:
        logger.info("PASCAL VOC 2012 dataset already downloaded.")

    logger.info("Extracting dataset...")
    with tarfile.open(dataset_tar_path) as tar:
        tar.extractall(path=save_dir)
    logger.info("Dataset extracted.")


def convert_voc_to_yolo(xml_path: Path, img_width: int, img_height: int):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_TO_ID:
            logger.warning(f"Unknown class '{class_name}' in {xml_path}. Skipping object...")
            continue

        class_id = CLASS_TO_ID[class_name]
        bndbox = obj.find("bndbox")
        x_min = int(bndbox.find("xmin").text)
        y_min = int(bndbox.find("ymin").text)
        x_max = int(bndbox.find("xmax").text)
        y_max = int(bndbox.find("ymax").text)

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_annotations.append([class_id, x_center, y_center, width, height])

    return yolo_annotations


def preprocess_data(raw_dir: Path, processed_dir: Path, splits: dict, image_size: tuple = (256, 256)):
    """
    Preprocesses raw image and annotation data for object detection.

    Args:
        raw_dir (Path): Path to the directory containing raw data.
        processed_dir (Path): Path to the directory where processed data will be saved.
        splits (dict): Dictionary specifying the dataset splits and their respective sample counts.
                       Example: {'train': 1000, 'val': 200, 'test': 200}
        image_size (tuple, optional): Desired size of the output images. Defaults to (256, 256).

    Raises:
        Exception: If there is an error processing an image or its corresponding annotation file.

    Notes:
        - The function expects the raw data to be in Pascal VOC format.
        - The processed data will be saved in YOLO format.
        - The function creates necessary directories for each split (train, val, test) and saves the processed images and labels in respective directories.
        - If an annotation file is not found for an image, the image is skipped.
        - Logs information, warnings, and errors during the processing.

    Example:
        preprocess_data(Path('/path/to/raw'), Path('/path/to/processed'), {'train': 1000, 'val': 200, 'test': 200})
    """
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    os.makedirs(processed_dir, exist_ok=True)

    images_dir = raw_dir / "VOCdevkit/VOC2012/JPEGImages"
    annotations_dir = raw_dir / "VOCdevkit/VOC2012/Annotations"

    all_files = list(images_dir.iterdir())
    shuffle(all_files)

    split_names = list(splits.keys())
    split_dirs = {}

    for split in split_names:
        split_dirs[split] = {"images": processed_dir / split / "images", "labels": processed_dir / split / "labels"}
        os.makedirs(split_dirs[split]["images"], exist_ok=True)
        os.makedirs(split_dirs[split]["labels"], exist_ok=True)

    start_idx = 0
    for split, count in splits.items():
        logger.info(f"Processing {count} samples for {split} set...")
        for i, image_file in enumerate(all_files[start_idx : start_idx + count]):
            annotation_file = annotations_dir / f"{image_file.stem}.xml"

            if not annotation_file.exists():
                logger.warning(f"Annotation file not found for {image_file}. Skipping...")
                continue

            try:
                with Image.open(image_file) as img:
                    img = img.convert("RGB")
                    img = img.resize(image_size)
                    img.save(split_dirs[split]["images"] / f"{image_file.stem}.jpg")

                yolo_annotations = convert_voc_to_yolo(annotation_file, image_size[1], image_size[0])

                with open(split_dirs[split]["labels"] / f"{image_file.stem}.txt", "w") as label_file:
                    for annotation in yolo_annotations:
                        label_file.write(" ".join(map(str, annotation)) + "\n")

            except Exception as e:
                logger.error(f"Error processing {image_file} or {annotation_file}: {e}")
                continue

        start_idx += count

    logger.info(f"Processed data saved to {processed_dir}")


def load_data():
    raw_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\data\raw")
    processed_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\data\processed")
    splits = {"train": 600, "val": 100, "test": 50}
    download_voc_dataset(raw_dir)
    preprocess_data(raw_dir, processed_dir, splits)


if __name__ == "__main__":
    typer.run(load_data)
