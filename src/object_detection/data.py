import os
from pathlib import Path
import tarfile
import requests
import shutil
import torch
import typer
from PIL import Image
import xml.etree.ElementTree as ET


def download_dataset(url: str, output_dir: Path) -> None:
    """
    Download and extract the dataset.
    Args:
        url (str): URL of the dataset tar file.
        output_dir (Path): Directory where raw data will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "VOC2012.tar"
    extracted_dir = output_dir / "VOCdevkit"

    if not tar_path.exists():
        typer.echo(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        typer.echo(f"Downloaded dataset to {tar_path}")
    else:
        typer.echo(f"Dataset already exists at {tar_path}")

    # Extract the tar file
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)
    typer.echo(f"Extracted dataset to {output_dir}")

    # Move VOCdevkit to raw data directory if needed
    if extracted_dir.exists() and extracted_dir.parent != output_dir:
        final_dest = output_dir / "VOCdevkit"
        shutil.move(str(extracted_dir), str(final_dest))
        typer.echo(f"Moved extracted data to {final_dest}")


def parse_annotation(annotation_path: Path, image_width: int, image_height: int):
    """
    Parse a VOC XML annotation file to extract bounding boxes and class labels.
    Args:
        annotation_path (Path): Path to the annotation file.
        image_width (int): Width of the corresponding image.
        image_height (int): Height of the corresponding image.

    Returns:
        list: A list of dictionaries with 'bbox' and 'label' keys.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text) / image_width
        ymin = int(bbox.find("ymin").text) / image_height
        xmax = int(bbox.find("xmax").text) / image_width
        ymax = int(bbox.find("ymax").text) / image_height
        objects.append({"bbox": [xmin, ymin, xmax, ymax], "label": label})
    return objects


from PIL import Image
import numpy as np
import torch

def preprocess_data(
    raw_dir: Path,
    processed_dir: Path,
    max_samples: int = 100,
) -> None:
    """
    Preprocess raw data and save it to the processed directory.
    Args:
        raw_dir (Path): Directory containing raw data.
        processed_dir (Path): Directory where processed data will be saved.
        max_samples (int): Maximum number of samples to process.
    """
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    os.makedirs(processed_dir, exist_ok=True)

    # Define paths for images and annotations
    images_dir = raw_dir / "VOCdevkit/VOC2012/JPEGImages"
    annotations_dir = raw_dir / "VOCdevkit/VOC2012/Annotations"

    typer.echo(f"Images directory: {images_dir}")
    typer.echo(f"Annotations directory: {annotations_dir}")

    images = []
    targets = []

    typer.echo(f"Processing up to {max_samples} samples...")

    for i, image_file in enumerate(images_dir.iterdir()):
        if i >= max_samples:
            break

        annotation_file = annotations_dir / f"{image_file.stem}.xml"

        if not annotation_file.exists():
            continue

        # Load and process the image
        with Image.open(image_file) as img:
            img = img.convert("RGB")  # Ensure RGB format
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            image_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
            images.append(image_tensor)

        # Placeholder for processing targets (to be replaced with actual annotation parsing)
        targets.append(torch.tensor([]))  # Replace with actual parsing logic

    # Save processed data
    torch.save(images, processed_dir / "train_images.pt")
    torch.save(targets, processed_dir / "train_target.pt")
    typer.echo(f"Processed training data saved to {processed_dir}")



def load_data() -> None:
    """
    Download, preprocess, and save data.
    """
    # Dataset download URL (example for VOC 2012)
    dataset_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    raw_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\data\raw")
    processed_dir = Path(r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\data\processed")

    # Download dataset
    download_dataset(dataset_url, raw_dir)

    # Preprocess data
    preprocess_data(raw_dir, processed_dir, max_samples=100)


if __name__ == "__main__":
    typer.run(load_data)
