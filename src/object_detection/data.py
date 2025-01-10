import os
from pathlib import Path
import tarfile
import requests
import shutil
import torch
import typer


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


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images by subtracting mean and dividing by standard deviation.
    """
    return (images - images.mean()) / images.std()


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

    # Simulate processing (adjust this based on your actual preprocessing)
    typer.echo(f"Processing {max_samples} samples...")
    # Placeholder logic for now
    torch.save([], processed_dir / "train_images.pt")
    torch.save([], processed_dir / "train_target.pt")
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
