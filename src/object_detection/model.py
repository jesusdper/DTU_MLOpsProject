import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO  # Import YOLOv8
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple


class CustomDataset(Dataset):


        
    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        """
        Custom Dataset for YOLO using images and YOLO annotations (text files).

        Attributes:
            image_files (List[Path]): List of image file paths.
            label_files (Dict[str, Path]): Dictionary mapping image file stems to label file paths.

        Methods:
            __len__() -> int:
                Returns the number of images in the dataset.

            __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                Returns the image and corresponding labels at the specified index.

                    idx (int): Index of the image and label to retrieve.

                Raises:
                    FileNotFoundError: If the label file corresponding to the image is not found.

                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the labels tensor.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = list(images_dir.glob("*.jpg"))  # Assuming images are .jpg
        self.label_files = {f.stem: f for f in labels_dir.glob("*.txt")}  # Map label file names to text files

        # Debugging lines to ensure images and labels are being loaded correctly
        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.label_files)} label files")
        print(f"Sample image: {self.image_files[0] if self.image_files else 'None'}")
        print(f"Sample label: {list(self.label_files.keys())[0] if self.label_files else 'None'}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.image_files[idx]
        label_file = self.label_files.get(image_file.stem)

        if not label_file:
            raise FileNotFoundError(f"Label file not found for {image_file.name}")

        # Load image
        image = Image.open(image_file).convert("RGB")
        image = image.resize((640, 640))  # Resize to match YOLO input size (adjust as necessary)
        image = (
            torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        )  # Normalize and convert to tensor

        # Load labels (YOLO format: class_id, x_center, y_center, width, height)
        with open(label_file, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        labels = torch.tensor(labels, dtype=torch.float32)  # Convert labels to tensor

        return image, labels


def create_yolo_model(pretrained_weights: str, cfg: dict) -> YOLO:
    """
    Initialize the YOLO model.
    """
    model = YOLO(pretrained_weights)
    model.cfg = cfg  # Attach configuration for easier access
    return model
