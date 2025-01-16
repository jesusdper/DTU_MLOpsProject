import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from ultralytics import YOLO  # Import YOLOv8 # type: ignore
import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from pathlib import Path
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path):
        """
        Custom Dataset for YOLO using images and YOLO annotations (text files).

        Args:
            images_dir (Path): Directory containing images.
            labels_dir (Path): Directory containing YOLO annotations (text files).
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = list(images_dir.glob("*.jpg"))  # Assuming images are .jpg
        self.label_files = {
            f.stem: f for f in labels_dir.glob("*.txt")
        }  # Map label file names to text files

        # Debugging lines to ensure images and labels are being loaded correctly
        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.label_files)} label files")
        print(f"Sample image: {self.image_files[0] if self.image_files else 'None'}")
        print(
            f"Sample label: {list(self.label_files.keys())[0] if self.label_files else 'None'}"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files.get(image_file.stem)

        if not label_file:
            raise FileNotFoundError(f"Label file not found for {image_file.name}")

        # Load image
        image = Image.open(image_file).convert("RGB")
        image = image.resize(
            (640, 640)
        )  # Resize to match YOLO input size (adjust as necessary)
        image = (
            torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        )  # Normalize and convert to tensor

        # Load labels (YOLO format: class_id, x_center, y_center, width, height)
        with open(label_file, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        labels = torch.tensor(labels, dtype=torch.float32)  # Convert labels to tensor

        return image, labels


def create_yolo_model(model_path=None, num_classes=80):
    """
    Initializes the YOLO model.

    Args:
        model_path (str): Path to a pre-trained YOLO model. If None, it will load the default model.
        num_classes (int): Number of classes in your dataset.

    Returns:
        model: YOLO model instance.
    """
    # Load YOLOv8 model
    model = YOLO(model_path or "yolov8n.pt")  # Load YOLO Nano model by default

    # Print the model architecture to help identify the correct way to access the final layer
    # print(model)

    # Update the number of classes in the final detection layer (the last layer)
    # model.model[-1].nc = num_classes  # Correct access to final layer

    return model
