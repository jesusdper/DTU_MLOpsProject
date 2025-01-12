import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO  # Import YOLOv8


class CustomPTDataset(Dataset):
    def __init__(self, images_path, targets_path, transform=None):
        self.images = torch.load(images_path)  # Load images as tensors
        self.targets = torch.load(targets_path)  # Load targets as tensors
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


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
    #print(model)

    # Update the number of classes in the final detection layer (the last layer)
    #model.model[-1].nc = num_classes  # Correct access to final layer

    return model