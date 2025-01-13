# utils.py (create this file or add to your existing one)

import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    """
    Custom collate function for object detection.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets
