import torch  # type: ignore

def collate_fn(batch):
    """
    Custom collate function for object detection.
    
    Args:
        batch: A list of tuples of form (image, target).

    Returns:
        None
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, targets
