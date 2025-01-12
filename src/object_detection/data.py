import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import typer
import hydra
from omegaconf import DictConfig
from model import YOLOv5  # Replace with your model import
from torchvision import transforms
from utils import collate_fn  # For handling batch formation in DataLoader

# Custom Dataset class for loading VOC2012 data
class VOC2012Dataset(Dataset):
    def __init__(self, images_dir: Path, targets_dir: Path, transform=None):
        self.images = torch.load(images_dir)  # Loaded image tensors
        self.targets = torch.load(targets_dir)  # Loaded target tensors (bboxes + class_ids)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

@hydra.main(config_path="configs", config_name="config.yaml")
def train_model(config: DictConfig) -> None:
    """
    Train the object detection model with the preprocessed VOC2012 data.
    Args:
        config: Configurations, including hyperparameters, model architecture, etc.
    """
    # Set up directories
    processed_dir = Path(config.data.processed_dir)
    output_dir = Path(config.output_dir)
    images_dir = processed_dir / "train_images.pt"
    targets_dir = processed_dir / "train_target.pt"

    # Data transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloaders
    dataset = VOC2012Dataset(images_dir, targets_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    # Initialize the model
    model = YOLOv5()  # Replace with your actual model class
    model.train()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # This depends on your model's requirements
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move data to GPU if available
            images = images.cuda()  # Assuming you have GPU
            targets = [target.cuda() for target in targets]  # Assuming you want to move target to GPU as well

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, targets)  # Modify as per your model's output format and loss function

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                typer.echo(f"Epoch [{epoch}/{config.num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

        # Save model checkpoints
        torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch}.pth")

    # Save the final model
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    typer.echo(f"Training complete. Final model saved to {output_dir}")

if __name__ == "__main__":
    typer.run(train_model)
