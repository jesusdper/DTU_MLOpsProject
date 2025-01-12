import os
import torch
from pathlib import Path
import logging
import yaml
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from model import create_yolo_model, CustomPTDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path=r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra.
    """
    # Print the configuration
    print(f"Starting training with config: {yaml.dump(cfg)}")
    logger.info(f"Starting training with config: {cfg}")

    # Load preprocessed data paths from the configuration
    processed_dir = Path(cfg.data.processed_dir).resolve()
    train_images_path = processed_dir / "train_images.pt"
    train_targets_path = processed_dir / "train_target.pt"

    # Check if the .pt files exist, if not, modify them using CustomPTDataset and save them
    if not train_images_path.exists() or not train_targets_path.exists():
        logger.info(f"Processing and saving {train_images_path} and {train_targets_path}")

        # You would need to define the images and targets, below is a sample modification
        # Assuming images and targets are in some form of raw data (e.g., numpy arrays or lists)
        images, targets = load_raw_data()  # Function to load your raw data

        # Save the raw data as tensors using torch.save
        torch.save(images, train_images_path)
        torch.save(targets, train_targets_path)
        logger.info(f"Processed data saved to {train_images_path} and {train_targets_path}")

    # Setup the dataset and dataloaders
    train_dataset = CustomPTDataset(train_images_path, train_targets_path)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    logger.info(f"Loaded {len(train_loader.dataset)} training samples.")

    # Create YOLO model
    model = create_yolo_model(model_path=cfg.model.pretrained_weights, num_classes=cfg.data.num_classes)

    # Prepare dataset config file for YOLO
    data_config = {
        'train': str(processed_dir),  # Path to training images directory
        'val': str(processed_dir),  # Path to validation images directory
        'nc': cfg.data.num_classes,  # Number of classes
        'names': list(cfg.data.class_names)  # Ensure this is a simple Python list
    }

    # Save the YAML config file for YOLO
    data_yaml_path = r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs\data.yaml"  # processed_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)
    logger.info(f"Data configuration saved to {data_yaml_path}")

    # Train the model
    model.train(
        data=str(data_yaml_path),  # Path to dataset config file
        epochs=cfg.training.epochs,  # Number of epochs
        imgsz=cfg.training.img_size,  # Image size
        batch=cfg.training.batch_size,  # Batch size
        name=cfg.training.experiment_name,  # Experiment name
        project=cfg.training.output_dir,  # Output directory for saving results
        # Disable validation since data.yaml already includes the train and val paths
    )


if __name__ == "__main__":
    main()
