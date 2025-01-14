import os
import torch
from pathlib import Path
import logging
import yaml
from omegaconf import DictConfig
import hydra
from ultralytics import YOLO
from model import CustomDataset, create_yolo_model
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@hydra.main(config_path=r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra.
    """
    # Print the configuration
    logger.info(f"Starting training with config: {yaml.dump(cfg)}")

    # Load processed data paths from configuration
    processed_dir = Path(cfg.data.processed_dir).resolve()

    # Define paths to images and labels for train, val, and test
    train_images_dir = processed_dir / "train" / "images"
    train_labels_dir = processed_dir / "train" / "labels"
    val_images_dir = processed_dir / "val" / "images"
    val_labels_dir = processed_dir / "val" / "labels"
    test_images_dir = processed_dir / "test" / "images"
    test_labels_dir = processed_dir / "test" / "labels"

    # Setup the dataset and dataloaders
    train_dataset = CustomDataset(train_images_dir, train_labels_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    logger.info(f"Loaded {len(train_loader.dataset)} training samples.")


    # Prepare dataset config file for YOLO
    data_config = {
        'train': str(train_images_dir),  # Path to training images
        'val': str(val_images_dir),  # Path to validation images
        'test': str(test_images_dir),  # Path to test images
        'nc': cfg.data.num_classes,  # Number of classes
        'names': list(cfg.data.class_names)  # Class names
    }

    # Create YOLO Model
    model = create_yolo_model(cfg.model.pretrained_weights, cfg)

    # Save the YAML config file for YOLO
    data_yaml_path = processed_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)
    logger.info(f"Data configuration saved to {data_yaml_path}")

    # Create a TensorBoard writer for logging
    tb_log_dir = Path(cfg.training.output_dir) / "logs"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Train the model and profile the training loop
   # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True) as prof:
   #     with record_function("model_training"):
    model.train(
                data=str(data_yaml_path),  # Path to dataset config file
                epochs=cfg.training.epochs,  # Number of epochs
                imgsz=cfg.training.img_size,  # Image size
                batch=cfg.training.batch_size,  # Batch size
                name=cfg.training.experiment_name,  # Experiment name
                project=cfg.training.output_dir,  # Output directory for saving results
            )

    # Save profiling results for TensorBoard visualization
    #prof.export_chrome_trace(str(tb_log_dir / "train_profile.json"))  # Save profiling trace in JSON format
    #writer.add_graph(model.model, next(iter(train_loader))[0].unsqueeze(0))  # Log the model graph to TensorBoard

    # Close the TensorBoard writer
    #writer.close()

if __name__ == "__main__":
    main()
