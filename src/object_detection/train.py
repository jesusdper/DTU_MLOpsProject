import os
import torch
from pathlib import Path
import logging
import yaml
from omegaconf import DictConfig
import hydra
from ultralytics import YOLO
from model import CustomDataset, create_yolo_model
from model_registry_helper import upload_model
from torch.utils.tensorboard import SummaryWriter
import wandb  # For W&B integration
import shutil
import torch.profiler

wandb.login(key="b97463597a9b7425acac3f6390c6ec7515ba2585")  # Ensure your WandB login key is set here

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the callback for frozen layers
# Callback to log metrics to W&B
def wandb_callback(metrics):
    """Log metrics to W&B."""
    wandb.log(metrics)

# Callback to freeze specific layers in eval mode
def put_in_eval_mode(trainer):
    """Freeze the layers in eval mode."""
    n_layers = trainer.args.freeze
    if not isinstance(n_layers, int):
        return
    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split(".")[1]) < n_layers:
            module.eval()
            module.track_running_stats = False

@hydra.main(config_path=r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra with W&B tracking.
    """
    # Initialize W&B with the sweep configuration
    wandb.init(
        project=cfg.wandb.project_name,
        name=cfg.training.experiment_name,
        config=dict(cfg),  # Use the configuration from WandB
        dir=str(Path(cfg.training.output_dir) / "logs"),
        resume="allow",
    )

    # Load processed data paths from configuration
    processed_dir = Path(cfg.data.processed_dir).resolve()

    train_images_dir = processed_dir / "train" / "images"
    train_labels_dir = processed_dir / "train" / "labels"
    val_images_dir = processed_dir / "val" / "images"
    val_labels_dir = processed_dir / "val" / "labels"

    # Setup the dataset and dataloaders
    train_dataset = CustomDataset(train_images_dir, train_labels_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    logger.info(f"Loaded {len(train_loader.dataset)} training samples.")

    # Prepare dataset config file for YOLO
    data_config = {
        'train': str(train_images_dir),
        'val': str(val_images_dir),
        'nc': cfg.data.num_classes,
        'names': list(cfg.data.class_names)
    }

    model = create_yolo_model(cfg.model.pretrained_weights, cfg)

    data_yaml_path = processed_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)
    logger.info(f"Data configuration saved to {data_yaml_path}")

    # Create a TensorBoard writer
    tb_log_dir = Path(cfg.training.output_dir) / "logs"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Prepare the YOLO training parameters
    train_params = {
        'data': str(data_yaml_path),  # Dataset config file
        'epochs': wandb.config.epochs,  # Use WandB sweep configuration for epochs
        'imgsz': cfg.training.img_size,  # Image size
        'batch': wandb.config.batch_size,  # Use WandB sweep configuration for batch size
        'name': cfg.training.experiment_name,  # Experiment name
        'project': cfg.training.output_dir,  # Directory for results
        'device': '0' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    }

    # Add callbacks to the YOLO model
    #model.add_callback("on_train_epoch_start", put_in_eval_mode)

    # Start training and log metrics to W&B
    logger.info("Starting model training.")
    model.train(**train_params)

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
