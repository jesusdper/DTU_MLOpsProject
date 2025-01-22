import os
import torch
from pathlib import Path
import logging
import yaml
from omegaconf import DictConfig
import hydra
from ultralytics import YOLO
from model import CustomDataset, create_yolo_model
from torch.utils.tensorboard import SummaryWriter
import wandb  # For W&B integration
import shutil
import torch.profiler

# Ensure the W&B log directory exists
log_dir = Path("/models/logs/yolov8_voc_test_wandb/wandb/run-20250120_160320-omms6rqf/logs")
log_dir.mkdir(parents=True, exist_ok=True)

wandb.login(key="b97463597a9b7425acac3f6390c6ec7515ba2585")

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

@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra with W&B tracking.
    """
    # Create a log subfolder for the model
    log_dir = Path(cfg.training.output_dir) / "logs" / cfg.training.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the W&B log directory exists
    wandb_log_dir = log_dir / "wandb" / f"run-{wandb.util.generate_id()}"
    wandb_log_dir.mkdir(parents=True, exist_ok=True)

    # Path for the trace.json file
    trace_file_path = log_dir / "train_trace.pt.trace.json"

    # Initialize W&B
    # Initialize W&B
    wandb.init(
     project=cfg.wandb.project_name,
     name=cfg.training.experiment_name,
     config=dict(cfg),
     dir=str(log_dir),
     resume="allow",
     )

    try:
        # Start profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=False,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            logger.info("Profiler started.")

            # Print the configuration
            logger.info(f"Starting training with config: {yaml.dump(cfg)}")

            # Load processed data paths from configuration
            processed_dir = Path(cfg.data.processed_dir).resolve()

            train_images_dir = processed_dir / "train" / "images"
            train_labels_dir = processed_dir / "train" / "labels"
            val_images_dir = processed_dir / "val" / "images"
            val_labels_dir = processed_dir / "val" / "labels"

            # Setup the dataset and dataloaders
            train_dataset = CustomDataset(train_images_dir, train_labels_dir)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

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
                'epochs': cfg.training.epochs,  # Number of epochs
                'imgsz': cfg.training.img_size,  # Image size
                'batch': cfg.training.batch_size,  # Batch size
                'name': cfg.training.experiment_name,  # Experiment name
                'project': cfg.training.output_dir,  # Directory for results
                'device': '0' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
            }

            # Add callbacks to the YOLO model
            #model.add_callback("on_train_epoch_start", put_in_eval_mode)

            # Start training and log metrics to W&B
            logger.info("Starting model training.")
            model.train(**train_params)

        logger.info(f"Saving profiler trace to {trace_file_path}")
        prof.export_chrome_trace(str(trace_file_path))

    except Exception as e:
        logger.error(f"An error occurred during profiling: {str(e)}")

    output_dir = Path(cfg.training.output_dir)
    experiment_dir = list(output_dir.glob(f"{cfg.training.experiment_name}*"))
    if len(experiment_dir) == 1:
        trained_model_dir = experiment_dir[0] / "weights"
        trained_model_path = trained_model_dir / "best.pt"

        # Save the trained model as a W&B artifact
        wandb.log_artifact(
            str(trained_model_path),
            name=f"{cfg.training.experiment_name}_model",
            type="model",
        )
        logger.info(f"Model artifact logged to W&B: {trained_model_path}")

    wandb.finish()

if __name__ == "__main__":
    main()