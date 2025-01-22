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
import wandb
import shutil
import torch.profiler
import argparse

# Ensure the W&B log directory exists
log_dir = Path.home() / "models/logs/yolov8_voc_test_wandb/wandb/run-20250120_160320-omms6rqf/logs"
log_dir.mkdir(parents=True, exist_ok=True)

wandb.login(key="61ddce14f1719a9a246485c5859a9bedb6c44a51")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download yolov8n.pt model
model_dir = Path.home() / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "yolov8n.pt"
if not model_path.exists():
    torch.hub.download_url_to_file('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', model_path)
    logger.info(f"Downloaded yolov8n.pt to {model_path}")

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra with W&B tracking.
    """
    log_dir = Path(cfg.training.output_dir) / "logs" / cfg.training.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb_log_dir = log_dir / "wandb" / f"run-{wandb.util.generate_id()}"
    wandb_log_dir.mkdir(parents=True, exist_ok=True)

    trace_file_path = log_dir / "train_trace.pt.trace.json"

    wandb.init(
     project=cfg.wandb.project_name,
     name=cfg.training.experiment_name,
     config=dict(cfg),
     dir=str(log_dir),
     resume="allow",
     )

    try:
        #with torch.profiler.profile(
        #    activities=[torch.profiler.ProfilerActivity.CPU],
        #    profile_memory=False,
        #    record_shapes=False,
        #    with_stack=False,
        #) as prof:
        logger.info("Profiler started.")

        #logger.info(f"Starting training with config: {yaml.dump(cfg)}")
        breakpoint()
        processed_dir = Path(cfg.data.processed_dir).resolve()

        train_images_dir = processed_dir / "train" / "images"
        train_labels_dir = processed_dir / "train" / "labels"
        val_images_dir = processed_dir / "val" / "images"
        val_labels_dir = processed_dir / "val" / "labels"

        if not train_images_dir.exists() or not train_labels_dir.exists():
            raise FileNotFoundError(f"Training data not found in {train_images_dir} or {train_labels_dir}")

        if not val_images_dir.exists() or not val_labels_dir.exists():
            raise FileNotFoundError(f"Validation data not found in {val_images_dir} or {val_labels_dir}")

        train_dataset = CustomDataset(train_images_dir, train_labels_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

        if len(train_loader.dataset) == 0:
            raise ValueError("No training samples found.")

        logger.info(f"Loaded {len(train_loader.dataset)} training samples.")

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

        tb_log_dir = Path(cfg.training.output_dir) / "logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

        train_params = {
            'data': str(data_yaml_path),
            'epochs': cfg.training.epochs,
            'imgsz': cfg.training.img_size,
            'batch': cfg.training.batch_size,
            'name': cfg.training.experiment_name,
            'project': cfg.training.output_dir,
            'device': '0' if torch.cuda.is_available() else 'cpu',
        }

        logger.info("Starting model training.")
        model.train(**train_params)

        logger.info(f"Saving profiler trace to {trace_file_path}")
        #prof.export_chrome_trace(str(trace_file_path))

    except Exception as e:
        logger.error(f"An error occurred during profiling: {str(e)}")

    output_dir = Path(cfg.training.output_dir)
    experiment_dir = list(output_dir.glob(f"{cfg.training.experiment_name}*"))
    if len(experiment_dir) == 1:
        trained_model_dir = experiment_dir[0] / "weights"
        trained_model_path = trained_model_dir / "best.pt"

        wandb.log_artifact(
            str(trained_model_path),
            name=f"{cfg.training.experiment_name}_model",
            type="model",
        )
        logger.info(f"Model artifact logged to W&B: {trained_model_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--save_location", type=str, default=str(Path.home() / "models"), help="Location to save the models")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for training")

    args = parser.parse_args()

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../configs")
    cfg = hydra.compose(config_name="config.yaml", overrides=[f"training.output_dir={args.save_location}", f"training.epochs={args.n_epochs}"])

    main(cfg)