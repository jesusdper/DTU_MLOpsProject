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
import shutil  # For compressing the model
import torch.profiler  # For profiling

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path=r"C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to train YOLO model using Ultralytics and Hydra.
    """
    # Create a log subfolder for the model
    log_dir = Path(cfg.training.output_dir) / "logs" / cfg.training.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)  # Create model's log folder

    # Path for the trace.json file
    trace_file_path = log_dir / "train_trace.pt.trace.json"

    # Initialize profiler
    try:
        # Start profiler
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=False,  # Disable memory profiling to reduce data
                record_shapes=False,  # Skip shape profiling if not needed
                with_stack=False,
        ) as prof:
            logger.info("Profiler started.")

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

            # Prepare the YOLO training parameters dictionary
            train_params = {
                'data': str(data_yaml_path),  # Path to dataset config file
                'epochs': cfg.training.epochs,  # Number of epochs
                'imgsz': cfg.training.img_size,  # Image size
                'batch': cfg.training.batch_size,  # Batch size
                'name': cfg.training.experiment_name,  # Experiment name
                'project': cfg.training.output_dir,  # Output directory for saving results
                'device': '0' if torch.cuda.is_available() else 'cpu',  # Ensure the correct device
            }

            # Start training inside the profiler
            logger.info("Starting model training.")
            model.train(**train_params)

        # Ensure profiler stops and trace is saved after training
        logger.info(f"Saving profiler trace to {trace_file_path}")
        prof.export_chrome_trace(str(trace_file_path))
        logger.info(f"Profiler trace saved to {trace_file_path}")

    except Exception as e:
        logger.error(f"An error occurred during profiling: {str(e)}")

    # Dynamically find the correct model path
    output_dir = Path(cfg.training.output_dir)
    logger.info(f"Output dir {output_dir}")

    experiment_dir = list(output_dir.glob(f"{cfg.training.experiment_name}*"))
    if len(experiment_dir) == 1:
        trained_model_dir = experiment_dir[0] / "weights"
        trained_model_path = trained_model_dir / "best.pt"
    elif len(experiment_dir) > 1:
        logger.error(f"Multiple directories found for experiment name {cfg.training.experiment_name}.")
        return
    else:
        logger.error(f"No directory found for experiment name {cfg.training.experiment_name}.")
        return

    # Log the model path
    logger.info(f"Trained model path: {trained_model_path}")

    if trained_model_path.exists():
        # Compress the trained model before uploading
        #zip_model_path = str(trained_model_path).replace(".pt", ".zip")
        #shutil.make_archive(zip_model_path.replace(".zip", ""), 'zip', trained_model_path.parent,
        #                    trained_model_path.name)
        #logger.info(f"Zip model path: {zip_model_path}")

        # Use the existing upload_model function from model_registry_helper to upload the model
        upload_model(
            model_archive_path=trained_model_path,  # Path to the zipped model file
            model_name=cfg.training.experiment_name,  # Model name for registry
            api_key=cfg.upload.api_key,  # API key for uploading
            model_registry_url=cfg.upload.model_registry_url  # Model registry URL
        )
        logger.info(f"Model {trained_model_path} successfully uploaded.")
    else:
        logger.error("Trained model not found!")

if __name__ == "__main__":
    main()
