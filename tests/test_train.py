from unittest.mock import patch, MagicMock
from src.object_detection.train import main
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import sys

sys.path.append("../DTU_MLOpsProject")

@patch("ultralytics.YOLO")
@patch("src.object_detection.model.CustomDataset")
def test_full_training_script(mock_dataset, mock_yolo):
    """
    Test the full training script with mocked dataset and model.
    """
    # Clear Hydra's global state before initialization
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    # Initialize Hydra
    with initialize(config_path="../configs", job_name="test"):
        # Create a configuration with overrides
        cfg = compose(
            config_name="config", overrides=["data.processed_dir=../mock_data/processed"]
        )

        # Mock dataset and model
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Run the main function with the modified config
        main(cfg)

        # Assertions
        mock_dataset.assert_called()  # Ensure the dataset was initialized
        mock_yolo.assert_called_once_with("yolov8n.pt")  # Model creation
        mock_model_instance.train.assert_called_once()  # Training process
