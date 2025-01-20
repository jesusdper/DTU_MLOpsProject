from unittest.mock import patch, MagicMock
from src.object_detection.train import main
from hydra import initialize, compose  # type: ignore
from hydra.core.global_hydra import GlobalHydra  # type: ignore
import sys

sys.path.append("../DTU_MLOpsProject")

@patch("ultralytics.YOLO")
@patch("src.object_detection.model.CustomDataset")
def test_full_training_script(mock_dataset, mock_yolo):
    """
    Test the full training script with mocked dataset and model.
    """
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    with initialize(config_path="../configs", job_name="test"):
        cfg = compose(
            config_name="config", overrides=["data.processed_dir=../mock_data/processed"]
        )

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        main(cfg)

        mock_dataset.assert_called()
        mock_yolo.assert_called_once_with("yolov8n.pt")
        mock_model_instance.train.assert_called_once()
