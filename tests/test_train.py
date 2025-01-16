from unittest.mock import patch, MagicMock
from src.object_detection.train import main

@patch("ultralytics.YOLO")
@patch("src.object_detection.model.CustomDataset")
def test_full_training_script(mock_dataset, mock_yolo):
    """
    Test the full training script with mocked dataset and model.
    """
    # Mock dataset and model
    mock_dataset_instance = MagicMock()
    mock_dataset.return_value = mock_dataset_instance

    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    # Run the script
    main()

    # Assertions
    mock_dataset.assert_called()  # Ensure the dataset was initialized
    mock_yolo.assert_called_once_with("pretrained_weights.pt")  # Model creation
    mock_model_instance.train.assert_called_once()  # Training process
