from src.object_detection.model import create_yolo_model
from ultralytics import YOLO  # type: ignore
import pytest # type: ignore


@pytest.fixture
def mock_pretrained_weights(tmp_path):
    """
    Fixture to mock pretrained weights
    """
    weights_path = tmp_path / "yolov8n.pt"
    
    return weights_path

@pytest.fixture
def mock_config_file(tmp_path):
    """
    Fixture to mock a config file
    """
    mock_config = tmp_path / "mock_config.yaml"

    return mock_config


def test_create_yolo_model(mock_pretrained_weights, mock_config_file):
    """
    Test the YOLO model is created correctly with mock config.
    """
    config = mock_config_file 

    model = create_yolo_model(pretrained_weights=mock_pretrained_weights, cfg=config)
    
    assert model is not None 
    assert isinstance(model, YOLO) 