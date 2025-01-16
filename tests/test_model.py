from src.object_detection.model import create_yolo_model
from ultralytics import YOLO  # type: ignore
from pathlib import Path
import pytest # type: ignore
import os
import shutil
from unittest.mock import patch

@pytest.fixture
def tmp_model(tmp_path):
    """
    Create a temporary directory for the model.
    """
    model_dir = tmp_path
    os.makedirs(model_dir, exist_ok=True)

    return model_dir


@patch("src.object_detection.model.create_yolo_model")
def test_create_yolo_model(mock_download, tmp_model):
    """
    Test the model is correctly created.
    """
    mock_download.side_effect = lambda destination: destination.write_text("mock model content")

    model = create_yolo_model(model_path=tmp_model, num_classes=80)

    assert isinstance(model, YOLO)
    