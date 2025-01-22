from src.object_detection.model import CustomDataset, create_yolo_model
from ultralytics import YOLO  # type: ignore
import pytest  # type: ignore
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np


@pytest.fixture
def temp_data():
    """
    Fixture to create temporary image and label data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        images_dir = temp_dir / "images"
        labels_dir = temp_dir / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        image_path = images_dir / "sample.jpg"
        image = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        image.save(image_path)

        label_path = labels_dir / "sample.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        yield images_dir, labels_dir


def test_dataset_initialization(temp_data):
    """
    Test dataset initialization.
    """
    images_dir, labels_dir = temp_data
    dataset = CustomDataset(images_dir, labels_dir)

    assert len(dataset) == 1
    assert dataset.image_files[0].name == "sample.jpg"
    assert "sample" in dataset.label_files


def test_dataset_getitem(temp_data):
    """
    Test dataset getitem.
    """
    images_dir, labels_dir = temp_data
    dataset = CustomDataset(images_dir, labels_dir)

    image, labels = dataset[0]
    assert image.shape == (3, 640, 640)
    assert labels.shape == (1, 5)
    assert labels[0, 0] == 0
    assert labels[0, 1:].tolist() == pytest.approx([0.5, 0.5, 0.2, 0.2])


def test_missing_label_file(temp_data):
    """
    Test dataset with missing label file.
    """
    images_dir, labels_dir = temp_data
    (labels_dir / "sample.txt").unlink()

    dataset = CustomDataset(images_dir, labels_dir)

    with pytest.raises(FileNotFoundError, match="Label file not found"):
        _ = dataset[0]


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
