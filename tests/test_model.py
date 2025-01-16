from src.object_detection.model import create_yolo_model
from ultralytics import YOLO  # type: ignore

def test_create_yolo_model():
    """
    Test the model is correctly created.
    """
    model = create_yolo_model()

    assert isinstance(model, YOLO)
    assert model.yaml["nc"] == 80
