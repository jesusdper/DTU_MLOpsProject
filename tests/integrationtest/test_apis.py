import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
from src.object_detection.api import app  # Adjusted import path

client = TestClient(app)

# Test the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Object Detection API"}


# Test the models listing endpoint
def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "available_models" in response.json()


# Test uploading a file
@pytest.mark.parametrize("filename", ["model1.pt", "model2.pt"])
def test_upload_file(filename):
    with open(filename, "wb") as f:
        f.write(b"dummy model data")

    with open(filename, "rb") as f:
        response = client.post("/upload", files={"file": (filename, f, "application/octet-stream")})
        assert response.status_code == 200
        assert f"Model {filename} uploaded successfully." in response.json()["message"]

    # Clean up the dummy file
    os.remove(filename)


# Test the prediction endpoint with mocked model

@patch("src.object_detection.api.YOLO")  # Patch YOLO in the location where it is used in api.py
def test_predict(mock_yolo):
    # Create a mock model object
    mock_model = MagicMock()

    # Mock the `names` attribute to simulate class labels
    mock_model.names = {0: "mocked_object"}  # Assuming class_id 0 corresponds to 'mocked_object'

    # Mock the behavior of YOLO's output (this simulates the result of the prediction)
    mock_result = MagicMock()
    mock_result.boxes.data.tolist.return_value = [[0, 0, 1, 1, 0.99, 0]]  # Mocked bounding box data with tolist()

    # Set the model to return a list containing the mocked result
    mock_model.return_value = [mock_result]

    # Return the mock model when YOLO is instantiated
    mock_yolo.return_value = mock_model

    # Create a dummy image file
    dummy_image_path = "dummy_image.jpg"
    with open(dummy_image_path, "wb") as f:
        f.write(b"dummy image data")

    # Send a prediction request
    with open(dummy_image_path, "rb") as f:
        response = client.post(
            "/predict/model1",  # Assuming model1 is the name of the model you want to test
            files={"file": ("dummy_image.jpg", f, "image/jpeg")},
        )

    # Check if the response status is 200
    assert response.status_code == 200

    # Check that the prediction returned matches the mocked data
    assert response.json() == {
        "predictions": [
            {
                "x_min": 0,
                "y_min": 0,
                "x_max": 1,
                "y_max": 1,
                "confidence": 0.99,
                "class_id": 0,
                "class_name": "mocked_object"
            }
        ]
    }

    # Clean up the dummy image file
    os.remove(dummy_image_path)
# Test the model status endpoint
def test_model_status():
    response = client.get("/status/model1")
    assert response.status_code == 200
    assert "status" in response.json()
