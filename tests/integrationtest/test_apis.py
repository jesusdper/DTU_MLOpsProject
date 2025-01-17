import pytest
from fastapi.testclient import TestClient
import sys
import os
#from src.object_detection import api  # Adjusted import path
#from src.object_detection.api import app

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/object_detection/api.py")))
from api import app


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


# Test the prediction endpoint (use an actual model if available)
def test_predict():
    with open("dummy_image.jpg", "wb") as f:
        f.write(b"dummy image data")

    response = client.post(
        "/predict/model1", files={"file": ("dummy_image.jpg", open("dummy_image.jpg", "rb"), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()

    os.remove("dummy_image.jpg")


# Test the model status endpoint
def test_model_status():
    response = client.get("/status/model1")
    assert response.status_code == 200
    assert "status" in response.json()
