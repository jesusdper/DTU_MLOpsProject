import random
from locust import HttpUser, between, task, events
from locust.clients import HttpSession
import os


class FastAPILoadTest(HttpUser):
    """Simulate user interactions with the FastAPI app."""

    # Time between tasks for each simulated user
    wait_time = between(1, 2)

    # Define tasks for each endpoint of the FastAPI application

    @task
    def get_root(self):
        """Simulate a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task
    def get_models(self):
        """Simulate a user getting the list of available models."""
        self.client.get("/models")

    @task(3)  # This will make the `predict` endpoint 3x more likely to be hit than `get_root`
    def post_predict(self):
        """Simulate a user uploading a file and running a prediction."""
        file_path = "/path/to/sample_image.jpg"  # Replace this with an actual path to an image you want to test
        model_name = random.choice(["model1", "model2"])  # Replace with actual model names in your app

        with open(file_path, 'rb') as file:
            self.client.post(f"/predict/{model_name}", files={"file": file})

    @task
    def post_upload(self):
        """Simulate a user uploading a model."""
        file_path = "/path/to/sample_model.pt"  # Replace with an actual model file path

        with open(file_path, 'rb') as file:
            self.client.post("/upload", files={"file": file})

    @task
    def get_status(self):
        """Simulate a user checking the status of a model."""
        model_name = random.choice(["model1", "model2"])  # Replace with actual model names in your app
        self.client.get(f"/status/{model_name}")


# Optional: Use the events for handling test start and stop
@events.test_start.add_listener
def on_test_start(**kwargs):
    print("Test started!")


@events.test_stop.add_listener
def on_test_stop(**kwargs):
    print("Test stopped!")
