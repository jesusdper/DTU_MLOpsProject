from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, REGISTRY, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
import os
import tempfile
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()

# Instrument the app to collect system metrics
instrumentator.instrument(app)

# Define Prometheus custom metrics
INFERENCE_REQUESTS = Counter("inference_requests_total", "Total number of inference requests")
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions made")
INFERENCE_DURATION = Histogram("inference_duration_seconds", "Duration of inference requests in seconds")

# Define the path to the model storage
MODEL_DIR = './uploads'
os.makedirs(MODEL_DIR, exist_ok=True)


# Utility function to get available models in the upload directory
def get_available_models():
    logger.info(f"Files in MODEL_DIR: {os.listdir(MODEL_DIR)}")
    return [f.split('.')[0] for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]


# Create Prometheus metrics route
@app.get("/metrics")
async def metrics():
    # Generate and return the latest Prometheus metrics, including custom ones
    return Response(generate_latest(REGISTRY), media_type="text/plain")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Object Detection API"}


@app.get("/models")
def list_models():
    """
    Endpoint to list available models for inference.
    """
    available_models = get_available_models()
    if not available_models:
        raise HTTPException(status_code=404, detail="No models available.")
    return {"available_models": available_models}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure the uploads directory exists
        upload_dir = MODEL_DIR
        os.makedirs(upload_dir, exist_ok=True)

        # Define the file path where the model will be saved
        model_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file
        with open(model_path, "wb") as f:
            f.write(file.file.read())

        return {"message": f"Model {file.filename} uploaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


@app.post("/predict/{model_name}")
async def predict(file: UploadFile = File(...), model_name: str = None):
    INFERENCE_REQUESTS.inc()  # Increment the inference requests counter

    if model_name is None:
        available_models = get_available_models()
        if not available_models:
            raise HTTPException(status_code=404, detail="No models available.")
        return {"available_models": available_models}

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")

    model = YOLO(model_path)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Debug: Verify file creation
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=500, detail="Temporary file was not created.")

    # Start timing the inference
    start_time = time.time()

    # Perform prediction
    try:
        results = model(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Record inference duration
    inference_time = time.time() - start_time
    INFERENCE_DURATION.observe(inference_time)  # Record the duration

    predictions = []
    for result in results:
        for box in result.boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = box
            predictions.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "confidence": confidence,
                "class_id": int(class_id),
                "class_name": model.names[int(class_id)],
            })

    PREDICTIONS_TOTAL.inc(len(predictions))  # Increment the predictions counter

    os.remove(temp_file_path)
    return {"predictions": predictions}


@app.get("/status/{model_name}")
async def model_status(model_name: str):
    try:
        # Logic to check model status
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        if os.path.exists(model_path):
            return {"status": "Model exists"}
        else:
            return {"status": "Model not found"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
