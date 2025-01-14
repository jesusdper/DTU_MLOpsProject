from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import torch
import os

app = FastAPI()
model = YOLO("models/yolov8_voc5/weights/best.pt")



@app.get("/")
def read_root():
    return {"message": "Welcome to the Object Detection API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform prediction
    results = model(file_path)  # YOLOv8 inference

    # Parse results using the correct format
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
                "class_name": model.names[int(class_id)],  # Map class ID to class name
            })

    # Clean up the temporary file
    os.remove(file_path)

    return {"predictions": predictions}

