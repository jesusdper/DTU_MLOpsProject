from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import torch
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


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


import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure the uploads directory exists
        upload_dir = './uploads'
        os.makedirs(upload_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Define the file path where the model will be saved
        model_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file
        with open(model_path, "wb") as f:
            f.write(file.file.read())

        return {"message": f"Model {file.filename} uploaded successfully to {model_path}"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


@app.get("/status/{model_name}")
async def model_status(model_name: str):
    try:
        # Logic to check model status
        # For example, check if the file exists in the upload directory
        model_path = f"./uploads/{model_name}.zip"
        if os.path.exists(model_path):
            return {"status": "Model exists"}
        else:
            return {"status": "Model not found"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
