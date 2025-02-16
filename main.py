from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2 as cv
import numpy as np
import joblib
from ultralytics import YOLO
from pathlib import Path
import os

app = FastAPI()

model_path = "modelXGBoost_3F.pkl"
if os.path.exists(model_path):
    ml_model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")

model_path_forDL = "modelXGBoost_forDL.pkl"
if os.path.exists(model_path_forDL):
    ml_model_for_DL = joblib.load(model_path_forDL)
else:
    raise FileNotFoundError(f"❌ Model file '{model_path_forDL}' not found!")

yolo_model_path = "yolov8s-seg.pt"
if os.path.exists(yolo_model_path):
    yolo_model = YOLO(yolo_model_path)
else:
    raise FileNotFoundError(f"❌ YOLO model file '{yolo_model_path}' not found!")

dpi = 15
pixels_per_cm = dpi / 2.54  

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    oblique_length: float = Form(...),
    withers_height: float = Form(...),
    heart_girth: float = Form(...)
):
    try:
        features = np.array([[oblique_length, withers_height, heart_girth]])

        prediction = ml_model.predict(features).tolist()
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_path = f"static/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(file.file.read())

        img = cv.imread(image_path)
        if img is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        results = yolo_model.predict(img)

        for r in results:
            for c in r:
                mask = np.zeros(img.shape[:2], np.uint8)
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv.drawContours(mask, [contour], -1, 255, cv.FILLED)

        edges = cv.Canny(mask, 50, 150)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        x_min, x_max = float("inf"), 0
        y_min, y_max = float("inf"), 0

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            x_min, x_max = min(x_min, x), max(x_max, x + w)
            y_min, y_max = min(y_min, y), max(y_max, y + h)

        width_cm = (x_max - x_min) / pixels_per_cm
        height_cm = (y_max - y_min) / pixels_per_cm

        X_test = np.array([[width_cm, height_cm]])
        y_pred = ml_model_for_DL.predict(X_test).tolist()

        return JSONResponse(content={"width_cm": width_cm, "height_cm": height_cm, "prediction": y_pred})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
