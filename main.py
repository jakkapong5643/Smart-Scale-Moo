from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2 as cv
import numpy as np
import joblib
from ultralytics import YOLO
import os
import base64
from pathlib import Path

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

dpi = 46
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

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = cv.imdecode(np.frombuffer(image_data, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)
        
        results = yolo_model.predict(img)
        
        oblique_length_cm = None
        height_cm = None
        
        for r in results:
            img_name = Path(file.filename).stem

            for c in r:
                label = c.names[c.boxes.cls.tolist().pop()]
                
                if label.lower() == "cow":
                    mask = np.zeros(img.shape[:2], np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    cv.drawContours(mask, [contour], -1, 255, cv.FILLED)

                    y_min, y_max = np.min(contour[:, 0, 1]), np.max(contour[:, 0, 1])
                    x_min, x_max = np.min(contour[:, 0, 0]), np.max(contour[:, 0, 0])
                    
                    lower_cut = int(y_max - (y_max - y_min) * 0.40)
                    mask[lower_cut:y_max, :] = 0

                    upper_cut = int(y_min + (y_max - y_min) * 0.15)
                    mask[:upper_cut, :] = 0

                    left_cut = int(x_min + (x_max - x_min) * 0.30)
                    mask[:, :left_cut] = 0

                    edges = cv.Canny(mask, 50, 150)
                    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        x, y, w, h = cv.boundingRect(cnt)
                        oblique_length_pixels = np.sqrt(w ** 2 + h ** 2)
                        oblique_length_cm = oblique_length_pixels / pixels_per_cm
                    height_cm = (y_max - y_min) / pixels_per_cm

        if oblique_length_cm is not None and height_cm is not None:
            X_test = np.array([[height_cm, oblique_length_cm]])
            y_pred = ml_model_for_DL.predict(X_test).tolist()

            return JSONResponse(content={
                "oblique_length_cm": oblique_length_cm,
                "height_cm": height_cm,
                "prediction": y_pred,
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
            })
        else:
            return JSONResponse(content={"error": "No cattle detected in the image."}, status_code=400)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
