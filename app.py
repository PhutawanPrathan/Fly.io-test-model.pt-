from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch
import io
import time

app = FastAPI()

# Load model (ต้องแนบไฟล์ yolov10.pt ด้วย)
model = YOLO("mix(320x160).pt")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((320, 160))  # Resize to match training
    img_np = np.array(img_resized).astype("float32") / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    tensor = torch.tensor(img_np).unsqueeze(0)
    return tensor

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    start = time.time()

    try:
        image_bytes = await image.read()
        input_tensor = preprocess_image(image_bytes)

        results = model(input_tensor)
        confs = results[0].boxes.conf.cpu().tolist()

        rtt = round((time.time() - start) * 1000, 2)
        return JSONResponse(content={
            "RTT_ms": rtt,
            "confidences": confs,
            "num_detections": len(confs)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
