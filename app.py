from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import time
import os

app = Flask(__name__)
model = YOLO("mix(320x160).pt")  # โหลดจาก Ultralytics

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files['image'].stream).convert("RGB")

    t1 = time.time()
    results = model(img)
    t2 = time.time()

    detections = results[0].boxes.xyxy.cpu().tolist()
    confidences = results[0].boxes.conf.cpu().tolist()
    classes = results[0].boxes.cls.cpu().tolist()

    data = []
    for xyxy, conf, cls in zip(detections, confidences, classes):
        data.append({
            "bbox": xyxy,
            "confidence": round(conf, 2),
            "class_id": int(cls)
        })

    return jsonify({
        "inference_time_ms": round((t2 - t1) * 1000, 2),
        "results": data
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
