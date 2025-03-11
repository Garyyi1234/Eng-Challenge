from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

# Ensure correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = YOLO(MODEL_PATH)

CLASS_NAMES = ["empty_spot", "car"]
COLOR_MAP = {"empty_spot": (0, 0, 255), "car": (0, 255, 0)}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Could not read image"}), 500

        results = model(image)
        height, width, _ = image.shape  # Get original image dimensions

        empty_spots, occupied_spots = 0, 0
        detections = []

        for result in results:
            for box in result.boxes:
                x_center, y_center, w, h = box.xywh[0]  # Get YOLO format values
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES[class_id]
                color = COLOR_MAP[class_name]

                # Convert YOLO format to initial format
                x_min = int((x_center - w / 2).item())
                x_max = int((x_center + w / 2).item())
                y_min = int((y_center - h / 2).item())
                y_max = int((y_center + h / 2).item())

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                # Convert class labels
                class_name_numbered = "1" if class_name == "empty_spot" else "2"
                detections.append(f"{file.filename} {class_name_numbered} {x_min} {x_max} {y_min} {y_max}")

                # Count empty and occupied spots
                if class_name == "empty_spot":
                    empty_spots += 1
                elif class_name == "car":
                    occupied_spots += 1

        # Save processed image
        processed_file_path = os.path.join(PROCESSED_FOLDER, file.filename)
        cv2.imwrite(processed_file_path, image)

        # Save detection results in text file
        text_filename = os.path.splitext(file.filename)[0] + ".txt"
        text_file_path = os.path.join(PROCESSED_FOLDER, text_filename)
        with open(text_file_path, "w") as f:
            f.write("\n".join(detections))

        response = {
            "image_url": f"http://127.0.0.1:5000/processed/{file.filename}",
            "text_file_url": f"http://127.0.0.1:5000/processed/{text_filename}",
            "empty_spots": empty_spots,
            "occupied_spots": occupied_spots
        }
        return jsonify(response)

    except Exception as e:
        print("❌ Backend Error:", str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/processed/<filename>")
def get_processed_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
