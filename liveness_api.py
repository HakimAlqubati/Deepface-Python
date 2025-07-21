# liveness_api.py

from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface.modules import detection
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def analyze_frame(frame):
    faces = detection.extract_faces(
        img_path=frame,
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        anti_spoofing=True
    )
    if faces:
        face = faces[0]
        is_real = face.get("is_real", False)
        score = face.get("antispoof_score", 0)
        landmarks = face.get("landmarks", {})
        return {
            "liveness": bool(is_real),
            "score": float(score),
            "landmarks": landmarks
        }
    return None

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = analyze_frame(frame)
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No face detected"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
