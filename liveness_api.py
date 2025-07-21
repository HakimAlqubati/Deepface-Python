# liveness_api.py

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from deepface.modules import detection

liveness_blueprint = Blueprint('liveness', __name__)

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

@liveness_blueprint.route("/liveness", methods=["POST"])
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
