# liveness_api.py

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import logging

from deepface.modules import detection

liveness_blueprint = Blueprint('liveness', __name__)

def analyze_frame(frame):
    try:
        faces = detection.extract_faces(
            img_path=frame,
            detector_backend="mediapipe",
            enforce_detection=False,
            align=True,
            anti_spoofing=True
        )
        return faces[0] if faces else None
    except Exception as e:
        logging.exception("extract_faces failed")
        # fallback بدون anti_spoofing
        try:
            faces = detection.extract_faces(
                img_path=frame,
                detector_backend="mediapipe",
                enforce_detection=False,
                align=True,
                anti_spoofing=False
            )
            return faces[0] if faces else None
        except Exception:
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
