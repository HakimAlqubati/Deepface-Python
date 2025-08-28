# liveness_api.py
import os
# --- Force CPU & tame TF/BLAS threads BEFORE importing DeepFace/TF ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # disable GPU to avoid cuInit 303
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # reduce TF logging
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Blueprint, request, jsonify
import cv2, numpy as np, logging, base64
from io import BytesIO
from PIL import Image
from deepface import DeepFace

# Limit OpenCV thread usage as well (helps on small machines)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

liveness_blueprint = Blueprint('liveness', __name__)
log = logging.getLogger("liveness")

# ---------- Utilities ----------
def _ensure_rgb(img):
    if isinstance(img, str):
        im = Image.open(img).convert("RGB")
        return np.array(im)
    if img is None:
        raise ValueError("Empty image")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _boost_for_detection(rgb):
    # تصحيح جاما + شحذ خفيف. لا نُكبّر بقوة لتجنّب استهلاك الذاكرة.
    rgb_f = rgb.astype(np.float32) / 255.0
    m = float(rgb_f.mean())
    gamma = 0.8 if m < 0.35 else (1.2 if m > 0.7 else 1.0)
    rgb_f = np.power(rgb_f, gamma)
    rgb_f = np.clip(rgb_f, 0, 1)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    boosted = cv2.filter2D((rgb_f * 255).astype(np.uint8), -1, sharp_kernel)
    # حد أقصى لحجم الصورة لتفادي تحذيرات الذاكرة
    MAX_SIDE = 1280
    h, w = boosted.shape[:2]
    s = max(h, w)
    if s > MAX_SIDE:
        scale = MAX_SIDE / s
        boosted = cv2.resize(boosted, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # Upscale بسيط فقط لو الصورة صغيرة جدًا
    if min(h, w) < 400:
        scale = 480.0 / min(h, w)
        boosted = cv2.resize(boosted, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return boosted

def _to_base64(img_rgb, fmt="JPEG", max_side=384):
    h, w = img_rgb.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------- Core ----------
def analyze_frame(frame, anti_spoof=True):
    rgb = _ensure_rgb(frame)
    rgb = _boost_for_detection(rgb)

    backends = ["mediapipe", "opencv", "retinaface", "mtcnn"]
    last_err = None

    for be in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=rgb,
                detector_backend=be,
                enforce_detection=False,
                align=True,
                anti_spoofing=False
            )
            if faces:
                best = max(faces, key=lambda f: f.get("confidence", 0.0))
                face_arr = best.get("face")
                face_rgb = (face_arr * 255).astype(np.uint8) if face_arr is not None else None

                liveness = None
                if anti_spoof:
                    try:
                        live_input = face_rgb if face_rgb is not None else rgb
                        live_res = DeepFace.extract_faces(
                            img_path=live_input,
                            detector_backend=be,
                            enforce_detection=False,
                            align=True,
                            anti_spoofing=True
                        )
                        if live_res and isinstance(live_res, list):
                            lr = live_res[0] or {}
                            live_bool = None
                            live_score = None
                            threshold = lr.get("threshold", 0.85)

                            if "score" in lr:
                                live_score = float(lr.get("score") or 0.0)
                                live_bool = live_score >= float(threshold)
                            for key in ("liveness", "is_real", "real"):
                                if key in lr and isinstance(lr[key], (bool, np.bool_)):
                                    live_bool = bool(lr[key])
                                    break

                            liveness = {
                                "is_live": live_bool,
                                "score": live_score,
                                "threshold": float(threshold),
                                "raw": {k: lr.get(k) for k in ("liveness", "is_real", "real", "score", "threshold")}
                            }
                        else:
                            liveness = None
                    except Exception:
                        liveness = None

                return {
                    "ok": True,
                    "backend": be,
                    "confidence": float(best.get("confidence") or 0.0),
                    "box": best.get("facial_area"),
                    "liveness": liveness,
                    "face_thumb_b64": _to_base64(face_rgb) if face_rgb is not None else None,
                }
        except Exception as e:
            last_err = e
            log.exception("DeepFace backend '%s' failed", be)

    return {
        "ok": False,
        "error": "No face detected after trying multiple backends",
        "last_exception": str(last_err) if last_err else None,
        "tried_backends": backends
    }

# ---------- Route ----------
@liveness_blueprint.route("/liveness", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "field 'image' is required"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"ok": False, "error": "empty image"}), 400

        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok": False, "error": "failed to decode image"}), 400

        result = analyze_frame(frame, anti_spoof=True)

        status = 200 if result.get("ok") else 422
        return jsonify(result), status

    except Exception as e:
        log.exception("Unexpected error in /liveness")
        return jsonify({"ok": False, "error": "internal_error", "details": str(e)}), 500
