# liveness_api.py
from flask import Blueprint, request, jsonify
import cv2, numpy as np, logging, base64
from io import BytesIO
from PIL import Image
from deepface import DeepFace

liveness_blueprint = Blueprint('liveness', __name__)
log = logging.getLogger("liveness")

# ---------- Utilities ----------
def _ensure_rgb(img):
    # يدعم: np.ndarray (BGR أو RGB) أو مسار ملف
    if isinstance(img, str):
        im = Image.open(img).convert("RGB")
        return np.array(im)  # RGB بالفعل
    if img is None:
        raise ValueError("Empty image")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        # غالباً BGR من OpenCV -> حوّل إلى RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _boost_for_detection(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    m = float(rgb_f.mean())
    gamma = 0.8 if m < 0.35 else (1.2 if m > 0.7 else 1.0)
    rgb_f = np.power(rgb_f, gamma)
    rgb_f = np.clip(rgb_f, 0, 1)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    boosted = cv2.filter2D((rgb_f * 255).astype(np.uint8), -1, sharp_kernel)
    h, w = boosted.shape[:2]
    if min(h, w) < 600:
        scale = 600.0 / min(h, w)
        boosted = cv2.resize(boosted, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return boosted

def _to_base64(img_rgb, fmt="JPEG", max_side=384):
    # تصغير بسيط عشان حجم الاستجابة ما يكبر
    h, w = img_rgb.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img_rgb)  # RGB
    buf = BytesIO()
    pil.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------- Core ----------
def analyze_frame(frame, anti_spoof=True):
    rgb = _ensure_rgb(frame)
    rgb = _boost_for_detection(rgb)

    backends = ["retinaface", "mtcnn", "mediapipe", "opencv"]
    last_err = None

    for be in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=rgb,              # np.ndarray (RGB)
                detector_backend=be,
                enforce_detection=False,
                align=True,
                anti_spoofing=False        # نؤجلها
            )
            if faces:
                # اختَر أعلى ثقة إن وُجدت
                best = max(faces, key=lambda f: f.get("confidence", 0.0))
                face_arr = best.get("face")
                face_rgb = (face_arr * 255).astype(np.uint8) if face_arr is not None else None

                liveness = None
                if anti_spoof:
                    try:
                        live_res = DeepFace.extract_faces(
                            img_path=rgb,
                            detector_backend=be,
                            enforce_detection=False,
                            align=True,
                            anti_spoofing=True
                        )
                        if live_res and isinstance(live_res, list):
                            # بعض الإصدارات تعيد dict فيه "liveness" أو "score"
                            liveness = live_res[0].get("liveness", None)
                    except Exception as _:
                        # لا نفشل الكشف بسببها
                        liveness = None

                return {
                    "ok": True,
                    "backend": be,
                    "confidence": float(best.get("confidence") or 0.0),
                    "box": best.get("facial_area"),
                    "liveness": liveness,
                    # اختياري: قص الوجه كـ base64 للاختبار/المعاينة
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
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # BGR
        if frame is None:
            return jsonify({"ok": False, "error": "failed to decode image"}), 400

        result = analyze_frame(frame, anti_spoof=True)

        status = 200 if result.get("ok") else 422
        return jsonify(result), status

    except Exception as e:
        log.exception("Unexpected error in /liveness")
        return jsonify({"ok": False, "error": "internal_error", "details": str(e)}), 500
