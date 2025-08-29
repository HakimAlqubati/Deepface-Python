# liveness_api.py
from flask import Blueprint, request, jsonify
import logging
from io import BytesIO
import base64

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

liveness_blueprint = Blueprint("liveness", __name__)
log = logging.getLogger("liveness")


# ---------- Utilities ----------
def _ensure_rgb(img):
    """
    يقبل: np.ndarray (BGR أو RGB) أو مسار ملف (str).
    يعيد مصفوفة RGB.
    """
    if isinstance(img, str):
        im = Image.open(img).convert("RGB")
        return np.array(im)

    if img is None:
        raise ValueError("Empty image")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        # أغلب دوال OpenCV تُرجع BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _boost_for_detection(rgb):
    """
    تحسين بسيط (جاما + شاربن) مع رفع الدقة إن كانت صغيرة.
    الهدف: مساعدة الكواشف في الإضاءة الضعيفة أو الضبابية.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    m = float(rgb_f.mean())
    gamma = 0.8 if m < 0.35 else (1.2 if m > 0.7 else 1.0)
    rgb_f = np.power(np.clip(rgb_f, 0, 1), gamma)

    sharp_kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]], dtype=np.float32
    )
    boosted = cv2.filter2D((rgb_f * 255).astype(np.uint8), -1, sharp_kernel)

    h, w = boosted.shape[:2]
    if min(h, w) < 600:
        scale = 600.0 / min(h, w)
        boosted = cv2.resize(
            boosted,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    return boosted


def _to_base64(img_rgb, fmt="JPEG", max_side=384, quality=90):
    """
    يحول مصفوفة RGB إلى Base64 مع تصغير اختياري لتقليل حجم الاستجابة.
    """
    if img_rgb is None:
        return None
    h, w = img_rgb.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img_rgb = cv2.resize(
            img_rgb, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _available_backends():
    """
    يبني قائمة backends المدعومة حاليًا في البيئة.
    نتجنب MediaPipe إذا لم تكن مثبتة (لتفادي خطأ No space أثناء pip).
    """
    order = []

    # جرّب إضافة mediapipe فقط إن كانت مثبتة
    try:
        import mediapipe  # noqa: F401
        order.append("mediapipe")
    except Exception:
        pass

    # باقي الباكندات الشائعة مع DeepFace
    # (لا تحتاج لفحص وجود الحزمة؛ DeepFace يتكفل بالتحميل إن وجدت تبعياتها)
    for be in ["retinaface", "mtcnn", "opencv"]:
        if be not in order:
            order.append(be)

    return order


def _parse_liveness(record: dict):
    """
    توحيد ناتج الـ anti_spoofing باختلاف أسماء الحقول بين الإصدارات.
    """
    if not isinstance(record, dict):
        return None

    # بعض الإصدارات ترجع dict لكل وجه تحت مفاتيح مختلفة
    live_bool = None
    live_score = None
    threshold = float(record.get("threshold", 0.85))

    if "score" in record and record["score"] is not None:
        try:
            live_score = float(record["score"])
            live_bool = live_score >= threshold
        except Exception:
            live_score = None

    for key in ("liveness", "is_real", "real"):
        v = record.get(key)
        if isinstance(v, (bool, np.bool_)):
            live_bool = bool(v)
            break

    return {
        "is_live": live_bool,
        "score": live_score,
        "threshold": threshold,
        "raw": {
            k: record.get(k) for k in ("liveness", "is_real", "real", "score", "threshold")
        }
    }


# ---------- Core ----------
def analyze_frame(frame, anti_spoof=True):
    """
    يحلل إطار (صورة واحدة) ويُرجع:
    ok, backend, confidence, box, liveness?, face_thumb_b64?
    """
    rgb = _ensure_rgb(frame)
    rgb = _boost_for_detection(rgb)

    backends = _available_backends()
    last_err = None

    for be in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=rgb,               # numpy RGB
                detector_backend=be,
                enforce_detection=False,
                align=True,
                anti_spoofing=False         # نؤجلها الآن
            )
        except Exception as e:
            last_err = e
            log.exception("DeepFace backend '%s' failed in extract_faces", be)
            continue

        if not faces:
            continue

        # اختر أعلى ثقة
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
                if isinstance(live_res, list) and len(live_res) > 0:
                    # بعض الإصدارات تضع نتيجة الوجه الأول مباشرة
                    lr = live_res[0] if isinstance(live_res[0], dict) else {}
                    liveness = _parse_liveness(lr)
            except Exception:
                # لا نفشل الكاشف بسبب فشل anti_spoof
                log.debug("anti_spoofing failed for backend '%s'", be, exc_info=True)
                liveness = None

        return {
            "ok": True,
            "backend": be,
            "confidence": float(best.get("confidence") or 0.0),
            "box": best.get("facial_area"),
            "liveness": liveness,
            "face_thumb_b64": _to_base64(face_rgb),
        }

    # لم ننجح بأي باكند
    return {
        "ok": False,
        "error": "No face detected after trying available backends",
        "last_exception": str(last_err) if last_err else None,
        "tried_backends": backends,
    }


# ---------- Route ----------
@liveness_blueprint.route("/liveness", methods=["POST"])
def analyze():
    """
    يقبل multipart/form-data يحتوي حقل 'image'.
    يعيد JSON مع نتيجة الكشف.
    """
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "field 'image' is required"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"ok": False, "error": "empty image"}), 400

        # نفك الصورة بـ OpenCV (BGR)
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
