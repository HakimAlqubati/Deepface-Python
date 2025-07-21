# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
from deepface.commons.logger import Logger

from flask import jsonify
from deepface.modules import detection


def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.int32, np.int64)):
        return int(val)
    return val

def recursive_convert(data):
    if isinstance(data, dict):
        return {k: recursive_convert(v) for k, v in data.items()}
    if isinstance(data, list):
        return [recursive_convert(item) for item in data]
    return to_serializable(data)

logger = Logger()

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str) -> Union[str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        img = image_utils.load_image_from_file_storage(file)

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = request.get_json() or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=True,
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", True),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])
    # actions is the only argument instance of list or tuple
    # if request is form data, input args can either be text or file
    if isinstance(actions, str):
        actions = (
            actions.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "")
            .split(",")
        )

    demographies = service.analyze(
        img_path=img,
        actions=actions,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=True,
    )

    logger.debug(demographies)
    print('qubati',demographies)
    print('converted:', recursive_convert(demographies))
    print('hakimahmed',jsonify(recursive_convert(demographies)))
    return jsonify(recursive_convert(demographies))
    return demographies


@blueprint.route("/liveness", methods=["POST"])
def liveness():
    """
    كشف الحيوية للوجه (liveness / anti-spoofing)
    """
    try:
        # استخرج الصورة من الطلب (نفس منطق extract_image_from_request)
        if request.files:
            file = request.files.get("img")
            if file is None or file.filename == "":
                return jsonify({"error": "No image uploaded."}), 400
            img = file
        elif request.is_json or request.form:
            input_args = request.get_json() or request.form.to_dict()
            img = input_args.get("img")
            if not img:
                return jsonify({"error": "No image provided."}), 400
        else:
            return jsonify({"error": "No image in request."}), 400

        # كشف الوجه والحيوية عبر extract_faces
        faces = detection.extract_faces(
            img_path=img,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            anti_spoofing=True
        )

        # لنأخذ أول وجه فقط (إذا الصورة فيها أكثر من وجه)
        if not faces:
            return jsonify({"error": "No face detected."}), 404

        # اختصر الرد وركّز فقط على الحيوية والثقة ومعلومات الوجه
        main = faces[0]
        result = {
            "is_real": main.get("is_real"),
            "antispoof_score": main.get("antispoof_score"),
            "confidence": main.get("confidence"),
            "facial_area": main.get("facial_area"),
        }
        return jsonify(recursive_convert(result)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500