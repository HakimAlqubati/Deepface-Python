# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np
import requests
import tempfile
import os

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

api_blueprint = Blueprint("api", __name__)

# pylint: disable=no-else-return, broad-except


@api_blueprint.route("/")
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


@api_blueprint.route("/represent", methods=["POST"])
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


@api_blueprint.route("/verify", methods=["POST"])
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


@api_blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )
    print('sdf')


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

 


EMPLOYEE_API = "https://workbench.ressystem.com/api/employees/simple-list"
DISTANCE_THRESHOLD = 0.55  # Adjust based on your testing

@api_blueprint.route("/recognize", methods=["POST"])
def recognize():
    # 1. Receive uploaded face image
    if 'img' not in request.files:
        return jsonify({"error": "Please upload an image in the 'img' field."}), 400

    uploaded_file = request.files['img']
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    uploaded_file.save(temp_input.name)
    temp_input.close()

    # 2. Fetch employee list from Laravel API
    try:
        employees = requests.get(EMPLOYEE_API).json()
    except Exception as e:
        os.remove(temp_input.name)
        return jsonify({"error": "Failed to fetch employees.", "details": str(e)}), 500

    # 3. Compare with each employee's avatar, track the best match
    best_distance = None
    best_employee = None

    for emp in employees:
        avatar_url = emp["avatar_url"]
        try:
            # Download avatar temporarily
            response = requests.get(avatar_url, stream=True, timeout=6)
            if response.status_code != 200:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_avatar:
                for chunk in response.iter_content(1024):
                    temp_avatar.write(chunk)
                temp_avatar_path = temp_avatar.name

            # Run DeepFace verification
            result = DeepFace.verify(
                img1_path=temp_input.name,
                img2_path=temp_avatar_path,
                model_name="Facenet",     # Change model as needed
                detector_backend="opencv",
                enforce_detection=False   # Set True if you want strict detection
            )
            distance = result["distance"]

            if (best_distance is None) or (distance < best_distance):
                best_distance = distance
                best_employee = emp

            os.remove(temp_avatar_path)

        except Exception as ex:
            continue

    os.remove(temp_input.name)

    # 4. Return result
    if best_employee and best_distance < DISTANCE_THRESHOLD:
        return jsonify({
            "matched": True,
            "distance": best_distance,
            "employee": best_employee
        })
    else:
        return jsonify({
            "matched": False,
            "distance": best_distance,
            "message": "No sufficiently close match found."
        })
from flask import Flask

app = Flask(__name__)
app.register_blueprint(api_blueprint, url_prefix="/api")


from scipy.spatial.distance import cosine

@api_blueprint.route("/recognize-v2", methods=["POST"])
def recognize_v2():
    if 'img' not in request.files:
        return jsonify({"error": "Please upload an image in the 'img' field."}), 400

    uploaded_file = request.files['img']
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    uploaded_file.save(temp_input.name)
    temp_input.close()

    try:
        query_embedding = DeepFace.represent(
            img_path=temp_input.name,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]['embedding']
    except Exception as e:
        os.remove(temp_input.name)
        return jsonify({"error": "Failed to process input image", "details": str(e)}), 500

    os.remove(temp_input.name)

    try:
        response = requests.get("https://workbench.ressystem.com/api/face-data")
        employee_embeddings = response.json()
    except Exception as e:
        return jsonify({"error": "Failed to fetch stored embeddings", "details": str(e)}), 500

    best_employee = None
    best_distance = float("inf")

    for record in employee_embeddings:
        stored_embedding = record.get("embedding")
        if not stored_embedding:
            continue
        try:
            distance = cosine(query_embedding, stored_embedding)
        except Exception:
            continue

        if distance < best_distance:
            best_distance = distance
            best_employee = record

    if best_employee and best_distance < DISTANCE_THRESHOLD:
        return jsonify(recursive_convert({
            "matched": True,
            "distance": best_distance,
            "employee": best_employee
        }))
    else:
        return jsonify(recursive_convert({
            "matched": False,
            "distance": best_distance,
            "message": "No sufficiently close match found."
        }))

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

