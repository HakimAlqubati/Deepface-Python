from deepface import DeepFace

img1 = "face1.jpg"
img2 = "face1-1.jpg"

result = DeepFace.verify(
    img1_path=img1,
    img2_path=img2,
    enforce_detection=False,
    detector_backend="retinaface"  # أو "mtcnn" أو "ssd" أو "opencv"
)
print(result)
