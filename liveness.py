import cv2
from deepface.modules import detection

def get_head_direction(landmarks):
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not left_eye or not right_eye:
        return None
    dx = left_eye[0] - right_eye[0]
    if dx > 15:
        return "right"
    elif dx < -15:
        return "left"
    else:
        return "center"

def main():
    cap = cv2.VideoCapture(0)
    print("Please move your head LEFT and RIGHT during the test.")
    directions = set()
    real_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        try:
            faces = detection.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
                anti_spoofing=True
            )
            if faces:
                face = faces[0]
                # Draw rectangle around the face
                fa = face["facial_area"]
                cv2.rectangle(frame, (fa["x"], fa["y"]), (fa["x"]+fa["w"], fa["y"]+fa["h"]), (0,255,0), 2)
                # Detect head direction
                direction = get_head_direction(fa)
                if direction:
                    directions.add(direction)
                    cv2.putText(frame, f"Direction: {direction}", (fa["x"], fa["y"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                # Liveness detection
                is_real = face.get("is_real", False)
                score = face.get("antispoof_score", 0)
                color = (0,255,0) if is_real and score > 0.7 else (0,0,255)
                text = f"Liveness: {'Real' if is_real else 'Fake'} ({score:.2f})"
                cv2.putText(frame, text, (fa["x"], fa["y"]+fa["h"]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Stats
                total_frames += 1
                if is_real and score > 0.7:
                    real_frames += 1
        except Exception as e:
            pass

        cv2.imshow("Liveness & Head Movement Detection", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("------ Summary ------")
    print(f"Real (liveness) frames: {real_frames}/{total_frames}")
    print(f"Detected head directions: {directions}")
    if real_frames / (total_frames or 1) > 0.6 and {"left", "right"}.issubset(directions):
        print("✅ Liveness and movement challenge PASSED!")
    else:
        print("❌ Liveness or movement challenge FAILED.")

if __name__ == "__main__":
    main()
