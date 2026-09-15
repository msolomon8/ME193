"""Practice: track wrist positions with the webcam before adding motors."""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Indices into the 33-point BlazePose landmark list.
LEFT_WRIST = 15
RIGHT_WRIST = 16


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose tracking model (one-time setup)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def list_cameras(max_index=5):
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available


def choose_camera():
    cameras = list_cameras()
    if not cameras:
        raise RuntimeError("No cameras found.")
    if len(cameras) == 1:
        return cameras[0]

    print("Available cameras:")
    for i in cameras:
        print(f"  {i}")
    choice = int(input("Enter the camera index to use: "))
    return choice


def main():
    ensure_model_downloaded()
    camera_index = choose_camera()
    cap = cv2.VideoCapture(camera_index)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Mirror the image so moving your hand right moves the dot right on screen.
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        center_y = height // 2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Zero-velocity line: wrists above it will later mean "forward", below means "backward".
        cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 2)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            wrists = {
                "left": landmarks[LEFT_WRIST],
                "right": landmarks[RIGHT_WRIST],
            }

            for label, wrist in wrists.items():
                x = int(wrist.x * width)
                y = int(wrist.y * height)
                color = (0, 255, 0) if label == "left" else (0, 0, 255)
                cv2.circle(frame, (x, y), 12, color, -1)

                # Positive = wrist above the line, negative = below. This is the value
                # that will eventually become motor speed.
                speed = center_y - y
                cv2.putText(
                    frame,
                    f"{label}: {speed}",
                    (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        cv2.imshow("Wrist Tracking Practice", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
