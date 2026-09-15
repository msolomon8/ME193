"""Drive the LEGO Double Motor car using hand gestures.

Thumb up   -> drive forward
Thumb down -> drive backward
Anything else (including no hand detected) -> stop
"""

import os
import time
import urllib.request

import cv2
import legoeducation as le
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

DRIVE_SPEED = 40  # percent, -100 to 100


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading gesture recognition model (one-time setup)...")
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


def connect_car():
    print("Scanning for Double Motor hub...")
    car = le.DoubleMotor()
    car.connect()
    print("Connected.")
    return car


def drive(car, gesture_name, current_state):
    """Only sends a new command when the gesture actually changes,
    so we're not spamming Bluetooth commands every frame."""
    if gesture_name == "Thumb_Up":
        new_state = "forward"
    elif gesture_name == "Thumb_Down":
        new_state = "backward"
    else:
        new_state = "stop"

    if new_state == current_state:
        return current_state

    if new_state == "forward":
        car.movement_move_tank(DRIVE_SPEED, DRIVE_SPEED, blocking=False)
    elif new_state == "backward":
        car.movement_move_tank(-DRIVE_SPEED, -DRIVE_SPEED, blocking=False)
    else:
        car.movement_stop(blocking=False)

    return new_state


def main():
    ensure_model_downloaded()
    camera_index = choose_camera()
    cap = cv2.VideoCapture(camera_index)

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    start_time = time.time()

    car = connect_car()
    state = "stop"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            gesture_name = "None"
            if result.gestures:
                top_gesture = result.gestures[0][0]
                gesture_name = top_gesture.category_name

            state = drive(car, gesture_name, state)

            cv2.putText(
                frame,
                f"Gesture: {gesture_name}  |  State: {state}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Gesture Car Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        car.movement_stop(blocking=False)
        car.disconnect()
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
