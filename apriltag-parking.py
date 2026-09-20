"""Detect an AprilTag and show simple parking guidance.

Run the camera mode with:
    ./ME193/le-venv/bin/python apriltag-parking.py

Generate a printable tag with:
    ./ME193/le-venv/bin/python apriltag-parking.py --generate --id 0

    The motor moves the detected tag horizontally until it reaches the center
    line of the camera image.
"""

import argparse
import os

import cv2
import legoeducation as le
import numpy as np

from lelib import doubleMotor


CAMERA_INDEX = 0
CENTER_TOLERANCE_PIXELS = 45
MOTOR_SPEED = 20
WINDOW_NAME = "AprilTag Parking"
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = 0997  # green connection card serial for the 0991 card


def create_detector():
    """Create an OpenCV AprilTag detector using the 36h11 family."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def camera_matrix(width, height):
    """Return an approximate matrix for an uncalibrated webcam."""
    focal_length = width
    return np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def generate_tag(tag_id, output_path, pixels):
    if not 0 <= tag_id < 587:
        raise ValueError("--id must be between 0 and 586 for AprilTag 36h11")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    quiet_zone = max(20, pixels // 12)
    tag = np.zeros((pixels, pixels), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, tag_id, pixels, tag, 1)
    marker = np.full(
        (pixels + quiet_zone * 2, pixels + quiet_zone * 2), 255, dtype=np.uint8
    )
    marker[quiet_zone:-quiet_zone, quiet_zone:-quiet_zone] = tag
    if not cv2.imwrite(output_path, marker):
        raise RuntimeError(f"Could not write marker image: {output_path}")
    print(f"Generated AprilTag {tag_id}: {os.path.abspath(output_path)}")


def centering_message(center_x, frame_width):
    horizontal_offset = center_x - frame_width / 2
    if horizontal_offset < -CENTER_TOLERANCE_PIXELS:
        return "MOVE RIGHT", (0, 255, 255), MOTOR_SPEED, MOTOR_SPEED
    if horizontal_offset > CENTER_TOLERANCE_PIXELS:
        return "MOVE LEFT", (0, 165, 255), -MOTOR_SPEED, -MOTOR_SPEED
    return "CENTERED - STOP", (0, 220, 0), 0, 0


def run_camera(camera_index, motor):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    detector = create_detector()
    last_motor_command = None

    def set_motor_speed(left_speed, right_speed):
        nonlocal last_motor_command
        command = (left_speed, right_speed)
        if command == last_motor_command:
            return
        motor.movement_move_tank(left_speed, right_speed, blocking=False)
        last_motor_command = command

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            frame_height, frame_width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            message = "NO APRILTAG"
            message_color = (0, 0, 255)
            set_motor_speed(0, 0)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for index, tag_id in enumerate(ids.flatten()):
                    points = corners[index][0]
                    center_x = float(points[:, 0].mean())
                    message, message_color, left_speed, right_speed = centering_message(
                        center_x, frame_width
                    )
                    set_motor_speed(left_speed, right_speed)
                    cv2.putText(
                        frame,
                        f"ID {tag_id}  center: {center_x:.0f}px",
                        (int(points[0, 0]), int(points[0, 1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            cv2.line(
                frame,
                (frame_width // 2, 0),
                (frame_width // 2, frame_height),
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                message_color,
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        set_motor_speed(0, 0)
        capture.release()
        cv2.destroyAllWindows()
        motor.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--id", type=int, default=0, dest="tag_id")
    parser.add_argument("--output", default="apriltag-36h11-id0.png")
    parser.add_argument("--pixels", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generate:
        generate_tag(args.tag_id, args.output, args.pixels)
    else:
        motor = doubleMotor()
        print("Connecting to LEGO Double Motor on green 0991 card...")
        motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        print("Connected. Press q to stop.")
        run_camera(args.camera, motor)


if __name__ == "__main__":
    main()