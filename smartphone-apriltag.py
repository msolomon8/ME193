"""Center a robot on a stationary AprilTag using a smartphone camera stream.

This script is intended for a robot with:
- a phone mounted in front of the robot, streaming video back to the laptop
- a LEGO Double Motor on the green 0991 connection card
- a stationary AprilTag in the environment

Typical smartphone sources:
- IP Webcam: http://PHONE_IP:8080/video
- DroidCam: http://PHONE_IP:4747/mjpegfeed
- Camo / other MJPEG/RTSP sources, if you prefer to use a different URL

Example:
    python smartphone-apriltag.py --stream http://192.168.1.25:8080/video
    python smartphone-apriltag.py --generate --id 0

The laptop does the computer vision work and then commands the robot to center
on the detected AprilTag using the LEGO Double Motor.
"""

import argparse
import os
import time

import cv2
import legoeducation as le
import numpy as np

from lelib import doubleMotor


CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = 997
WINDOW_NAME = "AprilTag Centering"

# A reasonable starting point for a small robot using a phone camera.
# These can be tuned after the first test run.
TURN_SPEED = 30
FORWARD_SPEED = 35
REVERSE_SPEED = 25
CENTER_TOLERANCE = 0.08
# When the tag is very small, it means the robot is far away.
TARGET_AREA = 3000
MIN_AREA = 800
MAX_AREA = 9000


def create_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


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


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def choose_tag(corners, ids):
    if ids is None:
        return None

    best_idx = None
    best_area = -1.0
    for index, _tag_id in enumerate(ids.flatten()):
        points = corners[index][0]
        area = cv2.contourArea(points.astype(np.float32))
        if area > best_area:
            best_area = area
            best_idx = index

    if best_idx is None:
        return None

    return corners[best_idx][0], int(ids[best_idx][0])


def center_command(center_x, frame_width, tag_area):
    """Return left/right motor commands to center the tag in the camera image.

    The robot can both turn to fix lateral error and move forward/backward to
    maintain a usable distance from the tag.
    """
    offset = (center_x - frame_width / 2) / max(frame_width / 2, 1)
    left_speed = 0
    right_speed = 0

    # Lateral correction: turn toward the tag when it is off-center.
    if abs(offset) > CENTER_TOLERANCE:
        turn = TURN_SPEED * (1.0 if abs(offset) > 0.30 else 0.55)
        if offset < 0:
            # Tag is left of center: robot should rotate toward the left.
            left_speed = turn
            right_speed = -turn
        else:
            # Tag is right of center: robot should rotate toward the right.
            left_speed = -turn
            right_speed = turn

    # Depth correction: if the tag is too small, move forward. If it is too large,
    # back up so the robot doesn't crash into the tag.
    if tag_area < MIN_AREA:
        forward_bias = FORWARD_SPEED
    elif tag_area > MAX_AREA:
        forward_bias = -REVERSE_SPEED
    else:
        forward_bias = 0

    # Combine the turn and forward/back correction.
    left_speed += forward_bias
    right_speed += forward_bias

    # Keep the resulting commands in the valid signed range.
    left_speed = clamp(int(round(left_speed)), -100, 100)
    right_speed = clamp(int(round(right_speed)), -100, 100)
    return left_speed, right_speed


def run_stream(args, motor):
    detector = create_detector()
    last_command = None
    last_print = time.monotonic()

    def send_command(left_speed, right_speed):
        nonlocal last_command
        command = (left_speed, right_speed)
        if command == last_command:
            return
        motor.movement_move_tank(left_speed, right_speed, blocking=False)
        last_command = command

    cap = cv2.VideoCapture(args.stream if args.stream else args.camera)
    if not cap.isOpened():
        if args.stream:
            raise RuntimeError(f"Could not open smartphone video stream: {args.stream}")
        raise RuntimeError(f"Could not open camera {args.camera}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Lost connection to video stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_height, frame_width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                send_command(0, 0)
                cv2.putText(
                    frame,
                    "NO APRILTAG",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                tag = choose_tag(corners, ids)
                if tag is None:
                    send_command(0, 0)
                    cv2.putText(
                        frame,
                        "TAG LOST",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 165, 255),
                        2,
                    )
                else:
                    points, tag_id = tag
                    center_x = float(points[:, 0].mean())
                    tag_area = cv2.contourArea(points.astype(np.float32))
                    left_speed, right_speed = center_command(center_x, frame_width, tag_area)
                    send_command(left_speed, right_speed)

                    cv2.putText(
                        frame,
                        f"ID {tag_id}  center={center_x:.0f}px  area={tag_area:.0f}",
                        (int(points[0, 0]), int(points[0, 1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"L={left_speed}  R={right_speed}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

            cv2.line(
                frame,
                (frame_width // 2, 0),
                (frame_width // 2, frame_height),
                (255, 255, 255),
                1,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if time.monotonic() - last_print > 2.0:
                last_print = time.monotonic()

    finally:
        send_command(0, 0)
        cap.release()
        cv2.destroyAllWindows()
        motor.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        default=os.environ.get("SMARTPHONE_STREAM_URL"),
        help=(
            "Phone video stream URL, such as http://192.168.1.25:8080/video or "
            "http://192.168.1.25:4747/mjpegfeed"
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Local webcam index to use as a fallback when no stream URL is provided",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--id", type=int, default=0, dest="tag_id")
    parser.add_argument("--output", default="apriltag-36h11-id0.png")
    parser.add_argument("--pixels", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate:
        generate_tag(args.tag_id, args.output, args.pixels)
        return

    motor = doubleMotor()
    print("Connecting to LEGO Double Motor on green 0991 card...")
    motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    print("Connected. Press q to stop.")
    print("Streaming from:", args.stream if args.stream else f"local camera {args.camera}")
    run_stream(args, motor)


if __name__ == "__main__":
    main()
