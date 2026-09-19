"""AprilTag centering using an IP Webcam stream from the phone.

This version is intended for the IP Webcam app on the phone. The app exposes the
phone camera as a MJPEG stream, and the laptop connects to that stream directly.
You do not need a browser camera workaround for this version.

Example:
    python phone-browser-apriltag.py --stream http://192.168.1.25:8080/video

The phone and laptop must be on the same Wi‑Fi network.
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

TURN_SPEED = 30
FORWARD_SPEED = 35
REVERSE_SPEED = 25
CENTER_TOLERANCE = 0.08
MIN_AREA = 800
MAX_AREA = 9000


def create_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


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
    offset = (center_x - frame_width / 2) / max(frame_width / 2, 1)
    left_speed = 0
    right_speed = 0

    if abs(offset) > CENTER_TOLERANCE:
        turn = TURN_SPEED * (1.0 if abs(offset) > 0.30 else 0.55)
        if offset < 0:
            left_speed = turn
            right_speed = -turn
        else:
            left_speed = -turn
            right_speed = turn

    if tag_area < MIN_AREA:
        forward_bias = FORWARD_SPEED
    elif tag_area > MAX_AREA:
        forward_bias = -REVERSE_SPEED
    else:
        forward_bias = 0

    left_speed += forward_bias
    right_speed += forward_bias

    return (
        clamp(int(round(left_speed)), -100, 100),
        clamp(int(round(right_speed)), -100, 100),
    )


def run_stream(args, motor):
    detector = create_detector()
    last_command = None

    source = args.stream if args.stream else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if args.stream:
            raise RuntimeError(f"Could not open phone stream: {args.stream}")
        raise RuntimeError(f"Could not open local camera: {args.camera}")

    print(f"Opened stream: {source}")

    def send_command(left_speed, right_speed):
        nonlocal last_command
        command = (left_speed, right_speed)
        if command == last_command:
            return
        motor.movement_move_tank(left_speed, right_speed, blocking=False)
        last_command = command

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost connection to stream. Retrying...")
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
        help="IP Webcam stream URL, usually http://PHONE_IP:8080/video",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Fallback local webcam if you do not provide --stream",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--id", type=int, default=0, dest="tag_id")
    parser.add_argument("--output", default="apriltag-36h11-id0.png")
    parser.add_argument("--pixels", type=int, default=600)
    return parser.parse_args()


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


def main():
    args = parse_args()

    if args.generate:
        generate_tag(args.tag_id, args.output, args.pixels)
        return

    motor = doubleMotor()
    print("Connecting to LEGO Double Motor on green 0997 card...")
    try:
        motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        print("Connected.")
    except Exception as exc:
        print(f"Green-card connection failed: {exc}")
        try:
            print("Trying first available Double Motor...")
            motor.connect(card_serial=None, card_color=None)
            print("Connected to first available Double Motor.")
        except Exception as exc2:
            print(f"No double motor found: {exc2}")
            raise

    if args.stream:
        print(f"Using IP Webcam stream: {args.stream}")
    else:
        print(f"Using local camera index: {args.camera}")

    print("Press q in the OpenCV window to stop.")
    run_stream(args, motor)


if __name__ == "__main__":
    main()
