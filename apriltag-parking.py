"""Detect an AprilTag and show simple parking guidance.

Run the camera mode with:
    ./ME193/le-venv/bin/python apriltag-parking.py

Generate a printable tag with:
    ./ME193/le-venv/bin/python apriltag-parking.py --generate --id 0

The distance estimate assumes the camera has been calibrated.  The default
camera matrix is only a reasonable starting point; replace it with measured
calibration values for accurate distance.
"""

import argparse
import os

import cv2
import numpy as np


CAMERA_INDEX = 0
TAG_SIZE_METERS = 0.10
PARKING_DISTANCE_METERS = 0.40
CENTER_TOLERANCE_PIXELS = 45
WINDOW_NAME = "AprilTag Parking"


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


def parking_message(center_x, frame_width, distance):
    offset = center_x - frame_width / 2
    if distance is not None and distance <= PARKING_DISTANCE_METERS:
        return "STOP - parked", (0, 220, 0)
    if offset < -CENTER_TOLERANCE_PIXELS:
        return "STEER LEFT", (0, 165, 255)
    if offset > CENTER_TOLERANCE_PIXELS:
        return "STEER RIGHT", (0, 165, 255)
    return "DRIVE FORWARD", (0, 255, 255)


def run_camera(camera_index, tag_size):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    detector = create_detector()
    camera = None
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            frame_height, frame_width = frame.shape[:2]
            if camera is None:
                camera = camera_matrix(frame_width, frame_height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            message = "NO APRILTAG"
            message_color = (0, 0, 255)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rotations, translations, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, tag_size, camera, np.zeros((5, 1), dtype=np.float32)
                )

                for index, tag_id in enumerate(ids.flatten()):
                    cv2.drawFrameAxes(
                        frame,
                        camera,
                        np.zeros((5, 1), dtype=np.float32),
                        rotations[index],
                        translations[index],
                        tag_size * 0.5,
                    )
                    points = corners[index][0]
                    center_x = float(points[:, 0].mean())
                    distance = float(translations[index][2])
                    message, message_color = parking_message(
                        center_x, frame_width, distance
                    )
                    cv2.putText(
                        frame,
                        f"ID {tag_id}  distance: {distance:.2f} m",
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
        capture.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_METERS)
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
        run_camera(args.camera, args.tag_size)


if __name__ == "__main__":
    main()