"""Test a phone camera feed on the Mac.

This test does not connect to the LEGO robot. It checks whether the Mac can
receive and display video from either an HTTP stream or a Mac camera device.

Example:
    python test-ip-webcam.py --url http://PHONE_IP:8080/video
    python test-ip-webcam.py --camera 0

Press q in the video window to quit.
"""

import argparse
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        help="An HTTP/MJPEG video URL from a phone camera app",
    )
    source.add_argument(
        "--camera",
        type=int,
        help="Mac camera index, such as the NDI virtual camera index",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.url if args.url else args.camera
    print(f"Trying to connect to: {source}")
    print("Press q in the video window to stop.")

    stream = cv2.VideoCapture(source)
    if not stream.isOpened():
        raise RuntimeError(
            "Could not open the camera. For NDI, install NDI Tools on the Mac, "
            "start NDI Webcam Input, and try another --camera index."
        )

    frame_count = 0
    started_at = time.monotonic()
    last_report = started_at

    try:
        while True:
            received, frame = stream.read()
            if not received or frame is None:
                print("The stream opened, but no video frame was received.")
                break

            frame_count += 1
            cv2.putText(
                frame,
                f"IP Webcam connected | frames: {frame_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("IP Webcam Test", frame)

            now = time.monotonic()
            if now - last_report >= 2:
                elapsed = now - started_at
                fps = frame_count / elapsed if elapsed else 0
                print(f"Receiving frames: {frame_count} total ({fps:.1f} FPS)")
                last_report = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()

    if frame_count:
        print(f"Success: received {frame_count} video frames from the phone.")
    else:
        print("Failure: received zero video frames.")


if __name__ == "__main__":
    main()
