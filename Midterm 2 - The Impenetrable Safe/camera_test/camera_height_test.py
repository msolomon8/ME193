"""
Camera height test - find how high the webcam must be above the hand area.

HOW TO USE
  1. Tape a ruler (or a tape measure) flat on the desk.
  2. Hold/clamp the webcam pointing straight down at a measured height H (lens to desk).
  3. Run:   ../.venv/bin/python camera_height_test.py          (tries camera 1, then 0)
            ../.venv/bin/python camera_height_test.py 0        (force a camera index)
     (camera 0 is usually the Mac's built-in camera - you want the USB webcam)
  4. Read off the ruler how many cm fit across the image from the left edge to the right edge
     (the white tick marks at the edges help). Type that into the table below as "visible width".
  5. Repeat at 2-3 heights. The script also finds a red square on the desk and tells you how much
     of the frame it fills.

KEYS:  s = save snapshot    q = quit

MATH:  visible_width = 2 * H * tan(HFOV / 2)  ->  HFOV = 2 * atan(visible_width / (2H))
       Needed height for a target zone of size Z (use the SHORT side of the image):
           H_needed = (Z / 2) / tan(VFOV / 2)
"""
import sys
import time

import cv2
import numpy as np


def open_camera():
    indexes = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 0]
    for i in indexes:
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera index {i}")
            return cap
        cap.release()
    sys.exit("No camera found. Plug in the webcam, or pass an index, e.g. `python camera_height_test.py 2`.")


def find_red_square(frame):
    """Return the bounding box (x, y, w, h) of the biggest red region, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # red wraps around hue 0, so use two ranges
    mask = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 70), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 2000:
        return None
    return cv2.boundingRect(c)


def main():
    cap = open_camera()
    # ask for 1280x720; the camera may give something else
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        view = frame.copy()

        # crosshair + edge ticks so you can read the ruler at the image edges
        cv2.line(view, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
        cv2.line(view, (0, H // 2), (W, H // 2), (255, 255, 255), 1)
        for x in (2, W - 3):
            cv2.line(view, (x, H // 2 - 30), (x, H // 2 + 30), (255, 255, 255), 3)

        # centered square = the biggest square crop we could train on
        s = min(H, W)
        x0, y0 = (W - s) // 2, (H - s) // 2
        cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (255, 200, 0), 2)

        # inner box at 80% of the square = where a spread hand should fit (leaves a margin)
        m = int(s * 0.10)
        cv2.rectangle(view, (x0 + m, y0 + m), (x0 + s - m, y0 + s - m), (0, 165, 255), 2)

        info = [f"frame {W}x{H}px   cyan = square crop   orange = keep spread hand inside"]
        box = find_red_square(frame)
        if box:
            x, y, w, h = box
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fill = max(w, h) / min(W, H) * 100
            info.append(f"red square: {w}x{h}px = {fill:.0f}% of the short side")
            if fill > 90:
                info.append("-> TOO CLOSE: raise the camera")
            elif fill < 50:
                info.append("-> far: lower the camera (or crop is fine, just less detail)")
            else:
                info.append("-> good range (60-85% is ideal)")
        else:
            info.append("no red square detected")

        for i, t in enumerate(info):
            cv2.putText(view, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(view, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("camera height test", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"snapshot_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(name, frame)
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
