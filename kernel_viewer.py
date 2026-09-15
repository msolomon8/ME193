"""Step a kernel across a binary image, pixel by pixel, to see what it sees.

Usage:
    python kernel_viewer.py <path-to-image>

Loads the image, converts it to grayscale, and thresholds it (Threshold
slider, default 128) to get a binary image. The Row/Col sliders (or the
i/j/k/l keys) move a kernel window one pixel at a time so you can inspect
exactly which neighborhood a kernel sees at each position, and what
dilation or erosion would output for the center pixel.

Sliders:
    Threshold             black/white cutoff for binarizing (default 128)
    Kernel Size (x2+1)    0-7, mapped to kernel sizes 1x1-15x15 (default 3x3)
    Mode (0=Dilate,1=Erode)   which operation's output to preview
    Row / Col              center position of the kernel

Keys:
    i / k   move the kernel up / down one pixel
    j / l   move the kernel left / right one pixel
    m       toggle between dilate and erode
    q       quit (or Esc)

The left panel is a zoomed-in, gridded view of the neighborhood around the
kernel (orange box = kernel, red box = center pixel). The right panel shows
the whole image with a red crosshair marking where the kernel currently is.
The banner on top shows the coordinates, mode, and what that operation
would output for the center pixel given the current kernel: dilation
outputs white if ANY pixel under the kernel is white, erosion outputs
white only if ALL pixels under the kernel are white.
"""

import sys

import cv2
import numpy as np

WINDOW_NAME = "Kernel Viewer (i/j/k/l = move, m = mode, q = quit)"
ZOOM = 24  # pixels per cell in the zoomed view
CONTEXT_CELLS = 6  # extra cells of context shown around the kernel
FULL_VIEW_WIDTH = 360


def clamp(value, low, high):
    return max(low, min(high, value))


def build_zoom_panel(binary_image, row, col, half_k):
    height, width = binary_image.shape
    half_view = half_k + CONTEXT_CELLS
    r0 = clamp(row - half_view, 0, height - 1)
    r1 = clamp(row + half_view, 0, height - 1)
    c0 = clamp(col - half_view, 0, width - 1)
    c1 = clamp(col + half_view, 0, width - 1)

    crop = binary_image[r0 : r1 + 1, c0 : c1 + 1]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    zoomed = cv2.resize(crop_bgr, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

    for i in range(crop.shape[0] + 1):
        y = i * ZOOM
        cv2.line(zoomed, (0, y), (zoomed.shape[1], y), (60, 60, 60), 1)
    for j in range(crop.shape[1] + 1):
        x = j * ZOOM
        cv2.line(zoomed, (x, 0), (x, zoomed.shape[0]), (60, 60, 60), 1)

    kernel_size = 2 * half_k + 1
    kr0 = (row - half_k) - r0
    kc0 = (col - half_k) - c0
    top_left = (kc0 * ZOOM, kr0 * ZOOM)
    bottom_right = ((kc0 + kernel_size) * ZOOM, (kr0 + kernel_size) * ZOOM)
    cv2.rectangle(zoomed, top_left, bottom_right, (0, 165, 255), 2)

    center_r = row - r0
    center_c = col - c0
    cv2.rectangle(
        zoomed,
        (center_c * ZOOM, center_r * ZOOM),
        ((center_c + 1) * ZOOM, (center_r + 1) * ZOOM),
        (0, 0, 255),
        2,
    )
    return zoomed


def build_context_panel(binary_image, row, col):
    height, width = binary_image.shape
    scale = FULL_VIEW_WIDTH / width
    small = cv2.resize(
        binary_image, (FULL_VIEW_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA
    )
    small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    marker = (int(col * scale), int(row * scale))
    cv2.drawMarker(small_bgr, marker, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    return small_bgr


def pad_to_height(panel, height):
    if panel.shape[0] == height:
        return panel
    pad = np.zeros((height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
    return np.vstack([panel, pad])


def main():
    if len(sys.argv) < 2:
        print("Usage: python kernel_viewer.py <path-to-image>")
        sys.exit(1)

    gray_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print(f"Could not open image: {sys.argv[1]}")
        sys.exit(1)

    height, width = gray_image.shape
    row, col = height // 2, width // 2

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Threshold", WINDOW_NAME, 128, 255, lambda _: None)
    cv2.createTrackbar("Kernel Size (x2+1)", WINDOW_NAME, 1, 7, lambda _: None)
    cv2.createTrackbar("Mode (0=Dilate,1=Erode)", WINDOW_NAME, 0, 1, lambda _: None)
    cv2.createTrackbar("Row", WINDOW_NAME, row, height - 1, lambda _: None)
    cv2.createTrackbar("Col", WINDOW_NAME, col, width - 1, lambda _: None)

    while True:
        threshold = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        half_k = cv2.getTrackbarPos("Kernel Size (x2+1)", WINDOW_NAME)
        erode_mode = cv2.getTrackbarPos("Mode (0=Dilate,1=Erode)", WINDOW_NAME) == 1
        row = cv2.getTrackbarPos("Row", WINDOW_NAME)
        col = cv2.getTrackbarPos("Col", WINDOW_NAME)

        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        neighborhood = binary_image[
            clamp(row - half_k, 0, height - 1) : row + half_k + 1,
            clamp(col - half_k, 0, width - 1) : col + half_k + 1,
        ]
        would_be_white = bool(neighborhood.min() == 255) if erode_mode else bool(neighborhood.max() == 255)

        zoom_panel = build_zoom_panel(binary_image, row, col, half_k)
        context_panel = build_context_panel(binary_image, row, col)
        target_height = max(zoom_panel.shape[0], context_panel.shape[0])
        zoom_panel = pad_to_height(zoom_panel, target_height)
        context_panel = pad_to_height(context_panel, target_height)
        combined = np.hstack([zoom_panel, context_panel])

        kernel_size = 2 * half_k + 1
        mode_name = "erode" if erode_mode else "dilate"
        status = (
            f"row={row} col={col}  kernel={kernel_size}x{kernel_size}  mode={mode_name}  "
            f"output -> {'WHITE' if would_be_white else 'black'}"
        )
        banner = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            banner, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        combined = np.vstack([banner, combined])

        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("i"):
            cv2.setTrackbarPos("Row", WINDOW_NAME, clamp(row - 1, 0, height - 1))
        elif key == ord("k"):
            cv2.setTrackbarPos("Row", WINDOW_NAME, clamp(row + 1, 0, height - 1))
        elif key == ord("j"):
            cv2.setTrackbarPos("Col", WINDOW_NAME, clamp(col - 1, 0, width - 1))
        elif key == ord("l"):
            cv2.setTrackbarPos("Col", WINDOW_NAME, clamp(col + 1, 0, width - 1))
        elif key == ord("m"):
            cv2.setTrackbarPos("Mode (0=Dilate,1=Erode)", WINDOW_NAME, 0 if erode_mode else 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
