"""Binarize an image, dilate the white, and outline it.

Usage:
    python binarize.py <path-to-image> [output-path]

Opens a window with two sliders:
    Threshold   (0-255, default 128) - pixels above this become white
    Dilate      (0-20, default 0)    - how many iterations to grow the
                                        white regions by

The outline is computed as (dilated image - original thresholded
image), which leaves only the ring of pixels that got added by the
dilation - i.e. the outline of the white regions. The preview updates
live as you drag either slider.

Controls:
    s   save the current outline image
    q   quit (or press Esc)
"""

import os
import sys

import cv2
import numpy as np

DEFAULT_THRESHOLD = 128
DEFAULT_DILATE = 0
WINDOW_NAME = "Binarize (s = save, q = quit)"


def binarize(gray_image, threshold):
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def dilate_white(binary_image, iterations):
    if iterations <= 0:
        return binary_image
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary_image, kernel, iterations=iterations)


def outline(binary_image, dilated_image):
    return cv2.subtract(dilated_image, binary_image)


def main():
    if len(sys.argv) < 2:
        print("Usage: python binarize.py <path-to-image> [output-path]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_binarized{ext}"

    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print(f"Could not open image: {input_path}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Threshold", WINDOW_NAME, DEFAULT_THRESHOLD, 255, lambda _: None)
    cv2.createTrackbar("Dilate", WINDOW_NAME, DEFAULT_DILATE, 20, lambda _: None)

    while True:
        threshold = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        dilate_iterations = cv2.getTrackbarPos("Dilate", WINDOW_NAME)
        binary_image = binarize(gray_image, threshold)
        dilated_image = dilate_white(binary_image, dilate_iterations)
        result_image = outline(binary_image, dilated_image)
        cv2.imshow(WINDOW_NAME, result_image)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):  # 'q' or Esc
            break
        if key == ord("s"):
            cv2.imwrite(output_path, result_image)
            print(
                f"Saved outline (threshold={threshold}, dilate={dilate_iterations}) "
                f"to {output_path}"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
