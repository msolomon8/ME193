"""Convert an image to grayscale.

Usage:
    python grayscale.py <path-or-url-to-image> [output-path]

If no output path is given, the result is saved next to the input
with "_grayscale" added to the filename.
"""

import os
import sys
import urllib.request

from PIL import Image


def resolve_input(source):
    if source.startswith("http://") or source.startswith("https://"):
        local_path = "downloaded_image" + os.path.splitext(source.split("?")[0])[1]
        print(f"Downloading image from {source} ...")
        urllib.request.urlretrieve(source, local_path)
        return local_path
    return source


def main():
    if len(sys.argv) < 2:
        print("Usage: python grayscale.py <path-or-url-to-image> [output-path]")
        sys.exit(1)

    source = sys.argv[1]
    input_path = resolve_input(source)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_grayscale{ext}"

    image = Image.open(input_path)
    grayscale_image = image.convert("L")
    grayscale_image.save(output_path)
    print(f"Saved grayscale image to {output_path}")


if __name__ == "__main__":
    main()
