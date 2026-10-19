"""How fast could this board run a neural-network detector?  (a feasibility check, BEFORE you spend an afternoon collecting data)

Run it ON THE UNO Q (it only needs OpenCV, which the sumo-vision app already uses):

    python3 bench_detector.py model.onnx --size 320

Make a model to test with on the laptop, in the YOLO setup you used for HW4:

    pip install ultralytics
    yolo export model=yolov8n.pt format=onnx imgsz=320

It doesn't need to be a model that knows about robots: speed doesn't depend on what the network was trained on, only on its size.
The app can't use a neural network that is too slow, and the answer here tells you whether to go on.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def verdict(fps):
    """My rules of thumb for the sumo-vision app, which publishes about 10 times a second and also runs its colour code."""
    if fps >= 8:
        return "FAST ENOUGH to run on every frame alongside the colour code."
    if fps >= 3:
        return "USABLE if it only runs on every 3rd or 4th frame, or at a smaller --size."
    return "TOO SLOW on this board. Run it on the laptop instead (needs a way to get the picture there), or skip the network."


def benchmark(path, size=320, seconds=10.0, warmup=3):
    """Returns a dict: times in ms, fps, threads. Raises with a plain message if OpenCV can't load the model."""
    try:
        net = cv2.dnn.readNetFromONNX(path)
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV couldn't load {path}: {str(exc).strip().splitlines()[-1]}\n"
                           "Is it really an ONNX file? If it is, export it with a lower opset (opset=12), or update OpenCV on the board.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    rng = np.random.default_rng(0)
    blob = cv2.dnn.blobFromImage(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), 1 / 255.0, (size, size), swapRB=True)
    for _ in range(warmup):
        net.setInput(blob)
        net.forward()
    times = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(times) < 5:
        t = time.perf_counter()
        net.setInput(blob)
        net.forward()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    return {"n": len(times), "median_ms": median, "p95_ms": times[int(len(times) * 0.95)], "fps": 1000.0 / median,
            "threads": cv2.getNumThreads(), "cpus": os.cpu_count()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="an .onnx file")
    p.add_argument("--size", type=int, default=320, help="the square size the model was exported for")
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    if not os.path.exists(a.model):
        sys.exit(f"No such file: {a.model}")
    try:
        r = benchmark(a.model, a.size, a.seconds)
    except RuntimeError as exc:
        sys.exit(str(exc))
    print(f"{a.model} at {a.size}x{a.size}: {r['n']} runs on {r['cpus']} CPUs ({r['threads']} threads)")
    print(f"  median {r['median_ms']:.0f} ms per picture, 95% under {r['p95_ms']:.0f} ms  ->  {r['fps']:.1f} pictures per second")
    print(verdict(r["fps"]))
    print("(this times the network only; the colour code takes extra time on top, and the app has about 100 ms per loop)")


if __name__ == "__main__":
    main()
