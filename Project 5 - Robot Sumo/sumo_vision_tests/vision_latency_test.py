"""How long from a blue edge appearing in front of the camera to the laptop KNOWING about it?

This is the delay that matters for the front edge: camera exposure, the UnoQ's processing and its 10-per-second publishing,
the broker, and the trip to this computer. It flashes a window between black and pure blue. Point the robot's camera at the
window so the BLUE FILLS THE BOTTOM THIRD of the picture (that is where the vision app looks for tape). The vision app reports
tape_ahead jumping up; this times how long that takes, using this computer's clock for both ends.

    python vision_latency_test.py --brain "Project 5 - Robot Sumo/sumo_brain.py"

The result includes the screen's own delay (about 20 ms), so it reads slightly high.
No robot handy? Check the tool itself (needs a broker; test.mosquitto.org is the default):   python vision_latency_test.py --self-test
"""

import argparse
import random
import sys
import threading
import time

import numpy as np

from common import VirtualScreen, fmt_ms, load_thresholds, summarize
from vision_check import DEFAULT_BROKER, DEFAULT_TOPIC, Listener


class BlueWindow:
    """The real flashing window: pure blue (on) or black (off)."""

    def __init__(self, fullscreen=False, size=(720, 1280)):
        import cv2
        self.cv2, self.size = cv2, size
        cv2.namedWindow("flash", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow("flash", size[1], size[0])

    def set(self, on):
        img = np.zeros((*self.size, 3), np.uint8)
        if on:
            img[:] = (255, 0, 0)                                  # BGR: pure blue
        self.cv2.imshow("flash", img)
        self.cv2.waitKey(1)

    def instructions(self):
        img = np.full((*self.size, 3), 90, np.uint8)
        lines = ["Point the robot's camera at this window, 20 to 30 cm away,",
                 "so the window fills at LEAST the bottom third of the picture.",
                 "Dim the room lights. Keep the robot still.", "", "Press SPACE to start (q to quit)."]
        for i, text in enumerate(lines):
            self.cv2.putText(img, text, (60, 150 + 60 * i), self.cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        while True:
            self.cv2.imshow("flash", img)
            k = self.cv2.waitKey(50) & 0xFF
            if k == ord(" "):
                return True
            if k == ord("q"):
                return False

    def close(self):
        self.cv2.destroyAllWindows()


def level_between(listener, t0, t1):
    vals = [float(m.get("tape_ahead", 0.0)) for _, m in listener.between(t0, t1)]
    return float(np.median(vals)) if vals else float("nan")


def measure(listener, screen, trials=10, settle=(1.2, 2.0), timeout=3.0, log=print):
    """Flash the screen and time how long tape_ahead takes to follow. Returns a dict with the delays in ms."""
    screen.set(False)
    time.sleep(1.5)
    t = time.perf_counter()
    time.sleep(0.8)
    low = level_between(listener, t, time.perf_counter())
    screen.set(True)
    time.sleep(1.5)
    t = time.perf_counter()
    time.sleep(0.8)
    high = level_between(listener, t, time.perf_counter())
    log(f"tape_ahead with a black screen: {low:.2f};  with the blue screen: {high:.2f}")
    if not (low == low and high == high):
        return {"error": "No camera messages arrived. Is the sumo-vision app running, and is this the right broker and topic?"}
    if low > 0.3:
        return {"error": f"tape_ahead is already {low:.2f} with a BLACK screen. The camera is seeing blue that isn't the flash "
                         "(its own apron?). Fix that first: run vision_check.py --guided."}
    if high - low < 0.3:
        return {"error": f"The camera barely reports the blue flash (only {high:.2f}). Move it closer so the blue fills the bottom "
                         "third of the picture, and dim the room lights. If it still fails, BLUE_LO/BLUE_HI may not match a pure blue screen."}
    threshold = (low + high) / 2
    out = {"rise": [], "fall": [], "missed": 0, "low": low, "high": high}
    state = True
    for _ in range(trials * 2):
        time.sleep(random.uniform(*settle))
        new = not state
        screen.set(new)
        t_flip = time.perf_counter()
        found = None
        while found is None and time.perf_counter() < t_flip + timeout:
            for ft, msg in listener.between(t_flip, time.perf_counter()):
                if (float(msg.get("tape_ahead", 0.0)) > threshold) == new:
                    found = ft
                    break
            time.sleep(0.005)
        state = new
        if found is None:
            out["missed"] += 1
        else:
            out["rise" if new else "fall"].append((found - t_flip) * 1000.0)
    return out


def verdict(median_ms, speed_mps):
    cm = median_ms / 1000.0 * speed_mps * 100.0
    if median_ms < 250:
        feel = "GOOD (rule of thumb: under 250 ms)"
    elif median_ms < 500:
        feel = "USABLE, but the robot needs a wide margin from the edge (250 to 500 ms)"
    else:
        feel = "SLOW: over 500 ms, so the camera's edge warning arrives late"
    return (f"{feel}.\nAt {speed_mps:.2f} m/s (an example speed: use your robot's real one) the robot moves about {cm:.0f} cm in that time. "
            "The front camera only helps if it sees the tape at least that far ahead.")


def report(res, speed_mps, log=print):
    if "error" in res:
        log("ERROR: " + res["error"])
        return None
    allv = res["rise"] + res["fall"]
    log("\n                       n   median      95%     worst   (ms)")
    for name, key in (("black -> blue", "rise"), ("blue -> black", "fall"), ("both", None)):
        s = summarize(allv if key is None else res[key])
        if s["n"]:
            log(f"  {name:<16}{s['n']:>4}   {fmt_ms(s['median'])}   {fmt_ms(s['p95'])}   {fmt_ms(s['max'])}")
    log(f"  flashes the camera never reported: {res['missed']}")
    if not allv:
        log("No flash was ever detected.")
        return None
    median = summarize(allv)["median"]
    log(verdict(median, speed_mps))
    return median


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--brain", help="path to sumo_brain.py (kept so every tool takes the same option)")
    p.add_argument("--trials", type=int, default=10, help="flashes in each direction")
    p.add_argument("--speed-mps", type=float, default=0.30, help="robot speed in metres per second, to turn delay into distance")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--self-test", action="store_true", help="measure a simulated camera instead of the robot")
    p.add_argument("--simulate-ms", type=float, default=300.0)
    a = p.parse_args()
    load_thresholds(a.brain)

    stop = threading.Event()
    topic = a.topic
    if a.self_test:
        from fake_vision import run_vision
        screen, topic, settle = VirtualScreen(), "selftest/vision", (0.4, 0.7)
        threading.Thread(target=run_vision, daemon=True, kwargs=dict(
            broker=a.broker, port=a.port, topic=topic, scenario="empty", delay_s=a.simulate_ms / 1000.0,
            screen=screen, stop=stop, quiet=True)).start()
        time.sleep(1.0)
        print(f"Self-test: a simulated camera pipeline that is {a.simulate_ms:.0f} ms behind, publishing 10 times a second.")
    else:
        screen, settle = BlueWindow(a.fullscreen), (1.2, 2.0)
        if not screen.instructions():
            return
    listener = Listener(a.broker, a.port, topic)
    try:
        report(measure(listener, screen, a.trials, settle), a.speed_mps)
    finally:
        listener.close()
        stop.set()
        if hasattr(screen, "close"):
            screen.close()


if __name__ == "__main__":
    main()
