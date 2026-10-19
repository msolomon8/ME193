"""Shared helpers for the sumo vision tests: statistics, an MQTT client that works with paho 1.x and 2.x,
the thresholds read from the behavior tree (so these tests always use YOUR numbers), and a pretend screen for self-tests."""

import importlib.util
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------- statistics
def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(values):
    v = sorted(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": v[0], "median": percentile(v, 50), "p95": percentile(v, 95),
            "p99": percentile(v, 99), "max": v[-1], "mean": sum(v) / len(v)}


def fmt_ms(x):
    return "   n/a" if x is None or x != x else f"{x:6.0f}"


# ------------------------------------------------- the behavior tree's thresholds
DEFAULTS = {"VISION_TIMEOUT": 0.5, "TAPE_AHEAD": 0.08, "OUT_CONFIRM": 0.3, "OUT_WINDOW": 3.0}


def load_thresholds(brain_path=None):
    """The numbers the behavior tree compares the camera's messages with. Reads them from sumo_brain.py if a path is given
    (so the tests follow any tuning your teammate does); anything it can't find there keeps the value from when this was written.
    Returns (dict, note about where they came from)."""
    th = dict(DEFAULTS)
    if not brain_path:
        return th, "built-in defaults (pass --brain path/to/sumo_brain.py to use your own)"
    try:
        spec = importlib.util.spec_from_file_location("sumo_brain_for_tests", str(Path(brain_path)))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:                                  # noqa: BLE001 - say why and carry on with defaults
        return th, f"built-in defaults (couldn't read {brain_path}: {type(exc).__name__}: {exc})"
    missing = []
    for key in DEFAULTS:
        if hasattr(mod, key):
            th[key] = float(getattr(mod, key))
        else:
            missing.append(key)
    note = f"read from {brain_path}"
    if missing:
        note += f" (not found there, using defaults for: {', '.join(missing)})"
    return th, note


# ---------------------------------------------------------------------- MQTT
def make_client(client_id=""):
    import paho.mqtt.client as mqtt
    try:                                                  # paho 2.x
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:                                # paho 1.x
        return mqtt.Client(client_id=client_id)


def connect(client, host, port=1883, timeout=8.0):
    """Connect and start the network thread, or exit with a message that says what to check."""
    ok = threading.Event()

    def on_connect(c, userdata, flags, rc, *extra):
        failed = rc.is_failure if hasattr(rc, "is_failure") else rc != 0
        if not failed:
            ok.set()

    client.on_connect = on_connect
    hint = ("  Is the computer online? (the class broker, test.mosquitto.org, needs the internet)\n"
            "  Does the network allow port 1883? Some school Wi-Fi blocks it: try a phone hotspot.\n"
            "  Using your own broker? Is Mosquitto running and is the firewall allowing it?")
    try:
        client.connect(host, port, keepalive=30)
    except OSError as exc:
        raise SystemExit(f"Can't reach the MQTT broker at {host}:{port} ({exc}).\n{hint}")
    client.loop_start()
    if not ok.wait(timeout):
        raise SystemExit(f"Connected to {host}:{port} but the broker didn't accept the connection in {timeout:.0f} s.\n{hint}")


# ------------------------------------------------------------- pretend screen
class VirtualScreen:
    """A stand-in for the flashing window, used by self-tests: remembers when it was switched on or off."""

    def __init__(self):
        self.history = [(0.0, False)]
        self.lock = threading.Lock()

    def set(self, on):
        with self.lock:
            self.history.append((time.perf_counter(), bool(on)))

    def state_at(self, t):
        with self.lock:
            state = False
            for ts, s in self.history:
                if ts > t:
                    break
                state = s
            return state
