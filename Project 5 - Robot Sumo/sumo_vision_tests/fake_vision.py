"""A stand-in for the UnoQ's sumo-vision app, so the tests can be tried (and tested) without the robot.

It publishes the same message the real app does, about 10 times a second:
    {"opp_seen": bool, "opp_x": -1..1, "opp_size": 0..1, "tape_ahead": 0..1, "tape_x": -1..1, "ask": "opponent" | "out" | null}
and with --chatter it also mixes in the commands that share the topic ({"charge": true}, {"out_detected": true}).

    python fake_vision.py --broker localhost --topic demo/vision --scenario opponent
Scenarios: empty, tape, opponent, left, right.   Faults to try:
    --self-view 0.15      the camera sees 15% blue in the strip even in an empty ring (like its own apron)
    --gap-after 5         go silent for 2 seconds after 5 s
    --stray-after 3       publish a bare number on the topic once, 3 s in (what another project's score would look like)
    --chatter             also publish {"charge": true} commands on the topic, like the real system does
"""

import argparse
import json
import random
import threading
import time

from common import connect, make_client

SCENARIOS = {
    "empty":   {"opp_seen": False, "opp_x": 0.0, "opp_size": 0.0, "tape_ahead": 0.0, "tape_x": 0.0},
    "tape":    {"opp_seen": False, "opp_x": 0.0, "opp_size": 0.0, "tape_ahead": 0.40, "tape_x": 0.30},
    "opponent": {"opp_seen": True, "opp_x": 0.0, "opp_size": 0.20, "tape_ahead": 0.0, "tape_x": 0.0},
    "left":    {"opp_seen": True, "opp_x": -0.6, "opp_size": 0.12, "tape_ahead": 0.0, "tape_x": 0.0},
    "right":   {"opp_seen": True, "opp_x": 0.6, "opp_size": 0.12, "tape_ahead": 0.0, "tape_x": 0.0},
}


def run_vision(broker="localhost", port=1883, topic="demo/vision", hz=10.0, scenario="empty", self_view=0.0,
               delay_s=0.0, screen=None, gap_after=None, stray_after=None, chatter=False, stop=None, quiet=False):
    """Publish until `stop` is set. With `screen` (a VirtualScreen): tape_ahead is 0.9 while the screen was ON
    `delay_s` seconds ago, else `self_view`. That is a camera pointed at a monitor, plus a pipeline delay."""
    stop = stop or threading.Event()
    client = make_client("fake-vision")
    connect(client, broker, port)
    if not quiet:
        print(f"[fake vision] publishing '{scenario}' to {topic} on {broker}:{port} at {hz:.0f} Hz")
    t0 = next_t = time.perf_counter()
    stray_sent = False
    opp_frames = 0
    next_chatter = t0 + 0.5
    while not stop.is_set():
        now = time.perf_counter()
        el = now - t0
        if stray_after is not None and not stray_sent and el >= stray_after:
            client.publish(topic, "20.0")
            stray_sent = True
        silent = gap_after is not None and gap_after <= el < gap_after + 2.0
        if chatter and now >= next_chatter:
            next_chatter += 0.5
            client.publish(topic, json.dumps({"charge": True}))
        if now >= next_t:
            next_t += 1.0 / hz
            if not silent:
                msg = dict(SCENARIOS[scenario])
                if screen is not None:
                    msg["tape_ahead"] = 0.9 if screen.state_at(now - delay_s) else self_view
                else:
                    msg["tape_ahead"] = max(msg["tape_ahead"], self_view)
                if msg["opp_seen"]:
                    msg["opp_x"] = round(msg["opp_x"] + random.gauss(0, 0.02), 2)
                opp_frames = opp_frames + 1 if msg["opp_seen"] else 0
                msg["ask"] = "opponent" if opp_frames >= 3 else None      # the real app asks after 3 frames in a row
                client.publish(topic, json.dumps(msg))
        time.sleep(0.002)
    client.loop_stop()
    client.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default="demo/vision")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="empty")
    p.add_argument("--self-view", type=float, default=0.0)
    p.add_argument("--gap-after", type=float, default=None)
    p.add_argument("--stray-after", type=float, default=None)
    p.add_argument("--chatter", action="store_true")
    a = p.parse_args()
    try:
        run_vision(a.broker, a.port, a.topic, a.hz, a.scenario, a.self_view, gap_after=a.gap_after, stray_after=a.stray_after,
                   chatter=a.chatter)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
