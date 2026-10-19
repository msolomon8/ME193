"""Tests for the sumo vision tools: run  python test_tools.py
Each tool is checked against a known answer (a fake camera app with a planted gap, a self-view fault, a stray message, commands
mixed into the topic, a set delay), so you can trust what the tools tell you about the real robot."""

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import common
import fake_vision
import vision_check as vc
import vision_latency_test as vlt

TH = dict(common.DEFAULTS, _source="test")


class Skip(Exception):
    pass


def msg(**kw):
    base = {"opp_seen": False, "opp_x": 0.0, "opp_size": 0.0, "tape_ahead": 0.0, "tape_x": 0.0, "ask": None}
    base.update(kw)
    return base


# ------------------------------------------------------------------ pure logic
def test_stats_and_thresholds():
    s = common.summarize([10, 20, 30, 40, 100])
    assert s["median"] == 30 and s["max"] == 100 and common.summarize([]) == {"n": 0}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("VISION_TIMEOUT = 0.7\nTAPE_AHEAD = 0.1\nOUT_CONFIRM = 0.4\nOUT_WINDOW = 2.0\nCHARGE_TIME = 1.5\n")
    th, note = common.load_thresholds(f.name)
    assert th == {"VISION_TIMEOUT": 0.7, "TAPE_AHEAD": 0.1, "OUT_CONFIRM": 0.4, "OUT_WINDOW": 2.0} and note == f"read from {f.name}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:               # a brain that lacks some of the names
        f.write("TAPE_AHEAD = 0.2\n")
    th, note = common.load_thresholds(f.name)
    assert th["TAPE_AHEAD"] == 0.2 and th["VISION_TIMEOUT"] == 0.5 and "using defaults for" in note and "VISION_TIMEOUT" in note
    th, note = common.load_thresholds("/no/such/file.py")
    assert th == common.DEFAULTS and "couldn't read" in note
    assert common.load_thresholds(None)[0] == common.DEFAULTS


def test_watch_stats_finds_rate_gaps_and_blind_time():
    recs = [(t / 10.0, msg()) for t in range(0, 20)] + [(t / 10.0, msg()) for t in range(32, 50)]      # a 1.3 s silence from 1.9 s to 3.2 s
    s = vc.watch_stats(recs, 0.0, 5.0, timeout=0.5, bad=1, not_dict=2, unknown=3)
    assert abs(s["rate"] - 38 / 5.0) < 1e-9
    assert 1.2 < s["max_gap"] < 1.4, s["max_gap"]
    assert 0.7 < s["blind_s"] < 0.9, s["blind_s"]                    # 1.3 s of silence minus the 0.5 s the tree tolerates
    assert (s["bad"], s["not_dict"], s["unknown"]) == (1, 2, 3)
    assert vc.watch_stats([(t / 10.0, msg()) for t in range(50)], 0.0, 5.0, 0.5)["blind_s"] == 0.0
    silent = vc.watch_stats([], 0.0, 5.0, 0.5)
    assert silent["n"] == 0 and silent["blind_s"] == 4.5


def values(**kw):
    base = {"n": 50, "tape_med": 0.0, "tape_p95": 0.0, "opp_frac": 0.0, "x_med": 0.0, "ask_frac": 0.0}
    base.update(kw)
    return base


def passed(checks):
    return all(ok for ok, _ in checks)


def test_judge_empty_ring_catches_phantom_tape_and_phantom_opponent():
    checks, hints = vc.judge("empty", values(), TH)
    assert passed(checks) and hints == []
    checks, hints = vc.judge("empty", values(tape_p95=0.28), TH)
    assert not passed(checks) and hints == ["phantom_tape"]
    checks, hints = vc.judge("empty", values(opp_frac=0.9), TH)
    assert not passed(checks) and hints == ["phantom_opp"]
    checks, hints = vc.judge("empty", values(ask_frac=0.5), TH)
    assert not passed(checks) and hints == ["phantom_opp"], "a false 'Opponent?' prompt is a failure even if opp_seen flickered"
    checks, hints = vc.judge("empty", values(n=2), TH)
    assert not passed(checks) and "only 2 messages" in checks[0][1]


def test_judge_tape_opponent_left_right():
    assert passed(vc.judge("tape", values(tape_med=0.4), TH)[0])
    assert vc.judge("tape", values(tape_med=0.02), TH)[1] == ["tape_miss"]
    assert passed(vc.judge("opponent", values(opp_frac=1.0, ask_frac=0.8), TH)[0])
    assert vc.judge("opponent", values(opp_frac=1.0, ask_frac=0.0), TH)[1] == ["no_prompt"], "seen but never asked"
    assert vc.judge("opponent", values(opp_frac=0.0), TH)[1] == ["opp_miss"]
    assert passed(vc.judge("left", values(opp_frac=1.0, x_med=-0.6), TH)[0])
    assert passed(vc.judge("right", values(opp_frac=1.0, x_med=0.6), TH)[0])
    assert vc.judge("left", values(opp_frac=1.0, x_med=0.6), TH)[1] == ["flipped"], "an opponent on the left reported on the right means FLIP_X"
    assert vc.judge("right", values(opp_frac=1.0, x_med=-0.6), TH)[1] == ["flipped"]
    assert vc.judge("left", values(opp_frac=1.0, x_med=0.0), TH)[1] == [], "centred is wrong but not evidence of a flip"


def test_scenario_values_reads_the_ask_field():
    v = vc.scenario_values([(0.0, msg(opp_seen=True, opp_x=-0.5, ask="opponent")), (0.1, msg(opp_seen=True, opp_x=-0.5)), (0.2, msg())])
    assert abs(v["opp_frac"] - 2 / 3) < 1e-9 and abs(v["ask_frac"] - 1 / 3) < 1e-9 and v["x_med"] == -0.5


class ScriptedListener:
    """Stands in for the broker: hands back the messages of whichever scenario the 'user' is currently on."""

    def __init__(self, per_scenario):
        self.per_scenario = per_scenario
        self.index = -1

    def advance(self, *_):
        self.index += 1

    def between(self, t0, t1):
        name = vc.SCENARIOS[self.index][0]
        return [(t0 + i * 0.02, m) for i, m in enumerate(self.per_scenario[name])]


def healthy():
    return {"empty": [msg()] * 40, "tape": [msg(tape_ahead=0.4)] * 40,
            "opponent": [msg(opp_seen=True, opp_size=0.2, ask="opponent")] * 40,
            "left": [msg(opp_seen=True, opp_x=-0.6, opp_size=0.1, ask="opponent")] * 40,
            "right": [msg(opp_seen=True, opp_x=0.6, opp_size=0.1, ask="opponent")] * 40}


def run_scripted(per_scenario):
    lines = []
    lis = ScriptedListener(per_scenario)
    results = vc.run_guided(lis, TH, seconds=0.05, settle=0.0, wait=lis.advance, out=lines.append)
    return results, "\n".join(lines)


def test_guided_run_passes_a_healthy_camera():
    results, text = run_scripted(healthy())
    assert all(passed(c) for c, _ in results.values()) and "ALL SCENARIOS PASSED" in text


def test_guided_run_suggests_a_size_limit_when_the_false_opponent_is_small():
    """Real run: a thin sliver (size 0.02) in the empty scene, real opponents 0.10 and up: the tool should suggest a MIN_OPP_AREA in between."""
    bad = healthy()
    bad["empty"] = [msg(opp_seen=True, opp_size=0.02, ask="opponent")] * 40
    results, text = run_scripted(bad)
    assert "size up to 0.02" in text and "MIN_OPP_AREA = 0.04" in text and "yellow = too small" in text
    assert not passed(results["empty"][0])
    ok_text = run_scripted(healthy())[1]
    assert "SIZE:" not in ok_text and "MIN_OPP_AREA" not in ok_text


def test_guided_run_says_when_a_size_limit_cannot_help():
    """If the false opponent is as big as the real ones, do NOT suggest a limit that would also blind the robot to real opponents."""
    bad = healthy()
    bad["empty"] = [msg(opp_seen=True, opp_size=0.09, ask="opponent")] * 40
    text = run_scripted(bad)[1]
    assert "cannot separate them" in text and "MIN_OPP_AREA =" not in text


def test_guided_run_explains_the_apron_problem_and_a_mirrored_camera():
    bad = healthy()
    bad["empty"] = [msg(tape_ahead=0.30)] * 40                       # its own apron in view
    bad["left"], bad["right"] = bad["right"], bad["left"]            # left and right swapped
    results, text = run_scripted(bad)
    assert "SOMETHING TO FIX" in text and "OWN" in text and "apron" in text and "SELF_MASK_BOTTOM" in text and "we're OUT?" in text
    assert "real ring tape" in text, "the hint must also cover the case where the blue is genuine ring tape in view (not a fault)"
    assert "FLIP_X" in text
    assert not passed(results["empty"][0]) and not passed(results["left"][0]) and passed(results["tape"][0])


def test_the_listener_sorts_every_kind_of_message_correctly():
    from types import SimpleNamespace as NS
    lis = vc.Listener.__new__(vc.Listener)                               # no broker: feed it messages directly
    lis.records, lis.commands, lis.bad, lis.not_dict, lis.unknown = [], [], 0, 0, 0
    lis.lock = threading.Lock()
    for payload in (b'{"opp_seen": false, "tape_ahead": 0.1, "ask": null}',     # the camera
                    b'{"charge": true}', b'{"out": false}', b'{"out_detected": true}', b'{"answer": "approve_opponent"}',   # commands
                    b'{"hello": "someone else"}',                               # a JSON object that is neither
                    b"20.0", b"[1, 2]",                                         # JSON that isn't an object (e.g. a ping-pong score)
                    b"not json at all"):
        lis._on_message(None, None, NS(payload=payload))
    assert len(lis.records) == 1 and len(lis.commands) == 4
    assert (lis.unknown, lis.not_dict, lis.bad) == (1, 2, 1)


# ------------------------------------------------------------- the output of --watch
class FixedListener:
    """Hands back the same messages, spread evenly over whatever window is asked for."""

    def __init__(self, messages, commands=()):
        self.messages, self.cmds, self.bad, self.not_dict, self.unknown = messages, list(commands), 0, 0, 0

    def between(self, t0, t1):
        n = len(self.messages)
        return [(t0 + (i + 0.5) * (t1 - t0) / n, m) for i, m in enumerate(self.messages)]

    def commands_between(self, t0, t1):
        return [(t0 + 0.01, m) for m in self.cmds]


def watch_text(messages, commands=()):
    out = []
    vc.run_watch(FixedListener(messages, commands), TH, 0.1, out=out.append)
    return "\n".join(out)


def test_watch_does_not_say_healthy_when_the_camera_may_be_seeing_its_own_apron():
    healthy_text = watch_text([msg()] * 30)
    assert "LOOKS HEALTHY" in healthy_text and "CHECK:" not in healthy_text
    apron = watch_text([msg(tape_ahead=0.15)] * 30)
    assert "LOOKS HEALTHY" not in apron and "own blue apron" in apron and "SELF_MASK_BOTTOM" in apron
    ghost = watch_text([msg(opp_seen=True, opp_size=0.07)] * 30)
    assert "LOOKS HEALTHY" not in ghost and "reported in most messages" in ghost
    nothing = watch_text([])
    assert "NO CAMERA MESSAGES" in nothing and "LOOKS HEALTHY" not in nothing


def test_watch_says_so_when_one_silence_was_longer_than_the_tree_allows():
    """The real watch output once said 'no gaps' right after '664 ms' (limit 500 ms): a healthy verdict must still mention the gap, and a low rate."""
    import vision_check as vc
    th = {"VISION_TIMEOUT": 0.5, "TAPE_AHEAD": 0.08}
    msg = {"opp_seen": False, "opp_x": 0.0, "opp_size": 0.0, "tape_ahead": 0.0, "tape_x": 0.0, "ask": None}

    def watch(times):
        class L:
            bad = not_dict = unknown = 0
            def between(self, a, b): return [(a + x, dict(msg)) for x in times]
            def commands_between(self, a, b): return []
        lines = []
        real_sleep, vc.time.sleep = vc.time.sleep, lambda s: None
        real_perf, ticks = vc.time.perf_counter, iter([100.0, 115.0])
        vc.time.perf_counter = lambda: next(ticks)
        try:
            vc.run_watch(L(), th, 15, out=lines.append)
        finally:
            vc.time.sleep, vc.time.perf_counter = real_sleep, real_perf
        return "\n".join(lines)

    steady = watch([i * 0.1 for i in range(150)])                         # 10 a second, no gap
    assert "LOOKS HEALTHY" in steady and "Note:" not in steady and "no gaps" not in steady
    one_gap = watch([i * 0.1 for i in range(60)] + [6.0 + 0.664 + i * 0.1 for i in range(80)])    # one 664 ms silence
    assert "LOOKS HEALTHY" in one_gap and "longer than the tree's 500 ms limit" in one_gap and "no gaps" not in one_gap
    slow = watch([i * 0.154 for i in range(97)])                          # 6.5 a second
    assert "LOOKS HEALTHY" in slow and "under the ~10 the app aims for" in slow


def test_guided_says_it_is_recording_right_after_enter():
    """On a real run, 'nothing happened' for six seconds after Enter: the tool must announce the recording at once."""
    import vision_check as vc
    th = {"VISION_TIMEOUT": 0.5, "TAPE_AHEAD": 0.08, "OUT_CONFIRM": 0.3, "OUT_WINDOW": 3.0, "_source": "test"}
    events = []

    class L:
        bad = not_dict = unknown = 0
        def between(self, a, b): return []
        def commands_between(self, a, b): return []
    vc.run_guided(L(), th, seconds=0.01, settle=0.01, wait=lambda prompt: events.append("enter"), out=lambda text: events.append(text))
    first_enter = events.index("enter")
    assert "Recording for about" in events[first_enter + 1], events[first_enter:first_enter + 3]


def test_watch_reports_the_commands_that_share_the_topic():
    text = watch_text([msg()] * 30, commands=[{"charge": True}, {"charge": True}, {"out_detected": True}])
    assert "charge x2" in text and "out_detected x1" in text and "LOOKS HEALTHY" in text, "commands are normal traffic, not a problem"


def test_check_detector_passes_a_good_result_and_fails_each_bad_one():
    import check_detector as cd
    good = {key: 1.0 if kind == "min" else 0.0 for _, key, kind, _ in cd.CRITERIA}
    good[cd.APRON_KEY] = 0.0
    assert all(ok for ok, _ in cd.judge(good))
    for _, key, kind, limit in cd.CRITERIA:                                   # break each criterion in turn
        bad = dict(good)
        bad[key] = 0.5
        assert [ok for ok, _ in cd.judge(bad)].count(False) == 1, key
    no_mask = {k: v for k, v in good.items() if k != cd.APRON_KEY}
    assert any((not ok) and "no SELF_MASK_BOTTOM" in text for ok, text in cd.judge(no_mask)), "a detector with no mask option must fail that line"
    old_style = {k: v for k, v in good.items() if k not in cd.OPTIONAL}        # a detector with no click-to-learn at all
    assert [ok for ok, _ in cd.judge(old_style)].count(False) == 0, "an optional check that cannot apply must not fail"
    assert any("wooden floor" in text for _, text in cd.judge(good)), "the click-on-the-floor check is in the list"


# ------------------------------------- the real thing, against a real broker
def _broker(port):
    exe = shutil.which("mosquitto")
    if not exe:
        raise Skip("mosquitto is not installed")
    conf = tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False)
    conf.write(f"listener {port} 127.0.0.1\nallow_anonymous true\n")
    conf.close()
    proc = subprocess.Popen([exe, "-c", conf.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.8)
    return proc


class Node:
    """A running fake vision app, stopped cleanly afterwards."""

    def __init__(self, port, topic, **kw):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=fake_vision.run_vision, daemon=True, kwargs=dict(
            broker="127.0.0.1", port=port, topic=topic, quiet=True, stop=self.stop, **kw))
        self.thread.start()
        time.sleep(0.8)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def test_against_a_real_broker_rate_gap_stray_message_commands_and_self_view():
    port = 18831
    broker = _broker(port)
    try:
        node = Node(port, "t/steady", scenario="empty")                       # steady: about 10 camera messages a second
        lis = vc.Listener("127.0.0.1", port, "t/steady")
        out = []
        s = vc.run_watch(lis, TH, 3.0, out=out.append)
        lis.close(); node.close()
        assert 8.0 <= s["rate"] <= 11.0, s["rate"]
        assert s["blind_s"] == 0.0 and "LOOKS HEALTHY" in out[-1], out[-1]
        node = Node(port, "t/gap", scenario="empty", gap_after=0.5)           # a 2 s silence is noticed
        lis = vc.Listener("127.0.0.1", port, "t/gap")
        out = []
        s = vc.run_watch(lis, TH, 4.0, out=out.append)
        lis.close(); node.close()
        assert s["max_gap"] > 1.5 and s["blind_s"] > 0.8 and "BLIND TOO OFTEN" in out[-1], (s["max_gap"], s["blind_s"])
        node = Node(port, "t/stray", scenario="empty", stray_after=1.5)       # a bare number (after the listener has subscribed)
        lis = vc.Listener("127.0.0.1", port, "t/stray")
        out = []
        s = vc.run_watch(lis, TH, 2.0, out=out.append)
        lis.close(); node.close()
        assert s["not_dict"] >= 1 and s["n"] > 10 and "SOMETHING ELSE IS PUBLISHING" in out[-1]
        node = Node(port, "t/chat", scenario="empty", chatter=True)           # charge commands share the topic: not camera messages
        lis = vc.Listener("127.0.0.1", port, "t/chat")
        out = []
        s = vc.run_watch(lis, TH, 3.0, out=out.append)
        lis.close(); node.close()
        assert 8.0 <= s["rate"] <= 11.0, f"commands must not inflate the camera rate (got {s['rate']:.1f})"
        assert "charge x" in "\n".join(out) and "LOOKS HEALTHY" in out[-1], out
        node = Node(port, "t/self", scenario="empty", self_view=0.15)         # a camera that sees its own apron
        lis = vc.Listener("127.0.0.1", port, "t/self")
        t0 = time.perf_counter(); time.sleep(2.0)
        v = vc.scenario_values(lis.between(t0, time.perf_counter()))
        lis.close(); node.close()
        checks, hints = vc.judge("empty", v, TH)
        assert hints == ["phantom_tape"] and not passed(checks), (v, hints)
    finally:
        broker.terminate()


def test_against_a_real_broker_every_scenario_the_fake_camera_can_show():
    port = 18832
    broker = _broker(port)
    try:
        for scenario in ("empty", "tape", "opponent", "left", "right"):
            topic = f"t/{scenario}"
            node = Node(port, topic, scenario=scenario)
            lis = vc.Listener("127.0.0.1", port, topic)
            t0 = time.perf_counter(); time.sleep(2.0)
            v = vc.scenario_values(lis.between(t0, time.perf_counter()))
            lis.close(); node.close()
            checks, hints = vc.judge(scenario, v, TH)
            assert passed(checks) and hints == [], (scenario, v, checks)
    finally:
        broker.terminate()


def test_latency_matches_the_simulated_pipeline_delay():
    port = 18833
    broker = _broker(port)
    try:
        for delay_ms, topic in ((300, "t/lat1"), (600, "t/lat2")):
            screen = common.VirtualScreen()
            node = Node(port, topic, scenario="empty", delay_s=delay_ms / 1000.0, screen=screen)
            lis = vc.Listener("127.0.0.1", port, topic)
            res = vlt.measure(lis, screen, trials=3, settle=(0.3, 0.5), log=lambda *_: None)
            lis.close(); node.close()
            allv = res["rise"] + res["fall"]
            assert res["missed"] == 0 and len(allv) == 6, res
            med = common.summarize(allv)["median"]
            # the camera app publishes 10 times a second, so a flash waits up to 100 ms for the next message
            assert delay_ms <= med <= delay_ms + 130, f"simulated {delay_ms} ms, measured {med:.0f} ms"
        assert "GOOD" in vlt.verdict(200, 0.3) and "SLOW" in vlt.verdict(700, 0.3) and "9 cm" in vlt.verdict(300, 0.3)
    finally:
        broker.terminate()


def test_latency_explains_a_camera_that_already_sees_blue_or_sees_nothing():
    port = 18834
    broker = _broker(port)
    try:
        screen = common.VirtualScreen()
        node = Node(port, "t/blue", scenario="empty", self_view=0.5, screen=screen)         # tape_ahead stuck high
        lis = vc.Listener("127.0.0.1", port, "t/blue")
        res = vlt.measure(lis, screen, trials=1, settle=(0.2, 0.3), log=lambda *_: None)
        lis.close(); node.close()
        assert "error" in res and "already" in res["error"] and "own apron" in res["error"], res
        lis = vc.Listener("127.0.0.1", port, "t/nobody")                                    # nobody publishing
        res = vlt.measure(lis, common.VirtualScreen(), trials=1, settle=(0.2, 0.3), log=lambda *_: None)
        lis.close()
        assert "error" in res and "No camera messages" in res["error"], res
    finally:
        broker.terminate()


def test_a_missing_broker_gives_a_helpful_message():
    c = common.make_client("x")
    try:
        common.connect(c, "127.0.0.1", 18999, timeout=2)
    except SystemExit as exc:
        assert "phone hotspot" in str(exc) and "1883" in str(exc)
    else:
        raise AssertionError("expected a SystemExit with advice")


def run():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    ok = skipped = 0
    for name, fn in tests:
        try:
            fn()
            ok += 1
        except Skip as why:
            skipped += 1
            print(f"  skipped {name}: {why}")
        except Exception:                                                    # noqa: BLE001
            print(f"FAIL  {name}")
            traceback.print_exc(limit=3)
    print(f"{ok}/{len(tests) - skipped} passed" + (f", {skipped} skipped" if skipped else ""))
    return ok == len(tests) - skipped


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
