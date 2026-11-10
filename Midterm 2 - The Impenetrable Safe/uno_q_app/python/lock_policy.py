"""
The lock policy - how the safe makes decisions. (Notion question 1.)

States:
  IDLE        locked, lights off. Press the button to start.
  READING     lights on. Waiting for a digit 0-5 (0 = fist) held steady for `stable_seconds`
              with confidence >= `min_confidence`.
  WAIT_CLEAR  a digit was accepted. Waiting for a blank box (hand removed) for
              `clear_seconds`, so one long hold can't count as two digits.
  UNLOCKED    the 3 digits matched the secret code: servo opens for `unlock_seconds`
              (or until the button is pressed again).
  DENIED      wrong code: red light, back to IDLE after `denied_seconds`.
  LOCKOUT     `max_failures` wrong codes in a row: ignore everything for `lockout_seconds`.

Any time no progress is made for `digit_timeout` seconds while reading, it gives up
and goes back to IDLE (so a half-entered code doesn't sit there forever).

The policy only sees the network's probabilities, never the image, so it is easy to test.
"""
NOT_A_DIGIT = (6,)   # 6 = blank (copied from config.py)

IDLE, READING, WAIT_CLEAR, UNLOCKED, DENIED, LOCKOUT = (
    "IDLE", "READING", "WAIT_CLEAR", "UNLOCKED", "DENIED", "LOCKOUT")


class LockPolicy:
    def __init__(self, cfg, on_event=None):
        self.cfg = cfg
        self.on_event = on_event or (lambda name, **info: None)
        self.state = IDLE
        self.digits = []
        self.failures = 0
        self.candidate = None
        self.candidate_since = None
        self.clear_since = None
        self.until = None
        self.last_progress = None
        self.hold_fraction = 0.0

    # ---- inputs -------------------------------------------------------------
    def press_button(self, now):
        if self.state == IDLE:
            self._start_reading(now)
        elif self.state == UNLOCKED:
            self._relock("button")
        elif self.state in (READING, WAIT_CLEAR):
            self._start_reading(now)      # pressing again restarts the code
            self._emit("restart")

    def update(self, probs, now):
        """Call every frame with the 7 probabilities (or None if no camera/model)."""
        if self.state in (UNLOCKED, DENIED, LOCKOUT):
            if now >= self.until:
                if self.state == UNLOCKED:
                    self._relock("timer")
                else:
                    self._set(IDLE)
                    self._emit("ready_idle")
            return

        if self.state not in (READING, WAIT_CLEAR):
            return
        if now - self.last_progress > self.cfg["digit_timeout"]:
            self._emit("timeout")
            self._reset()
            return
        if probs is None:
            return

        pred = max(range(len(probs)), key=lambda i: probs[i])
        sure = probs[pred] >= self.cfg["min_confidence"]

        if self.state == READING:
            if sure and pred not in NOT_A_DIGIT:
                if pred != self.candidate:
                    self.candidate, self.candidate_since = pred, now
                held = now - self.candidate_since
                self.hold_fraction = min(1.0, held / self.cfg["stable_seconds"])
                if held >= self.cfg["stable_seconds"]:
                    self._accept(pred, now)
            else:
                self.candidate, self.candidate_since, self.hold_fraction = None, None, 0.0

        elif self.state == WAIT_CLEAR:
            if sure and pred in NOT_A_DIGIT:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= self.cfg["clear_seconds"]:
                    self._set(READING)
                    self.last_progress = now
                    self._emit("ready_next", position=len(self.digits) + 1)
            else:
                self.clear_since = None

    # ---- internals ----------------------------------------------------------
    def _accept(self, digit, now):
        self.digits.append(digit)
        self.candidate, self.candidate_since, self.hold_fraction = None, None, 0.0
        self.last_progress = now
        self._emit("digit", digit=digit, position=len(self.digits))
        if len(self.digits) == 3:
            self._check(now)
        else:
            self.clear_since = None
            self._set(WAIT_CLEAR)

    def _check(self, now):
        if self.digits == list(self.cfg["secret_code"]):
            self.failures = 0
            self._set(UNLOCKED)
            self.until = now + self.cfg["unlock_seconds"]
            self._emit("unlock")
        else:
            self.failures += 1
            if self.failures >= self.cfg["max_failures"]:
                self._set(LOCKOUT)
                self.until = now + self.cfg["lockout_seconds"]
                self.failures = 0
                self._emit("lockout", entered=list(self.digits))
            else:
                self._set(DENIED)
                self.until = now + self.cfg["denied_seconds"]
                self._emit("denied", entered=list(self.digits), failures=self.failures)

    def _start_reading(self, now):
        self.digits = []
        self.candidate = self.candidate_since = self.clear_since = None
        self.hold_fraction = 0.0
        self.last_progress = now
        self._set(READING)
        self._emit("start")

    def _relock(self, why):
        self._reset()
        self._emit("relock", why=why)

    def _reset(self):
        self.digits = []
        self.candidate = self.candidate_since = self.clear_since = self.until = None
        self.hold_fraction = 0.0
        self._set(IDLE)

    def _set(self, state):
        self.state = state

    def _emit(self, name, **info):
        self.on_event(name, **info)

    # ---- for the interface / hardware ---------------------------------------
    def status(self, now):
        remaining = None if self.until is None else max(0.0, self.until - now)
        return {
            "state": self.state,
            "digits": list(self.digits),
            "failures": self.failures,
            "hold": round(self.hold_fraction, 2),
            "candidate": self.candidate,
            "remaining": None if remaining is None else round(remaining, 1),
            # what the physical outputs should be doing
            "scan_light": self.state in (READING, WAIT_CLEAR),
            "progress_leds": len(self.digits),
            "green": self.state == UNLOCKED,
            "red": self.state in (DENIED, LOCKOUT),
            "servo_open": self.state == UNLOCKED,
        }
