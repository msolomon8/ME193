"""
The bridge to the physical safe (servo, LEDs, buzzer).

Right now this only PRINTS what the hardware should do, so the whole lock can be
tested on the computer. When the UNO Q side is wired up, `send` will pass these same
commands to the UNO Q's microcontroller (servo / LEDs / buzzer), and nothing else in
the program has to change.
"""


class Hardware:
    def __init__(self):
        self.last = None

    def apply(self, status):
        """Called every frame with LockPolicy.status(); only reacts to changes."""
        outputs = (status["scan_light"], status["progress_leds"], status["green"],
                   status["red"], status["servo_open"])
        if outputs != self.last:
            self.last = outputs
            self.send(f"SCAN {int(outputs[0])} PROGRESS {outputs[1]} GREEN {int(outputs[2])} "
                      f"RED {int(outputs[3])} SERVO {'OPEN' if outputs[4] else 'LOCKED'}")

    def sound(self, event, **info):
        self.send(f"SOUND {event.upper()}")

    def send(self, command):
        print(f"[hardware] {command}")
