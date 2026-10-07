# Both Car Mini Figures (Two Car Door to Door Parking)

Two cars, one computer. The computer's camera finds both minifigs; the green car parks
its minifig on the left line and the blue car parks its minifig on the right line.
Same setup as the one-car version (`minifig_mqtt.py` + `unoq_app`), doubled.

```
computer camera → both_minifigs_mqtt.py (pop-up window, finds BOTH minifigs)
                     ├─ MQTT ME193/minifig/tashamia/green → GREEN car (CAR = "green") → stops at the left line  (x = 0.25)
                     └─ MQTT ME193/minifig/tashamia/blue  → BLUE car  (CAR = "blue")  → stops at the right line (x = 0.75)
```

- `both_minifigs_mqtt.py` - runs on the computer. Uses the trained `../green_minifig.pt` and
  `../blue_minifig/blue_minifig.pt`, shows the pop-up with both boxes and both lines.
- `two_car_door_to_door_parking_app/` - ONE App Lab app for both cars: `unoq_app` plus a
  `CAR` setting at the top of `python/main.py` that picks the topic, MQTT client ID and stopping line.

## Setup

1. Zip the app (in Finder: right-click `two_car_door_to_door_parking_app` → Compress).
2. In App Lab, import the zip on the **green** car's UNO Q and leave `CAR = "green"`.
3. Import the **same** zip on the **blue** car's UNO Q and change it to `CAR = "blue"`.
4. Start the app on both cars. Each console prints
   `MQTT connected (Success), subscribing to ME193/minifig/tashamia/<green|blue>` (can take up to a minute).
5. On the computer (venv active), from this folder: `python both_minifigs_mqtt.py`.
   The pop-up opens.
6. Put both minifigs in view. Each car drives until its minifig is on its line. Press **q** in the
   pop-up to stop; both cars stop within ~0.5 s.

## If something's off

- A car drives **away** from its line: flip `DIRECTION` in that car's `python/main.py`.
- Cars overshoot: lower `MAX_SPEED`.
- To move a line, change `CARS` in the app **and** `MINIFIGS` in `both_minifigs_mqtt.py`.
