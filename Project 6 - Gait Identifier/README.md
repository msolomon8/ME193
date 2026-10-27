# Gait Identifier (ME 193, Project 6)

A wearable built from an Arduino UNO Q, an MPU-6050 accelerometer/gyro and a battery pack. It tells **standing, walking, running and biking** apart. Two models are compared, and neither is a neural net:

| | Looks at | Model |
|---|---|---|
| **Model 1** | the **peak acceleration along one axis** in each 2 s window (`train.py` keeps the best of x, y and z) | a small decision tree on that one number, which amounts to a few thresholds |
| **Model 2** | **all IMU outputs**: accel and gyro means (tilt), spread (shake), peaks, rotation and cadence | a depth-4 decision tree |

Both run **on the wearable**. Its LED matrix shows Model 2's answer as a little stick figure that stands, walks, runs or rides a bike, animated at your measured stride rate (`make_led_frames.py` draws the frames). It shows **?** before a model has been trained, and **–** if the IMU isn't found.

## How it works

1. **Sketch (MCU).** Reads the MPU-6050 50 times a second over I²C (Qwiic connector or A4/A5) at ±16 g and ±2000 °/s, and buffers the readings.
2. **Python on the board** (`gait-identifier/python/main.py`).
   - Every 0.1 s it collects the new samples and streams them over MQTT (`ME193/Rogers/MiaGait/raw`).
   - Every 0.5 s it turns the last 2 s into features (`gait_features.py`), runs both trees, smooths each answer with a vote over the last 3 decisions, updates the LED figure and publishes the answers on `.../pred`.
3. **Laptop.**
   - `record.py` saves labeled recordings.
   - `train.py` trains both models, scores them on people they never saw, draws the figures and sends the new model to the wearable over MQTT (`.../model`, retained), so it doesn't have to be redeployed.
   - `live.html` is the demo page. A stick figure walks, runs or pedals at your measured stride rate next to Model 1's guess, with an estimated speed, cadence and distance. Open it in a browser, or add `?demo` to the address to try it without the wearable.
   - `live.py` is a simpler matplotlib fallback, in case the browser can't reach the broker.

The board app lives in the ArduinoApps repo, and `gait-identifier/` here is a copy of it.

## Running it

```bash
# On the Mac, in the ME193 folder, with the venv active:
cd "Project 6 - Gait Identifier"
python record.py mia walking           # 5 s countdown, then 60 s; a sound marks start and end
python record.py mia biking --seconds 90
python train.py --send                 # train, write figures/ and model.json, send to the wearable
open live.html                         # live page with the animated figure and speed (add ?demo to try it without the wearable)
```

The wearable needs internet for MQTT. A phone hotspot works anywhere, including outside on a bike.

## Speed estimate

Cadence comes from the gyro axis that swings the most: one cycle per stride at the ankle, measured with autocorrelation over the 2 s window. Speed is cadence × stride length:

- **Walking:** stride ≈ height × (0.45 + 0.42 f), about 0.83 × height at 0.9 Hz.
- **Running:** stride ≈ height × (0.2 + 0.9 f).
- **Biking:** 5.5 m per pedal turn.

These are textbook ratios, not measurements, so treat the speed as an estimate. Height is set on the page.

## Files

| File | What it is |
|---|---|
| `gait-identifier/` | Copy of the UNO Q app (sketch, `main.py`, `gait_features.py`, `model.json`) |
| `record.py` | Records labeled data into `data/` |
| `gait_tree.py` | A small CART decision tree written with numpy (Gini impurity) |
| `train.py` | Leave-one-person-out comparison of the two models, figures and model export |
| `live.html` | Live demo page: animated figure for each model, speed, cadence, distance and signals |
| `live.py` | Matplotlib fallback for the live view |
| `make_led_frames.py` | Draws the LED-matrix stick-figure frames and writes them into the sketch (preview: `figures/led_frames.png`) |
| `data/` | The recordings (`person_activity_time.csv`) |
| `figures/` | Model 1 thresholds, model 2 tree, confusion matrices, feature space, example signals |
