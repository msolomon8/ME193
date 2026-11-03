# Step Counter (ME 193, Project 7)

This builds on the [Gait Identifier](../Project%206%20-%20Gait%20Identifier/). A wearable just below the **right knee** counts steps with a **neural net built and trained in PyTorch**, estimates how far you walked, and reports how sure it is. The wearable is an Arduino UNO Q with an MPU-6050 accelerometer/gyro, an HC-SR04 ultrasonic sensor and a battery pack.

## How it works

```
UNO Q sketch ── 50 Hz: accel x,y,z + gyro x,y,z (MPU-6050) + distance to the left leg (HC-SR04)
     │
UNO Q Python ── streams the 7 channels over MQTT (broker.emqx.io, ME193/Rogers/MiaSteps/raw)
     │                                          ▲ the LED matrix shows the count (…/count)
Laptop ── count_live.py: every 0.5 s, the last 2 s → 5 neural nets → steps in the middle 0.5 s → add up
     │
live.html ── step count ± uncertainty, distance, cadence, the 5 nets' answers, a walking figure leaving footprints
```

**The sensors.**
- The **gyro** sees one big forward swing of the right shin per stride.
- The **accelerometer** sees every foot strike.
- The **ultrasonic sensor** points sideways at the left shin. Its reading dips each time the legs pass, which happens on every step, including the left steps that a right-leg IMU can only guess at.

**The neural net** (`step_net.py`, 1,297 weights):
- **Input:** a 2 s window, 7 channels × 100 samples.
- **Conv layer 1:** 8 filters, each 9 samples long. Then ReLU, and max-pooling down to 50.
- **Conv layer 2:** 8 filters, each 9 long. Then ReLU, and pooling down to 25.
- **Output:** the 200 numbers are flattened into a linear layer, followed by softplus. The result is the number of steps in the **middle 0.5 s** of the window.
- **Counting:** the window slides 0.5 s at a time, so every moment is in exactly one middle, and the count is the sum.
- **Training:** mean squared error, Adam, backprop. See `train_net()`.

**Labels.**
- You count steps out loud between two sounds (`record.py`).
- `steps_data.mark_steps()` places those steps in time. It keeps the strongest shin-swing peaks so that 2 × strides = your count, and puts the left steps halfway between the right ones.
- Each recording is also replayed 15 % slower and faster, so the net learns more paces than you recorded.
- `figures/labels/` has one plot per recording so you can check the marks.

## Uncertainty analysis (how good is it?)

`train.py` does all of this.

- **Leave-one-recording-out.** Each recording is counted by nets trained on all the *other* recordings, then compared with your count. This gives the average error, the bias, a 95 % range and the worst case (`figures/cv_counts.png`, `figures/cv_errors.png`).
- **Ensemble of 5 nets** started from different random weights. Their disagreement is the ± on each count.
- **Baseline.** A plain peak counter with no learning is run on the same recordings, to show what the net adds.
- **Sensor test.** The same test is repeated with sensors switched off (all 7 channels, IMU only, gyro only, accel only, ultrasonic only), to show what each sensor contributes (`figures/sensors.png`).
- **Not walking.** Standing, fidgeting and biking recordings give the false steps per minute.
- **Live ±** (95 %) combines two things: the cross-validated error, which grows with the count, and how far the 5 nets' running totals disagree.

## Distance

Distance = Σ steps × step length, where step length = height × (a + b × cadence), so faster steps are longer.
- a and b are fitted to your **measured-distance walks**, and each walk is scored by a fit that left it out (`figures/distance.png`).
- With no measured walks, a textbook ratio is used instead.

**Limitations:**
- Running or walking **in place** counts steps but goes nowhere, yet the estimate still adds distance.
- Step length is calibrated on one person and one height. Other people, hills, stairs, turns and shuffling change it.
- Errors build up, so a 2 % step-length bias is 20 m per km.
- A missed step costs a whole step length.

## Running it

```bash
# 1. Board: deploy the app (Mac, ArduinoApps folder). It also opens live.html.
python3 tools/deploy.py step-counter

# 2. Record (Mac, ME193 folder, venv active). Count every step out loud between the two sounds.
cd "Project 7 - Step Counter"
python record.py walking --note normal
python record.py walking --seconds 10           # a short walk
python record.py walking --course 5             # walk a measured 5 m, press Enter at the end (distance saved for you)
python record.py standing --seconds 30

# 3. Train and test (about 10 min; --quick for a first look)
python train.py --height-in 66

# 4. Live: count steps and watch
python count_live.py          # then open live.html (or press Demo on it to try without the wearable)

# 5. The animation of how the net works, filled with your real data and nets
python explain_export.py      # then open explain.html and screen-record it
```

### Recording plan (about 20 sessions, 25 minutes)

| Sessions | What | Why |
|---|---|---|
| 3 × 45 s | normal walking | the main case |
| 2 + 2 | slow walking, fast walking | other paces |
| 3 | walk a **measured course** (e.g. a hallway you measured, a 400 m track), different paces | distance calibration and its error |
| 2 × 30 s | jogging or running (type 0 m if in place) | fast steps |
| 2 × 30 s | standing still | zero steps |
| 2 × 30 s | sitting, swinging your leg, shifting around | things that are *not* steps |
| 2 × 30 s | biking or air-biking | periodic but zero steps, the hardest negative |

Record every kind **at least twice**. A recording is only tested on nets trained on the others, so a one-off activity is always unseen during testing.

### Wiring the HC-SR04
- VCC → 5V, GND → GND, Trig → D9, Echo → D10, all wired straight in (the UNO Q digital pins take the 5 V Echo signal).
- Mount it on the MPU-6050 just below the right knee, aimed horizontally **inward at the left shin**.

## Files

| File | What it is |
|---|---|
| `step-counter/` | Copy of the UNO Q app (sketch: MPU-6050 + HC-SR04 + LED count; Python: MQTT stream) |
| `record.py` | Records a session and asks for the counted steps and measured distance |
| `steps_data.py` | Scaling, step labels from counts, time-stretch augmentation, windows |
| `step_net.py` | The PyTorch net, training, counting, parallel training |
| `train.py` | Cross-validation, ensemble, baseline, sensor test, distance calibration, figures, `models/` |
| `count_live.py` | Live counting for the page and the LED matrix |
| `live.html` | Live page (reuses the Gait Identifier page) |
| `explain.html`, `explain_export.py` | Animation from the sensors through PyTorch to backprop, with real numbers |
| `data/`, `models/`, `figures/` | Recordings, trained nets and summary, plots |
