# Gait Identifier: handoff from the 2026-10-26 chat

Paste or @-mention this file in a new chat to continue. It covers what was asked, what was decided and built, the current state, and what's left.

**Due: 2026-10-27 (tomorrow as of this chat).**

---

## 1. The assignment (verbatim from Mia)

> Your goal here is to make a wearable out of the UnoQ and 5060 that can differentiate between standing still, walking, running, and biking (or lying on your back and air-biking). You should have two models: (1) only uses the peak acceleration in one direction, (2) uses all of the IMU outputs. You are welcome to place your wearable anywhere you want - but the hip is definitely harder than the ankle. This project will be graded by 3 of your peers (although I will give you the 4th point) and you get points in the following fashion. You cannot use neural nets for this one.
>
> Grading Rubric
> - +1 for your gait identifier working for your 3 peers (each does one)
> - +1 for a clear and cogent explanation of how your system works, including which model works better and why to your peers (you are allowed multiple tries, incorporating feedback).
> - +1 for getting your peers to say "wow, that was cool"
> - +1 for following all directions and having a Notion page with an introductory paragraph, demonstration video, graphical representation of your model, GitHub link, and answers to my 4 reflection questions

## 2. Facts Mia confirmed

- The "5060" is an **MPU-6050** (accelerometer + gyro, I²C address 0x68).
- A **battery pack** powers the UNO Q so the wearable is wireless.
- **No phone element.** Mia was confused when I said "phone layout"; I had only meant a responsive check of the page at a narrow width. The UI is for the laptop.
- **Training data: only Mia, and running is done in place** (decided 2026-10-26).
- **The 4 reflection questions have not been provided yet.** Ask for them before writing the Notion page.

## 3. Working rules for Mia (from memory and this chat)

- **Never connect to the board yourself** (no deploy.py, ssh or adb). Write the app in `~/Documents/GitHub/ArduinoApps/<app>`; Mia deploys it.
- Always say **where a command runs**: Mac terminal, board, or github.com.
- Go one step at a time and get approval, though the deadline means building fast.
- Don't commit or push unless asked. **Nothing from this project has been committed yet.**
- The `Midterm - Ping Pong/q_table.json` and `record.json` changes are deliberately left uncommitted.

## 4. Design decisions

| Decision | Choice and why |
|---|---|
| Placement | **Ankle.** It gives clear heel strikes, swing rotation and pedal circles; the assignment says the hip is harder. Wear it in the same orientation every time, and mark "this side up" for peers. |
| Sampling | 50 Hz, ±16 g accelerometer (running ankles can pass 8 g), ±2000 °/s gyro |
| Windows | 2 s (100 samples), with a new decision every 0.5 s and a majority vote over the last 3 decisions |
| Model 1 | A decision tree (depth 3) on **one feature**: the peak acceleration along one axis. `train.py` tries `max_ax`, `max_ay` and `max_az` and keeps the best under cross-validation. In effect it is a few thresholds. |
| Model 2 | A decision tree (depth 4) on **all 33 features**: per axis (accel and gyro) the mean (tilt and gravity direction), std, max and min, plus accel and gyro magnitude statistics, cadence (Hz) and periodicity |
| No sklearn | The venv has no scikit-learn, so `gait_tree.py` is a hand-written numpy CART (Gini impurity). The same tree format is walked in pure Python on the board. |
| Evaluation | **Leave-one-person-out** cross-validation, which matches peers trying it. With only one person it holds out half of each recording at a time (leaving a whole recording out would drop that activity from training). |
| Cadence | Autocorrelation of the **gyro axis with the largest std**, not the gyro magnitude (\|sin\| would double the frequency). At the ankle that's one cycle per stride. On test signals: 0.9 → 0.94, 1.45 → 1.47, 1.2 → 1.19 Hz. |
| Speed (UI only) | Stride Hz × stride length. Walking: H·(0.45 + 0.42f). Running: H·(0.2 + 0.9f). Biking: 5.5 m per pedal turn. H is the height entered on the page. These are estimates from textbook ratios, not calibrated. |
| Comms | MQTT on `broker.hivemq.com` (switched from test.mosquitto.org, which went down on 2026-10-26): `ME193/Rogers/MiaGait/raw` (samples), `/pred` (answers), `/model` (retained; the laptop pushes new models without a redeploy) |
| LED matrix | Shows **Model 2 only**, as an animated 4-frame stick figure (stand/walk/run/bike) at the measured stride rate. `?` means no model yet; `–` means no IMU. Model 1 is shown only on live.html. I offered to squeeze a small Model 1 letter into a corner; Mia hasn't answered. |
| "Wow" | The live.html animated figures move at your real cadence, the LED figure animates, and the two models visibly disagree, e.g. on biking |

## 5. Files

### Board app: `~/Documents/GitHub/ArduinoApps/gait-identifier/` (mirrored in `ME193/Project 6 - Gait Identifier/gait-identifier/`)
- `app.yaml`: no bricks.
- `sketch/sketch.ino`
  - Finds the MPU-6050 on `Wire1` (Qwiic), then on `Wire` (A4/A5), at 0x68 or 0x69.
  - Fills a 50 Hz ring buffer.
  - `Bridge.provide_safe("imu")` returns `"seq;ax,ay,az,gx,gy,gz;..."` as raw counts, or `"NOIMU"`.
  - `Bridge.provide_safe("show", activity, frame_ms)` sets the animation.
  - The animation frames sit between the `// FRAMES BEGIN` and `// FRAMES END` markers; `make_led_frames.py` generates them.
- `python/main.py`
  - Polls the IMU every 0.1 s and publishes `/raw`.
  - Every 0.5 s it computes features, runs both trees, smooths the answers and calls `show`, then publishes `/pred` with `m1`, `m2`, `peak`, `cadence`, `periodicity` and `gyro`.
  - Loads `model.json` from disk, or from the retained `/model` MQTT message.
- `python/gait_features.py`: feature extraction plus tree `predict`, in pure Python. It is shared with the laptop scripts.
- `python/requirements.txt`: `paho-mqtt>=2.0`.

### Laptop: `ME193/Project 6 - Gait Identifier/`
- `record.py`: `python record.py <person> <activity> [--seconds 60]` waits for the stream, counts down 5 s, plays a sound, records, plays a sound again, and saves `data/<person>_<activity>_<time>.csv`.
- `train.py` (add `--send` to push the model to the wearable)
  - Trims 2 s from each end of every recording, then windows it.
  - Picks Model 1's axis, then runs the cross-validation and prints the accuracies, confusion matrices and per-person scores.
  - Writes `model.json`, both into the mirror and into the ArduinoApps app.
  - Draws `figures/`: `model1_thresholds.png`, `model2_tree.png`, `confusion.png`, `feature_space.png` and `signals.png`.
- `gait_tree.py`: the numpy CART (`fit`, `predict`, `importance`).
- `live.html`: the demo UI. Open it in a browser.
  - Connects over MQTT websockets to `wss://broker.hivemq.com:8884/mqtt`.
  - Two animated SVG stick figures (Model 2 large, Model 1 small) walk, run or pedal at the measured cadence, with a bike for biking and a scrolling ground.
  - Tiles for speed (mph, km/h, m/s), cadence (steps/min or rpm), distance and Agree/Disagree.
  - 10 s strips of acceleration and gyro magnitude, a height input saved in localStorage, and a Demo button (or `?demo` in the URL).
  - Checked by screenshot in headless Chrome. Headless Chrome on this Mac doesn't fire requestAnimationFrame, so the test copy swapped it for a timer; the real page is fine.
- `live.py`: matplotlib fallback dashboard.
- `make_led_frames.py`: hand-placed pixel joints drawn with Bresenham lines; writes the frames into both copies of the sketch and a preview to `figures/led_frames.png`.
- `README.md`: overview, run steps and the speed formula.

## 6. What has and hasn't been tested

- ✅ train.py runs end to end on **synthetic** data for 3 fake people; all figures render. The synthetic data is too easy, so its 100% accuracy means nothing.
- ✅ main.py runs against a stubbed `arduino.app_utils`: a simulated walk led to `show(1, 270)`.
- ✅ sketch.ino passes `clang++ -fsyntax-only` against stand-in Arduino headers.
- ✅ live.html demo mode, checked by screenshot at desktop width and at 375 px.
- ❌ **It has never compiled for the real board.** If it fails, `Wire1` and `Bridge.provide_safe` are the likeliest culprits.
- ❌ It has never run with the real MPU-6050, and **no real data has been recorded yet.**

## 7. Current blocker: deploy can't reach the board

Mia ran `python3 tools/deploy.py gait-identifier` on her Mac, in the ArduinoApps folder, and got:

```
Trying 192.168.1.20 ... no answer
Trying 192.168.1.193 ... no answer
Trying 172.20.10.8 ... no answer
Trying msUNO.local ... no answer
Couldn't reach the board over USB (adb) or at any address in HOSTS. ...
```

Likely fixes, all done by Mia:
- Make sure the board is powered and on the same network as the Mac.
- Find the board's current IP, for example from the phone hotspot's connected-devices list or the router. A board on Mia's hotspot used to be at 172.20.10.7.
- Add the IP to `HOSTS` in `ArduinoApps/tools/board_secrets.py`.
- Or plug the board in over USB so adb works.

Remember that Claude must not try to reach the board itself.

## 8. Remaining steps (in order)

1. **Deploy** once the board is reachable. Fix any compile errors, then check that the matrix shows `?` (sensor found) and not `–`.
2. **Wire and strap**: MPU-6050 on the Qwiic connector (or 3.3V/GND/SDA→A4/SCL→A5), worn on the ankle.
3. **Record.** For each activity, Mia plus 2–3 friends, at slow and fast paces, with both real biking and air-biking on your back. The board needs internet for MQTT, so use the phone hotspot outdoors. Run `python record.py mia walking` and so on, on the Mac, in the project folder, with the venv active.
4. **Train**: `python train.py --send`. Look at the confusion matrices, try other `--depth` values for Model 2, and check that Model 1 fails where expected (biking vs. standing or walking, since pedaling has no impact peak).
5. **Demo**: open `live.html` and the LED figure, and test with peers.
6. **Notion page**: intro paragraph, demo video, model graphics (`figures/model1_thresholds.png`, `model2_tree.png`, `confusion.png`), GitHub link and the **4 reflection questions (get them from Mia)**. Notion drafts stay local and are never committed.
7. **Commit and push** both repos when Mia says so.

## 9. Talking points: "which model is better and why"

- Model 1 sees only how hard the foot hits. Standing (~1 g), walking (a moderate peak) and running (a big peak) separate well.
- Biking has **no impact**, so its peak looks like standing or slow walking, and Model 1 confuses them.
- Model 2 adds **rotation** (the gyro sees every pedal stroke and leg swing), **tilt** (the mean accelerometer reading shows the leg's angle on a bike or when lying on your back) and **cadence**. That separates biking, and it holds up better across different people's strides.
- Show it with the cross-validated confusion matrices, and live by watching the two figures disagree on live.html.
