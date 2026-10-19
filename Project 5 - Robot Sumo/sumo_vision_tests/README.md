# Sumo vision: tests and checks for the camera app

Tools to check that the camera app tells the behavior tree the truth, and on time, plus a description of the upgrade already built into `../sumo-vision/`. Run everything from this folder.

```
 UnoQ: webcam -> sumo-vision app -> MQTT  ME193/Rogers/MK  ->  sumo_brain.py (behavior tree, asks YOU to approve)
                  ^ the upgrade described below is already applied here   ^ vision_check.py and vision_latency_test.py listen here
```

## 1. What the vision upgrade changed (already applied in this repo)

The upgrade is part of `../sumo-vision/python/main.py`, `../sumo-vision/assets/index.html` and `../sumo-vision/assets/app.js`; there is nothing to apply.

| What it adds | Why |
|---|---|
| The floor is learned **per row**, and each picture's brightness and white-balance drift is measured on the floor and removed before comparing | The old code learned ONE colour from the bottom strip and compared raw colours, so a floor that looks different far away, or the camera's auto-exposure wandering, became one big false "opponent" |
| **Click on the tape** in the web page's picture: the app learns the tape's real colour from the spot you clicked, widened for light drift, and remembers it in `tape_calibration.json` | The fixed blue range can miss tape that the webcam renders pale, cyan or purple |
| `SELF_MASK_BOTTOM` (default 0, changes nothing) | Ignores the bottom rows of the picture, where the robot's own blue apron shows. tape_ahead now feeds the "we're OUT?" check, so a phantom reading stops the robot after back-edge escapes |
| Clicking on something that **isn't blue** is refused (a saturated orange wood floor has plenty of colour, so without this, one stray click would teach the app that orange is the tape and save it) | A mistaken click could break the tape detection and survive a restart |
| Learning the floor, and estimating drift, leave out blue tape (`masked_row_median`) | Pressing Relearn floor with a big ring corner in view used to teach the app that blue is floor: afterwards an empty ring with no tape reported an opponent in 100% of pictures |
| The web page says for 90 s after start-up whether the saved tape colour was loaded, missing or unreadable | App Lab shows the app's prints in its **Python** tab, not in **App launch**, so the console is easy to miss |
| Every print appears at once (`line_buffering`) | A container holds its output back, so the console can look empty even while the app runs |
| `OPP_CONFIRM = 3` | One stray frame no longer raises the "Opponent?" question |
| The app no longer crashes at boot if the network or broker isn't ready | `connect()` raised; it now retries in the background |
| `LOCK_CAMERA` (default False) | Freezes auto exposure and white balance once the floor is learned. **Untested on a real camera**: try it only if the picture still drifts |

To undo a tape colour you taught it, delete `tape_calibration.json` (next to `main.py`) and restart the app.

After changing the app: restart it with the robot in an **empty** ring (it learns the floor from its first 20 pictures), click on the
tape in the web page, then raise `SELF_MASK_BOTTOM` (0.10, 0.20 ...) until `vision_check.py --guided` scenario 1 passes.

## 2. Check it on the real robot

Setup: `pip install -r requirements.txt` (run it INSIDE this folder; the course repo has its own requirements.txt and running it from the wrong
folder installs that one instead). That file only asks for `paho-mqtt` and `numpy`. `vision_check.py` needs just `paho-mqtt`. The latency
test and the stress tests also need `cv2`: if you already have `opencv-python` OR `opencv-contrib-python` (the course environment has
the contrib one) you are set; never install both, they provide the same module and clash. The tools use the app's own broker and topic by default. Add
`--brain ../sumo_brain.py` so they use the tree's own thresholds.

| Command | What it tells you |
|---|---|
| `python vision_check.py --watch 15 --brain ...` | Message rate, gaps, commands sharing the topic, anything strange |
| `python vision_check.py --guided --brain ...` | Five scenarios (empty ring, tape, opponent, left, right) with pass or fail and what to change. If the empty scene reports a small false opponent while the real opponents read much bigger, it suggests a `MIN_OPP_AREA` in between (and says so when the sizes are too close for that to help). Scenario 1 needs the robot AWAY from the ring tape; scenario 2 needs the tape below the orange line |
| `python vision_latency_test.py --brain ...` | How long a blue edge takes to become a `tape_ahead` reading on the laptop (point the camera at the flashing window) |
| `python check_detector.py ../sumo-vision/python/main.py` | The stress test as a gate: run it after ANY change to the detection code |
| `python synthetic_eval.py "old/main.py" "new/main.py"` | The full side-by-side table below (compare any two versions of the detector) |
| `python test_tools.py` | The tools' own 20 tests: 16 run anywhere; 4 need Mosquitto installed and are skipped without it |

The pass marks are our own rules of thumb, not course rules.

## 3. The stress test (synthetic pictures: NOT a measurement of your camera)

Same scenes for both detectors; the detection functions are the real ones, pulled out of each `main.py`. It runs on a grey floor and on
an orange wood floor (`--all-floors`); the table below is the grey floor, and the orange floor gives the same picture (an empty ring
reported as an opponent: 44% before, 0% after).

| | original main.py | upgraded |
|---|---|---|
| Empty ring reported as an opponent (floor gradient, exposure +-25%, white balance drift) | **48%** | **0%** |
| Own apron in 3% of the picture, read as tape | 100% | 100% with the mask at 0, **0%** with `SELF_MASK_BOTTOM = 0.10` |
| Dark / red / white / LEGO-teal opponent found | 100% | 100% |
| Opponent the colour of the floor found | 33% (by accident) | 0% (it cannot be seen by colour) |
| Normal blue tape seen | 100% | 100% |
| Pale, cyan-leaning, purple-leaning tape seen | 22%, 33%, 67% | 67%, 100%, 100% after ONE click |
| Careless click on the tape's edge: tape seen / false tape in an empty ring | - | 100% / 0% |
| Ring corner in view (tape_ahead about 0.21), empty ring: false opponent | - | 0% |
| Floor relearned with a big ring corner in view (about 0.24), then an empty ring with no tape: false opponent | - | 0% (the first version of the upgrade: 100%) |
| A stray click on the plain floor is refused (grey floor / orange wood floor) | - | 100% / 100% (the first version of the upgrade accepted it on the orange floor: 0% refused) |
| LEGO-teal robot misread as tape | 33% | 0% after one click |
| Time per picture on this computer | 2 ms | 4 ms |

Still weak: pale, low-saturation tape (67%), and any robot the colour of the floor.

## 4. Is a trained model worth it? (`unoq/bench_detector.py`)

Do the checks in section 2 first. Go on to a model only if `--guided` still fails on real opponents after the upgrade.

1. **On the laptop**, export any small YOLO: `pip install ultralytics`, then `yolo export model=yolov8n.pt format=onnx imgsz=320`.
2. **On the UnoQ**: `python3 bench_detector.py yolov8n.onnx --size 320`. Speed depends on the network's size, not on what it was trained on.
3. At 8 pictures a second or better, an on-board model is realistic. Between 3 and 8, run it on every 3rd or 4th picture. Under 3, skip it.
The tool is tested on a toy network here; **it has not been run on a UnoQ**.

## What was tested, and what was not

The stress test, the guided check and the other tools are covered by `test_tools.py` (20 tests, including planted faults on a real Mosquitto broker), and the web page script passes `node --check`. The upgraded detector also passed the guided check on the real robot (`vision_check.py --guided`, all five scenarios, on our test floor).
**Not tested: the stress test as a measurement of the real camera (it uses synthetic pictures), `LOCK_CAMERA`, `unoq/bench_detector.py` on a real UNO Q, and a trained model.**
