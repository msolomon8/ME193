# Both Car Mini Figures

One app detects both minifigs with the computer's webcam and parks two cars:
the green car with its minifig on the left, the blue car with its minifig on the right.

```
laptop webcam → web page → GREEN car (two_car_parking_app: detects BOTH minifigs)
                                ├─ drives itself (green minifig → left line, x = 0.25)
                                └─ MQTT → BLUE car (car_blue_app: just drives, blue minifig → right line, x = 0.75)
```

- `two_car_parking_app/` - goes on the **green** car's UNO Q. Serves the web page, runs the
  trained green + blue YOLO models (ONNX), drives the green car, and sends the blue minifig's
  position over MQTT (`ME193/minifig/tashamia/blue`).
- `car_blue_app/` - goes on the **blue** car's UNO Q. Receives the blue minifig's position and drives.

## Setup

1. Zip each app folder (in Finder: right-click → Compress), or in the terminal from this folder:
   `zip -r ~/Desktop/two_car_parking_app.zip two_car_parking_app` and
   `zip -r ~/Desktop/car_blue_app.zip car_blue_app`
2. In App Lab, import `two_car_parking_app.zip` on the **green** car's UNO Q and
   `car_blue_app.zip` on the **blue** car's UNO Q.
3. Make sure both boards and the computer are on the same Wi-Fi with internet (for MQTT).
4. Start **both** apps in App Lab.
5. Find the green car's IP address (shown in App Lab), and on the computer open
   `https://<green car's IP>:7000`.
6. The browser warns the connection isn't private (the board makes its own certificate):
   click **Advanced → Proceed**. HTTPS is required for webcam access.
7. Click **Start camera** and allow camera access. Both minifigs get boxes on the page,
   and each car drives until its minifig reaches its dashed line.

## If something's off

- A car drives **away** from its line: flip `DIRECTION` in that app's `python/main.py`.
- Cars overshoot: lower `MAX_SPEED` (the page shows the board's ms per frame).
- `No module named 'onnxruntime'` / `'paho'`: App Lab didn't install `python/requirements.txt`.
