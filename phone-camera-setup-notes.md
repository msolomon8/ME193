# Phone Camera Setup Notes for AprilTag Robot Control

Date: 2026-09-19

## Goal
Use a phone-mounted camera to stream footage to the Mac, detect a stationary AprilTag, and steer the LEGO robot to center on it.

## Key findings
- The Mac cannot directly access the phone camera like a normal computer webcam unless the phone is exposing a stream or acting like a webcam device.
- The normal phone camera app alone is not enough; it must either:
  - stream over Wi‑Fi as an HTTP/MJPEG stream, or
  - appear as a webcam to the Mac through a companion app.
- For an iPhone + Mac setup, the no-app route is the browser-based workaround, but it is less reliable than a proper webcam app.
- For a Mac, app-based camera solutions are usually the simplest and most stable path.

## Recommended paths
### Option 1: no app
- Use a phone browser page served by the Mac.
- The browser requests camera access and sends frames back to the Mac over Wi‑Fi.
- This is the only true no-app path.
- It is workable but less reliable than an app-based webcam.

### Option 2: app-based webcam stream
Recommended apps for iPhone + Mac:
- Camo
- EpocCam
- iVCam

These apps make the phone appear like a webcam to the Mac. Then OpenCV can read the feed with the normal camera index (`--camera 0`, `--camera 1`, etc.).

## Practical workflow
1. Connect the iPhone and Mac to the same Wi‑Fi or connect the phone with the app’s recommended method.
2. Start the camera app or browser-based streaming page.
3. Verify the phone is serving frames or appearing as a webcam.
4. Run the AprilTag script on the Mac.
5. Keep the AprilTag in view of the phone camera.
6. The Mac computes the target offset and drives the robot to center the tag.

## Current project files
- `phone-browser-apriltag.py`: browser-based workaround for the no-app route.
- `smartphone-apriltag.py`: stream-based script intended for a phone camera stream or webcam feed.
- `apriltag-parking.py`: earlier local-camera / AprilTag parking prototype.

## Important caveat
The robot control part only works if:
- the phone feed is actually reaching the Mac,
- the AprilTag is visible in the frame,
- the LEGO double motor is connected,
- the correct green card serial is used for the hardware.

## Hardware note
The green connection card serial used in the latest scripts was updated to the 0997 card when that was identified as the active hardware.

## Commands used during setup
```bash
cd /Users/miasolomon/Documents/GitHub/ME193
source ME193/le-venv/bin/activate
python phone-browser-apriltag.py --help
python phone-browser-apriltag.py --stream http://10.243.41.54:8080/video
```

## Recommendation
For a Mac + iPhone workflow, the most reliable path is to use a webcam app such as Camo or EpocCam and then run the AprilTag script against the Mac camera feed rather than trying to force a raw IP stream URL.

## Current Browser Controller
The active controller is `iphone-apriltag.py`.

It uses:
- iPhone Safari back-camera access through an HTTPS page served by the Mac.
- The Mac as the AprilTag detector and robot controller.
- A LEGO Single Motor for slow, continuous camera scanning and reacquisition.
- A LEGO Double Motor for car yaw alignment and parking.
- Green connection card serial `0997`.
- AprilTag 36h11 detection.
- A tag size of 2.025 inches (`0.051435` meters).

The camera scan is continuous in one direction at speed `3`. It stops when the tag is detected. If the tag is later lost, the car stops and the camera slowly reacquires it.

## Parking Control
The controller uses separate measurements for separate jobs:
- AprilTag yaw controls whether the car is parallel to the tag.
- AprilTag image center controls whether the tag is centered in the camera view.
- AprilTag pose distance controls forward/backward parking.

The car cannot report `STOP - PARKED` unless distance, yaw, and image centering are all within tolerance. The current target distance is `0.45` meters and the center tolerance is `35` pixels.

The single motor is not used to park the car. It only scans or reacquires the AprilTag. The double motor performs the car alignment and forward/backward parking.

## Known Failure Causes and Fixes
- The old controller used horizontal pixel error as forward/backward error. This confused left/right position with distance. It was replaced with pose-estimated distance control.
- A camera return-to-zero phase blocked the double motor. It was removed so the car alignment phase can run immediately after tag detection.
- Finite camera degree commands were interrupted by timed reversals. Scanning is now continuous so the camera can complete full rotations.
- The car could declare parked without the tag being centered. Parking now requires distance, yaw, and center checks.
- Aggressive yaw commands caused the car to lose the tag. Yaw gain, correction cap, and slew rate were reduced.

## Current Run Command
From the project root on the Mac:

```bash
cd /Users/miasolomon/Documents/GitHub/ME193
source ME193/le-venv/bin/activate
python iphone-apriltag.py --https --host 0.0.0.0 --port 8444
```

Open Safari on the iPhone at the Mac's current Wi-Fi address:

```text
https://<MAC-IP>:8444/
```

The Mac and iPhone must be on the same Wi-Fi network. The local HTTPS certificate warning is expected; accept it so Safari can use the camera.

## 2026-09-20 working notes

### Current script under test
- Active file: `updatedphone-apriltag.py`
- The script is a browser-based iPhone camera -> Mac AprilTag controller.
- It binds to `--host 0.0.0.0` and uses the port specified by `--port`.
- The iPhone must open the Mac's actual local IP, not `0.0.0.0`.

### Correct URL pattern
Use the Mac's actual Wi-Fi IP on the phone, for example:

```text
https://10.243.29.88:8443/
```

Do not use `0.0.0.0` in the phone browser. `0.0.0.0` is only a bind address for the server, not a reachable client URL.

### Certificate issue
A certificate generated for an older address (`192.168.1.213`) will not match the current machine IP (`10.243.29.88`).

The cert was regenerated with the correct IP:

```bash
cd /Users/miasolomon/Documents/GitHub/ME193
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout phone-camera-key.pem \
  -out phone-camera-cert.pem \
  -days 365 \
  -subj "/CN=10.243.29.88" \
  -addext "subjectAltName=IP:10.243.29.88"
```

### Run command that works
```bash
cd /Users/miasolomon/Documents/GitHub/ME193
source ME193/le-venv/bin/activate
python updatedphone-apriltag.py --https --host 0.0.0.0 --port 8443
```

### Browser symptom and root cause
- Browser loads the page but the status changes to `Mac connection lost` after a refresh.
- That is coming from the JavaScript fetch to `/frame` in the script.
- The script succeeds in serving the page, but the POST from the phone camera does not reach the server reliably when:
  - the cert is for the wrong IP,
  - the phone is not on the same network,
  - the URL is wrong,
  - or Safari blocks the self-signed HTTPS certificate.

### Current recommendation
- Start the script with `--https --host 0.0.0.0 --port 8443`.
- Open the phone browser to `https://10.243.29.88:8443/`.
- If Safari warns about the cert, trust the certificate once.
- Make sure the phone and Mac are on the same Wi-Fi network or a trusted local network.
- If the browser still fails, verify the Mac IP with:

```bash
ipconfig getifaddr en0
```

This is the current status of the phone-camera setup and the troubleshooting steps from today.
