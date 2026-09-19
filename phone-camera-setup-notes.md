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
