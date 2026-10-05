# Runs one of the minifig YOLOv8 models (exported to ONNX) on a camera frame.
# Uses onnxruntime instead of PyTorch/Ultralytics, which are too heavy for the UNO Q.
#
# The models were exported from green_minifig.pt / blue_minifig.pt with
#   YOLO(pt).export(format="onnx", imgsz=320, opset=12, simplify=True)
import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 320      # must match the export size


class MinifigDetector:
    def __init__(self, onnx_path, confidence=0.5):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.confidence = confidence

    def detect(self, frame):
        """Return the most confident box as (x, y, w, h, conf), normalized 0-1
        to the frame size, or None if no minifig is found. frame is BGR (OpenCV)."""
        h0, w0 = frame.shape[:2]

        # Letterbox: shrink to fit 320x320 keeping the aspect ratio, pad with gray
        scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
        w1, h1 = round(w0 * scale), round(h0 * scale)
        pad_x, pad_y = (IMG_SIZE - w1) // 2, (IMG_SIZE - h1) // 2
        img = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        img[pad_y:pad_y + h1, pad_x:pad_x + w1] = cv2.resize(frame, (w1, h1))

        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = blob.astype(np.float32) / 255.0

        # Output is (1, 5, N): rows are cx, cy, w, h, conf for N candidate boxes
        out = self.session.run(None, {self.input_name: blob})[0][0]
        best = int(out[4].argmax())
        cx, cy, w, h, conf = out[:, best]
        if conf < self.confidence:
            return None

        # Undo the letterbox, then normalize to the original frame
        cx = (cx - pad_x) / scale / w0
        cy = (cy - pad_y) / scale / h0
        return float(cx), float(cy), float(w / scale / w0), float(h / scale / h0), float(conf)
