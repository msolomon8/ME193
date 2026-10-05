# Fine-tunes a YOLOv8 detector to find the blue LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export, class "Blue-Minifug",
# 53 train / 15 valid / 7 test images).
# Training progress and the final weights land under
# ME193/runs/detect/blue_minifig/weights/best.pt (Ultralytics runs_dir setting);
# best.pt is then copied to blue_minifig.pt next to this script.

import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
EPOCHS = 100  # small dataset benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = "mps"  # Apple Silicon GPU; falls back to CPU below if unavailable

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with ~75 images.
model = YOLO("yolov8n.pt")

try:
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="blue_minifig",
        exist_ok=True,  # reuse runs/detect/blue_minifig instead of blue_minifig2, 3, ...
        device=DEVICE,
    )
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="blue_minifig",
        exist_ok=True,
        device="cpu",
    )

# Copy the best weights next to this script, where blue_minifig_mqtt.py loads them
# (runs/ isn't in git, but blue_minifig.pt is).
best = Path(model.trainer.save_dir) / "weights" / "best.pt"
shutil.copy(best, Path(__file__).parent / "blue_minifig.pt")
print(f"Done. Weights saved to {best} and copied to blue_minifig.pt")
