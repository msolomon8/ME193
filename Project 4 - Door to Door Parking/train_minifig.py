# Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export, class "Lego-Minifig",
# 81 train / 23 valid / 11 test images).
# Training progress and the final weights land under
# ME193/runs/detect/green_minifig/weights/best.pt (Ultralytics runs_dir setting)

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
EPOCHS = 100  # small dataset benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = "mps"  # Apple Silicon GPU; falls back to CPU below if unavailable

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with ~100 images.
model = YOLO("yolov8n.pt")

try:
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        exist_ok=True,  # reuse runs/detect/green_minifig instead of green_minifig2, 3, ...
        device=DEVICE,
    )
except Exception as e:
    print(f"Training on device={DEVICE!r} failed ({e}). Retrying on CPU...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name="green_minifig",
        exist_ok=True,
        device="cpu",
    )

print("Done. Weights saved to ME193/runs/detect/green_minifig/weights/best.pt")
