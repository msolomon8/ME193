"""
The neural network: a small convolutional neural net (CNN) built from scratch in PyTorch.

    input: the whole camera frame as a 3 x 64 x 112 color image (height 64, width 112)
      |
    conv 5x5, 16 filters -> ReLU -> max-pool 2x2      -> 16 x 32 x 56
    conv 3x3, 32 filters -> ReLU -> max-pool 2x2      -> 32 x 16 x 28
    conv 3x3, 64 filters -> ReLU -> max-pool 2x2      -> 64 x  8 x 14
      |
    flatten (64*8*14 = 7168 numbers) -> dropout -> linear -> 7 scores (0..5 fingers, blank)

- A convolution slides small filters (5x5 or 3x3 grids of weights) across the image.
  Each filter learns to light up for a pattern: edges first, then finger shapes.
- ReLU = max(0, z), the same "bend" that let us fit the V.
- Max-pooling keeps the strongest value in each 2x2 block, halving the image size.
  The net gets smaller and more abstract, and a finger moved a few pixels still counts.
- BatchNorm rescales each layer's outputs to mean 0 / spread 1 while training
  (the same reason we normalized the V data) so it trains faster and more reliably.
- Dropout randomly switches off 30% of the inputs to the last layer during training,
  so the net can't memorize single pixels.
- The 7 scores go through softmax -> probabilities, trained with cross-entropy loss.

Width = number of filters per layer (16, 32, 64). Depth = 3 conv layers + 1 linear layer.
"""
import torch
from torch import nn

from config import CLASSES, IMG_H, IMG_W


class FingerCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * (IMG_H // 8) * (IMG_W // 8), n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_model(model, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = FingerCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("meta", {})
