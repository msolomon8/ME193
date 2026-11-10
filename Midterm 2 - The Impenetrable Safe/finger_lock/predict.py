"""
Run the trained network on a single image. Used by the interface and (later) the UNO Q.
Inference runs on the CPU - same as it will on the UNO Q.
"""
import torch

from camera import to_model_input
from config import MODEL_PATH
from model import load_model


class Predictor:
    def __init__(self, path=MODEL_PATH):
        torch.set_num_threads(2)
        self.model, self.meta = load_model(path, "cpu")

    @torch.no_grad()
    def probabilities(self, frame_bgr):
        """Returns a list of 7 probabilities: P(0 fingers), ..., P(5 fingers), P(blank)."""
        x = to_model_input(frame_bgr, self.meta.get("roi")).unsqueeze(0)   # same crop as training
        scores = self.model(x)
        return torch.softmax(scores, dim=1)[0].tolist()
