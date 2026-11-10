"""
Loading the hand photos and enlarging the dataset with random changes (augmentation).

Photos live in data/0/ ... data/5/ and data/blank/ (folder name = class).
File names look like  <burst id>_<frame>.png . All frames from one burst are nearly
identical, so a whole burst goes EITHER to training OR to validation - never split.
Otherwise validation accuracy would be fake-high (the net would have seen almost the
same photo in training).
"""
import random
from collections import defaultdict

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from camera import normalize, to_uint8_tensor
from config import CLASS_NAMES, CLASSES, DATA_DIR

# Random changes applied every time a training photo is used, so each epoch the net
# sees slightly different versions: rotated, shifted, zoomed, mirrored (left vs. right
# hand), brighter/darker. This is the professor's "rotate and shift" hint.
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    # fill=200: the gap left after rotating/shifting is filled with paper-white, not black
    v2.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=200),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])


def class_dir(label):
    return DATA_DIR / CLASS_NAMES[label]


def list_samples():
    samples = []
    for c in CLASSES:
        for p in sorted(class_dir(c).glob("*.png")):
            samples.append((p, c))
    return samples


def counts():
    return {c: len(list(class_dir(c).glob("*.png"))) for c in CLASSES}


def burst_id(path):
    return path.stem.rsplit("_", 1)[0]


def split(samples, val_fraction=0.2, seed=0):
    """Split by burst, separately for each class, so every class appears in validation."""
    rng = random.Random(seed)
    train, val = [], []
    for c in CLASSES:
        groups = defaultdict(list)
        for p, label in samples:
            if label == c:
                groups[burst_id(p)].append((p, label))
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = int(round(len(keys) * val_fraction))
        if len(keys) >= 2:
            n_val = max(1, n_val)
        for i, k in enumerate(keys):
            (val if i < n_val else train).extend(groups[k])
    return train, val


class FingerDataset(Dataset):
    def __init__(self, samples, augment=False, roi=None):
        self.samples = samples
        self.augment = augment
        # load every photo once into memory (small: 3 x 64 x 112 bytes each)
        self.images = [to_uint8_tensor(cv2.imread(str(p)), roi) for p, _ in samples]
        self.labels = torch.tensor([label for _, label in samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            img = AUGMENT(img)
        return normalize(img), self.labels[i]
