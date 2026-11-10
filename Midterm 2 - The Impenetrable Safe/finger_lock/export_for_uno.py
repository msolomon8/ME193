"""
Export the trained network for the UNO Q (which runs it with numpy, no PyTorch).

    ~/.venvs/impenetrable-safe/bin/python export_for_uno.py

Writes ../uno_q_app/python/finger_cnn.npz and rebuilds ../impenetrable-safe.zip (import it in App Lab). BatchNorm is folded into each conv layer:
    BN(conv(x)) = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
               = conv'(x)  with  w' = w * s,  b' = (b - mean) * s + beta,  s = gamma / sqrt(var + eps)
Then it checks that the numpy version gives the same answers as PyTorch on real photos.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import to_model_input
from config import CLASSES, MODEL_PATH
from dataset import class_dir
from model import load_model

OUT = Path(__file__).parent.parent / "uno_q_app" / "python" / "finger_cnn.npz"


def export():
    model, meta = load_model(MODEL_PATH, "cpu")
    f = model.features
    arrays = {}
    for i, (conv, bn) in enumerate([(f[0], f[1]), (f[4], f[5]), (f[8], f[9])], start=1):
        s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        arrays[f"conv{i}_w"] = (conv.weight.detach() * s[:, None, None, None]).numpy().astype(np.float32)
        arrays[f"conv{i}_b"] = ((conv.bias.detach() - bn.running_mean) * s + bn.bias.detach()).numpy().astype(np.float32)
    lin = model.classifier[2]
    arrays["lin_w"] = lin.weight.detach().numpy().astype(np.float32)
    arrays["lin_b"] = lin.bias.detach().numpy().astype(np.float32)
    arrays["roi"] = np.array(meta.get("roi") or [], dtype=np.float32)
    arrays["img_w"], arrays["img_h"] = np.array(meta["img_size"][0]), np.array(meta["img_size"][1])
    arrays["val_acc"] = np.array(meta.get("val_acc", 0.0))
    arrays["white_balance"] = np.array(bool(meta.get("white_balance", False)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"Saved {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, model from {meta.get('trained_at')}, "
          f"{meta.get('val_acc', 0):.0%} validation)")
    return model, meta


def check(model, meta, n_per_class=8):
    sys.path.insert(0, str(OUT.parent))
    from safe_net import NumpyCNN
    npnet = NumpyCNN(OUT)
    worst, agree, total = 0.0, 0, 0
    for c in CLASSES:
        for p in sorted(class_dir(c).glob("*.png"))[::40][:n_per_class]:
            img = cv2.imread(str(p))
            x = to_model_input(img, meta.get("roi"))
            with torch.no_grad():
                pt = torch.softmax(model(x.unsqueeze(0)), 1)[0].numpy()
            nppr = np.array(npnet.probabilities(npnet.prepare(img)))   # numpy's OWN preprocessing
            worst = max(worst, float(np.abs(pt - nppr).max()))
            agree += int(pt.argmax() == nppr.argmax())
            total += 1
    print(f"numpy vs PyTorch on {total} photos: same answer {agree}/{total}, biggest probability difference {worst:.6f}")


def copy_lock_policy():
    """The UNO Q uses the exact same lock policy file (minus the laptop config import)."""
    src = (Path(__file__).parent / "lock_policy.py").read_text()
    src = src.replace("from config import NOT_A_DIGIT", "NOT_A_DIGIT = (6,)   # 6 = blank (copied from config.py)")
    (OUT.parent / "lock_policy.py").write_text(src)
    print(f"Copied lock_policy.py to {OUT.parent}")


def build_zip():
    """Package uno_q_app/ as impenetrable-safe.zip for App Lab's "Import an App"."""
    import zipfile
    app_dir = OUT.parent.parent
    zip_path = app_dir.parent / "impenetrable-safe.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(app_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store":
                z.write(f, "impenetrable-safe/" + str(f.relative_to(app_dir)))
    print(f"Built {zip_path}")


if __name__ == "__main__":
    m, meta = export()
    copy_lock_policy()
    check(m, meta)
    build_zip()
