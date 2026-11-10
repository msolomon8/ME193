"""
Train the finger-counting CNN on your computer.

Run from the interface (Train tab), or from the terminal:
    ../.venv/bin/python train.py --epochs 30

Each epoch:
  1. forward pass: images -> network -> 7 scores
  2. loss: cross-entropy (how wrong the scores are for the true finger count)
  3. backward pass: PyTorch uses the chain rule to get the gradient of the loss
     for every weight (the same backprop we did by hand for the V)
  4. update: Adam optimizer nudges every weight downhill (like m -= lr * dm)
Then we check accuracy on the validation photos the network never trained on.

Saves: models/finger_cnn.pt (best model), history.json, training_curves.png,
       confusion_matrix.png, confusion.json
"""
import argparse
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from config import CLASS_NAMES, CLASSES, IMG_H, IMG_W, MODEL_DIR, MODEL_PATH
from dataset import FingerDataset, list_samples, split
from model import FingerCNN, count_parameters, save_model


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"      # Mac GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss_sum += loss_fn(scores, y).item() * len(y)
            pred = scores.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1
    return loss_sum / max(total, 1), correct / max(total, 1), confusion


RUNS_DIR = MODEL_DIR / "runs"


def archive_current_model(log=print):
    """Copy the current model into models/runs/ before training replaces it."""
    if not MODEL_PATH.exists():
        return
    import shutil
    try:
        meta = torch.load(MODEL_PATH, map_location="cpu", weights_only=True).get("meta", {})
        stamp = meta.get("trained_at", "old").replace(" ", "_").replace(":", "")
        name = f"{stamp}_{meta.get('val_acc', 0) * 100:.0f}pct.pt"
    except Exception:
        name = time.strftime("%Y-%m-%d_%H%M%S") + "_unknown.pt"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not (RUNS_DIR / name).exists():
        shutil.copy2(MODEL_PATH, RUNS_DIR / name)
        log(f"Saved the previous model as runs/{name}")


def train(epochs=30, lr=1e-3, batch_size=32, progress=None, should_stop=None, log=print):
    samples = list_samples()
    if len(samples) < 12:
        raise ValueError(f"Only {len(samples)} photos. Collect at least a few bursts of each class first.")
    train_s, val_s = split(samples)
    if not val_s:
        raise ValueError("Not enough separate bursts for a validation set - take at least 2 bursts per class.")

    present = sorted({c for _, c in samples})
    missing = [c for c in CLASSES if c not in present]
    if missing:
        log(f"Warning: no photos for {missing} fingers - the model can't learn those yet.")

    device = pick_device()
    log(f"Training on {len(train_s)} photos, validating on {len(val_s)}, device = {device}")

    roi = config.load().get("roi")     # crop to the inside of the red frame (same as live)
    log(f"Crop (inside red frame): {roi or 'whole picture'}")
    train_dl = DataLoader(FingerDataset(train_s, augment=True, roi=roi), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(FingerDataset(val_s, augment=False, roi=roi), batch_size=128)

    archive_current_model(log)
    torch.manual_seed(0)
    model = FingerCNN().to(device)
    log(f"Network has {count_parameters(model):,} weights")
    # class balancing: a class with fewer photos counts more per photo, so having e.g.
    # 800 photos of 3 but 400 of blank doesn't pull the answers toward 3
    n_per = torch.tensor([max(1, sum(1 for _, c in train_s if c == k)) for k in CLASSES], dtype=torch.float)
    weights = n_per.sum() / (len(CLASSES) * n_per)
    log("Class weights: " + ", ".join(f"{CLASS_NAMES[k]}={w:.2f}" for k, w in zip(CLASSES, weights.tolist())))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # learning rate starts at lr and smoothly shrinks to ~0 by the last epoch (cosine curve):
    # big steps early to learn fast, tiny steps at the end so it SETTLES instead of bouncing
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_conf, best_epoch = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if should_stop and should_stop():
            log("Stopped early.")
            break
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            scores = model(x)                 # 1. forward
            loss = loss_fn(scores, y)         # 2. loss
            opt.zero_grad()
            loss.backward()                   # 3. backprop
            opt.step()                        # 4. gradient step
            loss_sum += loss.item() * len(y)
            correct += (scores.argmax(1) == y).sum().item()
            total += len(y)

        sched.step()
        val_loss, val_acc, conf = evaluate(model, val_dl, device, nn.CrossEntropyLoss())
        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc, best_conf, best_epoch = val_acc, conf, epoch
            save_model(model, MODEL_PATH, {
                "classes": CLASSES, "img_size": [IMG_W, IMG_H], "roi": roi, "white_balance": True, "val_acc": val_acc, "epoch": epoch,
                "n_train": len(train_s), "n_val": len(val_s),
                "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            })

        msg = (f"epoch {epoch:3d}/{epochs}  train loss {loss_sum/total:.3f} acc {correct/total:.0%}   "
               f"val loss {val_loss:.3f} acc {val_acc:.0%}")
        log(msg)
        if progress:
            progress({"epoch": epoch, "epochs": epochs, "history": history,
                      "best_acc": best_acc, "best_epoch": best_epoch, "confusion": best_conf})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result = {"history": history, "best_acc": best_acc, "best_epoch": best_epoch,
              "confusion": best_conf, "seconds": round(time.time() - t0, 1)}
    (MODEL_DIR / "history.json").write_text(json.dumps(result, indent=1))
    save_plots(result)
    log(f"Best validation accuracy {best_acc:.0%} at epoch {best_epoch}. Saved {MODEL_PATH.name}")
    return result


def save_plots(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = result["history"]
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="training")
    ax[0].plot(ep, h["val_loss"], label="validation")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss vs. epoch")
    ax[1].plot(ep, [a * 100 for a in h["train_acc"]], label="training")
    ax[1].plot(ep, [a * 100 for a in h["val_acc"]], label="validation")
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy vs. epoch", ylim=(0, 100))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "training_curves.png", dpi=150)
    plt.close(fig)

    conf = result["confusion"]
    if conf:
        fig, a = plt.subplots(figsize=(5.5, 5))
        a.imshow(conf, cmap="Blues")
        for t in range(len(CLASSES)):
            for p in range(len(CLASSES)):
                a.text(p, t, conf[t][p], ha="center", va="center",
                       color="white" if conf[t][p] > max(map(max, conf)) / 2 else "black")
        a.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
              xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
              xlabel="predicted", ylabel="true",
              title=f"Confusion matrix (validation, {result['best_acc']:.0%})")
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    (MODEL_DIR / "confusion.json").write_text(json.dumps(conf))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    train(args.epochs, args.lr, args.batch)
