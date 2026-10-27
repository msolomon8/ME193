"""Train and compare the two gait models from the recordings in data/ (runs on the laptop).

    python train.py            # train, cross-validate, draw figures/, write model.json
    python train.py --send     # ...and send the model to the wearable over MQTT

Model 1: a decision tree that may only look at ONE number per 2 s window, the peak
         acceleration along one axis (it tries x, y and z and keeps the best axis).
Model 2: a decision tree on all IMU features (accel + gyro: tilt, shake, peaks, cadence).

Both are scored with leave-one-person-out cross-validation: train on everyone else, test
on the person left out. That is what happens when a peer puts the wearable on.
"""
import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gait-identifier" / "python"))
import gait_features as gf
import gait_tree

BROKER, PORT, TOPIC = "broker.hivemq.com", 1883, "ME193/Rogers/MiaGait"
APP_MODEL = Path.home() / "Documents/GitHub/ArduinoApps/gait-identifier/python/model.json"
TRIM = 2 * gf.RATE         # drop 2 s at each end of a recording (putting the laptop down, stopping)
TESTED = ("unseen people", "Tested on people the model never saw (leave-one-person-out)")
MAX_GAP = 10               # a gap of more than 10 lost samples splits a recording
PEAK_FEATURES = ["max_ax", "max_ay", "max_az"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # standing, walking, running, biking


def plain(name):
    """Feature name -> words a peer can read, e.g. std_ax -> 'shake along x'."""
    special = {"cadence_hz": "stride rate (Hz)", "periodicity": "how rhythmic",
               "mean_amag": "average total acceleration", "std_amag": "shake (total acceleration)",
               "max_amag": "hardest hit (total)", "range_amag": "acceleration range (total)",
               "mean_gmag": "average spin speed", "std_gmag": "spin variation", "max_gmag": "fastest spin"}
    if name in special:
        return special[name]
    stat, axis = name.split("_")
    words = {"mean": "tilt: gravity on" if axis[0] == "a" else "average spin around",
             "std": "shake along" if axis[0] == "a" else "spin variation around",
             "max": "peak acceleration along" if axis[0] == "a" else "fastest spin around",
             "min": "lowest acceleration along" if axis[0] == "a" else "fastest backward spin around"}
    return f"{words[stat]} {axis[1]}"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def load():
    """Every recording -> windows of features. Returns X, y, person, recording id, names, raw."""
    X, y, person, rec, raw = [], [], [], [], {}
    files = sorted((HERE / "data").glob("*.csv"))
    if not files:
        sys.exit("No recordings in data/. Run record.py first.")
    for r, path in enumerate(files):
        who, activity = path.stem.split("_")[:2]
        with open(path) as fh:
            rows = [[float(v) for v in row] for row in list(csv.reader(fh))[1:]]
        rows = rows[TRIM:-TRIM]
        raw.setdefault(activity, rows)
        segment = []
        for i, row in enumerate(rows + [None]):           # None flushes the last segment
            if row is None or (segment and row[0] - segment[-1][0] > MAX_GAP):
                for s in range(0, len(segment) - gf.WINDOW + 1, gf.HOP):
                    f = gf.features([tuple(x[1:]) for x in segment[s:s + gf.WINDOW]])
                    X.append([f[n] for n in gf.FEATURE_NAMES])
                    y.append(gf.CLASSES.index(activity))
                    person.append(who)
                    rec.append(r)
                segment = []
            if row is not None:
                segment.append(row)
    return np.array(X), np.array(y), np.array(person), np.array(rec), raw


def cross_validate(X, y, groups, names, depth):
    """Leave-one-group-out predictions for every window."""
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        tree = gait_tree.fit(X[~test], y[~test], names, len(gf.CLASSES), max_depth=depth)
        pred[test] = gait_tree.predict(tree, X[test], names)
    return pred


def confusion(y, pred):
    m = np.zeros((len(gf.CLASSES),) * 2, int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def export(tree):
    t = dict(tree)
    t["label"] = [gf.CLASSES[i] for i in tree["label"]]   # the board works with names
    return t


# ---------------- figures ----------------

def fig_model1(X1, y, tree, name, acc, out):
    """Model 1 drawn as what it is: thresholds on one number."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    lo, hi = X1.min() - 0.3, X1.max() + 0.3
    grid = np.linspace(lo, hi, 2000)
    regions = gait_tree.predict(tree, grid[:, None], [name])
    start = 0
    for i in range(1, len(grid) + 1):                     # shaded band per predicted class
        if i == len(grid) or regions[i] != regions[start]:
            ax.axvspan(grid[start], grid[i - 1], color=COLORS[regions[start]], alpha=0.12, lw=0)
            if i < len(grid):
                ax.axvline(grid[i - 1], color=MUTED, lw=1, ls="--")
                ax.text(grid[i - 1], len(gf.CLASSES) - 0.35, f"{grid[i - 1]:.2f} g", color=MUTED,
                        ha="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
            start = i
    rng = np.random.default_rng(0)
    for c, label in enumerate(gf.CLASSES):
        v = X1[y == c]
        ax.scatter(v, c + rng.uniform(-0.25, 0.25, len(v)), s=10, color=COLORS[c], alpha=0.7,
                   edgecolors="none")
    ax.set_yticks(range(len(gf.CLASSES)), gf.CLASSES)
    ax.set_ylim(-0.6, len(gf.CLASSES) - 0.2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"peak acceleration along {name[-1]} in a 2 s window (g)")
    ax.set_title(f"Model 1: one number, a few thresholds  ·  {acc:.0%} on {TESTED[0]}",
                 loc="left", color=INK)
    fig.text(0.01, 0.01, "Dots: real 2 s windows (true activity on the left). Shading: what model 1 answers.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_tree(tree, acc, out):
    """Model 2's decision tree, leaves in a row, each split above its children."""
    xs, ys = {}, {}
    leaf_x = [0]

    def place(n, depth):
        ys[n] = -depth
        if tree["left"][n] == -1:
            xs[n] = leaf_x[0]
            leaf_x[0] += 1
        else:
            place(tree["left"][n], depth + 1)
            place(tree["right"][n], depth + 1)
            xs[n] = (xs[tree["left"][n]] + xs[tree["right"][n]]) / 2

    place(0, 0)
    depth = -min(ys.values())
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * leaf_x[0]), 1.6 * (depth + 1) + 0.6))
    for n in xs:
        for child, word in ((tree["left"][n], "yes"), (tree["right"][n], "no")):
            if child != -1:
                ax.plot([xs[n], xs[child]], [ys[n], ys[child]], color=GRID, lw=1.5, zorder=0)
                ax.text((xs[n] + xs[child]) / 2, (ys[n] + ys[child]) / 2, word, color=MUTED,
                        fontsize=7, ha="center", va="center", backgroundcolor="white")
    for n in xs:
        counts = tree["counts"][n]
        if tree["left"][n] == -1:
            c = tree["label"][n]
            ax.text(xs[n], ys[n], f"{gf.CLASSES[c].upper()}\n{counts[c]}/{sum(counts)} windows",
                    ha="center", va="center", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.4", fc=COLORS[c], alpha=0.35, ec=COLORS[c]))
        else:
            ax.text(xs[n], ys[n], f"{plain(tree['feature'][n])}\n{tree['feature'][n]} ≤ {tree['threshold'][n]:.2f}?",
                    ha="center", va="center", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MUTED))
    ax.set_xlim(-0.7, leaf_x[0] - 0.3)
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.axis("off")
    ax.set_title(f"Model 2: decision tree on all IMU features  ·  {acc:.0%} on {TESTED[0]}",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_confusion(m1, m2, acc1, acc2, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, m, title in ((axes[0], m1, f"Model 1 (one peak)  {acc1:.0%}"),
                         (axes[1], m2, f"Model 2 (all IMU)  {acc2:.0%}")):
        frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(gf.CLASSES)):
            for j in range(len(gf.CLASSES)):
                ax.text(j, i, f"{frac[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(gf.CLASSES)), gf.CLASSES)
        ax.set_yticks(range(len(gf.CLASSES)), gf.CLASSES)
        ax.set_xlabel("model said")
        ax.set_ylabel("really doing")
        ax.set_title(title, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(TESTED[1], x=0.01, ha="left",
                 color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_features(X, y, names, f1, f2, out):
    i, j = names.index(f1), names.index(f2)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for c, label in enumerate(gf.CLASSES):
        ax.scatter(X[y == c, i], X[y == c, j], s=12, color=COLORS[c], alpha=0.7, label=label,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"{plain(f1)} ({f1})")
    ax.set_ylabel(f"{plain(f2)} ({f2})")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False)
    ax.set_title("Model 2's two most useful features", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_signals(raw, out):
    """Four seconds of each activity: what the sensor actually sees."""
    n = 4 * gf.RATE
    fig, axes = plt.subplots(len(gf.CLASSES), 2, figsize=(10, 7), sharex=True)
    t = np.arange(n) / gf.RATE
    for r, label in enumerate(gf.CLASSES):
        rows = np.array(raw.get(label, [])[:n])
        for col, (lo, title, unit) in enumerate(((1, "acceleration", "g"), (4, "rotation", "deg/s"))):
            ax = axes[r, col]
            if len(rows):
                for k, axis in enumerate("xyz"):
                    ax.plot(t[:len(rows)], rows[:, lo + k], lw=1.2, color=["#2a78d6", "#eb6834", "#1baf7a"][k],
                            label=axis)
            ax.grid(color=GRID, lw=0.6)
            if col == 0:
                ax.set_ylabel(f"{label}\n({unit})")
            else:
                ax.set_ylabel(unit)
            if r == 0:
                ax.set_title(title, loc="left", color=INK)
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4, help="model 2 tree depth")
    ap.add_argument("--send", action="store_true", help="publish the model to the wearable")
    args = ap.parse_args()

    X, y, person, rec, raw = load()
    names = gf.FEATURE_NAMES
    people = sorted(set(person))
    if len(people) > 1:
        groups, group_word = person, "person"
    else:
        # One person, maybe one recording per activity: test on each half of a recording
        # with the other half still in training, so no activity drops out entirely.
        second = np.zeros(len(rec), dtype=int)
        for r in np.unique(rec):
            idx = np.flatnonzero(rec == r)
            second[idx] = np.arange(len(idx)) >= len(idx) / 2
        groups, group_word = rec * 2 + second, "half-recording"
        global TESTED
        TESTED = ("held-out halves", "Tested on the half of each recording left out of training (one person)")
    print(f"{len(y)} windows from {len(set(rec))} recordings, people: {', '.join(people)}")
    for c, label in enumerate(gf.CLASSES):
        print(f"  {label:9s} {np.sum(y == c):5d} windows  ({', '.join(sorted(set(person[y == c]))) or 'none!'})")
    missing = [c for c in gf.CLASSES if not np.any(y == gf.CLASSES.index(c))]
    if missing:
        print(f"WARNING: no data for {', '.join(missing)}")
    if len(people) == 1:
        print("Only one person: each recording is split in half, and each half is tested "
              "on a model trained on everything else.")

    # Model 1: pick the axis whose peak works best on unseen people
    print(f"\nModel 1, leave-one-{group_word}-out accuracy:")
    best = None
    for k in PEAK_FEATURES:
        Xk = X[:, [names.index(k)]]
        acc = np.mean(cross_validate(Xk, y, groups, [k], depth=3) == y)
        print(f"  {k}: {acc:.1%}")
        if best is None or acc > best[1]:
            best = (k, acc)
    k1, acc1 = best
    X1 = X[:, [names.index(k1)]]
    pred1 = cross_validate(X1, y, groups, [k1], depth=3)
    tree1 = gait_tree.fit(X1, y, [k1], len(gf.CLASSES), max_depth=3)

    pred2 = cross_validate(X, y, groups, names, depth=args.depth)
    acc2 = np.mean(pred2 == y)
    tree2 = gait_tree.fit(X, y, names, len(gf.CLASSES), max_depth=args.depth)
    print(f"\nModel 2 (depth {args.depth}), leave-one-{group_word}-out accuracy: {acc2:.1%}")
    imp = gait_tree.importance(tree2)
    print("  features it relies on: " + ", ".join(f"{f} {w:.0%}" for f, w in imp))

    m1, m2 = confusion(y, pred1), confusion(y, pred2)
    print("\nConfusion (rows = really doing, columns = model said):")
    print(" " * 22 + " ".join(f"{c[:5]:>6s}" for c in gf.CLASSES))
    for tag, m in (("model 1", m1), ("model 2", m2)):
        for i, c in enumerate(gf.CLASSES):
            print(f"  {tag} {c:12s}" + " ".join(f"{v:6d}" for v in m[i]))

    if len(people) > 1:
        print("\nPer person (model 1 / model 2):")
        for p in people:
            s = person == p
            print(f"  {p:10s} {np.mean(pred1[s] == y[s]):.0%} / {np.mean(pred2[s] == y[s]):.0%}")

    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    fig_model1(X1[:, 0], y, tree1, k1, acc1, figs / "model1_thresholds.png")
    fig_tree(tree2, acc2, figs / "model2_tree.png")
    fig_confusion(m1, m2, acc1, acc2, figs / "confusion.png")
    top = [f for f, _ in imp][:2] + ["std_gmag", "mean_amag"]
    fig_features(X, y, names, top[0], next(f for f in top[1:] if f != top[0]), figs / "feature_space.png")
    fig_signals(raw, figs / "signals.png")
    print(f"\nFigures in {figs.relative_to(HERE)}/")

    model = {"classes": gf.CLASSES, "trained": datetime.now().isoformat(timespec="seconds"),
             "people": people, "windows": int(len(y)),
             "model1": {"feature": k1, "cv_accuracy": round(float(acc1), 3), "tree": export(tree1)},
             "model2": {"features": names, "cv_accuracy": round(float(acc2), 3), "tree": export(tree2)}}
    text = json.dumps(model)
    local = HERE / "gait-identifier" / "python" / "model.json"
    for path in (local, APP_MODEL):
        if path.parent.exists():
            path.write_text(text)
            print(f"Wrote {path}")

    if args.send:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(BROKER, PORT)
        client.loop_start()
        client.publish(TOPIC + "/model", text, qos=1, retain=True).wait_for_publish(10)
        client.loop_stop()
        print(f"Sent to the wearable on {TOPIC}/model (it switches to the new model right away)")


if __name__ == "__main__":
    main()
