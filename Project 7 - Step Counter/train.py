"""Train the step-counting neural net and measure how good it is (runs on the laptop).

    python train.py                  # full run: cross-validation, sensor test, final models, figures
    python train.py --quick          # fewer nets, no sensor test (for a first look)
    python train.py --height-in 66   # your height, for the distance estimate

How good is it (the uncertainty analysis):
  * Leave-one-session-out: for each recording, train on all the OTHER recordings, then count
    the steps in the one it never saw and compare with the number you counted out loud.
  * An ensemble of 5 nets, each started from different random weights. How much they
    disagree is the +- on each count.
  * A plain peak counter (no learning) on the same recordings, to see what the net adds.
  * Sensor test: the same cross-validation with some channels switched off.
  * Distance: steps x step length, with the step length calibrated on the measured walks.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steps_data as sd
import step_net as sn

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
MASKS = {                                   # which channels each sensor test keeps
    "all 7 channels": [1, 1, 1, 1, 1, 1, 1],
    "IMU only (no ultrasonic)": [1, 1, 1, 1, 1, 1, 0],
    "gyro only": [0, 0, 0, 1, 1, 1, 0],
    "accel only": [1, 1, 1, 0, 0, 0, 0],
    "ultrasonic only": [0, 0, 0, 0, 0, 0, 1],
}
DEFAULT_STEP = (0.225, 0.105)   # step length = height x (a + b x steps per second); textbook walking ratio


# ---------------- baseline: a peak counter with no learning ----------------

def peak_threshold(sessions):
    """Half the typical swing-peak height in the training sessions."""
    heights = []
    for s in sessions:
        m = s["meta"]
        if m["steps"] > 0:
            sig, _ = sd.swing_signal(s["x"], m["start"], m["end"])
            p = sd.find_peaks(sig, m["start"], m["end"], int(0.4 * sd.RATE))[:max(1, m["steps"] // 2)]
            heights += list(sig[p])
    return 0.5 * float(np.median(heights)) if heights else 1.0


def peak_count(s, thr):
    m = s["meta"]
    sig, _ = sd.swing_signal(s["x"], m["start"], m["end"])
    p = sd.find_peaks(sig, m["start"], m["end"], int(0.4 * sd.RATE))
    return 2 * int(np.sum(sig[p] > thr)) if len(p) else 0


# ---------------- distance ----------------

def hop_cadence(per_hop):
    """Steps per second at every hop, smoothed over 2 s."""
    k = np.ones(4) / 4
    return np.convolve(per_hop, k, mode="same") / (sd.HOP / sd.RATE)


def distance(per_hop, height_m, a, b):
    return float(np.sum(per_hop * height_m * (a + b * hop_cadence(per_hop))))


def fit_step_length(items, height_m):
    """items: list of (per_hop counts, true distance). Least squares for a, b (or a only if < 3 walks)."""
    if not items:
        return DEFAULT_STEP
    A = np.array([[np.sum(p) * height_m, np.sum(p * hop_cadence(p)) * height_m] for p, _ in items])
    d = np.array([t for _, t in items])
    if len(items) >= 3:
        a, b = np.linalg.lstsq(A, d, rcond=None)[0]
        if b < 0:                     # longer steps when slower makes no sense: keep the textbook slope
            a = float(np.sum((d - DEFAULT_STEP[1] * A[:, 1]) * A[:, 0]) / np.sum(A[:, 0] ** 2))
            b = DEFAULT_STEP[1]
        return float(a), float(b)
    b = DEFAULT_STEP[1]
    a = float(np.sum((d - b * A[:, 1]) * A[:, 0]) / np.sum(A[:, 0] ** 2))
    return a, b


# ---------------- figures ----------------

def fig_labels(s, events, out):
    m = s["meta"]
    sig, axis = sd.swing_signal(s["x"], m["start"], m["end"])
    t = np.arange(len(sig)) / sd.RATE
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    axes[0].plot(t, sig * sd.GYRO_SCALE, color=BLUE, lw=1)
    for e in events:
        axes[0].axvline(e / sd.RATE, color=ORANGE, lw=0.8, alpha=0.7)
    axes[0].set_ylabel(f"swing gyro {'xyz'[axis]} (°/s)")
    axes[0].set_title(f"{s['name']}: counted {m['steps']} steps, {len(events)} marks (orange)", loc="left", color=INK)
    axes[1].plot(t, s["x"][:, 6], color=AQUA, lw=1)
    axes[1].set_ylabel("ultrasonic (m)")
    axes[1].set_xlabel("seconds")
    for ax in axes:
        ax.axvspan(0, m["start"] / sd.RATE, color=GRID, alpha=0.6, lw=0)
        ax.axvspan(m["end"] / sd.RATE, t[-1], color=GRID, alpha=0.6, lw=0)
        ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_scatter(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    true = np.array([r["true"] for r in rows])
    top = max(true.max(), max(r["nn"] for r in rows)) * 1.08 + 5
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--")
    ax.fill_between([0, top], [0, top * 0.95], [0, top * 1.05], color=GRID, alpha=0.6, lw=0, label="±5 %")
    ax.scatter(true, [r["peak"] for r in rows], s=28, color=ORANGE, marker="x", label="peak counter (no learning)")
    ax.errorbar(true, [r["nn"] for r in rows], yerr=[2 * r["sd"] for r in rows], fmt="o", ms=6, color=BLUE,
                ecolor=BLUE, elinewidth=1.2, capsize=3, label="neural net (±2 sd of the 5 nets)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("steps you counted")
    ax.set_ylabel("steps the counter said")
    ax.set_title("Each recording, counted by nets that never saw it", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_errors(rows, out):
    rows = sorted(rows, key=lambda r: (r["activity"], r["name"]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(rows))
    nn_err = [r["nn"] - r["true"] for r in rows]
    pk_err = [r["peak"] - r["true"] for r in rows]
    ax.bar(x - 0.2, nn_err, 0.38, color=BLUE, label="neural net")
    ax.bar(x + 0.2, pk_err, 0.38, color=ORANGE, label="peak counter")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, [f"{r['activity']}\n{r['true']}" for r in rows], fontsize=8)
    ax.set_ylabel("error (steps)")
    ax.set_title("Error on each held-out recording (label: activity and counted steps)", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ablation(results, out):
    names = list(results)
    vals = [results[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[BLUE if i == 0 else "#9ec5f4" for i in range(len(names))], height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, f"{v:.1f} %", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlabel("average error on held-out recordings (% of steps, lower is better)")
    ax.set_title("Which sensors matter", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_trace(s, per_hop, events, out):
    m = s["meta"]
    sig, _ = sd.swing_signal(s["x"], m["start"], m["end"])
    t = np.arange(len(sig)) / sd.RATE
    th = (m["start"] + sd.HOP * np.arange(len(per_hop) + 1)) / sd.RATE
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, height_ratios=[1, 1.3])
    axes[0].plot(t, sig * sd.GYRO_SCALE, color=MUTED, lw=1)
    axes[0].set_ylabel("swing gyro (°/s)")
    axes[0].set_title(f"{s['name']} (held out): counted {m['steps']}, net said {per_hop.sum():.1f}", loc="left", color=INK)
    true_cum = [np.sum(events < m["start"] + sd.HOP * k) for k in range(len(per_hop) + 1)]
    axes[1].step(th, true_cum, where="post", color=ORANGE, lw=2, label="your steps (marks)")
    axes[1].step(th, np.concatenate([[0], np.cumsum(per_hop)]), where="post", color=BLUE, lw=2, label="neural net")
    axes[1].set_ylabel("steps so far")
    axes[1].set_xlabel("seconds")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_curve(history, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for h in history:
        ax.plot(h, color=BLUE, lw=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch (one pass over all the training windows)")
    ax.set_ylabel("loss (mean squared error)")
    ax.set_title("Training: the 5 final nets learning", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_distance(items, out):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    true = [d["true"] for d in items]
    top = max(max(true), max(d["pred"] for d in items)) * 1.1
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--")
    ax.scatter(true, [d["pred"] for d in items], color=BLUE, s=36, zorder=3)
    for d in items:
        ax.annotate(d["activity"], (d["true"], d["pred"]), textcoords="offset points", xytext=(5, -10), fontsize=8, color=MUTED)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("measured distance (m)")
    ax.set_ylabel("estimated distance (m)")
    ax.set_title("Distance on held-out walks", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 nets per fold, no sensor test")
    ap.add_argument("--nets", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--height-in", type=float, default=66, help="your height in inches")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--workers", type=int, default=4, help="nets trained at the same time")
    args = ap.parse_args()
    if args.quick:
        args.nets = 2
    height = args.height_in * 0.0254
    figs = HERE / "figures"
    (figs / "labels").mkdir(parents=True, exist_ok=True)
    (HERE / "models").mkdir(exist_ok=True)

    sessions = sd.load_sessions(args.data)
    if len(sessions) < 3:
        sys.exit("Need at least 3 recordings with step counts in data/ (run record.py).")
    print(f"{len(sessions)} recordings:")
    marks = []
    for s in sessions:
        m = s["meta"]
        ev, info = sd.mark_steps(s["x"], m["start"], m["end"], m["steps"])
        marks.append(ev)
        fig_labels(s, ev, figs / "labels" / f"{s['name']}.png")
        warn = ""
        if info.get("weakest") is not None and info["weakest"] < 0.35:
            warn = "  <- weak peaks: check figures/labels (count may be off)"
        if s["gaps"]:
            warn += f"  <- {s['gaps']} gaps in the data"
        dist = f"{m['distance_m']:.1f} m" if m.get("distance_m") is not None else "-"
        print(f"  {s['name']:28s} {m['activity']:10s} {m['steps']:4d} steps  {dist:>8s}{warn}")

    # ---- leave-one-session-out ----
    print(f"\nLeave-one-session-out: {args.nets} nets per held-out recording...")
    t0 = time.time()
    n = len(sessions)
    others = lambda k: tuple(j for j in range(n) if j != k)
    jobs = [(others(k), None, seed, args.epochs) for k in range(n) for seed in range(args.nets)]
    fold_nets = [t[0] for t in sn.train_many(args.data, jobs, args.workers)]
    print(f"  trained {len(jobs)} nets in {time.time() - t0:.0f} s")
    rows, cv_hops = [], {}
    for k, s in enumerate(sessions):
        train = [x for j, x in enumerate(sessions) if j != k]
        nets = fold_nets[k * args.nets:(k + 1) * args.nets]
        mean, sdv, per_hop = sn.count_session(nets, s)
        cv_hops[s["name"]] = per_hop
        pk = peak_count(s, peak_threshold(train))
        rows.append({"name": s["name"], "activity": s["meta"]["activity"], "true": s["meta"]["steps"],
                     "nn": mean, "sd": sdv, "peak": pk})
        print(f"  {s['name']:28s} counted {s['meta']['steps']:4d}   net {mean:6.1f} ± {sdv:4.1f}   peak counter {pk:4d}")

    walking = [r for r in rows if r["true"] > 0]
    still = [r for r in rows if r["true"] == 0]
    rel = np.array([(r["nn"] - r["true"]) / r["true"] for r in walking]) if walking else np.array([0.0])
    rel_pk = np.array([(r["peak"] - r["true"]) / r["true"] for r in walking]) if walking else np.array([0.0])
    summary = {
        "recordings": len(rows), "with_steps": len(walking),
        "nn_mean_abs_error_pct": float(np.mean(np.abs(rel)) * 100),
        "nn_bias_pct": float(np.mean(rel) * 100), "nn_sd_pct": float(np.std(rel, ddof=1) * 100) if len(rel) > 1 else 0.0,
        "nn_worst_pct": float(np.max(np.abs(rel)) * 100),
        "peak_mean_abs_error_pct": float(np.mean(np.abs(rel_pk)) * 100),
        "false_steps_per_min_when_not_walking": float(np.mean([r["nn"] / (sessions[i]["meta"]["seconds"] / 60)
                                                               for i, r in enumerate(rows) if r["true"] == 0])) if still else None,
        "mean_ensemble_sd_steps": float(np.mean([r["sd"] for r in rows])),
    }
    lo = summary["nn_bias_pct"] - 1.96 * summary["nn_sd_pct"]
    hi = summary["nn_bias_pct"] + 1.96 * summary["nn_sd_pct"]
    summary["nn_95pct_range_pct"] = [lo, hi]
    print("\nHow good is it (held-out recordings):")
    print(f"  neural net:   average error {summary['nn_mean_abs_error_pct']:.1f} %   bias {summary['nn_bias_pct']:+.1f} %   "
          f"95 % of counts within {lo:+.1f} % to {hi:+.1f} %   worst {summary['nn_worst_pct']:.1f} %")
    print(f"  peak counter: average error {summary['peak_mean_abs_error_pct']:.1f} %")
    if still:
        print(f"  not walking:  {summary['false_steps_per_min_when_not_walking']:.1f} false steps per minute")

    # ---- which sensors matter (one net per fold per sensor set) ----
    ablation = {}
    if not args.quick:
        print("\nSensor test (1 net per held-out recording per sensor set)...")
        folds = [k for k in range(n) if sessions[k]["meta"]["steps"] > 0]
        jobs = [(others(k), mask, 100, args.epochs) for mask in MASKS.values() for k in folds]
        ab_nets = iter(t[0] for t in sn.train_many(args.data, jobs, args.workers))
        for name, mask in MASKS.items():
            errs = []
            for k in folds:
                s = sessions[k]
                mean, _, _ = sn.count_session([next(ab_nets)], s)
                errs.append(abs(mean - s["meta"]["steps"]) / s["meta"]["steps"] * 100)
            ablation[name] = float(np.mean(errs)) if errs else float("nan")
            print(f"  {name:26s} {ablation[name]:5.1f} % average error")
        fig_ablation(ablation, figs / "sensors.png")
        summary["sensor_test_pct"] = ablation

    # ---- distance ----
    measured = [(s, cv_hops[s["name"]]) for s in sessions
                if s["meta"].get("distance_m") not in (None, 0) and s["meta"]["steps"] > 0]
    dist_rows = []
    for k, (s, hops) in enumerate(measured):                    # leave-one-walk-out calibration
        others = [(h, x["meta"]["distance_m"]) for j, (x, h) in enumerate(measured) if j != k]
        a, b = fit_step_length(others, height)
        dist_rows.append({"name": s["name"], "activity": s["meta"]["activity"], "true": s["meta"]["distance_m"],
                          "pred": distance(hops, height, a, b)})
    a, b = fit_step_length([(h, s["meta"]["distance_m"]) for s, h in measured], height)
    if dist_rows:
        drel = np.array([(d["pred"] - d["true"]) / d["true"] for d in dist_rows])
        summary["distance_mean_abs_error_pct"] = float(np.mean(np.abs(drel)) * 100)
        summary["distance_bias_pct"] = float(np.mean(drel) * 100)
        summary["distance_sd_pct"] = float(np.std(drel, ddof=1) * 100) if len(drel) > 1 else None
        fig_distance(dist_rows, figs / "distance.png")
        print(f"\nDistance on held-out measured walks: average error {summary['distance_mean_abs_error_pct']:.1f} %, "
              f"bias {summary['distance_bias_pct']:+.1f} %")
        for d in dist_rows:
            print(f"  {d['name']:28s} measured {d['true']:6.1f} m   estimated {d['pred']:6.1f} m")
    else:
        print("\nNo measured-distance walks yet: distance uses the textbook step length.")
    in_place = [s for s in sessions if s["meta"].get("distance_m") == 0 and s["meta"]["steps"] > 0]
    for s in in_place:
        print(f"  limitation: {s['name']} was in place (0 m) but the estimate says "
              f"{distance(cv_hops[s['name']], height, a, b):.1f} m")
    summary["step_length"] = {"a": a, "b": b, "height_m": height,
                              "meaning": "step length (m) = height x (a + b x steps per second)"}
    summary["distance"] = dist_rows

    # ---- final ensemble on everything ----
    print(f"\nTraining the final {args.nets} nets on all recordings...")
    trained = sn.train_many(args.data, [(tuple(range(n)), None, seed, args.epochs) for seed in range(args.nets)],
                            args.workers)
    nets = [t[0] for t in trained]
    sn.save(nets, HERE / "models" / "stepnet.pt", extra=summary)
    (HERE / "models" / "summary.json").write_text(json.dumps(
        {"summary": summary, "cv": rows, "history": [t[1] for t in trained],
         "cv_hops": {k: v.tolist() for k, v in cv_hops.items()}}, indent=2))
    print(f"Saved models/stepnet.pt and models/summary.json ({sum(p.numel() for p in nets[0].parameters())} weights per net)")

    fig_scatter(rows, figs / "cv_counts.png")
    fig_errors(rows, figs / "cv_errors.png")
    fig_curve([t[1] for t in trained], figs / "training.png")
    best = max(walking, key=lambda r: r["true"]) if walking else rows[0]
    k = [s["name"] for s in sessions].index(best["name"])
    fig_trace(sessions[k], cv_hops[best["name"]], marks[k], figs / "example_count.png")
    print("Figures in figures/ (labels/ has one per recording: check the orange marks)")


if __name__ == "__main__":
    main()
