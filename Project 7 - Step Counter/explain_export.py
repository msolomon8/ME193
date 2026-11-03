"""Put one real window of data, the trained net and real gradients into explain.html (runs on the laptop).

Run after train.py:  python explain_export.py
Then open explain.html: an animation that follows that window through the network,
the loss, back-propagation and training, and finally counts a whole recording.
Everything it shows is a real number from your data and your nets.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steps_data as sd
import step_net as sn

torch.set_num_threads(1)
r = lambda a, n=3: np.round(np.asarray(a, dtype=float), n).tolist()


def main(data=None):
    sessions = sd.load_sessions(data)
    nets, info = sn.load(HERE / "models" / "stepnet.pt")
    summary = json.loads((HERE / "models" / "summary.json").read_text())
    walk = max((s for s in sessions if s["meta"]["steps"] > 0), key=lambda s: s["meta"]["steps"])
    m = walk["meta"]
    events, _ = sd.mark_steps(walk["x"], m["start"], m["end"], m["steps"])

    # a window in the middle of the walk whose middle 0.5 s holds exactly one step
    starts, w = sd.windows(walk["x"], m["start"], m["end"])
    y = sd.targets(events, starts, m["start"], m["end"], m["steps"])
    mid = len(starts) // 2
    k = min((i for i in range(len(starts)) if y[i] == 1), key=lambda i: abs(i - mid), default=mid)
    i0 = int(starts[k])
    x = walk["x"][i0:i0 + sd.WINDOW].T.copy()
    target = float(y[k])

    # forward pass through net 0, keeping every stage
    net = nets[0]
    xt = torch.tensor(x[None])
    with torch.no_grad():
        z1 = net.conv1(xt)                          # before ReLU
        keep = {}
        out = net(xt, keep=keep)
        zfc = net.fc(keep["a2"].flatten(1))
    out_v = float(out)

    # one backward pass: the real gradients for this window
    net.zero_grad()
    out_g = net(xt)
    loss = (out_g - target) ** 2
    loss.sum().backward()
    grads = {n: p.grad.detach().numpy() for n, p in net.named_parameters()}

    # training from random weights: how this window's answer and the loss change, epoch by epoch
    X, Y, _ = sd.dataset(sessions)
    torch.manual_seed(7)
    fresh = sn.StepNet()
    opt = torch.optim.Adam(fresh.parameters(), lr=3e-3)
    rng = np.random.default_rng(7)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    learn = []
    for epoch in range(25):
        with torch.no_grad():
            learn.append({"epoch": epoch, "out": float(fresh(xt)), "loss": float(((fresh(Xt) - Yt) ** 2).mean())})
        for b in range(0, len(Yt), 128):
            idx = torch.tensor(rng.permutation(len(Yt))[:128])
            loss_b = ((fresh(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss_b.backward()
            opt.step()

    # counting the whole recording with all nets
    _, _, per_hop = sn.count_session(nets, walk)
    true_cum = [int(np.sum(events < m["start"] + sd.HOP * j)) for j in range(len(per_hop) + 1)]
    cv = next((c for c in summary["cv"] if c["name"] == walk["name"]), None)

    show_from = max(0, i0 - 2 * sd.RATE)
    data = {
        "channels": sd.CHANNELS, "rate": sd.RATE, "window": sd.WINDOW, "hop": sd.HOP, "center": sd.CENTER,
        "session": walk["name"], "counted": m["steps"],
        "context": r(walk["x"][show_from:i0 + sd.WINDOW + 2 * sd.RATE].T), "context_start": i0 - show_from,
        "context_events": r([e - show_from for e in events if show_from <= e < i0 + sd.WINDOW + 2 * sd.RATE], 1),
        "x": r(x), "events": r([e - i0 for e in events if i0 <= e < i0 + sd.WINDOW], 1), "target": target,
        "conv1_w": r(net.conv1.weight.detach().numpy()), "conv1_b": r(net.conv1.bias.detach().numpy()),
        "z1": r(z1[0].numpy()), "a1": r(keep["a1"][0].numpy()), "a2": r(keep["a2"][0].numpy()),
        "fc_w": r(net.fc.weight.detach().numpy()[0], 4), "fc_b": float(net.fc.bias.detach()),
        "z": float(zfc), "out": out_v, "loss": float(loss.detach()), "lr": 3e-3, "windows": int(len(Y)),
        "grad_fc": r(grads["fc.weight"][0], 5), "grad_conv2": r(np.abs(grads["conv2.weight"]).sum(axis=(1, 2)), 5),
        "grad_conv1": r(grads["conv1.weight"], 5),
        "params": sum(p.numel() for p in net.parameters()),
        "learn": learn, "history": summary.get("history", []),
        "per_hop": r(per_hop), "true_cum": true_cum,
        "cv": cv, "summary": {k2: v for k2, v in summary["summary"].items() if not isinstance(v, (list, dict))},
    }
    page = HERE / "explain.html"
    text = page.read_text()
    block = "/*DATA-BEGIN*/" + json.dumps(data, separators=(",", ":")) + "/*DATA-END*/"
    page.write_text(re.sub(r"/\*DATA-BEGIN\*/.*?/\*DATA-END\*/", lambda _: block, text, flags=re.S))
    print(f"explain.html now shows {walk['name']} (window at {i0 / sd.RATE:.1f} s, target {target:g}, "
          f"net said {out_v:.2f}). Open it in a browser.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
