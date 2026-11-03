"""The step-counting neural net (PyTorch) and how a stream of samples becomes a count.

StepNet looks at 2 s of all 7 sensor channels and answers one number: how many steps
happened in the middle 0.5 s of that window. Sliding the window forward 0.5 s at a time
and adding the answers counts every step exactly once.

    input (7 channels x 100 samples)
      -> conv1: 8 filters, each 9 samples (0.18 s) long, look for short shapes: a heel strike,
                a swing, the other leg passing the ultrasonic sensor ...      -> ReLU -> max-pool /2
      -> conv2: 8 filters combine those shapes over a longer stretch           -> ReLU -> max-pool /2
      -> 8 x 25 numbers, flattened -> one linear layer -> softplus (never negative) = steps
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import steps_data as sd

DEAD = 0.1    # a window answer below this is counted as 0 (standing still never adds up)


class StepNet(nn.Module):
    def __init__(self, channels=len(sd.CHANNELS), filters=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, filters, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(filters * (sd.WINDOW // 4), 1)

    def forward(self, x, keep=None):
        a1 = self.pool(torch.relu(self.conv1(x)))
        a2 = self.pool(torch.relu(self.conv2(a1)))
        out = nn.functional.softplus(self.fc(a2.flatten(1))).squeeze(-1)
        if keep is not None:          # for the animation: keep the in-between activations
            keep.update(a1=a1, a2=a2)
        return out


def train_net(X, y, mask=None, epochs=40, seed=0, lr=3e-3, batch=128, verbose=False):
    """Train one StepNet. mask (bool per channel) zeroes channels out, for the 'which sensors matter' test."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = StepNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    m = torch.ones(len(sd.CHANNELS)) if mask is None else torch.tensor(mask, dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        order = torch.tensor(rng.permutation(len(yt)))
        total = 0.0
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            xb = Xt[idx]
            # augmentation: a little gain change and noise, so it doesn't memorise exact amplitudes
            xb = xb * torch.empty(len(idx), 1, 1).uniform_(0.9, 1.1) + 0.02 * torch.randn_like(xb)
            # sometimes the ultrasonic sensor sees nothing: teach the net to cope
            lost = torch.rand(len(idx)) < 0.1
            xb[lost, 6] = sd.MAX_DIST
            xb = xb * m[None, :, None]
            loss = ((net(xb) - yt[idx]) ** 2).mean()        # mean squared error
            opt.zero_grad()
            loss.backward()                                 # back-propagation: d(loss)/d(every weight)
            opt.step()                                      # nudge every weight a little downhill
            total += loss.item() * len(idx)
        history.append(total / len(yt))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss {history[-1]:.4f}")
    net.eval()
    net.mask = m
    return net, history


@torch.no_grad()
def window_counts(nets, X):
    """(n_nets, n_windows) answers, with the dead band applied."""
    Xt = torch.tensor(X)
    out = np.stack([net(Xt * net.mask[None, :, None]).numpy() for net in nets])
    out[out < DEAD] = 0
    return out


def count_session(nets, session):
    """Count the steps between the start and end sounds. Returns (mean, std over nets, per-hop counts)."""
    m = session["meta"]
    starts, w = sd.windows(session["x"], m["start"], m["end"])
    X = np.stack([session["x"][i:i + sd.WINDOW].T for i in starts])
    per = window_counts(nets, X) * w[None, :]
    totals = per.sum(axis=1)
    return float(totals.mean()), float(totals.std()), per.mean(axis=0)


# ---- training many nets at once (one process per CPU core) ----
_SESSIONS = None


def _init_worker(data_folder):
    global _SESSIONS
    torch.set_num_threads(1)         # tiny nets: one thread each, several processes, is fastest
    _SESSIONS = sd.load_sessions(data_folder)


def _job(job):
    train_idx, mask, seed, epochs = job
    X, y, _ = sd.dataset([_SESSIONS[i] for i in train_idx])
    net, history = train_net(X, y, mask=mask, epochs=epochs, seed=seed)
    return net.state_dict(), net.mask.tolist(), history


def train_many(data_folder, jobs, workers=4):
    """jobs: list of (training session indices, channel mask or None, seed, epochs).
    Returns a list of (net, history) in the same order."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(data_folder),)) as pool:
        out = []
        for state, mask, history in pool.map(_job, jobs):
            net = StepNet()
            net.load_state_dict(state)
            net.eval()
            net.mask = torch.tensor(mask)
            out.append((net, history))
    return out


def save(nets, path, extra=None):
    torch.save({"state_dicts": [n.state_dict() for n in nets], "masks": [n.mask.tolist() for n in nets],
                "extra": extra or {}}, path)


def load(path):
    d = torch.load(path, weights_only=False)
    nets = []
    for sdict, m in zip(d["state_dicts"], d["masks"]):
        n = StepNet()
        n.load_state_dict(sdict)
        n.eval()
        n.mask = torch.tensor(m)
        nets.append(n)
    return nets, d["extra"]
