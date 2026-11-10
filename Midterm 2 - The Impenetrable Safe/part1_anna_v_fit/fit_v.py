r"""
Fit the V data from Anna's Code with a tiny neural network, trained by hand-written
gradient descent (no PyTorch autograd) so every line matches the math.

Network: depth = 2 layers (1 hidden + 1 output), hidden width = HIDDEN neurons.
With HIDDEN = 2:

    x --> h1 = ReLU(w1*x + b1) --\
      \-> h2 = ReLU(w2*x + b2) ---+--> y_hat = v1*h1 + v2*h2 + c     (no activation)

Two neurons CAN make a perfect V (h1 = ReLU(2x-120), h2 = ReLU(-2x+120), y = h1 + h2),
but gradient descent often gets stuck in a "half V" local minimum. Testing 50 random
starts: width 2 fit the V 8/50 times, width 3 25/50, width 4 41/50, width 6 46/50.

Loss: mean squared error  L = (1/N) * sum( (y_hat - y)^2 )

Run:  ../.venv/bin/python fit_v.py
Saves: v_fit.png (data + fit) and v_loss.png (loss vs. epoch)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Load the data (columns: time, x, y)
# ---------------------------------------------------------------------------
data = np.loadtxt(HERE / "v_data.tsv", delimiter="\t", skiprows=1)
x_raw, y_raw = data[:, 1], data[:, 2]
N = len(x_raw)

# Normalize to z-scores (mean 0, std 1). Without this the gradient for the
# weights is ~60x bigger than for the biases (because x is ~60), and one
# learning rate can't suit both. Anna's Code does the same thing.
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
x = (x_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std

# ---------------------------------------------------------------------------
# 2. Initialize weights (small random numbers so the two neurons start different)
# ---------------------------------------------------------------------------
HIDDEN = 4                       # width of the hidden layer (try 2, 3, 4, 6)
rng = np.random.default_rng(1)
w = rng.uniform(-1, 1, size=HIDDEN)   # hidden weights
# Hidden biases: put each neuron's bend (where w*x + b = 0) at a random data point.
# If a bend starts outside the data, that ReLU is "off" for every point, its
# gradient is always 0, and it never learns (a "dead neuron").
b = -w * rng.choice(x, size=HIDDEN)   # hidden biases
v = rng.uniform(-1, 1, size=HIDDEN) / np.sqrt(HIDDEN)   # output weights
c = 0.0                          # output bias

def relu(z):
    return np.maximum(0, z)

def forward(x):
    z = np.outer(x, w) + b       # pre-activations, shape (N, HIDDEN)
    h = relu(z)                  # hidden outputs
    y_hat = h @ v + c            # network output
    return z, h, y_hat

# ---------------------------------------------------------------------------
# 3. Gradient descent
# ---------------------------------------------------------------------------
lr = 0.1
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward pass
    z, h, y_hat = forward(x)
    err = y_hat - y                       # (y_hat - y) for every point
    loss = np.mean(err ** 2)
    losses.append(loss)

    # Backward pass (chain rule)
    dL_dyhat = 2 * err / N                # dL/dy_hat for each point

    # Output layer: y_hat = v . h + c
    dL_dv = h.T @ dL_dyhat                # sum over points of dL/dy_hat * h
    dL_dc = dL_dyhat.sum()

    # Hidden layer: h = ReLU(z), z = w*x + b
    dL_dh = np.outer(dL_dyhat, v)         # error sent back through v
    dL_dz = dL_dh * (z > 0)               # ReLU'(z) = 1 if z > 0 else 0  (the "gate")
    dL_dw = x @ dL_dz                     # dz/dw = x
    dL_db = dL_dz.sum(axis=0)             # dz/db = 1

    # Update: step downhill
    w -= lr * dL_dw
    b -= lr * dL_db
    v -= lr * dL_dv
    c -= lr * dL_dc

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d}   loss {loss:.4f}")

# ---------------------------------------------------------------------------
# 4. Convert weights back to raw units (so they're readable)
#    z = w*(x_raw - x_mean)/x_std + b  ->  w_raw = w/x_std,  b_raw = b - w*x_mean/x_std
#    y_raw = y_std*y_hat + y_mean      ->  v_raw = y_std*v,  c_raw = y_std*c + y_mean
# ---------------------------------------------------------------------------
w_raw = w / x_std
b_raw = b - w * x_mean / x_std
v_raw = v * y_std
c_raw = c * y_std + y_mean
print("\nLearned weights (raw units):")
for i in range(HIDDEN):
    print(f"  h{i+1} = ReLU({w_raw[i]:+.3f}*x {b_raw[i]:+.2f})   bend at x = {-b_raw[i]/w_raw[i]:.1f}")
print("  y = " + " ".join(f"{v_raw[i]:+.3f}*h{i+1}" for i in range(HIDDEN)) + f" {c_raw:+.2f}")
print(f"Final loss (normalized units): {losses[-1]:.4f}")

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
x_line = np.linspace(x_raw.min(), x_raw.max(), 300)
_, _, y_line = forward((x_line - x_mean) / x_std)
y_line = y_line * y_std + y_mean

plt.figure(figsize=(7, 4.5))
plt.scatter(x_raw, y_raw, s=14, alpha=0.7, label="V data (Anna's Code)")
plt.plot(x_line, y_line, "r-", lw=2.5, label="network fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"V data fit: 1 input -> {HIDDEN} ReLU neurons -> 1 output")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(HERE / "v_fit.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.plot(losses, lw=2)
plt.xlabel("epoch")
plt.ylabel("loss (mean squared error, normalized)")
plt.yscale("log")
plt.title("Loss vs. epoch")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(HERE / "v_loss.png", dpi=150)
print("\nSaved v_fit.png and v_loss.png")
