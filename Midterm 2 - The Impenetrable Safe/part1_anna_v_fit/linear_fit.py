"""
Straight-line fit (y = m*x + b) to the V data with gradient descent -
the SAME math as the Excel sheet, just run for many more epochs.

Follows the professor's 6 steps. Like his Python, the "2" in the derivative is
left out (it just gets absorbed into the learning rate).

Run:  ../.venv/bin/python linear_fit.py
Saves: linear_fit.png (data + line) and linear_loss.png (loss vs. epoch)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# 1. Data arrays (columns in the file: time, x, y)
data = np.loadtxt(HERE / "v_data.tsv", delimiter="\t", skiprows=1)
X, Y = data[:, 1], data[:, 2]

# 2. Hyperparameters
learning_rate = 0.0001       # same as the Excel sheet
epochs = 1000

# 3. Initial guess
m, b = 0.0, 0.0

losses = []
for epoch in range(epochs):
    # 4. Forward pass + gradients (same rows as Excel: guess, db, dm)
    guess = m * X + b
    db = guess - Y           # Excel "db" row
    dm = db * X              # Excel "dm" row
    losses.append(np.mean(db ** 2))

    # 5. Step downhill
    m -= learning_rate * np.mean(dm)
    b -= learning_rate * np.mean(db)

    if epoch <= 5:
        print(f"epoch {epoch}:  m = {m:.5f}   b = {b:.5f}   (loss before step {losses[-1]:.1f})")

print(f"\nAfter {epochs} epochs: y = {m:.4f}x + {b:.4f}   loss = {losses[-1]:.1f}")
best_m, best_b = np.polyfit(X, Y, 1)
print(f"Exact best line:      y = {best_m:.4f}x + {best_b:.4f}   loss = {np.mean((best_m*X+best_b-Y)**2):.1f}")
print("b is still far from the best value: with raw x, b learns very slowly (x is ~60, so m dominates).")

# Same thing with centered x (the professor's trick) - converges fast
Xc = X - X.mean()
mc, bc, lc = 0.0, 0.0, []
for epoch in range(epochs):
    err = mc * Xc + bc - Y
    lc.append(np.mean(err ** 2))
    mc -= 0.001 * np.mean(err * Xc)
    bc -= 0.1 * np.mean(err)   # could use one rate; b converges in a few steps once x is centered
print(f"Centered x:           y = {mc:.4f}x + {bc - mc*X.mean():.4f}   loss = {lc[-1]:.1f}")

# Plots
xs = np.linspace(X.min(), X.max(), 2)
plt.figure(figsize=(7, 4.5))
plt.scatter(X, Y, s=14, alpha=0.7, label="V data (Anna's Code)")
plt.plot(xs, m * xs + b, "r-", lw=2.5, label=f"gradient descent, raw x: y = {m:.3f}x + {b:.2f}")
plt.plot(xs, mc * xs + (bc - mc * X.mean()), "g--", lw=2, label="gradient descent, centered x")
plt.xlabel("x"); plt.ylabel("y")
plt.title("A straight line can't fit a V")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(HERE / "linear_fit.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.plot(losses, lw=2, label="raw x (lr = 0.0001)")
plt.plot(lc, lw=2, ls="--", label="centered x")
plt.xlabel("epoch"); plt.ylabel("loss (mean squared error)")
plt.title("Loss vs. epoch: straight-line fit")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(HERE / "linear_loss.png", dpi=150)
print("\nSaved linear_fit.png and linear_loss.png")
