"""A small CART decision tree (Gini impurity), written with numpy.

No neural nets and no scikit-learn needed. The tree is stored as parallel lists
(feature, threshold, left, right, label, counts), the same format that
gait_features.predict() walks on the UNO Q.
"""
import numpy as np


def gini(counts):
    n = counts.sum()
    return 0.0 if n == 0 else 1.0 - ((counts / n) ** 2).sum()


def fit(X, y, names, n_classes, max_depth=4, min_leaf=5):
    """X: (n, d) features, y: (n,) class indices, names: d feature names."""
    tree = {"feature": [], "threshold": [], "left": [], "right": [], "label": [], "counts": [], "gain": []}

    def new_node(idx):
        counts = np.bincount(y[idx], minlength=n_classes)
        tree["feature"].append(None)
        tree["threshold"].append(None)
        tree["left"].append(-1)
        tree["right"].append(-1)
        tree["label"].append(int(counts.argmax()))
        tree["counts"].append(counts.tolist())
        tree["gain"].append(0.0)
        return len(tree["left"]) - 1

    def best_split(idx):
        parent = np.bincount(y[idx], minlength=n_classes)
        best = (0.0, None, None)                      # (impurity drop, feature, threshold)
        g0 = gini(parent)
        n = len(idx)
        for j in range(X.shape[1]):
            order = idx[np.argsort(X[idx, j], kind="stable")]
            xs = X[order, j]
            onehot = np.eye(n_classes, dtype=int)[y[order]]
            left = onehot.cumsum(axis=0)              # class counts left of each cut
            for i in range(min_leaf - 1, n - min_leaf):
                if xs[i] == xs[i + 1]:
                    continue
                lc, rc = left[i], parent - left[i]
                drop = g0 - ((i + 1) * gini(lc) + (n - i - 1) * gini(rc)) / n
                if drop > best[0] + 1e-12:
                    best = (drop, j, (xs[i] + xs[i + 1]) / 2)
        return best

    def grow(idx, depth):
        node = new_node(idx)
        if depth >= max_depth or len(set(y[idx])) == 1:
            return node
        drop, j, thr = best_split(idx)
        if j is None:
            return node
        tree["feature"][node] = names[j]
        tree["threshold"][node] = float(thr)
        tree["gain"][node] = float(drop * len(idx))
        go_left = X[idx, j] <= thr
        tree["left"][node] = grow(idx[go_left], depth + 1)
        tree["right"][node] = grow(idx[~go_left], depth + 1)
        return node

    grow(np.arange(len(y)), 0)
    return tree


def predict(tree, X, names):
    col = {n: i for i, n in enumerate(names)}
    out = []
    for row in X:
        n = 0
        while tree["left"][n] != -1:
            n = tree["left"][n] if row[col[tree["feature"][n]]] <= tree["threshold"][n] else tree["right"][n]
        out.append(tree["label"][n])
    return np.array(out)


def importance(tree):
    """Total impurity drop each feature is responsible for, largest first."""
    imp = {}
    for f, g in zip(tree["feature"], tree["gain"]):
        if f is not None:
            imp[f] = imp.get(f, 0.0) + g
    total = sum(imp.values()) or 1.0
    return sorted(((f, g / total) for f, g in imp.items()), key=lambda t: -t[1])
