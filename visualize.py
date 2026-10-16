import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest as SkIF

from iforest import IsolationForest, build_tree, make_data

CONTAMINATION = 0.06
X = make_data()
model = IsolationForest(contamination=CONTAMINATION, seed=0).fit(X)
sk = SkIF(n_estimators=100, max_samples=256, contamination=CONTAMINATION, random_state=0).fit(X)

xs = np.linspace(-7, 7, 80)
gx, gy = np.meshgrid(xs, xs)
grid = np.c_[gx.ravel(), gy.ravel()]

fig, ax = plt.subplots(2, 2, figsize=(12, 11))


def score_map(a, scores, flags, title):
    """Draw anomaly scores over the grid as a filled contour, then the data on top.

    scores: one score per grid point; flags: 1/0 anomaly label per data point,
    so flagged points show in red and normal ones in white.
    """
    cs = a.contourf(gx, gy, scores.reshape(gx.shape), levels=20, cmap="magma_r")
    fig.colorbar(cs, ax=a, label="anomaly score")
    a.scatter(*X[flags == 0].T, s=10, c="white", edgecolors="k", linewidths=0.3, label="normal")
    a.scatter(*X[flags == 1].T, s=40, c="red", edgecolors="k", label="anomaly")
    a.set_title(title)
    a.legend(loc="upper left")


score_map(ax[0, 0], model.score_samples(grid), model.predict(X), "From scratch: score map")
# sklearn score_samples is the negated paper score
score_map(ax[0, 1], -sk.score_samples(grid), (sk.predict(X) == -1).astype(int), "sklearn: score map")

s = model.score_samples(X)
ax[1, 0].hist(s, bins=40, color="steelblue")
ax[1, 0].axvline(model.threshold, color="red", ls="--", label=f"threshold {model.threshold:.3f}")
ax[1, 0].set(title="Score distribution (from scratch)", xlabel="anomaly score", ylabel="count")
ax[1, 0].legend()


def draw_splits(a, node, box, depth=0):
    """Draw every cut of one iTree as a line inside the region it splits.

    box = [xmin, xmax, ymin, ymax] is the region the current node covers. A cut
    on feature 0 is a vertical line, on feature 1 a horizontal one; the region
    is then split in two and each child is drawn inside its half. Early cuts
    are dark and thick, deeper cuts lighter and thinner.
    """
    if "size" in node:
        return
    f, v = node["feature"], node["split"]
    color = plt.cm.viridis(min(depth / 8, 1))
    if f == 0:
        a.plot([v, v], box[2:], color=color, lw=1.5 - depth * 0.12)
        left, right = [box[0], v, *box[2:]], [v, box[1], *box[2:]]
    else:
        a.plot(box[:2], [v, v], color=color, lw=1.5 - depth * 0.12)
        left, right = [*box[:2], box[2], v], [*box[:2], v, box[3]]
    draw_splits(a, node["left"], left, depth + 1)
    draw_splits(a, node["right"], right, depth + 1)


tree = build_tree(X, 0, 8, np.random.default_rng(1))
lo, hi = X.min(axis=0), X.max(axis=0)
draw_splits(ax[1, 1], tree, [lo[0], hi[0], lo[1], hi[1]])
ax[1, 1].scatter(*X.T, s=8, c="k")
ax[1, 1].set_title("One iTree's partitions (dark = early cut)\noutliers sit alone in big cells")

plt.tight_layout()
plt.savefig("isolation_forest.png", dpi=120)
print("saved isolation_forest.png")
if "--show" in sys.argv:
    plt.show()
