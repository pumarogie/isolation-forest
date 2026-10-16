"""Isolation Forest from scratch (Liu, Ting, Zhou 2008), NumPy only."""
import math

import numpy as np

EULER = 0.5772156649


def c(n):
    """Average path length of an unsuccessful BST search over n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (math.log(n - 1) + EULER) - 2 * (n - 1) / n


def build_tree(X, depth, limit, rng):
    """Leaf: {"size"}. Node: {"feature", "split", "left", "right"}."""
    n = len(X)
    if depth >= limit or n <= 1:
        return {"size": n}
    lo, hi = X.min(axis=0), X.max(axis=0)
    candidates = np.flatnonzero(hi > lo)
    if len(candidates) == 0:  # all points identical
        return {"size": n}
    f = rng.choice(candidates)
    split = rng.uniform(lo[f], hi[f])
    mask = X[:, f] < split
    return {
        "feature": f,
        "split": split,
        "left": build_tree(X[mask], depth + 1, limit, rng),
        "right": build_tree(X[~mask], depth + 1, limit, rng),
    }


def path_length(x, node, depth=0):
    while "size" not in node:
        node = node["left"] if x[node["feature"]] < node["split"] else node["right"]
        depth += 1
    return depth + c(node["size"])


class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, contamination=0.05, seed=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.psi = min(self.sample_size, len(X))
        limit = math.ceil(math.log2(self.psi))
        self.trees = [
            build_tree(X[self.rng.choice(len(X), self.psi, replace=False)], 0, limit, self.rng)
            for _ in range(self.n_trees)
        ]
        self.threshold = np.quantile(self.score_samples(X), 1 - self.contamination)
        return self

    def score_samples(self, X):
        """Anomaly score in (0, 1]; near 1 = anomaly, well below 0.5 = normal."""
        X = np.asarray(X, dtype=float)
        # ponytail: Python loop per point per tree, vectorize traversal if n gets large
        mean_h = np.array([np.mean([path_length(x, t) for t in self.trees]) for x in X])
        return 2.0 ** (-mean_h / c(self.psi))

    def predict(self, X):
        """1 = anomaly, 0 = normal."""
        return (self.score_samples(X) > self.threshold).astype(int)


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal([-2, -2], 0.6, size=(150, 2))
    b = rng.normal([2, 2], 0.6, size=(150, 2))
    outliers = rng.uniform(-6, 6, size=(20, 2))
    return np.vstack([a, b, outliers])


if __name__ == "__main__":
    from scipy.stats import spearmanr
    from sklearn.ensemble import IsolationForest as SkIF

    X = make_data()
    model = IsolationForest(contamination=0.06, seed=0).fit(X)
    s = model.score_samples(X)

    far, center = model.score_samples([[6, -6], [-2, -2]])
    assert far > 0.6, far
    assert center < 0.5, center

    flagged = model.predict(X).mean()
    assert abs(flagged - 0.06) < 0.02, flagged

    sk = SkIF(n_estimators=100, max_samples=256, random_state=0).fit(X)
    rho = spearmanr(s, -sk.score_samples(X)).statistic
    assert rho > 0.9, rho

    print(f"ok  far={far:.3f} center={center:.3f} flagged={flagged:.3f} spearman_vs_sklearn={rho:.3f}")
