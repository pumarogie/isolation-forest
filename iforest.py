"""Isolation Forest from scratch (Liu, Ting, Zhou 2008), NumPy only."""
import math

import numpy as np

EULER = 0.5772156649


def c(n):
    """Expected path length for a group of n points, used to normalize depths.

    Equals the average path length of an unsuccessful search in a binary
    search tree built from n points: 2*H(n-1) - 2*(n-1)/n, where H(i) is the
    harmonic number, approximated by ln(i) + Euler's constant.

    Used twice:
    - to turn the average depth into a score between 0 and 1 (divide by c(psi));
    - at a leaf that still holds several points (the tree stopped at its
      height limit), to estimate how many more cuts they would have needed.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (math.log(n - 1) + EULER) - 2 * (n - 1) / n


def build_tree(X, depth, limit, rng):
    """Build one isolation tree (iTree) by cutting the data randomly.

    At each node: pick a random feature, pick a random split value between that
    feature's min and max, send points below it left and the rest right, then
    repeat on both halves. Stops when a node holds at most one point, all its
    points are identical, or the depth reaches `limit`.

    Returns nested dicts: a leaf is {"size": n}, an inner node is
    {"feature", "split", "left", "right"}.
    """
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
    """Number of cuts needed to isolate point x in one tree.

    Walks x down the tree, counting edges, until a leaf. If that leaf still
    holds several points, adds c(size) as an estimate of the remaining depth.
    Anomalies get isolated early, so they have short paths.
    """
    while "size" not in node:
        node = node["left"] if x[node["feature"]] < node["split"] else node["right"]
        depth += 1
    return depth + c(node["size"])


class IsolationForest:
    """An ensemble of isolation trees; averages path lengths into one score."""

    def __init__(self, n_trees=100, sample_size=256, contamination=0.05, seed=0):
        """Store the settings.

        n_trees: number of trees; more trees give a more stable average.
        sample_size: points per tree (psi); small subsamples isolate anomalies
            better and keep trees cheap. The paper recommends 256.
        contamination: expected share of anomalies, used to set the threshold.
        seed: makes the random cuts reproducible.
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        """Grow n_trees trees, each on its own random subsample of X.

        Tree depth is capped at ceil(log2(psi)), about the average depth of a
        tree over psi points: normal points would go deeper, but their exact
        depth doesn't matter, only that anomalies come out shorter.
        Then sets the threshold so that the top `contamination` share of the
        training scores counts as anomalies.
        """
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
        """Anomaly score s = 2^(-E[h(x)] / c(psi)) for each row of X.

        E[h(x)] is the path length averaged over all trees. Short paths push s
        toward 1 (anomaly); paths around the typical length c(psi) give about
        0.5; long paths give values well below 0.5 (normal).
        """
        X = np.asarray(X, dtype=float)
        # ponytail: Python loop per point per tree, vectorize traversal if n gets large
        mean_h = np.array([np.mean([path_length(x, t) for t in self.trees]) for x in X])
        return 2.0 ** (-mean_h / c(self.psi))

    def predict(self, X):
        """Label each row: 1 = anomaly (score above the fitted threshold), 0 = normal."""
        return (self.score_samples(X) > self.threshold).astype(int)


def make_data(seed=42):
    """Toy 2D data: two Gaussian clusters of 150 points plus 20 uniform outliers."""
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
