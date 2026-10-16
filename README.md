# Isolation Forest from scratch

- `uv run python iforest.py` — self-check (asserts + rank correlation vs sklearn)
- `uv run python visualize.py [--show]` — writes `isolation_forest.png`

## Tradeoffs

### Algorithm
- **+ Fast, scales linearly.** Each tree is built on only psi=256 points with depth at most log2(psi), so training cost doesn't grow with dataset size. No distance or density computation (unlike kNN/LOF, which are O(n²)).
- **+ Few assumptions.** No distribution, no distance metric, no feature scaling needed: cuts happen between each feature's own min and max.
- **- Axis-parallel cuts.** Boxy score maps and "ghost" bands: lighter cross-shaped strips that run from the clusters along x and y (visible in the plot around x≈±2 and y≈±2). Points there score more normal than they should. Fix: Extended Isolation Forest, which cuts with random hyperplanes.
- **- Global, not local.** It finds points far from everything. Weak on local anomalies: a point just outside a dense cluster, or a sparse but normal cluster sitting next to a dense one.
- **- Clustered anomalies hide each other (masking).** Many anomalies close together need more cuts to isolate. Subsampling (small psi) helps, which is why 256 works well.
- **- High dimensions with irrelevant features.** Each cut picks a feature at random, so noise features waste cuts and dilute the signal.
- **- Threshold needs a guess.** `contamination` must be set by hand, and the quantile threshold always flags that share, even if the data has no anomalies.
- **- Random.** Scores change slightly with the seed. More trees give more stable scores but slower runs.

### This implementation
- **From scratch vs sklearn:** easy to read and close to the paper, but slower: about 3x on the 320-point demo and about 19x on 10,000 points (measured on one machine). Scoring is a Python loop per point per tree; sklearn uses compiled code and can run on several cores (`n_jobs`).
- **Trees as nested dicts:** readable, but use more memory and are slower than flat NumPy arrays.
- **Height limit ceil(log2 psi):** cheap, and anomalies are still isolated early. The cost: deep normal points all get roughly the same score, so scores below ~0.5 are less fine-grained. Fine for detection, worse for ranking normal points.
- **Threshold = quantile of training scores:** simple and matches the intent of `contamination`. sklearn does the same thing through `offset_`.
- **Only numeric features:** categorical data needs encoding first.

![Isolation Forest visualization](isolation_forest.png)
