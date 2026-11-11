# matching.py
#
# Retrieval / matching for the multimedia retrieval project.
#
# Usage (from project root):
#     python matching.py <path_to_obj>              # top-10 matches
#     python matching.py <path_to_obj> --k 5        # top-5
#     python matching.py <path_to_obj> --verbose    # per-feature breakdown
#     python matching.py --quick-eval               # leave-one-out NN accuracy
#
# Assumes Features.csv is in the project root with:
#   - one path column (any of: path, filename, file, obj_path)
#   - one class column (optional: class, category, label; falls back to parent folder)
#   - 7 global columns: surface_area, compactness, aabb_volume, diameter,
#                       eccentricity, convexity, volume
#   - 5 histograms × 25 bins: a3_*, d1_*, d2_*, d3_*, d4_*
#     (naming variants: a3_0..a3_24, a3_bin_0..a3_bin_24, a30..a324, a3[0]..a3[24])

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_CSV = "Matching/features.csv"
DEFAULT_QUERY = "ResampledShapeDatabase/Tree/D00765.obj"   # <-- change to the shape you want

NUM_BINS = 25

GLOBAL_FEATURES = [
    "surface_area",
    "compactness",
    "aabb_volume",
    "diameter",
    "eccentricity",   # == elongation (per PDF)
    "convexity",
    "volume",
]

HISTOGRAM_FEATURES = ["a3", "d1", "d2", "d3", "d4"]


# ---------------------------------------------------------------------------
# EDIT THESE WEIGHTS TO TUNE RETRIEVAL
#
# weight = 1.0  -> this feature contributes as much as any other (after
#                   distance standardization below)
# weight > 1.0  -> boost this feature's influence
# weight < 1.0  -> reduce it
# weight = 0    -> ignore this feature entirely
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "surface_area": 1.0,
    "compactness":  1.0,
    "aabb_volume":  1.0,
    "diameter":     1.0,
    "eccentricity": 1.0,
    "convexity":    1.0,
    "volume":       1.0,
    "a3":           1.0,
    "d1":           1.0,
    "d2":           1.0,
    "d3":           1.0,
    "d4":           1.0,
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Feature database
# ---------------------------------------------------------------------------

class FeatureDatabase:
    """
    Holds every shape in Features.csv with two-level normalization applied,
    and exposes retrieve(query_idx, k) -> [(db_idx, combined_distance), ...].
    """

    def __init__(self, csv_path=FEATURES_CSV, verbose=True):
        df = pd.read_csv(csv_path)

        self.path_col = _find_column(df, ["path", "filename", "file", "obj_path"])
        if self.path_col is None:
            self.path_col = df.columns[0]
        self.class_col = _find_column(df, ["class", "category", "label"])

        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)

        self.df = df
        self.paths = df[self.path_col].astype(str).values
        if self.class_col is not None:
            self.classes = df[self.class_col].astype(str).values
        else:
            # Derive class from parent folder name
            self.classes = np.array([Path(p).parent.name for p in self.paths])
        self.n = len(df)

        if verbose:
            print(f"Loaded {self.n} shapes from {csv_path}")
            if n_dropped:
                print(f"  Dropped {n_dropped} rows with NaN features")

        # ---- Level 1: feature-level normalization ----
        self.globals = {}
        for f in GLOBAL_FEATURES:
            if f not in df.columns:
                if verbose:
                    print(f"  Warning: global feature '{f}' not in CSV, skipping")
                continue
            vals = df[f].values.astype(float)
            mu, sd = float(vals.mean()), float(vals.std())
            if sd < 1e-12:
                sd = 1.0
            self.globals[f] = (vals - mu) / sd

        self.histograms = {}
        for h in HISTOGRAM_FEATURES:
            cols = [f"{h}_{i}" for i in range(NUM_BINS)]
            if not all(c in df.columns for c in cols):
                if verbose:
                    print(f"  Warning: histogram '{h}' not in CSV, skipping")
                continue
            H = df[cols].values.astype(float)
            row_sums = H.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.histograms[h] = H / row_sums

        if verbose:
            print(f"  Globals:    {list(self.globals.keys())}")
            print(f"  Histograms: {list(self.histograms.keys())}")

        # ---- Level 2: distance-level standardization ----
        self._calibrate_distances(n_samples=10000, verbose=verbose)

    def _calibrate_distances(self, n_samples, verbose):
        """
        For each feature, sample ~n_samples pairwise raw distances from the
        database, then store their (mean, std). At query time the combined
        distance uses (raw - mean) / std, so each feature contributes on a
        comparable scale.
        """
        rng = np.random.default_rng()
        self.dist_mean, self.dist_std = {}, {}

        for f, vals in self.globals.items():
            i = rng.integers(0, self.n, n_samples)
            j = rng.integers(0, self.n, n_samples)
            keep = i != j
            i, j = i[keep], j[keep]
            diffs = np.abs(vals[i] - vals[j])
            mu, sd = float(diffs.mean()), float(diffs.std())
            self.dist_mean[f] = mu
            self.dist_std[f] = sd if sd > 1e-12 else 1.0

        for h, H in self.histograms.items():
            cdfs = np.cumsum(H, axis=1) / NUM_BINS
            i = rng.integers(0, self.n, n_samples)
            j = rng.integers(0, self.n, n_samples)
            keep = i != j
            i, j = i[keep], j[keep]
            emds = np.abs(cdfs[i] - cdfs[j]).sum(axis=1)
            mu, sd = float(emds.mean()), float(emds.std())
            self.dist_mean[h] = mu
            self.dist_std[h] = sd if sd > 1e-12 else 1.0

        if verbose:
            print(f"\n  Distance calibration (mean, std):")
            for k in self.dist_mean:
                print(f"    {k:<15s}  mu={self.dist_mean[k]:.4f}  "
                      f"sd={self.dist_std[k]:.4f}")

    def export_normalized(self, out_path="normalized_features.csv"):
        """
        Write the level-1 normalized feature table to CSV:
          - globals: z-score standardized
          - histograms: each histogram divided by its bin sum
        """
        out = pd.DataFrame()
        out[self.path_col] = self.paths
        out["class"] = self.classes

        for f, vals in self.globals.items():
            out[f] = vals

        for h, H in self.histograms.items():
            for b in range(NUM_BINS):
                out[f"{h}_{b}"] = H[:, b]

        out.to_csv(out_path, index=False)
        print(f"Wrote {len(out)} normalized rows to {out_path}")

    # ------------------------------------------------------------------

    def find_index(self, path):
        """
        Match a query path to a row in the CSV.
        Basename first, then full-path, case-insensitive.
        """
        needle_name = Path(str(path)).name.lower()
        for idx, p in enumerate(self.paths):
            if Path(p).name.lower() == needle_name:
                return idx
        needle_full = str(path).replace("\\", "/").lower()
        for idx, p in enumerate(self.paths):
            if p.replace("\\", "/").lower() == needle_full:
                return idx
        return None

    def _standardized_global_distance(self, f, i, j):
        raw = abs(float(self.globals[f][i]) - float(self.globals[f][j]))
        return (raw - self.dist_mean[f]) / self.dist_std[f]

    def _standardized_histogram_distance(self, h, i, j):
        H = self.histograms[h]
        cdf_i = np.cumsum(H[i]) / NUM_BINS
        cdf_j = np.cumsum(H[j]) / NUM_BINS
        raw = float(np.abs(cdf_i - cdf_j).sum())   # 1D EMD
        return (raw - self.dist_mean[h]) / self.dist_std[h]

    def feature_distance(self, i, j, weights):
        total = 0.0
        for f in self.globals:
            total += weights.get(f, 1.0) * self._standardized_global_distance(f, i, j)
        for h in self.histograms:
            total += weights.get(h, 1.0) * self._standardized_histogram_distance(h, i, j)
        return total

    def distance_row(self, query_idx, weights):
        return np.array([
            self.feature_distance(query_idx, j, weights)
            for j in range(self.n)
        ])

    def retrieve(self, query_idx, k=10, weights=None):
        if weights is None:
            weights = DEFAULT_WEIGHTS
        dists = self.distance_row(query_idx, weights)
        order = np.argsort(dists)[:k]
        return [(int(idx), float(dists[idx])) for idx in order]

    def top1_same_class(self, weights=None):
        """LOO top-1 class accuracy — useful as a quick weight-tuning signal."""
        if weights is None:
            weights = DEFAULT_WEIGHTS
        correct = 0
        for i in range(self.n):
            (best, _), = self.retrieve(i, k=1, weights=weights)
            if self.classes[best] == self.classes[i]:
                correct += 1
        return correct / self.n


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(db, query_idx, results, verbose=False):
    qpath = db.paths[query_idx]
    qclass = db.classes[query_idx]
    print(f"\nQuery:  {qpath}")
    print(f"Class:  {qclass}")
    print(f"Top {len(results)} matches (combined standardized distance):\n")

    for rank, (idx, dist) in enumerate(results, 1):
        p = db.paths[idx]
        c = db.classes[idx]
        marker = "same" if c == qclass else "diff"
        print(f"  {rank:2d}. [{marker}] {c:<18s}  {p:<58s}  d={dist:+.4f}")

        if verbose:
            for f in db.globals:
                s = db._standardized_global_distance(f, query_idx, idx)
                print(f"           {f:<14s} std_dist = {s:+.3f}")
            for h in db.histograms:
                s = db._standardized_histogram_distance(h, query_idx, idx)
                print(f"           {h:<14s} std_dist = {s:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="OBJ path to query")
    parser.add_argument("--k", type=int, default=10, help="number of matches (default 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="show per-feature standardized distances")
    parser.add_argument("--quick-eval", action="store_true",
                        help="report leave-one-out top-1 accuracy for current weights")
    parser.add_argument("--norm", action="store_true",
                        help="write normalized_features.csv and exit")
    args = parser.parse_args()

    db = FeatureDatabase(FEATURES_CSV)
    if args.norm:
        db.export_normalized("Matching/normalized_features.csv")
        if args.query is None and not args.quick_eval:
            return

    if args.quick_eval:
        acc = db.top1_same_class()
        print(f"\nLOO top-1 class accuracy: {acc * 100:.2f}%  "
              f"({db.n} shapes, current weights)")
        if args.query is None:
            return

    if args.query is None:
        args.query = DEFAULT_QUERY

    idx = db.find_index(args.query)
    if idx is None:
        print(f"\nShape not found in Features.csv: {args.query}")
        return

    results = db.retrieve(idx, k=args.k)
    print_results(db, idx, results, verbose=args.verbose)


if __name__ == "__main__":
    main()