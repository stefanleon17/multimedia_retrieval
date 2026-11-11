# ANN (pynndescent) search over the normalized shape database, plus a t-SNE plot.
# Distance = the same weighted, scaled L1/EMD distance as matching.py (ranking-equivalent).
#
#   KNN : the K most similar shapes                 ShapeIndex.knn / search_item
#   RNN : at most K shapes, none farther than R     ShapeIndex.knn_within
#
# Inputs written by `python matching.py --norm`:
#   normalized_features.csv  z-scored globals + sum-normalized histograms
#   dist_scales.json         per-feature distance std + weights
# load_features() turns histograms into cumulative sums and builds the per-column scale
# vector, so plain manhattan distance on the result equals the weighted EMD.
#
# Run from the project root:
#     python nearest_neighbours.py                        # first item in DB (demo)
#     python nearest_neighbours.py <path_to_obj>          # query by filename
#     python nearest_neighbours.py <path_to_obj> --k 10  # top-10 results
#     python nearest_neighbours.py --tsne                 # interactive t-SNE plot (tsne.html)
#     python nearest_neighbours.py --rebuild              # force index rebuild

import argparse
import os.path
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json
import joblib
import plotly.express as px
from pynndescent import NNDescent
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

INDEX_CACHE = "Matching/ann_index.joblib"

class ShapeIndex:
    """Build once when the database is loaded, then query as often as you like."""

    def __init__(self, X, paths, classes, scale, weights=None,
                 cache_path=INDEX_CACHE, force_rebuild=False):
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("features contain NaN/inf (failed extractions?) - clean them first")

        self.paths = np.asarray(paths)
        self.classes = np.asarray(classes)
        self.X_raw = X
        self.scale = np.asarray(scale, dtype=np.float32)
        self.X = X * self.scale

        if not force_rebuild and cache_path and os.path.exists(cache_path):
            print(f"Loading ANN index from cache: {cache_path}")
            self.index = joblib.load(cache_path)
        else:
            print("Building ANN index…")
            self.index = NNDescent(
                self.X,
                metric="manhattan",
                n_neighbors=50,       # see note below on ANN precision
                random_state=0,
            )
            self.index.prepare()
            if cache_path:
                joblib.dump(self.index, cache_path)
                print(f"ANN index saved to: {cache_path}")

        self.nn = NearestNeighbors(algorithm="brute", metric="manhattan", n_jobs=-1)
        self.nn.fit(self.X)

    # ---- helpers ----------------------------------------------------------

    def _prepare(self, query):
        return (np.asarray(query, dtype=np.float32) * self.scale).reshape(1, -1)

    def _results(self, distances, indices):
        """List of (path, class, distance), closest first."""
        return [(self.paths[i], self.classes[i], float(d)) for d, i in zip(distances, indices)]

    # ---- queries ----------------------------------------------------------

    def knn(self, query, k=5):
        indices, distances = self.index.query(self._prepare(query), k=min(k, len(self.X)))
        return self._results(distances[0], indices[0])

    def search_item(self, i, k=5, only_class=None):
        """KNN for database row i, leaving the item itself out of its own results."""
        results = self.knn(self.X_raw[i], k + 1)
        return [r for r in results if r[0] != self.paths[i]][:k]

    def knn_within(self, query, k=143, max_distance=0.25, only_class=None):
        """At most k results, and none farther than max_distance (so possibly fewer than k)."""
        return [r for r in self.knn(query, k) if r[2] <= max_distance]

    # ---- choosing R -------------------------------------------------------

    def kth_neighbor_distances(self, k=5, n_queries=200, seed=0):
        """Distance to the k-th nearest neighbour for a random sample of database items.
        Look at its median / percentiles to pick a sensible radius R."""
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(self.X), size=min(n_queries, len(self.X)), replace=False)
        _, distances = self.index.query(self.X[rows], k=k + 1)
        return distances[:, k]  # column 0 is the item itself (distance 0)


GLOBALS = ["surface_area", "compactness", "aabb_volume", "diameter",
           "eccentricity", "convexity", "volume"]
HISTS = ["a3", "d1", "d2", "d3", "d4"]
NUM_BINS = 25

def load_features(csv_path="Matching/normalized_features.csv", scales_path="dist_scales.json"):
    df = pd.read_csv(csv_path)
    paths = df.iloc[:, 0].to_numpy()
    classes = df["class"].to_numpy()
    if os.path.exists((scales_path)):
        with open(scales_path) as fh:
            meta = json.load(fh)
        std, w = meta["dist_std"], meta["weights"]
    else:
        print(f"Warning: {scales_path} not found, using unit scales (std=1, weight=1)")
        std = {name: 1.0 for name in GLOBALS + HISTS}
        w = {name: 1.0 for name in GLOBALS + HISTS}

    blocks, scale = [], []
    for f in GLOBALS:
        blocks.append(df[[f]].to_numpy(np.float32))
        scale.append(np.full(1, w[f] / std[f]))
    for h in HISTS:
        H = df[[f"{h}_{b}" for b in range(NUM_BINS)]].to_numpy(np.float32)
        blocks.append(np.cumsum(H, axis=1))                       # L1 on this = 1D EMD
        scale.append(np.full(NUM_BINS, w[h] / (NUM_BINS * std[h])))
    return paths, classes, np.hstack(blocks), np.concatenate(scale).astype(np.float32)


# --------------------------------------------------------------------------
# t-SNE visualisation of how the classes cluster
# --------------------------------------------------------------------------
def embed_tsne(X, perplexity=30, seed=0):
    return TSNE(n_components=2, perplexity=perplexity, metric="manhattan",
                random_state=seed, n_jobs=-1).fit_transform(X)

def plot_tsne(emb, paths, classes, save_path="tsne.html"):
    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "class": classes, "shape": paths})
    fig = px.scatter(df, x="x", y="y", color="class", hover_name="shape",
                     hover_data={"x": False, "y": False}, render_mode="webgl")
    fig.update_traces(marker_size=5)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left",
                                  x=0, entrywidth=0.15, entrywidthmode="fraction", font=dict(size=10)),
                      width=1000, height=1000)

    # Interactive HTML
    fig.write_html(save_path)

    # Static PNG
    fig.write_image(save_path.replace(".html", ".png"), scale=2)

    fig.show()

def _cache_is_stale(csv_path, cache_path):
    """Return True if the index cache is older than the feature CSV, or missing."""
    if not os.path.exists(cache_path):
        return True
    return os.path.getmtime(csv_path) > os.path.getmtime(cache_path)


def _find_index(paths, query_path):
    """
    Look up a query .obj path in the database paths array.
    Tries basename match first, then full-path match (case-insensitive).
    Returns the integer index, or None if not found.
    """
    needle_name = Path(query_path).name.lower()
    for idx, p in enumerate(paths):
        if Path(p).name.lower() == needle_name:
            return idx
    needle_full = str(query_path).replace("\\", "/").lower()
    for idx, p in enumerate(paths):
        if str(p).replace("\\", "/").lower() == needle_full:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="ANN nearest-neighbour search over the shape database."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Path to a .obj file to query (basename match is fine). "
             "Omit to query the first item in the DB as a demo.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of nearest neighbours to return (default: 5).",
    )
    parser.add_argument(
        "--tsne", action="store_true",
        help="Render an interactive t-SNE plot instead of running a query.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force a full rebuild of the ANN index (ignores cache).",
    )
    args = parser.parse_args()

    csv_path = "Matching/normalized_features.csv"
    force_rebuild = args.rebuild
    if not force_rebuild and _cache_is_stale(csv_path, INDEX_CACHE):
        print("Feature CSV is newer than the cached index — rebuilding.")
        force_rebuild = True

    paths, classes, X, scale = load_features(csv_path)
    index = ShapeIndex(X, paths, classes, scale, force_rebuild=force_rebuild)

    if args.tsne:
        plot_tsne(embed_tsne(index.X), paths, classes)
        return

    # ---- resolve query -------------------------------------------------------
    if args.query is not None:
        i = _find_index(paths, args.query)
        if i is None:
            print(f"Error: '{args.query}' not found in {csv_path}.")
            print("Tip: you can pass just the filename (e.g. D00765.obj) instead of the full path.")
            sys.exit(1)
    else:
        i = 0   # demo: first item in the DB
        print("No query given — using first item in the DB as a demo.\n")

    # ---- results -------------------------------------------------------------
    print(f"Query: {paths[i]}  [{classes[i]}]\n")

    print(f"Top {args.k} nearest neighbours (query itself excluded):")
    print(f"  {'dist':>8}  {'class':<20}  path")
    print(f"  {'-'*8}  {'-'*20}  {'-'*40}")
    for path, cls, dist in index.search_item(i, k=args.k):
        match = "✓" if cls == classes[i] else " "
        print(f"  {dist:8.4f}  {match} {cls:<20}  {path}")

    radius = float(np.median(index.kth_neighbor_distances(k=args.k)))
    print(f"\nRadius query  R = {radius:.4f}  (median dist to the {args.k}-th neighbour):")
    within = index.knn_within(X[i], k=len(X), max_distance=radius)
    within = [r for r in within if r[0] != paths[i]]  # exclude self
    print(f"  {len(within)} shapes within R (showing up to 10):")
    for path, cls, dist in within[:10]:
        match = "✓" if cls == classes[i] else " "
        print(f"  {dist:8.4f}  {match} {cls:<20}  {path}")


if __name__ == "__main__":
    main()
