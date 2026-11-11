# evaluate_retrieval.py
#
# Leave-one-out evaluation of the retrieval system, following the
# pseudocode from the technical tips (Module 6):
#
#   for Q in DB:
#       R = query(Q)                          # top-K neighbors
#       m = metric(labels(R), label(Q))
#       M(C(Q)) += m ;  Mavg += m
#   for C in classes: M(C) /= K(C)
#   Mavg /= |DB|
#
# Usage (from project root):
#     python evaluate_retrieval.py                    # DEFAULT_WEIGHTS from matching.py
#     python evaluate_retrieval.py --weights w.json   # load weights from JSON
#     python evaluate_retrieval.py --quiet            # only grand totals
#     python evaluate_retrieval.py --save run1.json   # save metrics as JSON
#
# To test new weights:
#     1. Edit DEFAULT_WEIGHTS in matching.py
#     2. python evaluate_retrieval.py

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from Matching import FeatureDatabase, FEATURES_CSV, DEFAULT_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_FILE  = PROJECT_ROOT / "evaluation_report.txt"


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def load_weights(path=None):
    if path is None:
        return dict(DEFAULT_WEIGHTS)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(db, weights, ks=(1, 5, 10, 20)):
    """
    Leave-one-out over all shapes in db.

    Returns:
        per_class: {class_name: {metric_name: mean_value}}
        grand:     {metric_name: mean_value_over_all_queries}
        n_eval:    number of queries evaluated
        class_sizes: {class_name: size_in_db}
    """
    N = db.n
    class_names, classes_int = np.unique(
        np.asarray(db.classes), return_inverse=True
    )
    class_sizes = Counter(classes_int.tolist())

    per_class = defaultdict(lambda: defaultdict(list))

    for i in range(N):
        # Ranked list of neighbors (self excluded)
        dists = db.distance_row(i, weights)
        dists[i] = np.inf
        order = np.argsort(dists)

        q_cid = int(classes_int[i])
        n_rel = int(class_sizes[q_cid]) - 1     # exclude query itself
        if n_rel <= 0:
            continue

        is_rel = (classes_int[order] == q_cid).astype(np.int32)

        # P@K
        for k in ks:
            per_class[q_cid][f"P@{k}"].append(is_rel[:k].sum() / k)

        # R-Precision = P at rank n_rel
        per_class[q_cid]["R-Prec"].append(is_rel[:n_rel].sum() / n_rel)

        # AP = mean of P@k over the positions of relevant results
        precisions = np.cumsum(is_rel) / np.arange(1, len(is_rel) + 1)
        per_class[q_cid]["AP"].append(
            (precisions * is_rel).sum() / n_rel
        )

        # MRR = reciprocal rank of first TP
        if is_rel.any():
            first_tp = int(np.argmax(is_rel))
            per_class[q_cid]["MRR"].append(1.0 / (first_tp + 1))
        else:
            per_class[q_cid]["MRR"].append(0.0)

    # Per-class means
    per_class_mean = {}
    for cid, metrics in per_class.items():
        per_class_mean[class_names[cid]] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in metrics.items()
        }

    # Grand totals (mean over all queries)
    all_labels = [f"P@{k}" for k in ks] + ["R-Prec", "AP", "MRR"]
    grand = {}
    for key in all_labels:
        combined = []
        for metrics in per_class.values():
            combined.extend(metrics.get(key, []))
        grand[key] = float(np.mean(combined)) if combined else 0.0

    class_sizes_by_name = {
        class_names[cid]: int(size)
        for cid, size in class_sizes.items()
    }

    return per_class_mean, grand, N, class_sizes_by_name


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(per_class, grand, n_eval, class_sizes, ks, weights):
    lines = []
    def emit(s=""):
        lines.append(s)

    all_labels = [f"P@{k}" for k in ks] + ["R-Prec", "AP", "MRR"]
    n_classes = len(per_class)

    emit("=" * 92)
    emit("RETRIEVAL EVALUATION — leave-one-out over the entire database")
    emit("=" * 92)
    emit(f"Queries evaluated:       {n_eval} / {sum(class_sizes.values())}")
    emit(f"Classes:                 {n_classes}")
    emit(f"Metric aggregation:      per-class = mean over its queries; "
         f"grand = mean over all queries")
    emit()

    non_default = {k: v for k, v in weights.items() if abs(v - 1.0) > 1e-9}
    if non_default:
        emit("Weights (non-default only):")
        for k, v in non_default.items():
            emit(f"  {k:<18s} {v:>6.3f}")
    else:
        emit("Weights: all 1.0 (equal)")
    emit()

    # Grand totals
    emit("-" * 92)
    emit("GRAND TOTALS")
    emit("-" * 92)
    for lbl in all_labels:
        display = "mAP" if lbl == "AP" else lbl
        emit(f"  {display:<12s}  {grand.get(lbl, 0.0) * 100:6.2f}%")
    emit()

    # Per-class table
    emit("-" * 92)
    emit("PER-CLASS METRICS (sorted by P@1, worst → best)")
    emit("-" * 92)
    header = (f"  {'class':<22s} {'N':>4s}  "
              + "  ".join(f"{lbl:>6s}" for lbl in all_labels))
    emit(header)
    emit("  " + "-" * (22 + 4 + 4 + len(all_labels) * 8))

    sorted_classes = sorted(
        per_class.items(), key=lambda kv: kv[1].get("P@1", 0.0)
    )
    for cls, m in sorted_classes:
        row = f"  {cls:<22s} {class_sizes.get(cls, 0):>4d}  "
        row += "  ".join(
            f"{m.get(lbl, 0.0) * 100:>5.1f}%" for lbl in all_labels
        )
        emit(row)
    emit()

    # Summary buckets
    emit("-" * 92)
    emit("SUMMARY")
    emit("-" * 92)
    easy = [c for c, m in sorted_classes if m.get("P@1", 0.0) >= 0.70]
    mid  = [c for c, m in sorted_classes if 0.30 <= m.get("P@1", 0.0) < 0.70]
    hard = [c for c, m in sorted_classes if m.get("P@1", 0.0) <  0.30]
    emit(f"  Classes with P@1 >= 70%:  {len(easy):3d} / {n_classes}")
    emit(f"  Classes with P@1 30-70%:  {len(mid):3d} / {n_classes}")
    emit(f"  Classes with P@1 <  30%:  {len(hard):3d} / {n_classes}")
    emit()

    emit("-" * 92)
    emit("FIVE WORST CLASSES (by P@1)")
    emit("-" * 92)
    for cls, m in sorted_classes[:5]:
        emit(f"  {cls:<22s} P@1={m.get('P@1', 0.0)*100:5.1f}%  "
             f"P@10={m.get('P@10', 0.0)*100:5.1f}%  "
             f"AP={m.get('AP', 0.0)*100:5.1f}%")
    emit()
    emit("FIVE BEST CLASSES (by P@1)")
    emit("-" * 92)
    for cls, m in sorted_classes[-5:][::-1]:
        emit(f"  {cls:<22s} P@1={m.get('P@1', 0.0)*100:5.1f}%  "
             f"P@10={m.get('P@10', 0.0)*100:5.1f}%  "
             f"AP={m.get('AP', 0.0)*100:5.1f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None,
                    help="JSON file with weights (default: DEFAULT_WEIGHTS from matching.py)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20],
                    help="K values for P@K (default: 1 5 10 20)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print grand totals; skip full report to console")
    ap.add_argument("--save", type=str, default=None,
                    help="save full metrics as JSON to this path")
    args = ap.parse_args()

    weights = load_weights(args.weights)

    print("Loading feature database...")
    db = FeatureDatabase(FEATURES_CSV, verbose=False)
    print(f"Loaded {db.n} shapes.")

    print("Running leave-one-out evaluation (this takes ~10-30s)...\n")
    per_class, grand, n_eval, class_sizes = evaluate(
        db, weights, ks=tuple(args.k)
    )

    report = build_report(
        per_class, grand, n_eval, class_sizes, args.k, weights
    )

    if args.quiet:
        print("=" * 50)
        print(f"  {'Metric':<12s}  Value")
        print("=" * 50)
        for lbl in [f"P@{k}" for k in args.k] + ["R-Prec", "AP", "MRR"]:
            display = "mAP" if lbl == "AP" else lbl
            print(f"  {display:<12s}  {grand.get(lbl, 0.0) * 100:6.2f}%")
        print("=" * 50)
    else:
        print(report)

    # Always write the full report to disk
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFull report saved to: {REPORT_FILE}")

    if args.save:
        data = {
            "weights": weights,
            "grand": grand,
            "per_class": per_class,
            "class_sizes": class_sizes,
            "n_eval": n_eval,
            "ks": list(args.k),
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Metrics saved to: {args.save}")


if __name__ == "__main__":
    main()