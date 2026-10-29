# FeatureExtraction_Pipeline.py
#
# Run from the project root:
#     python FeatureExtraction_Pipeline.py                            # whole DB, quiet
#     python FeatureExtraction_Pipeline.py ResampledShapeDatabase/Car # one class, quiet
#     python FeatureExtraction_Pipeline.py -v                         # whole DB, per-feature detail
#
# Quiet output (default):
#     [   1/2483] ResampledShapeDatabase/AircraftBuoyant/m1339.obj — PASSED
#     [  18/2483] ResampledShapeDatabase/Apartment/D00086.obj — FAILED: convexity (nan); d3 (CRASH: ValueError: ...)
#
# Output:
#     features.csv  (or features_<folder>.csv when a sub-folder is given)
#         one row per shape: path, class, then every feature in FEATURE_COLUMNS.
#         Scalars are one column each; histograms (A3, D1-D4) are one column per bin,
#         e.g. a3_0 ... a3_9. A feature that fails is stored as NaN, so rows always line up.
#     feature_report.txt
#         one line per shape that had at least one failed feature (appended).
#
# Reload the vectors later with load_feature_vectors("features.csv").
#
# A crash in the processing of one feature causes that feature to becomes NaN.

from pathlib import Path
import csv
import sys
import traceback
import numpy as np

from shape import Shape
from FeatureExtraction.obj_surface_area import calculate_surface_area
from FeatureExtraction.obj_volume_calculation import get_volume
from FeatureExtraction.obj_compactness import calculate_compactness
from FeatureExtraction.obj_convexity import calculate_convexity
from FeatureExtraction.obj_aabb_volume import calculate_aabb_volume
from FeatureExtraction.obj_diameter import calculate_diameter
from FeatureExtraction.obj_eccentricity import calculate_eccentricity
from FeatureExtraction.obj_a3 import calculate_a3, NUM_BINS as A3_BINS
from FeatureExtraction.obj_d1 import calculate_d1, NUM_BINS as D1_BINS
from FeatureExtraction.obj_d2 import calculate_d2, NUM_BINS as D2_BINS
from FeatureExtraction.obj_d3 import calculate_d3, NUM_BINS as D3_BINS
from FeatureExtraction.obj_d4 import calculate_d4, NUM_BINS as D4_BINS


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT   = PROJECT_ROOT / "ResampledShapeDatabase"
OUTPUT_CSV   = PROJECT_ROOT / "features.csv"
REPORT_FILE  = PROJECT_ROOT / "feature_report.txt"
SEED         = 0          # histogram descriptors sample randomly; seeded per shape so runs repeat
HIST_EPS     = 1e-6       # tolerance for "histogram sums to 1"


# --------------------------------------------------------------------------
# Feature registry
#   Scalars:    name -> fn(shape) -> float
#   Histograms: name -> fn(shape) -> (histogram, bin_edges), plus its bin count
# --------------------------------------------------------------------------

SCALAR_FEATURES = [
    ("surface_area", calculate_surface_area),
    ("volume",       get_volume),
    ("compactness",  calculate_compactness),
    ("convexity",    calculate_convexity),
    ("aabb_volume",  calculate_aabb_volume),
    ("diameter",     calculate_diameter),
    ("eccentricity", calculate_eccentricity),
]

HIST_FEATURES = [
    ("a3", calculate_a3, A3_BINS),
    ("d1", calculate_d1, D1_BINS),
    ("d2", calculate_d2, D2_BINS),
    ("d3", calculate_d3, D3_BINS),
    ("d4", calculate_d4, D4_BINS),
]

# Fixed column order of the feature vector (this IS the vector layout).
FEATURE_COLUMNS = (
    [name for name, _ in SCALAR_FEATURES]
    + [f"{name}_{i}" for name, _, bins in HIST_FEATURES for i in range(bins)]
)


# --------------------------------------------------------------------------
# Per-feature checks — each returns (ok, message)
# --------------------------------------------------------------------------

def _check_scalar(value):
    ok = bool(np.isfinite(value)) and value > 0
    return ok, f"{value:.6g}"


def _check_hist(hist, bins):
    if len(hist) != bins:
        return False, f"{len(hist)} bins (expected {bins})"
    if not np.all(np.isfinite(hist)):
        return False, "non-finite bin values"
    total = float(np.sum(hist))
    return abs(total - 1.0) < HIST_EPS, f"sum = {total:.6f}"


# --------------------------------------------------------------------------
# One shape through all features
# --------------------------------------------------------------------------

def extract_one(filename, verbose=False):
    """Compute every feature for one shape.

    Returns:
        (vector, report)
        vector is a dict {column_name: float}, covering all of FEATURE_COLUMNS
        (NaN for anything that failed). report is a list of (name, ok, msg).
    """
    report = []
    vector = {col: float("nan") for col in FEATURE_COLUMNS}

    def record(name, ok, msg):
        report.append((name, ok, msg))
        if verbose:
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    # load
    try:
        shape = Shape(str(filename))
    except Exception as exc:
        record("load", False, f"CRASH: {type(exc).__name__}: {exc}")
        return vector, report
    ok = shape.num_faces > 0
    record("load", ok, f"v={shape.num_vertices} f={shape.num_faces}")
    if not ok:
        return vector, report

    np.random.seed(SEED)

    # scalar features
    for name, fn in SCALAR_FEATURES:
        try:
            value = float(fn(shape))
        except Exception as exc:
            record(name, False, f"CRASH: {type(exc).__name__}: {exc}")
            continue
        vector[name] = value
        ok, msg = _check_scalar(value)
        record(name, ok, msg)

    # histogram features
    for name, fn, bins in HIST_FEATURES:
        try:
            hist, _edges = fn(shape)
            hist = np.asarray(hist, dtype=float)
        except Exception as exc:
            record(name, False, f"CRASH: {type(exc).__name__}: {exc}")
            continue
        ok, msg = _check_hist(hist, bins)
        if len(hist) == bins:
            for i, v in enumerate(hist):
                vector[f"{name}_{i}"] = float(v)
        record(name, ok, msg)

    return vector, report


# --------------------------------------------------------------------------
# One-line per-shape formatter
# --------------------------------------------------------------------------

def format_result(rel_path, report):
    failures = [(name, msg) for name, ok, msg in report if not ok]
    if not failures:
        return f"{rel_path} — PASSED"
    detail = "; ".join(f"{name} ({msg})" for name, msg in failures)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{rel_path} — FAILED: {detail}\n")

    return f"{rel_path} — FAILED: {detail}"


# --------------------------------------------------------------------------
# Feature-vector storage
# --------------------------------------------------------------------------

def load_feature_vectors(csv_path=OUTPUT_CSV):
    """Load a features CSV.

    Returns:
        paths   list[str]            one per shape
        classes list[str]
        X       (n_shapes, n_features) float array, columns in FEATURE_COLUMNS order
    """
    paths, classes, rows = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            classes.append(row["class"])
            rows.append([float(row[c]) for c in FEATURE_COLUMNS])
    return paths, classes, np.array(rows, dtype=float)


# --------------------------------------------------------------------------
# Tests  (WIP — placeholders, not wired into anything yet)
# --------------------------------------------------------------------------

# WIP: test_surface_area — e.g. unit cube mesh -> 6.0
def test_surface_area():
    pass


# WIP: test_volume — e.g. unit cube -> 1.0; two disjoint components -> sum; inward-oriented
# component still counts positive
def test_volume():
    pass


# WIP: test_compactness — e.g. a finely tessellated sphere -> ~1.0; cube -> 6/pi ≈ 1.91
def test_compactness():
    pass


# WIP: test_convexity — e.g. a convex shape -> ~1.0; a concave shape -> < 1.0;
# broken mesh -> NaN rather than a crash
def test_convexity():
    pass


# WIP: test_aabb_volume — e.g. a box scaled to known dimensions -> Dx*Dy*Dz
def test_aabb_volume():
    pass


# WIP: test_diameter — e.g. unit cube -> sqrt(3); compare against a brute-force pairwise check
def test_diameter():
    pass


# WIP: test_eccentricity — e.g. a sphere -> ~1.0; an elongated box -> clearly > 1.0
def test_eccentricity():
    pass


# WIP: test_a3 — histogram has A3_BINS bins and sums to 1; right angles (cube corners) land in
# the expected bin
def test_a3():
    pass


# WIP: test_d1 — histogram has D1_BINS bins and sums to 1; sphere -> mass concentrated in one bin
def test_d1():
    pass


# WIP: test_d2 — histogram has D2_BINS bins and sums to 1; all distances fall inside [0, sqrt(3)]
def test_d2():
    pass


# WIP: test_d3 — histogram has D3_BINS bins and sums to 1
def test_d3():
    pass


# WIP: test_d4 — histogram has D4_BINS bins and sums to 1
def test_d4():
    pass


# WIP: test_feature_vector — FEATURE_COLUMNS has no duplicates; extract_one returns every column;
# same shape twice gives identical vectors (seeding works); CSV round-trips through
# load_feature_vectors
def test_feature_vector():
    pass


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def main():
    args    = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    root    = PROJECT_ROOT / args[0] if args else INPUT_ROOT
    out_csv = (PROJECT_ROOT / f"features_{Path(args[0]).name}.csv") if args else OUTPUT_CSV

    if not root.exists():
        print(f"Input path does not exist: {root}")
        sys.exit(1)

    files = sorted(root.rglob("*.obj"))
    n = len(files)
    print(f"Found {n} OBJ files under {root}")
    print(f"Writing feature vectors to {out_csv}\n")

    successes      = 0
    failed_files   = []          # list of (rel_path, [(feature_name, msg), ...])
    failed_checks  = {}          # feature_name -> count

    with open(out_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "class"] + FEATURE_COLUMNS)

        for i, f in enumerate(files, 1):
            rel = f.relative_to(PROJECT_ROOT)
            prefix = f"[{i:4d}/{n}]"

            try:
                vector, report = extract_one(f, verbose=verbose)
            except Exception as exc:
                # Should never happen: extract_one catches per-feature. Safety net.
                traceback.print_exc()
                print(f"{prefix} {rel} — FAILED: internal error ({exc})")
                failed_files.append((rel, [("<internal>", str(exc))]))
                continue

            # write the row immediately so a long run can be interrupted without losing work
            writer.writerow([rel.as_posix(), f.parent.name]
                            + [vector[c] for c in FEATURE_COLUMNS])
            csv_file.flush()

            failures = [(name, msg) for name, ok, msg in report if not ok]
            print(f"{prefix} {format_result(rel, report)}")

            if failures:
                failed_files.append((rel, failures))
                for name, _ in failures:
                    failed_checks[name] = failed_checks.get(name, 0) + 1
            else:
                successes += 1

    # ---------------- Summary ----------------
    print(f"\n{'=' * 60}")
    print(f"Succeeded: {successes} / {n}")
    print(f"Failed:    {n - successes} / {n}")

    if failed_checks:
        print("\nFailed by feature:")
        for name, count in sorted(failed_checks.items()):
            print(f"  {name}: {count} shape(s)")

    if failed_files and not verbose:
        print("\nFailed shapes (feature names only):")
        for rel, fails in failed_files:
            names = ", ".join(name for name, _ in fails)
            print(f"  {rel}: {names}")


if __name__ == "__main__":
    main()