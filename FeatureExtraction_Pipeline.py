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
import math
import tempfile

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
# Tests
#   Run with:  python FeatureExtraction_Pipeline.py --test
#
#   Reference shapes (already resampled), all with max extent 1 and radius / edge 0.5:
#     unit_cube   cube, side 1
#     sphere      radius 0.5
#     half_sphere radius 0.5, flat side closed by the hole-filling step (a closed solid)
#
#   Scalars: closed-form values, compared with a relative tolerance because the shapes are
#   10k-vertex meshes (measured error vs the exact value is <= ~1.2%).
#   Histograms: sphere D1 / D2 are closed-form; everything else is a reference computed
#   from these exact files (4M-sample numpy run, independent of the pipeline code).
#   Re-compute the histogram references if you ever re-resample the shapes.
#
#   Slow tests: diameter (brute force on ~10k vertices, x3 shapes), A3, and feature_vector.
# --------------------------------------------------------------------------

SHAPE_DIR   = PROJECT_ROOT / "TestShapesResampled"
SHAPE_NAMES = ["unit_cube", "sphere", "half_sphere"]

# feature -> (expected value, relative tolerance)
EXPECTED = {
    "unit_cube": {
        "surface_area": (6.0,              0.02),
        "volume":       (1.0,              0.02),
        "compactness":  (6.0 / math.pi,    0.03),   # S^3 / (36 pi V^2) = 216 / (36 pi)
        "convexity":    (1.0,              0.01),
        "aabb_volume":  (1.0,              0.02),
        "diameter":     (math.sqrt(3),     0.01),   # body diagonal
        "eccentricity": (1.0,              0.05),   # isotropic (measured 1.008)
    },
    "sphere": {
        "surface_area": (math.pi,          0.02),   # 4 pi R^2, R = 0.5
        "volume":       (math.pi / 6,      0.02),   # 4/3 pi R^3
        "compactness":  (1.0,              0.03),
        "convexity":    (1.0,              0.01),
        "aabb_volume":  (1.0,              0.02),
        "diameter":     (1.0,              0.01),
        "eccentricity": (1.0,              0.15),   # ideal 1.0; measured 1.10 (see note below)
    },
    "half_sphere": {
        "surface_area": (0.75 * math.pi,   0.02),   # dome 2 pi R^2 + flat cap pi R^2
        "volume":       (math.pi / 12,     0.02),   # 2/3 pi R^3 -> only valid if holes were filled
        "compactness":  (27.0 / 16.0,      0.03),   # 27 pi^3 / (16 pi^3) = 1.6875
        "convexity":    (1.0,              0.01),   # half-ball is convex
        "aabb_volume":  (0.5,              0.02),   # 1 x 1 x 0.5
        "diameter":     (1.0,              0.01),   # rim
        "eccentricity": (4.30,             0.05),   # reference from the file (vertex covariance)
    },
}

# feature -> 10 bin probabilities
EXPECTED_HIST = {
    "unit_cube": {
        "a3": [0.073, 0.166, 0.224, 0.213, 0.153, 0.097, 0.046, 0.018, 0.007, 0.003],
        "d1": [0.0, 0.0, 0.062, 0.641, 0.296, 0.0, 0.0, 0.0, 0.0, 0.0],
        "d2": [0.017, 0.054, 0.092, 0.129, 0.161, 0.213, 0.217, 0.098, 0.018, 0.001],
        "d3": [0.005, 0.029, 0.072, 0.127, 0.184, 0.221, 0.214, 0.128, 0.020, 0.0],
        "d4": [0.018, 0.073, 0.155, 0.222, 0.239, 0.188, 0.090, 0.015, 0.0, 0.0],
    },
    "sphere": {
        "a3": [0.064, 0.169, 0.221, 0.214, 0.165, 0.101, 0.048, 0.015, 0.003, 0.0],
        "d1": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # every vertex at R = 0.5
        "d2": [0.03, 0.09, 0.15, 0.21, 0.27, 0.25, 0.0, 0.0, 0.0, 0.0],  # CDF(d) = d^2 for R = 0.5
        "d3": [0.007, 0.056, 0.140, 0.232, 0.296, 0.257, 0.012, 0.0, 0.0, 0.0],
        "d4": [0.023, 0.134, 0.263, 0.312, 0.222, 0.045, 0.0, 0.0, 0.0, 0.0],
    },
    "half_sphere": {
        "a3": [0.095, 0.200, 0.203, 0.162, 0.122, 0.102, 0.075, 0.033, 0.007, 0.0],
        "d1": [0.0, 0.223, 0.651, 0.126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "d2": [0.058, 0.152, 0.215, 0.245, 0.231, 0.099, 0.0, 0.0, 0.0, 0.0],
        "d3": [0.022, 0.133, 0.257, 0.303, 0.221, 0.063, 0.0, 0.0, 0.0, 0.0],
        "d4": [0.061, 0.284, 0.382, 0.233, 0.040, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}

HIST_TOL = {"d1": 0.005}      # D1 is deterministic (no sampling)
HIST_TOL_DEFAULT = 0.02       # sampled descriptors: noise is ~0.002, rest is margin


_shape_cache = {}


def _shape(name):
    if name not in _shape_cache:
        _shape_cache[name] = Shape(str(SHAPE_DIR / f"{name}.obj"))
    return _shape_cache[name]


def _edge_counts(shape):
    counts = {}
    for face in shape.F:
        v = [c[0] for c in face]
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _t_rel(label, got, expected, rel):
    ok = bool(np.isfinite(got)) and abs(got - expected) <= rel * abs(expected)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: got {got:.6g}, "
          f"expected {expected:.6g} (±{rel:.0%})")
    return ok


def _t_true(label, cond, detail=""):
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return bool(cond)


def _t_hist_sum(label, hist):
    total = float(np.sum(hist))
    return _t_true(label, abs(total - 1.0) < 1e-9, f"sum = {total:.9f}")


def _t_hist(label, hist, expected, tol):
    got = np.asarray(hist, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if got.shape != expected.shape:
        return _t_true(label, False, f"{got.shape[0]} bins, expected {expected.shape[0]}")
    err = float(np.max(np.abs(got - expected)))
    ok = err <= tol
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: max bin error {err:.4f} (tol {tol})")
    if not ok:
        print(f"        got      {np.round(got, 3).tolist()}")
        print(f"        expected {np.round(expected, 3).tolist()}")
    return ok


def _scalar_test(feature, fn):
    """One scalar feature on all three shapes, against EXPECTED."""
    results = []
    for name in SHAPE_NAMES:
        expected, rel = EXPECTED[name][feature]
        results.append(_t_rel(name, fn(_shape(name)), expected, rel))
    return results


def _hist_test(feature, fn):
    """One histogram descriptor on all three shapes, against EXPECTED_HIST."""
    results = []
    for name in SHAPE_NAMES:
        np.random.seed(SEED)
        hist, _edges = fn(_shape(name))
        results.append(_t_hist_sum(f"{name}: sums to 1", hist))
        results.append(_t_hist(f"{name}: distribution", hist,
                               EXPECTED_HIST[name][feature],
                               HIST_TOL.get(feature, HIST_TOL_DEFAULT)))
    return results


def test_shapes_closed():
    # Precondition for volume / convexity / compactness. For half_sphere this checks that
    # the hole-filling really closed the flat side: every edge must be shared by 2 faces.
    results = []
    for name in SHAPE_NAMES:
        shape = _shape(name)
        bad = sum(1 for c in _edge_counts(shape).values() if c != 2)
        results.append(_t_true(f"{name}: closed manifold", bad == 0,
                               f"{shape.num_vertices} vertices, {shape.num_faces} faces, "
                               f"{bad} edges not shared by exactly 2 faces"))
    return all(results)


def test_surface_area():
    return all(_scalar_test("surface_area", calculate_surface_area))


def test_volume():
    # get_volume measures from the origin; half_sphere is the interesting case (its volume
    # is only right if the hole was filled and the mesh is closed)
    return all(_scalar_test("volume", get_volume))


def test_compactness():
    results = _scalar_test("compactness", calculate_compactness)
    c = {n: calculate_compactness(_shape(n)) for n in SHAPE_NAMES}
    results.append(_t_true("ordering: sphere < half_sphere < cube",
                           c["sphere"] < c["half_sphere"] < c["unit_cube"],
                           ", ".join(f"{n} {c[n]:.3f}" for n in SHAPE_NAMES)))
    return all(results)


def test_convexity():
    return all(_scalar_test("convexity", calculate_convexity))


def test_aabb_volume():
    return all(_scalar_test("aabb_volume", calculate_aabb_volume))


def test_diameter():
    return all(_scalar_test("diameter", calculate_diameter))


def test_eccentricity():
    results = _scalar_test("eccentricity", calculate_eccentricity)
    e = {n: calculate_eccentricity(_shape(n)) for n in SHAPE_NAMES}
    results.append(_t_true("ordering: half_sphere is far more eccentric than the others",
                           e["half_sphere"] > 2 * max(e["sphere"], e["unit_cube"]),
                           ", ".join(f"{n} {e[n]:.3f}" for n in SHAPE_NAMES)))
    return all(results)


def test_a3():
    return all(_hist_test("a3", calculate_a3))


def test_d1():
    return all(_hist_test("d1", calculate_d1))


def test_d2():
    return all(_hist_test("d2", calculate_d2))


def test_d3():
    return all(_hist_test("d3", calculate_d3))


def test_d4():
    return all(_hist_test("d4", calculate_d4))


def test_feature_vector():
    # Slow-ish: runs the full extraction twice on the cube.
    obj = SHAPE_DIR / "unit_cube.obj"
    vec1, report = extract_one(obj)
    vec2, _ = extract_one(obj)

    # CSV round trip, same layout main() writes
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "features.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "class"] + FEATURE_COLUMNS)
            writer.writerow(["TestShapesResampled/unit_cube.obj", "Test"]
                            + [vec1[c] for c in FEATURE_COLUMNS])
        paths, classes, X = load_feature_vectors(csv_path)

    a = np.array([vec1[c] for c in FEATURE_COLUMNS])
    b = np.array([vec2[c] for c in FEATURE_COLUMNS])
    return all([
        _t_true("no duplicate column names", len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))),
        _t_true("vector has every column", set(vec1) == set(FEATURE_COLUMNS),
                f"{len(FEATURE_COLUMNS)} columns"),
        _t_true("all feature checks pass on the cube", all(ok for _, ok, _ in report)),
        _t_true("no NaNs", bool(np.all(np.isfinite(a)))),
        _t_rel("volume stored in the vector", vec1["volume"], 1.0, 0.02),
        _t_true("deterministic (same shape twice)", bool(np.array_equal(a, b))),
        _t_true("CSV round trip", X.shape == (1, len(FEATURE_COLUMNS))
                and bool(np.allclose(X[0], a)) and classes == ["Test"]),
    ])


ALL_TESTS = [
    test_shapes_closed,
    test_surface_area, test_volume, test_compactness, test_convexity,
    test_aabb_volume, test_diameter, test_eccentricity,
    test_a3, test_d1, test_d2, test_d3, test_d4,
    test_feature_vector,
]


def run_tests():
    results = {}
    for test in ALL_TESTS:
        print(f"\n{test.__name__}")
        try:
            results[test.__name__] = test()
        except Exception as exc:
            traceback.print_exc()
            print(f"    [FAIL] crashed: {type(exc).__name__}: {exc}")
            results[test.__name__] = False

    print(f"\n{'=' * 60}")
    passed = [n for n, ok in results.items() if ok]
    print(f"Tests passed: {len(passed)} / {len(results)}")
    for name, ok in results.items():
        if not ok:
            print(f"  FAILED: {name}")
    return len(passed) == len(results)

# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def main():
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
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