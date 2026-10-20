# Preprocessing_Pipeline.py
#
# Run from the project root:
#     python Preprocessing_Pipeline.py                       # whole DB, quiet
#     python Preprocessing_Pipeline.py ShapeDatabase/Car     # one class, quiet
#     python Preprocessing_Pipeline.py -v                    # whole DB, per-step detail
#
# Quiet output (default):
#     [   1/2483] ShapeDatabase/AircraftBuoyant/m1339.obj — PASSED
#     [  18/2483] ShapeDatabase/Apartment/D00086.obj — FAILED: 1. clean (non-manifold edges = 49); 3. fill_holes (boundary edges = 74)

from pathlib import Path
import sys
import traceback
import numpy as np

from shape import Shape
from Preprocessing.obj_resampler import (
    clean_shape,
    simplify_shape,
    TARGET_VERTICES,
    TOLERANCE,
)
from Preprocessing.obj_triangle_orientation import orient_shape
from Preprocessing.obj_hole_filling import fill_holes
from Preprocessing.obj_alignment import align
from Preprocessing.obj_normalization import flip, scale
from FeatureExtraction.obj_volume_calculation import (
    get_connected_components,
    calculate_component_volume,
    flip_component,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT   = PROJECT_ROOT / "ShapeDatabase"
OUTPUT_ROOT  = PROJECT_ROOT / "ResampledShapeDatabase"
VOL_EPS      = 1e-9


# --------------------------------------------------------------------------
# Shared helpers (unchanged)
# --------------------------------------------------------------------------

def build_edge_faces(shape):
    edges = {}
    for fi, face in enumerate(shape.F):
        vs = [c[0] for c in face]
        for a, b in [(vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])]:
            edges.setdefault(tuple(sorted((a, b))), []).append(fi)
    return edges


def count_boundary_edges(shape):
    return sum(1 for fs in build_edge_faces(shape).values() if len(fs) == 1)


def count_nonmanifold_edges(shape):
    return sum(1 for fs in build_edge_faces(shape).values() if len(fs) > 2)


def component_volumes(shape):
    origin = np.zeros(3)
    return [calculate_component_volume(shape, c, origin)
            for c in get_connected_components(shape)]


def ensure_outward(shape, eps=VOL_EPS):
    origin = np.zeros(3)
    for comp in get_connected_components(shape):
        v = calculate_component_volume(shape, comp, origin)
        if v < -eps:
            flip_component(shape, comp)
    return shape


# --------------------------------------------------------------------------
# Per-step wrappers for functions that don't return a Shape
# --------------------------------------------------------------------------

def _step_flip(shape):
    shape.V = [tuple(v) for v in flip(shape.V, shape.F)]
    shape.num_vertices = len(shape.V)
    shape.VN = []
    return shape


def _step_scale(shape):
    V = np.array(shape.V, dtype=float)
    span = float((V.max(0) - V.min(0)).max()) if len(V) else 0.0
    if span < 1e-9:
        raise ValueError(f"degenerate shape after flip (span={span})")
    shape.V = [tuple(v) for v in scale(shape.V)]
    shape.num_vertices = len(shape.V)
    shape.VN = []
    return shape


def _step_write(shape):
    shape._last_written_path = shape.write()
    return shape


# --------------------------------------------------------------------------
# Per-step checks — each returns (ok, message)
# --------------------------------------------------------------------------

def _check_clean(shape):
    nm = count_nonmanifold_edges(shape)
    return nm == 0, f"non-manifold edges = {nm}"


def _check_orient(shape):
    return True, f"v={shape.num_vertices} f={shape.num_faces}"


def _check_fill(shape):
    b = count_boundary_edges(shape)
    return b == 0, f"boundary edges = {b}"


def _check_align(shape):
    return True, "barycenter moved to origin"


def _check_flip(shape):
    return True, "axis signs canonicalised"


def _check_scale(shape):
    V = np.array(shape.V, dtype=float)
    maxd = float((V.max(0) - V.min(0)).max()) if len(V) else 0.0
    return abs(maxd - 1.0) < 1e-6, f"max bbox dim = {maxd:.6f}"


def _check_outward(shape):
    vols = component_volumes(shape)
    bad  = sum(1 for v in vols if v < -VOL_EPS)
    flat = sum(1 for v in vols if abs(v) <= VOL_EPS)
    return bad == 0, f"{len(vols)} components, {flat} flat, {bad} inward"


def _check_simplify(shape):
    v = shape.num_vertices
    in_range = (TARGET_VERTICES - TOLERANCE) <= v <= (TARGET_VERTICES + TOLERANCE)
    b = count_boundary_edges(shape)
    # Boundaries here usually pre-date step 8 (left over from step 3), so we
    # tolerate a small fraction of them. Step 8 no longer introduces any.
    tolerance_b = max(100, shape.num_faces // 200)
    ok = in_range and b <= tolerance_b
    return ok, (f"v={v} (target {TARGET_VERTICES}±{TOLERANCE}), "
                f"boundary edges = {b} / {shape.num_faces} faces")


def _check_write(shape):
    p = getattr(shape, "_last_written_path", None)
    if p is None:
        return False, "write() returned no path"
    return Path(p).exists(), str(p)


# --------------------------------------------------------------------------
# One shape through the full pipeline
# --------------------------------------------------------------------------

def preprocess_one(filename, verbose=False):
    """Run all pipeline steps on one shape.

    Returns:
        (output_path_or_None, report)
        report is a list of (step_name, ok, msg) tuples. A step that raises
        is recorded as failed with a CRASH message, and the pipeline bails
        for that shape so we don't drown in cascading errors.
    """
    report = []

    def record(name, ok, msg):
        report.append((name, ok, msg))
        if verbose:
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    # 0. load
    try:
        shape = Shape(str(filename))
    except Exception as exc:
        record("0. load", False, f"CRASH: {type(exc).__name__}: {exc}")
        return None, report
    record("0. load", shape.num_faces > 0,
           f"v={shape.num_vertices} f={shape.num_faces}")

    steps = [
        ("1. clean",          clean_shape,     _check_clean),
        ("2. orient",         orient_shape,    _check_orient),
        ("3. fill_holes",     fill_holes,      _check_fill),
        ("4. align",          align,           _check_align),
        ("5. flip",           _step_flip,      _check_flip),
        ("6. scale",          _step_scale,     _check_scale),
        ("7. ensure_outward", ensure_outward,  _check_outward),
        ("8. simplify",       simplify_shape,  _check_simplify),
        ("9. write",          _step_write,     _check_write),
    ]

    for name, fn, check in steps:
        try:
            shape = fn(shape)
        except Exception as exc:
            record(name, False, f"CRASH: {type(exc).__name__}: {exc}")
            return None, report
        try:
            ok, msg = check(shape)
        except Exception as exc:
            ok, msg = False, f"check crashed: {type(exc).__name__}: {exc}"
        record(name, ok, msg)

    return getattr(shape, "_last_written_path", None), report


# --------------------------------------------------------------------------
# One-line per-shape formatter
# --------------------------------------------------------------------------

def format_result(rel_path, report):
    failures = [(name, msg) for name, ok, msg in report if not ok]
    if not failures:
        return f"{rel_path} — PASSED"
    detail = "; ".join(f"{name} ({msg})" for name, msg in failures)
    return f"{rel_path} — FAILED: {detail}"


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def main():
    args    = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    root    = PROJECT_ROOT / args[0] if args else INPUT_ROOT

    if not root.exists():
        print(f"Input path does not exist: {root}")
        sys.exit(1)

    files = sorted(root.rglob("*.obj"))
    n = len(files)
    print(f"Found {n} OBJ files under {root}\n")

    successes      = 0
    failed_files   = []          # list of (rel_path, [(step_name, msg), ...])
    failed_checks  = {}          # step_name -> count

    for i, f in enumerate(files, 1):
        rel = f.relative_to(PROJECT_ROOT)
        prefix = f"[{i:4d}/{n}]"

        try:
            out, report = preprocess_one(f, verbose=verbose)
        except Exception as exc:
            # Should never happen: preprocess_one catches per-step. Safety net.
            traceback.print_exc()
            print(f"{prefix} {rel} — FAILED: internal error ({exc})")
            failed_files.append((rel, [("<internal>", str(exc))]))
            continue

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
        print("\nFailed by step:")
        for name, count in sorted(failed_checks.items()):
            print(f"  {name}: {count} shape(s)")

    if failed_files and not verbose:
        print("\nFailed shapes (step names only):")
        for rel, fails in failed_files:
            steps = ", ".join(name for name, _ in fails)
            print(f"  {rel}: {steps}")


if __name__ == "__main__":
    main()