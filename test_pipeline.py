# step_test.py — interactive, step-by-step pipeline tester.
#
# Usage:
#     python step_test.py ShapeDatabase/Car/m1487.obj
#
# Controls:
#     n  /  SPACE   run the next preprocessing step
#     left drag     rotate
#     right drag    pan
#     + / -         zoom
#     q / ESC       quit (prints a pass/fail summary)

import sys
from pathlib import Path
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from shape import Shape
from Preprocessing.obj_resampler import (
    clean_shape, simplify_shape, TARGET_VERTICES, TOLERANCE,
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


# --------------------------------------------------------------------------
# Shared geometry analysis helpers (used by every check)
# --------------------------------------------------------------------------

def _directed_edges(face):
    vs = [c[0] for c in face]
    return [(vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])]


def build_edge_faces(shape):
    edges = {}
    for fi, face in enumerate(shape.F):
        for a, b in _directed_edges(face):
            edges.setdefault(tuple(sorted((a, b))), []).append(fi)
    return edges


def count_boundary_edges(shape):
    return sum(1 for fs in build_edge_faces(shape).values() if len(fs) == 1)


def count_nonmanifold_edges(shape):
    return sum(1 for fs in build_edge_faces(shape).values() if len(fs) > 2)


def count_inconsistent_shared_edges(shape):
    bad = 0
    for k, fs in build_edge_faces(shape).items():
        if len(fs) != 2:
            continue
        dirs = []
        for fi in fs:
            for a, b in _directed_edges(shape.F[fi]):
                if tuple(sorted((a, b))) == k:
                    dirs.append((a, b))
                    break
        if len(dirs) == 2 and dirs[0] == dirs[1]:
            bad += 1
    return bad


def compute_moments(shape):
    V = np.array(shape.V, dtype=float)
    moments = np.zeros(3)
    for face in shape.F:
        idx = [c[0] for c in face]
        if len(idx) != 3:
            continue
        a, b, c = V[idx[0]], V[idx[1]], V[idx[2]]
        center = (a + b + c) / 3.0
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        for ax in range(3):
            moments[ax] += np.sign(center[ax]) * area * center[ax] ** 2
    return moments


def compute_area_barycenter(shape):
    V = np.array(shape.V, dtype=float)
    total = 0.0
    weighted = np.zeros(3)
    for face in shape.F:
        idx = [c[0] for c in face]
        if len(idx) != 3:
            continue
        a, b, c = V[idx[0]], V[idx[1]], V[idx[2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        total += area
        weighted += area * (a + b + c) / 3.0
    return weighted / max(total, 1e-12)


def compute_component_volumes(shape):
    origin = np.zeros(3)
    return [calculate_component_volume(shape, c, origin)
            for c in get_connected_components(shape)]


def ensure_outward(shape, eps=1e-9):
    origin = np.zeros(3)
    for comp in get_connected_components(shape):
        v = calculate_component_volume(shape, comp, origin)
        if v < -eps:                  # clearly inward
            flip_component(shape, comp)
        elif abs(v) <= eps:           # flat / degenerate — leave alone
            pass
    return shape


def _clear_normals(shape):
    shape.VN = []
    return shape


def _apply_vertices(shape, new_V):
    """Replace shape.V with new_V, keep types consistent, invalidate normals."""
    shape.V = [tuple(map(float, v)) for v in new_V]
    shape.num_vertices = len(shape.V)
    shape.VN = []
    return shape


def _write(shape):
    shape._last_written_path = shape.write()
    return shape


# --------------------------------------------------------------------------
# Pipeline definition: (display name, step function, per-step check)
# Each check returns (ok: bool, message: str).
# --------------------------------------------------------------------------

def make_pipeline():
    def check_clean(shape, _prev):
        nm = count_nonmanifold_edges(shape)
        return nm == 0, f"non-manifold edges = {nm} (expected 0)"

    def check_orient(shape, _prev):
        bad = count_inconsistent_shared_edges(shape)
        return bad == 0, f"inconsistent shared edges = {bad} (expected 0)"

    def check_fill(shape, _prev):
        b = count_boundary_edges(shape)
        return b == 0, f"boundary edges = {b} (expected 0)"

    def check_align(shape, _prev):
        bary = compute_area_barycenter(shape)
        V = np.array(shape.V, dtype=float)
        size = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
        ok = np.linalg.norm(bary) < 1e-3 * size
        return ok, f"area barycenter = {np.round(bary, 5)} (expected ~0)"

    def check_flip(shape, _prev):
        m = compute_moments(shape)
        ref = max(abs(mi) for mi in m) if len(m) else 1.0
        ok = all(mi >= -1e-9 * max(ref, 1e-9) for mi in m)
        return ok, f"axis moments = {np.round(m, 5)} (expected all >= 0)"

    def check_scale(shape, _prev):
        V = np.array(shape.V, dtype=float)
        maxd = float((V.max(0) - V.min(0)).max())
        return abs(maxd - 1.0) < 1e-6, f"max bbox dim = {maxd:.6f} (expected 1.0)"

    def check_outward(shape, _prev):
        vols = compute_component_volumes(shape)
        ok = all(v > 0 for v in vols)
        return ok, f"component volumes = {np.round(vols, 5)} (expected all > 0)"

    def check_simplify(shape, _prev):
        v = shape.num_vertices
        in_range = (TARGET_VERTICES - TOLERANCE) <= v <= (TARGET_VERTICES + TOLERANCE)
        b = count_boundary_edges(shape)
        ok = in_range and b == 0
        return ok, (f"vertices = {v} (expected "
                    f"{TARGET_VERTICES - TOLERANCE}..{TARGET_VERTICES + TOLERANCE}), "
                    f"boundary edges = {b} (expected 0)")

    def check_write(shape, _prev):
        p = getattr(shape, "_last_written_path", None)
        if p is None:
            return False, "shape.write() did not return a path"
        exists = Path(p).exists()
        return exists, f"written to {p} (exists = {exists})"

    return [
        ("1. clean",          lambda s: _clear_normals(clean_shape(s)),         check_clean),
        ("2. orient",         lambda s: orient_shape(s),                        check_orient),
        ("3. fill_holes",     lambda s: fill_holes(s),                          check_fill),
        ("4. align",          lambda s: align(s),                               check_align),
        ("5. flip",           lambda s: _apply_vertices(s, flip(s.V, s.F)),     check_flip),
        ("6. scale",          lambda s: _apply_vertices(s, scale(s.V)),         check_scale),
        ("7. ensure_outward", lambda s: ensure_outward(s),                      check_outward),
        ("8. simplify",       lambda s: _clear_normals(simplify_shape(s)),      check_simplify),
        ("9. write",          lambda s: _write(s),                              check_write),
    ]


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------

def compute_view_bounds(V):
    if not V:
        return (0.0, 0.0, 0.0), 1.0
    V = np.array(V, dtype=float)
    lo, hi = V.min(0), V.max(0)
    return tuple((lo + hi) / 2.0), max(float((hi - lo).max()) / 2.0, 1e-6)


def face_normal(v1, v2, v3):
    n = np.cross(np.array(v2) - np.array(v1), np.array(v3) - np.array(v1))
    ln = np.linalg.norm(n)
    return tuple(n / ln) if ln > 0 else (0.0, 0.0, 0.0)


def run(filename, width=1000, height=800):
    filename = str(filename)
    shape = Shape(filename)
    steps = make_pipeline()
    step_index = -1          # -1 = only "loaded" has happened
    check_log = []

    center, radius = compute_view_bounds(shape.V)
    zoom = radius * 3.0
    rot_x = rot_y = pan_x = pan_y = 0.0
    last_x = last_y = 0
    active_button = None

    def run_next_step():
        nonlocal shape, step_index, center, radius, zoom, pan_x, pan_y, rot_x, rot_y
        if step_index >= len(steps) - 1:
            print("Pipeline already finished. Press q to quit.")
            return
        step_index += 1
        name, fn, check = steps[step_index]
        prev = shape
        print(f"\n=== {name} ===")
        try:
            shape = fn(shape)
        except Exception as exc:
            msg = f"CRASH: {type(exc).__name__}: {exc}"
            print(f"  {msg}")
            check_log.append((name, False, msg))
            glutPostRedisplay()
            return
        print(f"  vertices={shape.num_vertices}  faces={shape.num_faces}")
        try:
            ok, msg = check(shape, prev)
        except Exception as exc:
            ok, msg = False, f"check crashed: {type(exc).__name__}: {exc}"
        print(f"  check: {'PASS' if ok else 'FAIL'} — {msg}")
        check_log.append((name, ok, msg))

        # Refit the camera to the new geometry
        center, radius = compute_view_bounds(shape.V)
        zoom = radius * 3.0
        pan_x = pan_y = 0.0
        rot_x = rot_y = 0.0
        glutPostRedisplay()

    def draw_axes(length):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0); glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
        glColor3f(0.0, 1.0, 0.0); glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
        glColor3f(0.0, 0.0, 1.0); glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
        glEnd()

    def draw_text(x, y, text):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)
        glRasterPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
        glEnable(GL_DEPTH_TEST)
        if shape.VN:
            glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()

    def draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Perspective: recompute each frame since radius changes after each step.
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        near = max(zoom * 0.01, 1e-4)
        far = zoom + radius * 20.0 + 1.0
        gluPerspective(50.0, width / float(height), near, far)

        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(pan_x, pan_y, -zoom)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        glTranslatef(-center[0], -center[1], -center[2])

        draw_axes(radius * 1.2)

        use_vn = bool(shape.VN)
        glColor3f(0.8, 0.8, 0.8)
        if not use_vn:
            glDisable(GL_LIGHTING)

        for face in shape.F:
            glBegin(GL_TRIANGLES if len(face) == 3 else GL_POLYGON)
            if use_vn:
                for v, vn in face:
                    if vn is not None and vn < len(shape.VN):
                        glNormal3f(*shape.VN[vn])
                    glVertex3f(*shape.V[v])
            else:
                if len(face) >= 3:
                    n = face_normal(shape.V[face[0][0]],
                                    shape.V[face[1][0]],
                                    shape.V[face[2][0]])
                    glNormal3f(*n)
                for v, _ in face:
                    glVertex3f(*shape.V[v])
            glEnd()

        if use_vn:
            glEnable(GL_LIGHTING)

        # ---- HUD ----
        current = steps[step_index][0] if step_index >= 0 else "0. loaded"
        draw_text(20, height - 30,
                  f"Step: {current}   v={shape.num_vertices}  f={shape.num_faces}")
        draw_text(20, height - 55,
                  f"Press 'n' for next step   ({step_index + 1}/{len(steps)} done)")

        y = height - 85
        for name, ok, msg in reversed(check_log[-6:]):
            tag = "PASS" if ok else "FAIL"
            draw_text(20, y, f"[{tag}] {name}: {msg}")
            y -= 22

        glutSwapBuffers()

    def mouse(button, state, x, y):
        nonlocal active_button, last_x, last_y, zoom
        last_x, last_y = x, y
        if state == GLUT_DOWN:
            if button == 3:
                zoom *= 0.9; glutPostRedisplay(); return
            if button == 4:
                zoom *= 1.1; glutPostRedisplay(); return
            active_button = button
        else:
            active_button = None

    def motion(x, y):
        nonlocal last_x, last_y, rot_x, rot_y, pan_x, pan_y
        dx = x - last_x
        dy = y - last_y
        if active_button == GLUT_LEFT_BUTTON:
            rot_y += dx * 0.5
            rot_x += dy * 0.5
        elif active_button == GLUT_RIGHT_BUTTON:
            speed = radius * 0.002
            pan_x += dx * speed
            pan_y -= dy * speed
        last_x, last_y = x, y
        glutPostRedisplay()

    def keyboard(key, _x, _y):
        nonlocal zoom
        if key in (b'n', b' '):
            run_next_step()
        elif key in (b'+', b'='):
            zoom *= 0.9; glutPostRedisplay()
        elif key == b'-':
            zoom *= 1.1; glutPostRedisplay()
        elif key in (b'q', b'\x1b'):
            print("\n\n=== Summary ===")
            if not check_log:
                print("  (no steps were run)")
            for name, ok, msg in check_log:
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
            try:
                glutLeaveMainLoop()
            except Exception:
                import os; os._exit(0)

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(f"step test: {filename}".encode())

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.16, 0.17, 0.19, 1.0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 1.0])

    glutDisplayFunc(draw)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutKeyboardFunc(keyboard)
    glutMainLoop()


if __name__ == "__main__":
    run(r"./ShapeDatabase/AircraftBuoyant/m1337.obj")