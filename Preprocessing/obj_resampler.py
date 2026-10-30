# Preprocessing/obj_resampler.py
from pathlib import Path
import pymeshlab
import numpy as np
from shape import Shape

TARGET_VERTICES = 10000
TOLERANCE = 2000


def _mesh_from_shape(shape):
    """Build a fresh MeshSet from the *current* shape.V / shape.F."""
    V = np.array(shape.V, dtype=np.float64)
    F = np.array([[c[0] for c in face] for face in shape.F], dtype=np.int32)
    mesh = pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F))
    return mesh


def clean_shape(shape):
    """Remove duplicates & repair non-manifold edges/vertices.

    Keeps the original resolution. Must run *before* orient_shape /
    fill_holes so that boundary detection is meaningful.
    """
    mesh = _mesh_from_shape(shape)

    mesh.meshing_remove_duplicate_vertices()
    mesh.meshing_remove_duplicate_faces()
    mesh.meshing_remove_unreferenced_vertices()
    mesh.meshing_repair_non_manifold_edges(method='Split Vertices')
    mesh.meshing_repair_non_manifold_vertices()
    mesh.meshing_remove_unreferenced_vertices()

    shape.mesh = mesh
    shape._load_from_mesh()
    return shape


# Preprocessing/obj_resampler.py
import pymeshlab

def _pre_simplify_cleanup(mesh):
    """Re-run the same repairs clean_shape does, on a freshly built mesh.
    Catches non-manifold that fill_holes may have introduced."""
    mesh.meshing_remove_duplicate_vertices()
    mesh.meshing_remove_duplicate_faces()
    mesh.meshing_remove_unreferenced_vertices()
    mesh.meshing_repair_non_manifold_edges(method='Split Vertices')
    mesh.meshing_repair_non_manifold_vertices()
    mesh.meshing_remove_unreferenced_vertices()
    mesh.meshing_remove_duplicate_faces()   # split-vertices can leave dupes


def simplify_shape(shape, target=TARGET_VERTICES, tolerance=TOLERANCE, max_iters=50):
    mesh = _mesh_from_shape(shape)

    it = 0
    while mesh.current_mesh().vertex_number() < (target - tolerance) and it < max_iters:
        try:
            mesh.meshing_surface_subdivision_ls3_loop(iterations=1)
        except pymeshlab.PyMeshLabException:
            # LS3 failed — the mesh has a defect (bowtie vertex, non-manifold
            # edge, inconsistent winding). NOW we attempt repairs, and only
            # here, so clean meshes are never disturbed.
            print(f"    simplify: LS3 failed at {mesh.current_mesh().vertex_number()} vertices; repairing")
            _pre_simplify_cleanup(mesh)
            try:
                mesh.meshing_repair_non_manifold_edges(method='Remove Faces')
                mesh.meshing_surface_subdivision_ls3_loop(iterations=1)
            except pymeshlab.PyMeshLabException as exc:
                print(f"    simplify: WARNING still failing ({exc}); "
                      f"skipping subdivision at {mesh.current_mesh().vertex_number()} vertices")
                break
        it += 1

    target_faces = 100 + 2 * target
    it = 0
    while mesh.current_mesh().vertex_number() > (target + tolerance) and it < max_iters:
        mesh.meshing_decimation_quadric_edge_collapse(
            targetfacenum=max(4, target_faces),
            preservenormal=True,
            preservetopology=True,
            preserveboundary=True,
        )
        v = mesh.current_mesh().vertex_number()
        target_faces -= (v - target)
        it += 1

    shape.mesh = mesh
    shape._load_from_mesh()
    return shape


def resample(input_path, target=TARGET_VERTICES, tolerance=TOLERANCE, max_iter = 10):
    """Standalone: clean + simplify + write. Kept for the viewer ('s' key)."""
    shape = Shape(str(input_path))
    shape = clean_shape(shape)
    iter = 0
    while (shape.V > (target + tolerance) or shape.V < (target - tolerance)) and iter < max_iter:
        shape = simplify_shape(shape, target, tolerance)
        iter += 1
    return shape.write()