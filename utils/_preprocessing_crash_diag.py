import numpy as np, pymeshlab
from pathlib import Path

path = r"G:\Projects\Multimedia Retrieval\ShapeDatabase\PlantIndoors\D00080.obj"

def drop_overshared_faces(ms):
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    edges = np.sort(np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]), axis=1)
    face_ids = np.tile(np.arange(len(F)), 3)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    bad = np.unique(face_ids[counts[inv] > 2])
    keep = np.setdiff1d(np.arange(len(F)), bad)
    ms.clear()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F[keep]))
    ms.meshing_remove_unreferenced_vertices()
    return len(bad)

def repair_manifold(ms):
    try:
        ms.meshing_repair_non_manifold_edges(method='Split Vertices')
        return "split"
    except pymeshlab.PyMeshLabException:
        pass
    try:
        ms.meshing_repair_non_manifold_edges(method='Remove Faces')
        return "remove_faces"
    except pymeshlab.PyMeshLabException:
        pass
    n = drop_overshared_faces(ms)
    return f"numpy_drop_{n}"

def edge_stats(ms):
    F = ms.current_mesh().face_matrix()
    e = np.sort(np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return c.max(), (c > 2).sum()

from pathlib import Path

def repair_manifold(ms):
    try:
        ms.meshing_repair_non_manifold_edges(method='Split Vertices')
        return "split"
    except pymeshlab.PyMeshLabException:
        pass
    try:
        ms.meshing_repair_non_manifold_edges(method='Remove Faces')
        return "remove_faces"
    except pymeshlab.PyMeshLabException:
        pass
    n = drop_overshared_faces(ms)
    return f"numpy_drop_{n}"

def dump(ms, name, folder="debug"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    out = Path(folder) / name
    ms.save_current_mesh(str(out))
    print("saved", out)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(path)
dump(ms, "D00080_raw.obj")

ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_unreferenced_vertices()
method = repair_manifold(ms)
print(method)
try:
    ms.meshing_repair_non_manifold_vertices()
except pymeshlab.PyMeshLabException as e:
    print("vertex repair failed:", e)
ms.meshing_remove_unreferenced_vertices()
dump(ms, "D00080_repaired.obj")