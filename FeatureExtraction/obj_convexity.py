import pymeshlab
from FeatureExtraction.obj_mesh_utils import vertices_array, triangles_array
from FeatureExtraction.obj_mesh_utils import vertices_array
from FeatureExtraction.obj_volume_calculation import get_volume


def calculate_convex_hull_volume(shape):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=vertices_array(shape),
        face_matrix=triangles_array(shape),
    ))
    ms.generate_convex_hull()
    return float(ms.get_geometric_measures()["mesh_volume"])


def calculate_convexity(shape):
    try:
        hull_vol = calculate_convex_hull_volume(shape)
    except pymeshlab.PyMeshLabException:
        return float("nan")     # catching potential messed up shapes, in case the preprocessing was not effective
    if hull_vol <= 0:
        return float("nan")     # added so it doesn't crash on 0 division
    return get_volume(shape) / hull_vol
