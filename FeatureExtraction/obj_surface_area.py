import numpy as np

from FeatureExtraction.obj_mesh_utils import triangle_corners


def calculate_surface_area(shape):
    """Total surface area = sum over triangles of 0.5 * |(b - a) x (c - a)|."""
    a, b, c = triangle_corners(shape)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

