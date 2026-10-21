# Shared helpers for the feature extraction scripts. Interfaces directly with the Shape class:
#       shape.V = list of (x, y, z) tuples;
#       shape.F = list of (vertex_index, vertex_normal_index) tuple lists.

import numpy as np


def vertices_array(shape):
    """(n, 3) array of the vertex positions."""
    return np.asarray(shape.V, dtype=np.float64).reshape(-1, 3)


def triangles_array(shape):
    """(m, 3) int array of triangle vertex indices."""
    return np.array([[c[0] for c in face] for face in shape.F], dtype=np.int64)


def triangle_corners(shape):
    """Return (a, b, c): three (m, 3) arrays with the corner positions of every triangle."""
    V = vertices_array(shape)
    T = triangles_array(shape)
    return V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
