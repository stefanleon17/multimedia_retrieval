import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 100_000
NUM_BINS = 10

D4_MIN = 0.0
D4_MAX = np.cbrt(1.0 / 3.0)


def calculate_d4(shape):
    """
    Calculate the D4 descriptor.

    D4 = cube root of the volume of a tetrahedron
    formed by four randomly selected points.
    """

    vertices = vertices_array(shape)

    num_vertices = len(vertices)

    # Generate random groups of four vertices.
    indices = np.random.randint(
        0,
        num_vertices,
        size=(NUM_SAMPLES, 4)
    )

    # Do not allow duplicate vertices
    # inside one tetrahedron.
    valid = (
        (indices[:, 0] != indices[:, 1])
        & (indices[:, 0] != indices[:, 2])
        & (indices[:, 0] != indices[:, 3])
        & (indices[:, 1] != indices[:, 2])
        & (indices[:, 1] != indices[:, 3])
        & (indices[:, 2] != indices[:, 3])
    )

    indices = indices[valid]

    p1 = vertices[indices[:, 0]]
    p2 = vertices[indices[:, 1]]
    p3 = vertices[indices[:, 2]]
    p4 = vertices[indices[:, 3]]

    vector1 = p2 - p1
    vector2 = p3 - p1
    vector3 = p4 - p1

    # Cross product of two tetrahedron edges.
    cross_products = np.cross(
        vector2,
        vector3
    )

    # Scalar triple product:
    # |v1 · (v2 × v3)| / 6
    volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            vector1,
            cross_products
        )
    ) / 6.0

    # D4 uses cube root of tetrahedron volume.
    d4_values = np.cbrt(
        volumes
    )

    histogram, bin_edges = np.histogram(
        d4_values,
        bins=NUM_BINS,
        range=(D4_MIN, D4_MAX)
    )

    histogram = (
        histogram
        / np.sum(histogram)
    )

    return histogram, bin_edges


def test_d4(inputs):
    """
    Flexible D4 test.

    inputs can be:
    - one OBJ file
    - multiple OBJ files
    - one folder containing OBJ files
    """

    test_histogram_descriptor(
        inputs=inputs,
        descriptor_function=calculate_d4,
        title="D4 Descriptor",
        xlabel="Cube root of tetrahedron volume"
    )


if __name__ == "__main__":

    # One OBJ
    # test_d4(
    #     "../ResampledShapeDatabase/Car/m1487.obj"
    # )

    # Multiple OBJ files
    # test_d4([
    #     "../ResampledShapeDatabase/Car/m1489.obj",
    #     "../ResampledShapeDatabase/Car/m1488.obj",
    #     "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    # ])

    # Whole class
    test_d4(
        "../ResampledShapeDatabase/Car"
    )