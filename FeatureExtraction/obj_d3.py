import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 1_000_000
NUM_BINS = 10

D3_MIN = 0.0
D3_MAX = np.sqrt(np.sqrt(3) / 2)


def calculate_d3(shape):
    """
    Calculate the D3 descriptor.

    D3 = square root of the area of a triangle
    formed by three randomly selected points.
    """

    vertices = vertices_array(shape)

    num_vertices = len(vertices)

    # Generate random vertex triplets.
    indices = np.random.randint(
        0,
        num_vertices,
        size=(NUM_SAMPLES, 3)
    )

    # Remove triplets containing duplicate indices.
    valid = (
        (indices[:, 0] != indices[:, 1])
        &
        (indices[:, 0] != indices[:, 2])
        &
        (indices[:, 1] != indices[:, 2])
    )

    indices = indices[valid]

    p1 = vertices[indices[:, 0]]
    p2 = vertices[indices[:, 1]]
    p3 = vertices[indices[:, 2]]

    # Two triangle edges.
    vector1 = p2 - p1
    vector2 = p3 - p1

    # Cross product gives twice the triangle area.
    cross_products = np.cross(
        vector1,
        vector2
    )

    areas = (
        0.5
        * np.linalg.norm(
            cross_products,
            axis=1
        )
    )

    # D3 uses square root of triangle area.
    d3_values = np.sqrt(
        areas
    )

    histogram, bin_edges = np.histogram(
        d3_values,
        bins=NUM_BINS,
        range=(D3_MIN, D3_MAX)
    )

    histogram = (
        histogram
        / np.sum(histogram)
    )

    return histogram, bin_edges


def test_d3(inputs):
    """
    Flexible D3 test.

    inputs can be:
    - one OBJ file
    - multiple OBJ files
    - one folder containing OBJ files
    """

    test_histogram_descriptor(
        inputs=inputs,
        descriptor_function=calculate_d3,
        title="D3 Descriptor",
        xlabel="Square root of triangle area"
    )


if __name__ == "__main__":

    # One OBJ
    # test_d3(
    #     "../ResampledShapeDatabase/Car/m1487.obj"
    # )

    # Multiple OBJ files
    test_d3([
        "../ResampledShapeDatabase/Car/m1489.obj",
        "../ResampledShapeDatabase/Car/m1488.obj",
        "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    ])

    # Whole class
    # test_d3(
    #     "../ResampledShapeDatabase/Car"
    # )