import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 100_000
NUM_BINS = 10

D2_MIN = 0.0
D2_MAX = np.sqrt(3)


def calculate_d2(shape):
    """
    Calculate the D2 descriptor.

    D2 = distribution of distances between
    two randomly selected surface points.
    """

    vertices = vertices_array(shape)

    num_vertices = len(vertices)

    # Randomly select the first point.
    index1 = np.random.randint(
        0,
        num_vertices,
        size=NUM_SAMPLES
    )

    # Randomly select the second point.
    index2 = np.random.randint(
        0,
        num_vertices,
        size=NUM_SAMPLES
    )

    # Do not allow the same vertex
    # to be selected twice in one pair.
    same_indices = (
        index1 == index2
    )

    while np.any(same_indices):

        index2[same_indices] = np.random.randint(
            0,
            num_vertices,
            size=np.sum(same_indices)
        )

        same_indices = (
            index1 == index2
        )

    point1 = vertices[index1]
    point2 = vertices[index2]

    differences = (
        point1
        - point2
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    histogram, bin_edges = np.histogram(
        distances,
        bins=NUM_BINS,
        range=(D2_MIN, D2_MAX)
    )

    histogram = (
        histogram
        / np.sum(histogram)
    )

    return histogram, bin_edges


def test_d2(inputs):
    """
    Flexible D2 test.

    inputs can be:
    - one OBJ file
    - multiple OBJ files
    - one folder containing OBJ files
    """

    test_histogram_descriptor(
        inputs=inputs,
        descriptor_function=calculate_d2,
        title="D2 Descriptor",
        xlabel="Distance between points"
    )


if __name__ == "__main__":

    # One OBJ:
    # test_d2(
    #     "../ResampledShapeDatabase/Car/m1487.obj"
    # )

    # Multiple OBJ files:
    test_d2([
        "../ResampledShapeDatabase/Car/m1487.obj",
        "../ResampledShapeDatabase/Car/m1490.obj",
        "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    ])

    # Whole class:
    # test_d2(
    #     "../ResampledShapeDatabase/Car"
    # )