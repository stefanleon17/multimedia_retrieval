import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from shape import Shape
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_BINS = 10

D1_MIN = 0.0
D1_MAX = np.sqrt(3)


def calculate_d1(shape):
    """
    Calculate the D1 descriptor.

    D1 = distribution of distances between
    the shape barycenter and surface points.
    """

    vertices = vertices_array(shape)

    barycenter = np.mean(
        vertices,
        axis=0
    )

    differences = (
        vertices
        - barycenter
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    histogram, bin_edges = np.histogram(
        distances,
        bins=NUM_BINS,
        range=(D1_MIN, D1_MAX)
    )

    histogram = (
        histogram
        / np.sum(histogram)
    )

    return histogram, bin_edges


def test_d1(inputs):
    """
    Flexible D1 test.

    inputs can be:
    - one OBJ file
    - a list of OBJ files
    - one folder containing OBJ files
    """

    test_histogram_descriptor(
        inputs=inputs,
        descriptor_function=calculate_d1,
        title="D1 Descriptor",
        xlabel="Distance from barycenter"
    )


if __name__ == "__main__":

    # Example 1: single OBJ
    # test_d1("../ResampledShapeDatabase/Car/m1487.obj")

    # Example 2: multiple OBJ files
    # test_d1([
    #     "../ResampledShapeDatabase/Car/m1487.obj",
    #     "../ResampledShapeDatabase/Car/m1490.obj",
    #     "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    # ])

    # Example 3: all OBJ files in one class folder
    test_d1(
        "../ResampledShapeDatabase/Car"
    )