import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 100_000
NUM_BINS = 10


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle formed by three points.

    The angle is measured at p2.
    """

    vector1 = p1 - p2
    vector2 = p3 - p2

    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    # Angle is undefined when two sampled
    # points have the same coordinates.
    if length1 == 0 or length2 == 0:
        return None

    cosine = np.dot(
        vector1,
        vector2
    ) / (
        length1 * length2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.arccos(
        cosine
    )

    return angle


def calculate_a3(shape):
    """
    Calculate A3 descriptor.

    A3 = distribution of angles formed
    by three randomly selected points.
    """

    vertices = vertices_array(shape)

    angles = []

    while len(angles) < NUM_SAMPLES:

        indices = np.random.choice(
            len(vertices),
            size=3,
            replace=False
        )

        p1 = vertices[indices[0]]
        p2 = vertices[indices[1]]
        p3 = vertices[indices[2]]

        angle = calculate_angle(
            p1,
            p2,
            p3
        )

        if angle is not None:
            angles.append(angle)

    histogram, bin_edges = np.histogram(
        angles,
        bins=NUM_BINS,
        range=(0, np.pi)
    )

    histogram = (
        histogram
        / np.sum(histogram)
    )

    return histogram, bin_edges


def calculate_a3_for_test(shape):
    """
    Same A3 descriptor, but convert bin edges
    to degrees for easier visualization.
    """

    histogram, bin_edges = calculate_a3(
        shape
    )

    bin_edges = np.degrees(
        bin_edges
    )

    return histogram, bin_edges


def test_a3(inputs):
    """
    Flexible A3 test.

    inputs can be:
    - one OBJ file
    - a list of OBJ files
    - one folder containing OBJ files
    """

    test_histogram_descriptor(
        inputs=inputs,
        descriptor_function=calculate_a3_for_test,
        title="A3 Descriptor",
        xlabel="Angle (degrees)"
    )


if __name__ == "__main__":

    # Example 1: single OBJ
    # test_a3("../ResampledShapeDatabase/Car/m1487.obj")

    # Example 2: multiple OBJ files
    test_a3([
        "../ResampledShapeDatabase/Car/m1487.obj",
        "../ResampledShapeDatabase/Car/m1490.obj",
        "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    ])

    # Example 3: all OBJ files in one class folder
    # test_a3(
    #     "../ResampledShapeDatabase/Car"
    # )