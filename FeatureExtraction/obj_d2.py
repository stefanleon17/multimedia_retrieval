import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 100_000
NUM_BINS = 25

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
        0, num_vertices - 1, size=NUM_SAMPLES
    )

    index2 += (index2 >= index1)

    point1 = vertices[index1]
    point2 = vertices[index2]

    # Calculate Euclidean distances
    distances = np.linalg.norm(
        point1 - point2,
        axis=1
    )

    if not np.all(np.isfinite(distances)):
        raise ValueError("D2 contains invalid distances")

    # The maximum possible separation in a unit cube
    max_distance = np.sqrt(3)

    if np.any(distances > max_distance + 1e-10):
        print(
            f"Warning: D2 maximum {distances.max():.6f} "
            f"exceeds expected {max_distance:.6f}"
        )

    # Normalize distances to [0, 1]
    distances = np.clip(
        distances / max_distance, 0, 1
    )

    # Construct histogram
    histogram, bin_edges = np.histogram(
        distances,
        bins=NUM_BINS,
        range=(0, 1)
    )

    # Normalize histogram frequencies
    histogram = histogram.astype(float)
    histogram /= histogram.sum()

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
    # test_d2([
    #     "../ResampledShapeDatabase/Car/m1487.obj",
    #     "../ResampledShapeDatabase/Car/m1490.obj",
    #     "../ResampledShapeDatabase/Biplane/D00069.obj"
    # ])

    # Whole class:
    test_d2(
        "../ResampledShapeDatabase/Car"
    )