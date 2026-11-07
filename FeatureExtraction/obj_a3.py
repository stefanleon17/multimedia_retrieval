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


# def calculate_a3(shape):
#     """
#     Calculate A3 descriptor.
#
#     A3 = distribution of angles formed
#     by three randomly selected points.
#     """
#
#     vertices = vertices_array(shape)
#
#     angles = []
#
#     while len(angles) < NUM_SAMPLES:
#
#         indices = np.random.choice(
#             len(vertices),
#             size=3,
#             replace=False
#         )
#
#         p1 = vertices[indices[0]]
#         p2 = vertices[indices[1]]
#         p3 = vertices[indices[2]]
#
#         angle = calculate_angle(p1, p2, p3)
#
#         if angle is not None:
#             angles.append(angle / np.pi)
#
#     histogram, bin_edges = np.histogram(
#         angles,
#         bins=NUM_BINS,
#         range=(0, 1)
#     )
#
#     # Normalize frequencies so their sum is 1
#     histogram = histogram.astype(float)
#     histogram = histogram / np.sum(histogram)
#
#     return histogram, bin_edges

# using vector - faster!

def calculate_a3(shape):
    """
    Calculate A3 descriptor using vectorized random sampling.

    A3 = distribution of normalized angles formed
    by three randomly selected vertices.

    Angles are normalized to [0, 1].
    Histogram frequencies sum to 1.
    """

    vertices = vertices_array(shape)
    n_vertices = len(vertices)

    if n_vertices < 3:
        raise ValueError("A3 requires at least 3 vertices")

    angles = []
    remaining = NUM_SAMPLES

    while remaining > 0:

        batch_size = min(remaining, 100_000)

        # Sample three distinct vertices per triplet
        i1 = np.random.randint(0, n_vertices, batch_size)

        i2 = np.random.randint(0, n_vertices - 1, batch_size)
        i2 += (i2 >= i1)

        i3 = np.random.randint(0, n_vertices - 2, batch_size)

        low = np.minimum(i1, i2)
        high = np.maximum(i1, i2)

        i3 += (i3 >= low)
        i3 += (i3 >= high)

        p1 = vertices[i1]
        p2 = vertices[i2]
        p3 = vertices[i3]

        vector1 = p1 - p2
        vector2 = p3 - p2

        length1 = np.linalg.norm(vector1, axis=1)
        length2 = np.linalg.norm(vector2, axis=1)

        valid = (length1 > 0) & (length2 > 0)

        if not np.any(valid):
            raise ValueError(
                "Cannot calculate A3: degenerate vertices"
            )

        cosine = np.sum(
            vector1[valid] * vector2[valid],
            axis=1
        ) / (
            length1[valid] * length2[valid]
        )

        cosine = np.clip(cosine, -1.0, 1.0)

        # Normalize angles from [0, pi] to [0, 1]
        batch_angles = np.arccos(cosine) / np.pi

        angles.extend(batch_angles.tolist())

        remaining = NUM_SAMPLES - len(angles)

    # Create histogram with normalized X-axis
    histogram, bin_edges = np.histogram(
        angles,
        bins=NUM_BINS,
        range=(0, 1)
    )

    # Normalize frequencies
    histogram = histogram.astype(float)
    histogram = histogram / histogram.sum()

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
        descriptor_function=calculate_a3,
        title="A3 Descriptor",
        xlabel="Normalized angle (0-1)"
    )


if __name__ == "__main__":

    # Example 1: single OBJ
    test_a3("../ResampledShapeDatabase/Insect/D00787.obj")

    # Example 2: multiple OBJ files
    # test_a3([
    #     "../ResampledShapeDatabase/Car/m1487.obj",
    #     "../ResampledShapeDatabase/Car/m1490.obj",
    #     "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    # ])

    # Example 3: all OBJ files in one class folder
    # test_a3(
    #     "../ResampledShapeDatabase/Insect"
    # )