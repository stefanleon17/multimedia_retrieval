import numpy as np

from FeatureExtraction.obj_histogram_utils import (
    test_histogram_descriptor,
)
from FeatureExtraction.obj_mesh_utils import vertices_array


NUM_SAMPLES = 100_000
NUM_BINS = 25

D4_MIN = 0.0
D4_MAX = np.cbrt(1.0 / 3.0)


def calculate_d4(shape):
    """
    Calculate the D4 descriptor.

    D4 = cube root of the volume of a tetrahedron
    formed by four randomly selected points.
    """

    vertices = vertices_array(shape)

    n_vertices = len(vertices)

    if n_vertices < 4:
        raise ValueError("D4 requires at least 4 vertices")

    if not np.isfinite(vertices).all():
        raise ValueError("D4 contains invalid vertices")

    histogram = np.zeros(NUM_BINS, dtype=np.int64)
    bin_edges = np.linspace(0, 1, NUM_BINS + 1)

    remaining = NUM_SAMPLES

    while remaining > 0:
        batch_size = min(remaining, 50_000)

        # Generate random vertex indices
        indices = np.random.randint(
            0, n_vertices, size=(batch_size, 4)
        )

        # Keep only triplets of four distinct indices
        sorted_indices = np.sort(indices, axis=1)

        valid = np.all(
            np.diff(sorted_indices, axis=1) != 0,
            axis=1
        )

        indices = indices[valid]

        if len(indices) == 0:
            continue

        p1 = vertices[indices[:, 0]]
        p2 = vertices[indices[:, 1]]
        p3 = vertices[indices[:, 2]]
        p4 = vertices[indices[:, 3]]

        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1

        # Calculate tetrahedron volume
        cross = np.cross(v2, v3)

        volumes = np.abs(
            np.einsum("ij,ij->i", v1, cross)
        ) / 6.0

        # D4 = cube root of volume
        d4_values = np.cbrt(volumes)

        # Normalize with a shared fixed scale
        normalized = d4_values / D4_MAX

        if not np.isfinite(normalized).all():
            raise ValueError("Invalid D4 measurements")
        dimensions = np.ptp(vertices, axis=0)

        print("AABB dimensions:", dimensions)
        print("Maximum AABB dimension:", dimensions.max())
        if np.any(normalized > 1 + 1e-9):
            print(
                "D4 exceeds unit-cube range. "
                "Check mesh scaling."
            )

        normalized = np.minimum(normalized, 1.0)

        counts, _ = np.histogram(
            normalized,
            bins=bin_edges
        )

        histogram += counts
        remaining -= len(indices)

    histogram = histogram.astype(float)
    histogram /= histogram.sum()

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