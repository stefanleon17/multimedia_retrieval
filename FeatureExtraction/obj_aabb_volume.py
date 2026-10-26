import numpy as np

from shape import Shape
from FeatureExtraction.obj_mesh_utils import vertices_array


def calculate_aabb_dimensions(shape):
    vertices = vertices_array(shape)
    minimum = np.min(
        vertices,
        axis=0
    )

    maximum = np.max(
        vertices,
        axis=0
    )

    dimensions = maximum - minimum
    return dimensions

def calculate_aabb_volume(shape):
    dimensions = calculate_aabb_dimensions(
        shape
    )

    volume = np.prod(                       # V = Dx * Dy * Dz
        dimensions
    )
    return float(volume)


def test_aabb_volume(filename):
    shape = Shape(filename)
    dimensions = calculate_aabb_dimensions(
        shape
    )
    volume = calculate_aabb_volume(
        shape
    )
    print(
        f"Dx: {dimensions[0]}"
    )

    print(
        f"Dy: {dimensions[1]}"
    )

    print(
        f"Dz: {dimensions[2]}"
    )

    print(
        f"AABB volume: {volume}"
    )


if __name__ == "__main__":

    filename = (
        "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    )

    test_aabb_volume(
        filename
    )