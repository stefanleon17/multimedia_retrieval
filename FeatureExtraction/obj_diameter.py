import numpy as np

from shape import Shape
from FeatureExtraction.obj_mesh_utils import vertices_array


def calculate_diameter(shape):
    vertices = vertices_array(shape)
    max_distance = 0.0

    for vertex in vertices:
        differences = (vertices-vertex)
        distances = np.linalg.norm(differences,axis=1)
        current_max = np.max(distances)

        if current_max > max_distance:
            max_distance = current_max

    return float(max_distance)


def test_diameter(filename):
    shape = Shape(filename)
    diameter = calculate_diameter(shape)
    print(f"Diameter: {diameter}")

if __name__ == "__main__":

    filename = (
        # "../ResampledShapeDatabase/Car/m1487.obj"
        # "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
        "../ResampledShapeDatabase/PlantIndoors/m998.obj"
    )

    test_diameter(
        filename
    )