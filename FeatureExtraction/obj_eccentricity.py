import numpy as np

from shape import Shape
from FeatureExtraction.obj_mesh_utils import vertices_array


def calculate_eccentricity(shape):
    vertices = vertices_array(shape)

    covariance_matrix = np.cov(
        vertices,
        rowvar=False
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_matrix
    )

    eigenvalues = np.sort(
        np.abs(eigenvalues)
    )[::-1]

    eccentricity = (                    # E = |lambda_1| / |lambda_3|
        eigenvalues[0] / eigenvalues[2]
    )
    return float(eccentricity)


def test_eccentricity(filename):
    shape = Shape(filename)
    vertices = vertices_array(shape)
    covariance_matrix = np.cov(
        vertices,
        rowvar=False
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_matrix
    )

    eigenvalues = np.sort(
        np.abs(eigenvalues)
    )[::-1]

    eccentricity = calculate_eccentricity(
        shape
    )

    print(f"Lambda 1: {eigenvalues[0]}")
    print(f"Lambda 2: {eigenvalues[1]}")
    print(f"Lambda 3: {eigenvalues[2]}")
    print(f"Eccentricity: {eccentricity}")

if __name__ == "__main__":

    filename = (
        # "../ResampledShapeDatabase/Car/m1487.obj"
        # "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
        "../ResampledShapeDatabase/PlantIndoors/m998.obj"
    )

    test_eccentricity(
        filename
    )