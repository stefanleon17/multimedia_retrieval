import math

from shape import Shape
from FeatureExtraction.obj_surface_area import calculate_surface_area
from FeatureExtraction.obj_volume_calculation import get_volume

def calculate_compactness(shape):
    if (
        shape is None
        or len(shape.V) == 0
        or len(shape.F) == 0
    ):
        return float("nan")

    area = calculate_surface_area(shape)
    volume = get_volume(shape)

    if area <= 0 or volume <= 0:
        return float("nan")

    compactness = (         # c = S^3 / (36 * pi * V^2)
        area ** 3
        /
        (
            36
            * math.pi
            * volume ** 2
        )
    )

    return float(compactness)       # 1 is perfect sphere


def test_compactness(filename):
    shape = Shape(filename)
    compactness = calculate_compactness(shape)

    print(f"Compactness:  {compactness}")


if __name__ == "__main__":

    filename = (
        "../ResampledShapeDatabase/AircraftBuoyant/m1337.obj"
    )

    test_compactness(filename)