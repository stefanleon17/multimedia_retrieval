from pathlib import Path

from shape import Shape
from obj_hole_filling import get_boundary_edges


DATABASE_PATH = Path("ShapeDatabase")


def count_shapes_with_holes():
    total_shapes = 0
    shapes_with_holes = 0

    for category_folder in DATABASE_PATH.iterdir():

        if not category_folder.is_dir():
            continue

        for filename in category_folder.glob("*.obj"):

            total_shapes += 1

            shape = Shape(filename)

            boundary_edges = get_boundary_edges(shape)

            if boundary_edges:
                shapes_with_holes += 1

                print(
                    f"Hole found: {filename} "
                    f"({len(boundary_edges)} boundary edges)"
                )

    print()
    print("===================================")
    print(f"Total shapes:        {total_shapes}")
    print(f"Shapes with holes:   {shapes_with_holes}")
    print("Shapes without holes:", total_shapes - shapes_with_holes)
    print("===================================")


if __name__ == "__main__":
    count_shapes_with_holes()