import numpy as np

from shape import Shape
from Preprocessing.obj_alignment import align
from Preprocessing.obj_hole_filling import fill_holes
from Preprocessing.obj_triangle_orientation import build_edge_lookup, edge_key


def tetrahedron_volume(v1, v2, v3, barycenter):
    """
    Calculate the signed volume of the tetrahedron formed by:

        barycenter
        v1
        v2
        v3

    The sign depends on the orientation of the triangle.
    """

    a = v1 - barycenter
    b = v2 - barycenter
    c = v3 - barycenter

    return np.dot(a, np.cross(b, c)) / 6.0


def get_connected_components(shape):
    """
    Find all continuous components in the mesh.

    Two triangles belong to the same component if they share
    an edge.

    Returns:
        A list of components.

        Each component is a list of triangle indices.
    """

    edge_to_faces = build_edge_lookup(shape)

    visited = set()
    components = []

    for start_face in range(len(shape.F)):

        if start_face in visited:
            continue

        component = []
        queue = [start_face]
        visited.add(start_face)

        while queue:

            current_face_index = queue.pop(0)
            component.append(current_face_index)

            current_face = shape.F[current_face_index]

            vertices = [
                corner[0]
                for corner in current_face
            ]

            edges = [
                (vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[2], vertices[0])
            ]

            for edge in edges:

                neighbouring_faces = edge_to_faces[
                    edge_key(*edge)
                ]

                for neighbour_index in neighbouring_faces:

                    if neighbour_index in visited:
                        continue

                    visited.add(neighbour_index)
                    queue.append(neighbour_index)

        components.append(component)

    return components


def calculate_component_volume(shape, component, barycenter):
    """
    Calculate the signed volume of one continuous component.
    """

    volume = 0.0

    for face_index in component:

        face = shape.F[face_index]

        v1 = np.array(shape.V[face[0][0]], dtype=float)
        v2 = np.array(shape.V[face[1][0]], dtype=float)
        v3 = np.array(shape.V[face[2][0]], dtype=float)

        volume += tetrahedron_volume(
            v1,
            v2,
            v3,
            barycenter
        )

    return volume


def flip_component(shape, component):
    """
    Flip the orientation of every triangle in a component.
    """

    for face_index in component:
        shape.F[face_index] = list(
            reversed(shape.F[face_index])
        )


def calculate_total_volume(shape):
    """
    Fill holes and calculate the total volume of all continuous
    components in the mesh.

    Components with negative signed volume are flipped so that
    all
    component volumes contribute positively.

    Returns:
        Total volume of all continuous components.
    """

    # Fill holes before calculating volume
    shape = fill_holes(shape)

    # Calculate the barycenter of the complete mesh
    shape = align(shape)

    barycenter = np.zeros(3)

    # Get all continuous components
    components = get_connected_components(shape)

    print(f"Found {len(components)} continuous shape(s).")

    total_volume = 0.0

    for component_index, component in enumerate(components):

        volume = calculate_component_volume(
            shape,
            component,
            barycenter
        )

        print(
            f"Shape {component_index + 1}: "
            f"{len(component)} triangles, "
            f"signed volume = {volume}"
        )

        if volume < 0:

            print(
                f"Shape {component_index + 1} has negative "
                f"orientation. Flipping it."
            )

            flip_component(shape, component)

            volume = -volume

        total_volume += volume

    print(f"Total volume: {total_volume}")

    return total_volume


if __name__ == "__main__":

    filename = r"../ShapeDatabase/Car/m1487.obj"

    shape = Shape(filename)

    volume = calculate_total_volume(shape)

    print(f"\nFinal volume: {volume}")