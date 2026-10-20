import numpy as np


def flip(vertices, faces):
    vertices = np.array(vertices, dtype=float)
    flipped = vertices.copy()

    moments = np.zeros(3)

    for face in faces:
        # We only need the vertex indices
        indices = [item[0] for item in face]

        # If the face is not a triangle, skip it for now
        if len(indices) != 3:
            continue

        v1 = vertices[indices[0]]
        v2 = vertices[indices[1]]
        v3 = vertices[indices[2]]

        # Triangle center
        center = (v1 + v2 + v3) / 3.0

        # Triangle area
        edge1 = v2 - v1
        edge2 = v3 - v1

        area = 0.5 * np.linalg.norm(
            np.cross(edge1, edge2)
        )

        # Moment test for X, Y, Z
        for axis in range(3):
            c = center[axis]

            moments[axis] += (
                np.sign(c)
                * area
                * c**2
            )

    # Flip necessary axes
    for axis in range(3):
        if moments[axis] < 0:
            flipped[:, axis] *= -1

    # print("Moment X:", moments[0])
    # print("Moment Y:", moments[1])
    # print("Moment Z:", moments[2])

    return [tuple(v) for v in flipped]

def scale(vertices, eps=1e-12):
    vertices = np.array(vertices, dtype=float)

    if vertices.size == 0:
        return vertices

    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    dimensions = maximum - minimum
    largest_dimension = float(np.max(dimensions))

    if largest_dimension < eps:
        # Degenerate shape (all vertices coincide, or the file loaded wrong).
        # Return an unmodified copy so downstream code never sees inf/NaN.
        print(f"WARNING: scale(): largest dimension {largest_dimension} "
              f"< {eps}, skipping scaling.")
        return vertices

    return vertices / largest_dimension