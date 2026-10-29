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

def check_flip(shape, rel_tol=1e-9):
    V = np.array(shape.V, dtype=float)

    # Same as flip(): only triangles are considered
    faces = np.array([
        (f[0][0], f[1][0], f[2][0])
        for f in shape.F
        if len(f) == 3
    ])

    if len(faces) == 0:
        return True, "flip moment check skipped (no triangles)"

    a = V[faces[:, 0]]
    b = V[faces[:, 1]]
    c = V[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    centers = (a + b + c) / 3.0

    # Same moment as flip(): sum of sign(c) * area * c^2, per axis
    moments = (np.sign(centers) * areas[:, None] * centers**2).sum(axis=0)

    # Scale for the tolerance: the total unsigned moment per axis.
    # Symmetric shapes can give tiny negative values from float noise,
    # which shouldn't count as a failure.
    scale = (areas[:, None] * centers**2).sum(axis=0)
    ok_per_axis = moments >= -rel_tol * scale

    axis_names = "XYZ"
    parts = [
        f"{axis_names[i]}={moments[i]:.6g} ({'ok' if ok_per_axis[i] else 'FAIL'})"
        for i in range(3)
    ]

    return bool(ok_per_axis.all()), "moments: " + ", ".join(parts)

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