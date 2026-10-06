import random
from collections import defaultdict

from shape import Shape


def edge_key(v1, v2):
    """
    Return an edge in a canonical form.

    (a, b) and (b, a) represent the same undirected edge,
    so sorting the two vertex indices gives them the same key.
    """
    return tuple(sorted((v1, v2)))


def reverse_face(face):
    """
    Reverse the orientation of a face.

    A face contains (vertex_index, normal_index) tuples,
    so we reverse the complete tuples.
    """
    return list(reversed(face))


def build_edge_lookup(shape):
    """
    Build a dictionary mapping each undirected edge to the
    triangles that contain that edge.

    Example:
        (10, 25) -> [3, 17]

    means triangles 3 and 17 share the edge between
    vertices 10 and 25.
    """
    edge_to_faces = defaultdict(list)

    for face_index, face in enumerate(shape.F):
        # Every face is assumed to be a triangle
        v1 = face[0][0]
        v2 = face[1][0]
        v3 = face[2][0]

        edges = [
            (v1, v2),
            (v2, v3),
            (v3, v1)
        ]

        for v1, v2 in edges:
            edge_to_faces[edge_key(v1, v2)].append(face_index)

    return edge_to_faces


def get_shared_edge_direction(face, edge):
    """
    Determine the direction in which a face traverses a given edge.

    Returns:
        (a, b) if the face contains a -> b
        (b, a) if the face contains b -> a
        None if the edge isn't part of the face.
    """
    vertices = [corner[0] for corner in face]

    for i in range(3):
        a = vertices[i]
        b = vertices[(i + 1) % 3]

        if {a, b} == set(edge):
            return a, b

    return None


def orient_shape(shape):
    """
    Orient all triangles consistently.

    A random triangle is chosen as the starting triangle and
    considered correctly oriented. Orientation is then propagated
    through neighbouring triangles.

    The mesh geometry (shape.V) is not changed. Only the ordering
    of vertices in shape.F is changed.
    """

    if not shape.F:
        return shape

    # Build edge -> face lookup
    edge_to_faces = build_edge_lookup(shape)

    # Pick a random starting triangle
    start_face = random.randrange(len(shape.F))

    # Keep track of triangles whose orientation has been fixed
    oriented = {start_face}

    # Triangles that still need to be processed
    queue = [start_face]

    while queue:

        current_index = queue.pop(0)
        current_face = shape.F[current_index]

        current_vertices = [corner[0] for corner in current_face]

        # The three directed edges of the current triangle
        current_edges = [
            (current_vertices[0], current_vertices[1]),
            (current_vertices[1], current_vertices[2]),
            (current_vertices[2], current_vertices[0])
        ]

        for edge in current_edges:

            # Find all triangles sharing this edge
            neighbouring_faces = edge_to_faces[edge_key(*edge)]

            for neighbour_index in neighbouring_faces:

                # Don't process the current triangle or an already
                # oriented triangle.
                if neighbour_index == current_index:
                    continue

                if neighbour_index in oriented:
                    continue

                neighbour_face = shape.F[neighbour_index]

                # Determine how the neighbour traverses the shared edge
                neighbour_direction = get_shared_edge_direction(
                    neighbour_face,
                    edge
                )

                # Correct orientation requires opposite directions
                #
                # Current:   a -> b
                # Correct neighbour: b -> a
                #
                # If neighbour also has a -> b, it must be flipped.
                if neighbour_direction == edge:
                    shape.F[neighbour_index] = reverse_face(
                        neighbour_face
                    )

                # Mark the neighbour as oriented
                oriented.add(neighbour_index)

                # Process its neighbours later
                queue.append(neighbour_index)

    return shape


if __name__ == "__main__":

    # Change this to one OBJ from your ShapeDatabase
    filename = r"ShapeDatabase/Car/m1487.obj"

    shape = Shape(filename)

    print(f"Loaded {shape.num_faces} triangles.")

    shape = orient_shape(shape)

    print("Orientation finished.")

    # Shape.write() saves to ResampledShapeDatabase/
    output_path = shape.write()

    print(f"Saved oriented mesh to: {output_path}")