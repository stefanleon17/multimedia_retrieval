from collections import defaultdict
from shape import Shape

def edge_key(v1, v2):
    """
    An undirected edge.

    (a, b) and (b, a) represent the same edge.
    """
    return tuple(sorted((v1, v2)))


def get_boundary_edges(shape):
    """
    Find all edges that occur in exactly one triangle.

    Returns:
        boundary_edges:
            dictionary where:
            edge -> face index

        The face index tells us which existing triangle contains
        the boundary edge.
    """

    edge_faces = defaultdict(list)

    for face_index, face in enumerate(shape.F):

        v1 = face[0][0]
        v2 = face[1][0]
        v3 = face[2][0]

        edges = [
            (v1, v2),
            (v2, v3),
            (v3, v1)
        ]

        for a, b in edges:
            edge_faces[edge_key(a, b)].append(face_index)

    boundary_edges = {}

    for edge, faces in edge_faces.items():
        if len(faces) == 1:
            boundary_edges[edge] = faces[0]

    return boundary_edges


def find_boundary_loops(boundary_edges):
    """
    Group boundary edges into separate loops.

    boundary_edges has the form:

        (vertex_a, vertex_b) -> face_index

    Returns a list of loops.

    Each loop is a list of edges:

        [
            (a, b),
            (b, c),
            (c, d),
            (d, a)
        ]
    """

    # Build vertex -> boundary edges lookup
    vertex_edges = defaultdict(list)

    for edge in boundary_edges:
        a, b = edge

        vertex_edges[a].append(edge)
        vertex_edges[b].append(edge)

    unused_edges = set(boundary_edges.keys())
    loops = []

    while unused_edges:

        # Start a new loop
        start_edge = next(iter(unused_edges))
        unused_edges.remove(start_edge)

        loop = [start_edge]

        start_vertex = start_edge[0]
        current_vertex = start_edge[1]

        while current_vertex != start_vertex:

            # Find an unused boundary edge connected to current_vertex
            next_edge = None

            for edge in vertex_edges[current_vertex]:
                if edge in unused_edges:
                    next_edge = edge
                    break

            # No continuation means the boundary isn't a proper loop
            if next_edge is None:
                break

            unused_edges.remove(next_edge)
            loop.append(next_edge)

            a, b = next_edge

            if a == current_vertex:
                current_vertex = b
            else:
                current_vertex = a

        loops.append(loop)

    return loops


def get_directed_boundary_edge(shape, edge, face):
    """
    Find the direction in which the existing triangle traverses
    the boundary edge.

    For example, if the existing triangle contains:

        a -> b

    this returns (a, b).
    """

    vertices = [corner[0] for corner in face]

    for i in range(3):
        a = vertices[i]
        b = vertices[(i + 1) % 3]

        if edge_key(a, b) == edge_key(*edge):
            return a, b

    return None


def fill_holes(shape):
    """
    Find all holes in the mesh and fill each hole with triangles
    connecting its boundary loop to a new center vertex.

    The existing geometry is not changed.

    Returns:
        The modified Shape.
    """

    boundary_edges = get_boundary_edges(shape)

    # No holes / no boundary edges
    if not boundary_edges:
        print("No holes found.")
        return shape

    print(f"Found {len(boundary_edges)} boundary edges.")

    loops = find_boundary_loops(boundary_edges)

    print(f"Found {len(loops)} hole(s).")

    for loop_index, loop in enumerate(loops):

        # ---------------------------------------------------------
        # 1. Get all vertices belonging to this boundary loop
        # ---------------------------------------------------------

        vertices_in_loop = set()

        for edge in loop:
            vertices_in_loop.update(edge)

        # ---------------------------------------------------------
        # 2. Calculate barycenter of the boundary vertices
        # ---------------------------------------------------------

        center = [0.0, 0.0, 0.0]

        for vertex_index in vertices_in_loop:
            vertex = shape.V[vertex_index]

            center[0] += vertex[0]
            center[1] += vertex[1]
            center[2] += vertex[2]

        number_of_vertices = len(vertices_in_loop)

        center = (
            center[0] / number_of_vertices,
            center[1] / number_of_vertices,
            center[2] / number_of_vertices
        )

        # ---------------------------------------------------------
        # 3. Add the center as a new vertex
        # ---------------------------------------------------------

        center_index = len(shape.V)
        shape.V.append(center)
        shape.num_vertices += 1

        # ---------------------------------------------------------
        # 4. Create one triangle for every boundary edge
        # ---------------------------------------------------------

        for edge in loop:

            boundary_face_index = boundary_edges[edge]

            boundary_face = shape.F[boundary_face_index]

            # Determine the direction of the boundary edge
            # in the already-oriented existing triangle.
            directed_edge = get_directed_boundary_edge(
                shape,
                edge,
                boundary_face
            )

            if directed_edge is None:
                continue

            a, b = directed_edge

            # Existing triangle traverses:
            #
            #     a -> b
            #
            # Therefore the new triangle must traverse the
            # shared edge in the opposite direction:
            #
            #     b -> a
            #
            new_face = [
                (b, None),
                (a, None),
                (center_index, None)
            ]

            shape.F.append(new_face)
            shape.num_faces += 1

        print(
            f"Filled hole {loop_index + 1}: "
            f"{len(loop)} new triangles."
        )

    return shape


if __name__ == "__main__":

    # -------------------------------------------------------------
    # Change this to the OBJ you want to test
    # -------------------------------------------------------------

    filename = r"../ShapeDatabase/AircraftBuoyant/m1337.obj"

    shape = Shape(filename)

    print(f"Loaded {shape.num_faces} triangles.")

    shape = fill_holes(shape)

    print(f"Final number of triangles: {shape.num_faces}")

