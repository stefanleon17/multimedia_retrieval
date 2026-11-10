import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shape import Shape


def draw_shape(ax, filename):
    shape = Shape(str(filename))

    vertices = np.asarray(shape.V, dtype=float)
    faces = np.array(
        [[v for v, vn in face] for face in shape.F],
        dtype=int
    )

    # Center and scale for visualization only
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices - center

    extent = np.ptp(vertices, axis=0).max()
    if extent > 0:
        vertices = vertices / extent

    triangles = vertices[faces]

    # Compute face normals
    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(v1, v2)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] = normals[valid] / lengths[valid][:, None]

    # Simple light direction
    light_dir = np.array([1.0, 1.0, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Brightness from dot product
    brightness = normals @ light_dir
    brightness = np.clip(brightness, 0, 1)

    # Make it grey and still visible
    brightness = 0.35 + 0.45 * brightness

    facecolors = np.stack(
        [brightness, brightness, brightness],
        axis=1
    )

    mesh = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolor="none"
    )

    ax.add_collection3d(mesh)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=45)
    ax.set_axis_off()


def show_results(query_path, results):
    fig = plt.figure(figsize=(15, 6))

    # Query + top K results
    items = [(query_path, None)] + results

    for i, (path, distance) in enumerate(items):
        ax = fig.add_subplot(
            1, len(items), i + 1,
            projection="3d"
        )

        draw_shape(ax, path)

        name = Path(path).name
        shape_class = Path(path).parent.name

        if i == 0:
            ax.set_title(f"QUERY\n{shape_class}\n{name}")
        else:
            ax.set_title(
                f"#{i}  {shape_class}\n"
                f"{name}\nDistance: {distance:.4f}"
            )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from Matching.Matching import FeatureDatabase

    if len(sys.argv) < 2:
        raise ValueError("Please provide an OBJ file path")

    query_path = sys.argv[1]
    k = 5

    database = FeatureDatabase("Matching/features.csv")

    query_idx = database.find_index(query_path)

    if query_idx is None:
        raise ValueError(f"Query not found in feature database: {query_path}")

    # Retrieve more candidates to ensure we can exclude the query
    ranked = database.retrieve(query_idx, k=k + 1)

    results = []

    for idx, distance in ranked:
        if idx == query_idx:
            continue

        # Resolve path relative to project root
        result_path = str(Path(database.paths[idx]))

        results.append((result_path, distance))

        if len(results) == k:
            break

    show_results(query_path, results)