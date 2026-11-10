import numpy as np
import matplotlib.pyplot as plt

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

    # Preview only: limit faces for responsive display
    if len(faces) > 5000:
        indices = np.linspace(
            0, len(faces) - 1, 5000, dtype=int
        )
        faces = faces[indices]

    # Center and scale for visualization only
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices - center

    extent = np.ptp(vertices, axis=0).max()
    if extent > 0:
        vertices = vertices / extent

    triangles = vertices[faces]

    mesh = Poly3DCollection(
        triangles,
        facecolor="black",
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
    from pathlib import Path

    database = Path("RescaledShapeDatabase")

    # Find OBJ files from the database
    files = sorted(database.glob("*/*.obj"))

    if len(files) < 6:
        raise ValueError("Need at least 6 OBJ files for testing")

    # First object as query
    query = str(files[0])

    # Five dummy retrieval results
    results = [
        (str(path), float(i + 1) / 10)
        for i, path in enumerate(files[1:6])
    ]

    show_results(query, results)