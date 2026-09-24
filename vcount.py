import os
import csv
from collections import Counter

dataset_folder = "../ShapeDatabase/"
output_csv = "obj_analysis.csv"

results = []

for class_name in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):

        if not filename.lower().endswith(".obj"):
            continue

        file_path = os.path.join(class_path, filename)

        vertex_count = 0
        face_sizes = Counter()

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                # Count vertices
                if parts[0] == "v":
                    vertex_count += 1

                # Count faces based on number of vertices
                elif parts[0] == "f":
                    face_vertex_count = len(parts) - 1

                    if face_vertex_count >= 3:
                        face_sizes[face_vertex_count] += 1

        triangles = face_sizes[3]
        quads = face_sizes[4]
        pentagons = face_sizes[5]
        hexagons = face_sizes[6]

        # 7 or more vertices
        larger_ngons = sum(
            count
            for size, count in face_sizes.items()
            if size >= 7
        )

        total_faces = sum(face_sizes.values())

        # Determine which polygon types exist
        detected_types = []

        for size in sorted(face_sizes):
            if face_sizes[size] > 0:
                if size == 3:
                    detected_types.append("Triangle")
                elif size == 4:
                    detected_types.append("Quad")
                elif size == 5:
                    detected_types.append("Pentagon")
                elif size == 6:
                    detected_types.append("Hexagon")
                else:
                    detected_types.append(f"{size}-gon")

        if detected_types:
            face_type = ", ".join(detected_types)
        else:
            face_type = "No faces"

        results.append([
            class_name,
            filename,
            vertex_count,
            triangles,
            quads,
            pentagons,
            hexagons,
            larger_ngons,
            total_faces,
            face_type
        ])


results.sort(
    key=lambda x: (x[0].lower(), x[1].lower())
)


with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Class",
        "Object",
        "Vertex Count",
        "Triangles",
        "Quads",
        "Pentagons",
        "Hexagons",
        "7+ sided polygons",
        "Total Faces",
        "Face Types"
    ])

    writer.writerows(results)


print(f"Done. Found {len(results)} OBJ files.")
print(f"CSV saved as: {output_csv}")