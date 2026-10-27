from pathlib import Path
import csv

from shape import Shape

from FeatureExtraction.obj_surface_area import calculate_surface_area
from FeatureExtraction.obj_compactness import calculate_compactness
from FeatureExtraction.obj_aabb_volume import calculate_aabb_volume
from FeatureExtraction.obj_diameter import calculate_diameter
from FeatureExtraction.obj_eccentricity import calculate_eccentricity

from FeatureExtraction.obj_a3 import calculate_a3
from FeatureExtraction.obj_d1 import calculate_d1
from FeatureExtraction.obj_d2 import calculate_d2
from FeatureExtraction.obj_d3 import calculate_d3
from FeatureExtraction.obj_d4 import calculate_d4


NUM_BINS = 10


def histogram_columns(name):
    """
    Create column names
    """

    return [
        f"{name}_{i + 1}"
        for i in range(NUM_BINS)
    ]


def get_csv_header():
    """
    Columns of the feature database.
    """

    header = [
        "class",
        "filename",

        # Scalar descriptors
        "surface_area",
        "compactness",
        "aabb_volume",
        "diameter",
        "eccentricity",
    ]

    # Histogram descriptors
    header += histogram_columns("a3")
    header += histogram_columns("d1")
    header += histogram_columns("d2")
    header += histogram_columns("d3")
    header += histogram_columns("d4")

    return header


def extract_features(filename):
    """
    Extract all features from one OBJ file.
    """

    shape = Shape(str(filename))

    # ---------------------------------
    # Scalar descriptors
    # ---------------------------------

    surface_area = calculate_surface_area(
        shape
    )

    compactness = calculate_compactness(
        shape
    )

    aabb_volume = calculate_aabb_volume(
        shape
    )

    diameter = calculate_diameter(
        shape
    )

    eccentricity = calculate_eccentricity(
        shape
    )

    # ---------------------------------
    # Histogram descriptors
    # ---------------------------------

    a3, _ = calculate_a3(
        shape
    )

    d1, _ = calculate_d1(
        shape
    )

    d2, _ = calculate_d2(
        shape
    )

    d3, _ = calculate_d3(
        shape
    )

    d4, _ = calculate_d4(
        shape
    )

    return {
        "surface_area": surface_area,
        "compactness": compactness,
        "aabb_volume": aabb_volume,
        "diameter": diameter,
        "eccentricity": eccentricity,

        "a3": a3,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
    }


def create_csv_row(
    class_name,
    filename,
    features
):
    """
    Convert extracted features into one flat CSV row.
    """

    row = [
        class_name,
        filename.name,

        # Scalar features
        features["surface_area"],
        features["compactness"],
        features["aabb_volume"],
        features["diameter"],
        features["eccentricity"],
    ]

    # Histogram bins
    row.extend(features["a3"])
    row.extend(features["d1"])
    row.extend(features["d2"])
    row.extend(features["d3"])
    row.extend(features["d4"])

    return row


def extract_database(
    database_folder,
    output_csv
):
    """
    Extract features from all OBJ files.
    """

    database_folder = Path(
        database_folder
    )

    output_csv = Path(
        output_csv
    )

    obj_files = sorted(
        database_folder.rglob("*.obj")
    )

    if len(obj_files) == 0:
        print(
            f"No OBJ files found in: "
            f"{database_folder}"
        )
        return

    print(
        f"Found {len(obj_files)} OBJ files."
    )

    header = get_csv_header()

    with open(
        output_csv,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            header
        )

        for index, filename in enumerate(
            obj_files,
            start=1
        ):

            # Parent folder is the class name.
            class_name = (
                filename.parent.name
            )

            print(
                f"[{index}/{len(obj_files)}] "
                f"{class_name}/"
                f"{filename.name}"
            )

            try:

                features = extract_features(
                    filename
                )

                row = create_csv_row(
                    class_name,
                    filename,
                    features
                )

                writer.writerow(
                    row
                )

            except Exception as error:

                print(
                    f"ERROR: "
                    f"{filename}"
                )

                print(
                    error
                )

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION FINISHED")
    print("=" * 60)
    print(
        f"CSV saved to: {output_csv}"
    )


if __name__ == "__main__":

    database_folder = (
        "ResampledShapeDatabase/Hat"         #check only 1 class
    )

    output_csv = (
        "shape_features.csv"
    )

    extract_database(
        database_folder,
        output_csv
    )