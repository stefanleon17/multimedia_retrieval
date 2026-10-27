# FeatureExtraction/obj_histogram_utils.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def resolve_obj_files(inputs):
    """
    Resolve flexible inputs into a list of OBJ file paths.

    Accepted input:
    - single file path
    - single folder path
    - list/tuple of files and/or folders

    If a folder is given, all .obj files directly inside it are used.
    """

    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    obj_files = []

    for item in inputs:
        path = Path(item)

        if path.is_file():
            if path.suffix.lower() == ".obj":
                obj_files.append(path)

        elif path.is_dir():
            obj_files.extend(
                sorted(path.glob("*.obj"))
            )

    return obj_files


def plot_single_histogram(
    histogram,
    bin_edges,
    title,
    xlabel
):
    """
    Plot one histogram as a bar chart.
    """

    bin_centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2

    bin_widths = np.diff(
        bin_edges
    )

    plt.figure()

    plt.bar(
        bin_centers,
        histogram,
        width=bin_widths,
        align="center"
    )

    plt.xlabel(xlabel)
    plt.ylabel("Normalized frequency")
    plt.title(title)

    plt.show()


def plot_multiple_histograms(
    results,
    title,
    xlabel
):
    """
    Plot multiple histograms as line charts.

    results:
        list of (label, histogram, bin_edges)
    """

    plt.figure()

    for label, histogram, bin_edges in results:

        bin_centers = (
            bin_edges[:-1]
            + bin_edges[1:]
        ) / 2

        plt.plot(
            bin_centers,
            histogram,
            marker="o",
            label=label
        )

    plt.xlabel(xlabel)
    plt.ylabel("Normalized frequency")
    plt.title(title)
    plt.legend()
    plt.show()


def test_histogram_descriptor(
    inputs,
    descriptor_function,
    title,
    xlabel
):
    """
    Generic test function for histogram descriptors.

    Behavior:
    - 1 OBJ  -> bar chart
    - >1 OBJ -> line comparison plot

    descriptor_function must accept a Shape object
    and return:
        histogram, bin_edges
    """

    from shape import Shape

    obj_files = resolve_obj_files(
        inputs
    )

    if len(obj_files) == 0:
        print("No OBJ files found.")
        return

    results = []

    for file_path in obj_files:

        shape = Shape(str(file_path))

        histogram, bin_edges = descriptor_function(
            shape
        )

        results.append(
            (
                file_path.stem,
                histogram,
                bin_edges
            )
        )

    if len(results) == 1:

        label, histogram, bin_edges = results[0]

        print("=" * 50)
        print(title.upper())
        print("=" * 50)
        print(f"File: {obj_files[0]}")
        print()

        print("Histogram:")

        for i, value in enumerate(histogram):
            print(
                f"Bin {i + 1}: {value:.6f}"
            )

        print()
        print(
            f"Histogram sum: "
            f"{np.sum(histogram):.6f}"
        )
        print("=" * 50)

        plot_single_histogram(
            histogram,
            bin_edges,
            f"{title} - {label}",
            xlabel
        )

    else:

        print("=" * 50)
        print(title.upper() + " COMPARISON")
        print("=" * 50)
        print("Compared files:")

        for file_path in obj_files:
            print(file_path)

        print("=" * 50)

        plot_multiple_histograms(
            results,
            f"{title} Comparison",
            xlabel
        )