import os

folder_a = r"../ShapeDatabase"
folder_b = r"../ResampledShapeDatabase"

def get_files(folder):
    file_paths = set()

    for root, dirs, files in os.walk(folder):
        # Output the subfolder currently being scanned
        print(f"Scanning folder: {root}")

        for file in files:
            relative_path = os.path.relpath(
                os.path.join(root, file), folder
            )
            file_paths.add(relative_path)

    return file_paths

print("\n--- Scanning Folder A ---")
files_a = get_files(folder_a)

print("\n--- Scanning Folder B ---")
files_b = get_files(folder_b)

# Compare relative paths
missing = sorted(files_a - files_b)

print(f"\nFound {len(missing)} files missing from Folder B:\n")

for path in missing:
    print(path)