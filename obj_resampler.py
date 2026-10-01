from pathlib import Path
import pymeshlab
import numpy as np

TARGET_VERTICES = 10000
TOLERANCE = 2000

def resample(input_path,
                 target=TARGET_VERTICES,
                 tolerance=TOLERANCE):
    path = Path(input_path)
    class_name = path.parent.name
    name = path.name

    output_path = Path("./ResampledShapeDatabase") / class_name / name

    folder = Path("ResampledShapeDatabase") / class_name
    if not folder.exists():
        folder.mkdir()

    mesh = pymeshlab.MeshSet()
    mesh.load_new_mesh(input_path)

    # # Clean-up
    mesh.meshing_remove_duplicate_vertices()
    mesh.meshing_remove_duplicate_faces()
    mesh.meshing_remove_unreferenced_vertices()
    mesh.meshing_repair_non_manifold_edges(method='Split Vertices')
    mesh.meshing_repair_non_manifold_vertices()
    mesh.meshing_remove_unreferenced_vertices()

    # Subdivide until we have more than 10,000 vertices
    while mesh.current_mesh().vertex_number() < (target - tolerance):
        mesh.meshing_surface_subdivision_midpoint(iterations=1)

    ### DEBUG
    # m = mesh.current_mesh()
    # v, f = m.vertex_number(), m.face_number()
    # print(f"before decimation: {v} vertices, {f} faces, ratio {f / v:.2f}")
    #
    # faces = m.face_matrix()
    # unique_faces = np.unique(np.sort(faces, axis=1), axis=0)
    # print(f"duplicate faces: {len(faces) - len(unique_faces)}")
    #
    # print(mesh.get_topological_measures())

    # Decimate: for a closed manifold mesh, faces ~ 2 * vertices
    target_faces = 100 + 2 * target

    # Simplify the mesh. Only first simplification will be agressive
    while mesh.current_mesh().vertex_number() > (target + tolerance):
        mesh.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
            autoclean=False
        )
        print("Decimated to", target_faces, "faces mesh has", mesh.current_mesh().vertex_number(), "vertex")
        # Refine our estimation to slowly converge to target vertex number
        target_faces -= (mesh.current_mesh().vertex_number() - target)

    ### DEBUG
    # m = mesh.current_mesh()
    # v, f = m.vertex_number(), m.face_number()
    # print(f"before decimation: {v} vertices, {f} faces, ratio {f / v:.2f}")
    #
    # faces = m.face_matrix()
    # unique_faces = np.unique(np.sort(faces, axis=1), axis=0)
    # print(f"duplicate faces: {len(faces) - len(unique_faces)}")
    #
    # print(mesh.get_topological_measures())

    print(output_path, str(output_path))
    mesh.save_current_mesh(str(output_path))
    print(f"{input_path}: {mesh.current_mesh().vertex_number()} vertices")

    return output_path
