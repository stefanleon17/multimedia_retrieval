from pathlib import Path
import pymeshlab
import numpy as np

class Shape:
    def __init__(self, filename, mesh_set = None):
        self.V = []
        self.VN = []
        self.F = []

        self.num_vertices = 0
        self.num_faces = 0
        self.filename = filename
        self.mesh = mesh_set

        if mesh_set is not None:
            self._load_from_mesh(mesh_set)
        else:
            self._load_from_file(filename)


    def _load_from_file(self, filename):
        with open(filename) as f:
            for line in f:
                data = line.split()
                if not data or data[0] == '#':
                    continue
                if data[0] == 'v':
                    self.V.append(tuple(float(i) for i in data[1:4]))
                    self.num_vertices += 1
                elif data[0] == 'vn':
                    self.VN.append(tuple(float(i) for i in data[1:4]))
                elif data[0] == 'f':
                    face = []
                    for token in data[1:]:
                        parts = token.split("/")
                        v = int(parts[0]) - 1
                        # parts[1] is the texture index (ignored)
                        vn = None
                        if len(parts) >= 3 and parts[2] != "":
                            vn = int(parts[2]) - 1
                        face.append((v, vn))
                    self.F.append(face)
                    self.num_faces += 1

    def _load_from_mesh(self, mesh=None):
        m = (mesh if mesh is not None else self.mesh).current_mesh()
        self.V = [tuple(v) for v in m.vertex_matrix()]
        self.F = [[(int(i), None) for i in face] for face in m.face_matrix()]
        self.num_vertices = len(self.V)
        self.num_faces = len(self.F)

    def write(self):
        if self.mesh is None:
            self.mesh = pymeshlab.MeshSet()

        path = Path(self.filename)
        project_root = Path(__file__).resolve().parent
        output_path = project_root / "ResampledShapeDatabase" / path.parent.name / path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = np.array(self.V, dtype=np.float64)
        faces = np.array([[c[0] for c in face] for face in self.F], dtype=np.int32)
        assert faces.ndim == 2 and faces.shape[1] == 3, "Non-triangular face encountered"

        self.mesh.clear()
        self.mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        self.mesh.compute_normal_per_vertex()

        self.mesh.save_current_mesh(str(output_path), save_vertex_normal=True)
        return output_path