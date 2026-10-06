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
        for line in open(filename):
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
                    if '/' in token:
                        t = token.split("/")
                        v = int(t[0]) - 1  # iterating from 0 instead of 1
                        # texture = int(t[1]) -1    # Empty, hopefully
                        vn = int(t[2]) - 1  # iterating from 0 instead of 1
                        face.append((v, vn))
                    else:
                        v = int(token) - 1  # iterating from 0 instead of 1
                        face.append((v, None))
                self.F.append(face)
                self.num_faces += 1

    def _load_from_mesh(self, mesh):
        m = self.mesh.current_mesh()
        self.V = [tuple(v) for v in m.vertex_matrix()]
        self.F = [[(int(i), None) for i in face] for face in m.face_matrix()]
        self.num_vertices = len(self.V)
        self.num_faces = len(self.F)

    def write(self):
        path = Path(self.filename)
        output_path = Path("ResampledShapeDatabase") / path.parent.name / path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = np.array(self.V, dtype=np.float64)
        faces = np.array([[corner[0] for corner in face] for face in self.F], dtype=np.int32)

        self.mesh.clear()
        self.mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        self.mesh.compute_normal_per_vertex()  # rebuild normals for the new geometry

        self.mesh.save_current_mesh(str(output_path), save_vertex_normal=True)

        return output_path