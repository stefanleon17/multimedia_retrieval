class Shape:
    def __init__(self, filename):
        self.V = []
        self.VN = []
        self.F = []

        self.num_vertices = 0
        self.num_faces = 0

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
                        v = int(t[0]) - 1           # iterating from 0 instead of 1
                        # texture = int(t[1]) -1    # Empty, hopefully
                        vn = int(t[2]) - 1          # iterating from 0 instead of 1
                        face.append((v, vn))
                    else:
                        v = int(token) - 1          # iterating from 0 instead of 1
                        face.append((v, None))
                self.F.append(face)
                self.num_faces += 1

    def __iter__(self):
        # Lets old code keep doing: V, VN, F, num_vertices, num_faces = load_obj(...)
        return iter((self.V, self.VN, self.F, self.num_vertices, self.num_faces))


def load(filename):
    return Shape(filename)