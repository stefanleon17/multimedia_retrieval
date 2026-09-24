##################################
# Code for loading a .obj file
##################################

def load_obj(filename):
    V = []
    VN = []
    F = []

    num_vertices = 0
    num_faces = 0

    for line in open(filename):
        data = line.split()
        if not data or data[0] == '#':
            continue
        if data[0] == 'v':
            V.append(tuple(float(i) for i in data[1:4]))
            num_vertices += 1
        elif data[0] == 'vn':
            VN.append(tuple(float(i) for i in data[1:4]))
        elif data[0] == 'f':
            face = []
            for token in data[1:]:
                if '/' in token:
                    t = token.split("/")
                    v = int(t[0]) - 1           # iterating from 0 instead of 1
                    # texture = int(t[1]) -1    # Empty, I hope
                    vn = int(t[2]) - 1          # iterating from 0 instead of 1
                    face.append((v, vn))
                else:
                    v = int(token) - 1          # iterating from 0 instead of 1
                    face.append((v, None))
            F.append(face)
            num_faces += 1

    return V, VN, F, num_vertices, num_faces