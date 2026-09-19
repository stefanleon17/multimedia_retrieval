import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def load_obj(filename):
    V = []
    VN = []
    F = []

    for line in open(filename):
        data = line.split()
        if not data or data[0] == '#':
            continue
        if data[0] == 'v':
            V.append(tuple(float(i) for i in data[1:4]))
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

    return V, VN, F

def view(width, height, V, VN, F):

    # Perspective & viweport parameters
    lo = [min(v[i] for v in V) for i in range(3)]
    hi = [max(v[i] for v in V) for i in range(3)]
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    radius = max(max(hi[i] - lo[i] for i in range(3)) * 0.5, 1e-6)

    distance = radius * 3.0
    eye = (center[0] + distance, center[1] + distance, center[2] + distance)
    up_x, up_y, up_z = 0, 0.5, 0

    fov = 100
    near = distance * 0.01
    far = distance + radius * 10
    aspect = float(width) / float(height)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2],center[0], center[1], center[2], up_x, up_y, up_z)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect, near, far)

    glViewport(0, 0, width, height)


def display(filename, width, height):
    V, VN, F = load_obj(filename)

    def draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for face in F:
            glBegin(GL_POLYGON)
            for v, vn in face:
                if vn is not None:
                    glNormal3f(*VN[vn])
                glVertex3f(*V[v])
            glEnd()

        glutSwapBuffers()

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(filename.encode())

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.16, 0.17, 0.19, 1.0)
    glColor3f(0.75, 0.76, 0.80)
    if VN:
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 1.0])

    view(width, height, V, VN, F)

    glutDisplayFunc(draw)
    glutMainLoop()


