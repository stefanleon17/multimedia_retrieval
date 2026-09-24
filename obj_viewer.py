##############################
# Code for drawing an (.obj) 3D object.
###############################

import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from obj_loader import load_obj

def view(width, height, V, VN, F):

    # Perspective & viweport parameters
    # Horrible magic numbers, needs looking over
    lo = [min(v[i] for v in V) for i in range(3)]
    hi = [max(v[i] for v in V) for i in range(3)]
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    radius = max(max(hi[i] - lo[i] for i in range(3)) * 0.5, 1e-6)

    distance = radius * 3.0
    eye = (center[0] + distance, center[1] + distance, center[2] + distance)
    up_x, up_y, up_z = 0, 0.5, 0

    fov = 50
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

    return center, radius


def display(filename, width, height):
    V, VN, F, num_vertices, num_faces = load_obj(filename)

    lo = [min(v[i] for v in V) for i in range(3)]
    hi = [max(v[i] for v in V) for i in range(3)]

    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    radius = max(max(hi[i] - lo[i] for i in range(3)) * 0.5, 1e-6)

    # for mouse interaction
    rot_x, rot_y = 0, 0
    pan_x, pan_y = 0, 0
    zoom = radius * 3.0
    last_x, last_y = 0, 0
    active_button = None

    # Function defined here for compatibility with gluDisplayFunc()
    def draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(pan_x, pan_y, -zoom)

        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)

        glTranslatef(
            -center[0],
            -center[1],
            -center[2]
        )


        for face in F:
            glBegin(GL_POLYGON)
            for v, vn in face:
                if vn is not None:
                    glNormal3f(*VN[vn])
                glVertex3f(*V[v])
            glEnd()

        def draw_text(x, y, text):
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, width, 0, height, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)

            glColor3f(1.0, 1.0, 1.0)
            glRasterPos2f(x, y)
            for ch in text:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

            if VN:
                glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

        draw_text(width - 150, 40, f"Vertices: {num_vertices}")
        draw_text(width - 150, 20, f"Faces: {num_faces}")

        glutSwapBuffers()

    def mouse(button, state, x, y):
        nonlocal active_button
        nonlocal last_x, last_y, zoom
        last_x, last_y = x, y

        if state == GLUT_DOWN:

            if button == 3:
                zoom *= 0.9             #zoom in
                glutPostRedisplay()
                return
            elif button == 4:
                zoom *= 1.1             #zoom out
                glutPostRedisplay()
                return

            active_button = button
        else: active_button = None

    def motion(x, y):

        nonlocal last_x, last_y
        nonlocal rot_x, rot_y
        nonlocal pan_x, pan_y

        dx = x - last_x
        dy = y - last_y

        # Left drag = rotate
        if active_button == GLUT_LEFT_BUTTON:
            rot_y += dx * 0.5
            rot_x += dy * 0.5

        # Right drag = pan
        elif active_button == GLUT_RIGHT_BUTTON:

            pan_speed = radius * 0.002

            pan_x += dx * pan_speed
            pan_y -= dy * pan_speed

        last_x = x
        last_y = y

        glutPostRedisplay()

    def keyboard(key, x, y):        # zoom in/out with keyboard, because scroll func didnt work on mac, idk
        nonlocal zoom

        if key in (b'+', b'='):
            zoom *= 0.9

        elif key == b'-':
            zoom *= 1.1

        glutPostRedisplay()

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
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutKeyboardFunc(keyboard)
    glutMainLoop()


