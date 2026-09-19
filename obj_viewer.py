import sys
import math

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
                t = token.split("/")
                v = int(t[0]) - 1       # iterating from 0 instead of 1
                texture = int(t[1])     # Empty, I hope
                vn = int(t[2]) - 1      # iterating from 0 instead of 1
                face.append((v, vn))
            F.append(face)

        return V, VN, F


def draw_obj(filename):
    V, VN, F = load_obj(filename)

    # Empty file handling
    if not V:
        print('No vertices found')
        return




