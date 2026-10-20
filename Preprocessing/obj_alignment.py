from shape import Shape
import numpy as np

def align(shape):
    V = np.array(shape.V)

    faces = []
    for face in shape.F:
        v1 = face[0][0]
        v2 = face[1][0]
        v3 = face[2][0]
        faces.append((v1, v2, v3))
    faces = np.array(faces)

    a = V[faces[:, 0]]
    b = V[faces[:, 1]]
    c = V[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b-a, c-a), axis=1)

    if areas.sum() == 0:
        return shape

    center_points = (a + b + c) / 3
    barycenter = (areas.reshape(-1,1) * center_points).sum(axis=0) / areas.sum()

    # Translation
    translated = V - barycenter

    # Eigenvector alignment (area-weighted covariance of face centers)
    centers = center_points - barycenter
    cov = (areas.reshape(-1,1) * centers).T @ centers / areas.sum()
    evals, evecs = np.linalg.eigh(cov)

    # Sort axes by decreasing eigenvalue (align largest with x)
    evecs = evecs[:, np.argsort(evals)[::-1]]

    # Keep a proper rotation
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1

    rotated = translated @ evecs
    shape.V = [tuple(v) for v in rotated]

    return shape

