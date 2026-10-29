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

def check_align(shape, tol=0.01):
    V = np.array(shape.V)
    faces = np.array([(f[0][0], f[1][0], f[2][0]) for f in shape.F])

    a = V[faces[:, 0]]
    b = V[faces[:, 1]]
    c = V[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    if areas.sum() == 0:
        # align() leaves such shapes untouched, so there is nothing to check
        return True, "largest eigenvector angle to X axis=n/a (zero area)"

    # Same covariance as in align(): area-weighted covariance of face centers
    center_points = (a + b + c) / 3
    barycenter = (areas.reshape(-1, 1) * center_points).sum(axis=0) / areas.sum()
    centers = center_points - barycenter
    cov = (areas.reshape(-1, 1) * centers).T @ centers / areas.sum()

    evals, evecs = np.linalg.eigh(cov)
    largest = evecs[:, np.argmax(evals)]

    # Eigenvectors have an arbitrary sign, so use the absolute cosine
    cos_angle = np.clip(abs(largest[0]), 0.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))



    return (angle <= tol and np.linalg.norm(barycenter) < tol,
            f"largest eigenvector angle to X axis={angle} degrees; "
            f"barycenter distance from origin={np.linalg.norm(barycenter)}")

