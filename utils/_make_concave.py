# make_dented_half_sphere.py
import math
from pathlib import Path

SRC = Path("TestShapesResampled/half_sphere.obj")
DST = Path("TestShapesResampled/half_sphere_dented.obj")
C = 0.25            # how far the cap's center vertex is pushed into the dome (must be < R = 0.5)

lines = SRC.read_text().splitlines()
v_idx = [i for i, l in enumerate(lines) if l.startswith("v ")]
coords = [tuple(float(x) for x in lines[i].split()[1:4]) for i in v_idx]

z_min = min(z for _, _, z in coords)
on_plane = [k for k, (_, _, z) in enumerate(coords) if abs(z - z_min) < 2e-3]
center = min(on_plane, key=lambda k: math.hypot(coords[k][0], coords[k][1]))

x, y, z = coords[center]
lines[v_idx[center]] = f"v {x:.9f} {y:.9f} {z + C:.9f}"
DST.write_text("\n".join(lines) + "\n")
print(f"moved vertex {center} from z={z:.4f} to z={z + C:.4f}; wrote {DST}")