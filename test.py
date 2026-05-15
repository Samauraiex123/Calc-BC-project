import numpy as np
from stl import mesh
import os

def f(x):
    return np.sin(x) + 1.5  # radius in cm

X_MIN, X_MAX = 0, 2 * np.pi

# Resolution
NX = 200   # slices along x
NT = 200   # points around circumference

# Wall thickness in cm
THICKNESS = 0.75

os.makedirs("stl_output", exist_ok=True)

def build_mesh_triangles(nx, nt, radius_fn, x_min, x_max):
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    X, T = np.meshgrid(x, t)
    R = radius_fn(X)
    Y = R * np.cos(T)
    Z = R * np.sin(T)
    triangles = []
    for i in range(nt):
        for j in range(nx - 1):
            i1 = (i + 1) % nt
            # two triangles per quad
            triangles.append([
                [X[i,  j],   Y[i,  j],   Z[i,  j]],
                [X[i,  j+1], Y[i,  j+1], Z[i,  j+1]],
                [X[i1, j],   Y[i1, j],   Z[i1, j]],
            ])
            triangles.append([
                [X[i1, j],   Y[i1, j],   Z[i1, j]],
                [X[i,  j+1], Y[i,  j+1], Z[i,  j+1]],
                [X[i1, j+1], Y[i1, j+1], Z[i1, j+1]],
            ])
    return triangles

def build_end_cap_triangles(x_val, radius, nt, inner_radius=None):
    """Annular or full disc cap perpendicular to x-axis."""
    t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    triangles = []
    for i in range(nt):
        i1 = (i + 1) % nt
        if inner_radius is None:
            # full disc — fan from center
            triangles.append([
                [x_val, 0, 0],
                [x_val, radius * np.cos(t[i]),  radius * np.sin(t[i])],
                [x_val, radius * np.cos(t[i1]), radius * np.sin(t[i1])],
            ])
        else:
            # annular washer between inner and outer
            triangles.append([
                [x_val, radius       * np.cos(t[i]),  radius       * np.sin(t[i])],
                [x_val, inner_radius * np.cos(t[i1]), inner_radius * np.sin(t[i1])],
                [x_val, radius       * np.cos(t[i1]), radius       * np.sin(t[i1])],
            ])
            triangles.append([
                [x_val, radius       * np.cos(t[i]),  radius       * np.sin(t[i])],
                [x_val, inner_radius * np.cos(t[i]),  inner_radius * np.sin(t[i])],
                [x_val, inner_radius * np.cos(t[i1]), inner_radius * np.sin(t[i1])],
            ])
    return triangles

def save_stl(triangles, filename):
    tri_array = np.array(triangles)
    solid = mesh.Mesh(np.zeros(len(tri_array), dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(tri_array):
        solid.vectors[i] = tri
    solid.save(filename)
    print(f"Saved: {filename}  ({len(tri_array)} triangles)")

# --- CONTINUOUS STL ---
outer_fn = lambda x: f(x)
inner_fn = lambda x: f(x) - THICKNESS

outer_tris = build_mesh_triangles(NX, NT, outer_fn, X_MIN, X_MAX)
inner_tris = build_mesh_triangles(NX, NT, inner_fn, X_MIN, X_MAX)

# end caps (annular washers at x=0 and x=2pi)
cap_r0_outer = f(X_MIN)
cap_r0_inner = f(X_MIN) - THICKNESS
cap_r1_outer = f(X_MAX)
cap_r1_inner = f(X_MAX) - THICKNESS

left_cap  = build_end_cap_triangles(X_MIN, cap_r0_outer, NT, inner_radius=cap_r0_inner)
right_cap = build_end_cap_triangles(X_MAX, cap_r1_outer, NT, inner_radius=cap_r1_inner)

all_tris = outer_tris + inner_tris + left_cap + right_cap
save_stl(all_tris, "stl_output/continuous.stl")

# --- DISCRETE STL (14 washers) ---
N_WASHERS = 14
x_vals = np.linspace(X_MIN, X_MAX, N_WASHERS + 1)
disc_tris = []

for k in range(N_WASHERS):
    x0, x1 = x_vals[k], x_vals[k + 1]
    Ro = f(x0)           # outer radius of this disc
    Ri = Ro - THICKNESS  # inner radius

    t = np.linspace(0, 2 * np.pi, NT, endpoint=False)

    for i in range(NT):
        i1 = (i + 1) % NT

        co, si   = np.cos(t[i]),  np.sin(t[i])
        co1, si1 = np.cos(t[i1]), np.sin(t[i1])

        # outer barrel
        disc_tris += [
            [[x0, Ro*co, Ro*si], [x0, Ro*co1, Ro*si1], [x1, Ro*co,  Ro*si]],
            [[x1, Ro*co, Ro*si], [x0, Ro*co1, Ro*si1], [x1, Ro*co1, Ro*si1]],
        ]
        # inner barrel (reversed winding)
        disc_tris += [
            [[x0, Ri*co, Ri*si], [x1, Ri*co,  Ri*si],  [x0, Ri*co1, Ri*si1]],
            [[x1, Ri*co, Ri*si], [x1, Ri*co1, Ri*si1], [x0, Ri*co1, Ri*si1]],
        ]
        # left annular face
        disc_tris += [
            [[x0, Ro*co, Ro*si], [x0, Ri*co1, Ri*si1], [x0, Ro*co1, Ro*si1]],
            [[x0, Ro*co, Ro*si], [x0, Ri*co,  Ri*si],  [x0, Ri*co1, Ri*si1]],
        ]
        # right annular face
        disc_tris += [
            [[x1, Ro*co, Ro*si], [x1, Ro*co1, Ro*si1], [x1, Ri*co1, Ri*si1]],
            [[x1, Ro*co, Ro*si], [x1, Ri*co1, Ri*si1], [x1, Ri*co,  Ri*si]],
        ]

save_stl(disc_tris, "stl_output/discrete.stl")