import numpy as np
from stl import mesh
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

SCALE = 1

def f(x):
    return (np.sin(x) + 1.5) * SCALE

def inner_r(x):
    outer = f(x)
    return np.maximum(outer * 0.3, outer - THICKNESS)

X_MIN, X_MAX = 0, 2 * np.pi * SCALE
THICKNESS = 0.75 * SCALE

NX = 200
NT = 200

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
    t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    triangles = []
    if inner_radius is None:
        n_rings = 8
        for ring in range(n_rings):
            r0 = radius * ring / n_rings
            r1 = radius * (ring + 1) / n_rings
            for i in range(nt):
                i1 = (i + 1) % nt
                triangles.append([
                    [x_val, r0 * np.cos(t[i]),  r0 * np.sin(t[i])],
                    [x_val, r1 * np.cos(t[i]),  r1 * np.sin(t[i])],
                    [x_val, r1 * np.cos(t[i1]), r1 * np.sin(t[i1])],
                ])
                triangles.append([
                    [x_val, r0 * np.cos(t[i]),  r0 * np.sin(t[i])],
                    [x_val, r1 * np.cos(t[i1]), r1 * np.sin(t[i1])],
                    [x_val, r0 * np.cos(t[i1]), r0 * np.sin(t[i1])],
                ])
    else:
        for i in range(nt):
            i1 = (i + 1) % nt
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
outer_tris = build_mesh_triangles(NX, NT, lambda x: f(x),       X_MIN, X_MAX)
inner_tris = build_mesh_triangles(NX, NT, lambda x: inner_r(x), X_MIN, X_MAX)
left_cap   = build_end_cap_triangles(X_MIN, f(X_MIN), NT, inner_radius=None)
right_cap  = build_end_cap_triangles(X_MAX, f(X_MAX), NT, inner_radius=None)
save_stl(outer_tris + inner_tris + left_cap + right_cap, "stl_output/continuous.stl")

# --- DISCRETE STL ---
N_WASHERS = 14
x_vals = np.linspace(X_MIN, X_MAX, N_WASHERS + 1)
disc_tris = []

for k in range(N_WASHERS):
    x0, x1 = x_vals[k], x_vals[k + 1]
    Ro = f(x0)
    Ri = max(Ro * 0.3, Ro - THICKNESS)
    t  = np.linspace(0, 2 * np.pi, NT, endpoint=False)
    for i in range(NT):
        i1 = (i + 1) % NT
        co,  si  = np.cos(t[i]),  np.sin(t[i])
        co1, si1 = np.cos(t[i1]), np.sin(t[i1])
        disc_tris += [
            [[x0, Ro*co, Ro*si], [x0, Ro*co1, Ro*si1], [x1, Ro*co,  Ro*si]],
            [[x1, Ro*co, Ro*si], [x0, Ro*co1, Ro*si1], [x1, Ro*co1, Ro*si1]],
        ]
        disc_tris += [
            [[x0, Ri*co, Ri*si], [x1, Ri*co,  Ri*si],  [x0, Ri*co1, Ri*si1]],
            [[x1, Ri*co, Ri*si], [x1, Ri*co1, Ri*si1], [x0, Ri*co1, Ri*si1]],
        ]
        disc_tris += [
            [[x0, Ro*co, Ro*si], [x0, Ri*co1, Ri*si1], [x0, Ro*co1, Ro*si1]],
            [[x0, Ro*co, Ro*si], [x0, Ri*co,  Ri*si],  [x0, Ri*co1, Ri*si1]],
        ]
        disc_tris += [
            [[x1, Ro*co, Ro*si], [x1, Ro*co1, Ro*si1], [x1, Ri*co1, Ri*si1]],
            [[x1, Ro*co, Ro*si], [x1, Ri*co1, Ri*si1], [x1, Ri*co,  Ri*si]],
        ]

save_stl(disc_tris, "stl_output/discrete.stl")

# --- DIMENSIONS ---
max_r  = f(np.pi / 2)   # peak at middle of arch
min_r  = f(0)           # endpoints where sin(0) = sin(π) = 0, smallest radius
length = X_MAX - X_MIN

print(f"\n--- Dimensions (cm) ---")
print(f"Length along x-axis : {length:.2f} cm")
print(f"Max diameter        : {2 * max_r:.2f} cm")
print(f"Min diameter        : {2 * min_r:.2f} cm")
print(f"Wall thickness      : {THICKNESS:.2f} cm")
print(f"Num washers (disc)  : {N_WASHERS}")

# --- MATPLOTLIB PREVIEW ---
fig = plt.figure(figsize=(14, 6))
fig.subplots_adjust(bottom=0.15)
gs = gridspec.GridSpec(1, 2, figure=fig)

ax_disc = fig.add_subplot(gs[0], projection='3d')
ax_cont = fig.add_subplot(gs[1], projection='3d')

lim = f(np.pi / 2)

def style_ax(ax, title):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')
    ax.set_title(title, pad=10)

x_p = np.linspace(X_MIN, X_MAX, 120)
t_p = np.linspace(0, 2 * np.pi, 120)
Xm, Tm = np.meshgrid(x_p, t_p)
ax_cont.plot_surface(Xm, f(Xm) * np.cos(Tm), f(Xm) * np.sin(Tm),
                     cmap=plt.cm.YlGnBu_r, alpha=0.9)
style_ax(ax_cont, r'Continuous  $f(x) = \sin(x) + 1.5$')

def redraw_discrete(n):
    ax_disc.clear()
    x_d = np.linspace(X_MIN, X_MAX, n + 1)
    t_d = np.linspace(0, 2 * np.pi, 80)
    T1, _ = np.meshgrid(t_d, [0, 1])

    for k in range(n):
        x0, x1 = x_d[k], x_d[k + 1]
        Ro = f(x0)
        Ri = max(Ro * 0.3, Ro - THICKNESS)
        X2 = np.array([[x0] * len(t_d), [x1] * len(t_d)])

        ax_disc.plot_surface(X2, Ro * np.cos(T1), Ro * np.sin(T1),
                             color='steelblue', alpha=0.85)
        ax_disc.plot_surface(X2, Ri * np.cos(T1), Ri * np.sin(T1),
                             color='steelblue', alpha=0.85)
        R_face = np.array([[Ri] * len(t_d), [Ro] * len(t_d)])
        ax_disc.plot_surface(np.full_like(R_face, x0),
                             R_face * np.cos(T1), R_face * np.sin(T1),
                             color='steelblue', alpha=0.85)
        ax_disc.plot_surface(np.full_like(R_face, x1),
                             R_face * np.cos(T1), R_face * np.sin(T1),
                             color='steelblue', alpha=0.85)

    style_ax(ax_disc, f'Discrete  (n = {n})')
    fig.canvas.draw_idle()

redraw_discrete(N_WASHERS)

ax_slider = fig.add_axes([0.25, 0.04, 0.5, 0.03])
slider = Slider(ax_slider, 'Num discs', 3, 40, valinit=N_WASHERS, valstep=1)
slider.on_changed(lambda val: redraw_discrete(int(val)))

plt.show()