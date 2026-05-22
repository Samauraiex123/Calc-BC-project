import numpy as np
from stl import mesh
import os

os.makedirs("stl_output", exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================
SCALE = 1

THICKNESS = 0.75 * SCALE

def f(x):
    return (np.sin(x) + 1.5) * SCALE

def inner_r(x):
    outer = f(x)
    return np.maximum(outer * 0.3, outer - THICKNESS)

X_MIN, X_MAX = 0, 2 * np.pi * SCALE

# Higher resolution for cleaner engraving
NX = 320
NT = 320

N_WASHERS = 14

# ============================================================
# ENGRAVING SETTINGS
# ============================================================
TEXT = "sin(x)+1.5"

# deeper engraving for clearer shadows
ENGRAVE_D = 0.18

COLS = 5
ROWS = 9
CGAP = 2

# 0 = top of tube
TEXT_THETA = 0.0

TEXT_X_CEN = X_MAX / 2.0

# ============================================================
# AUTO SCALE TEXT
# ============================================================
TEXT_W_PIX = len(TEXT) * (COLS + CGAP) - CGAP

# fill 85% of tube length
PIX = (X_MAX - X_MIN) * 0.85 / TEXT_W_PIX

TEXT_W = TEXT_W_PIX * PIX
TEXT_H = ROWS * PIX

TX0 = TEXT_X_CEN - TEXT_W / 2.0

R_TEXT = f(TEXT_X_CEN)

DTHETA_TOT = TEXT_H / R_TEXT

THETA0 = TEXT_THETA - DTHETA_TOT / 2.0

# ============================================================
# 5x9 FONT
# 1 = ENGRAVED
# ============================================================
FONT = {
    's': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,0],[0,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[0,0,0,0,0]],
    'i': [[0,1,1,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,0,0,0]],
    'n': [[1,0,0,0,1],[1,1,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,0,1,1],[1,0,0,1,1],[1,0,0,0,1],[0,0,0,0,0]],
    '(': [[0,0,1,1,0],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,1,0],[0,0,0,0,0]],
    'x': [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,0,0,0,0]],
    ')': [[0,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,1,1,0,0],[0,0,0,0,0]],
    '+': [[0,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    '1': [[0,0,1,0,0],[0,1,1,0,0],[1,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,0,0,0]],
    '.': [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0]],
    '5': [[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[0,0,0,0,0]],
}

# ============================================================
# PRECOMPUTE ENGRAVED PIXELS
# ============================================================
def get_engraved_pixels():
    pixels = []
    cx = TX0

    for ch in TEXT:
        if ch not in FONT:
            cx += (COLS + CGAP) * PIX
            continue

        grid = FONT[ch]
        for row in range(ROWS):
            for col in range(COLS):
                if grid[row][col] == 1:
                    px_x0 = cx + col * PIX
                    px_x1 = px_x0 + PIX
                    dtheta = PIX / R_TEXT
                    px_t0 = THETA0 + (ROWS - 1 - row) * dtheta
                    px_t1 = px_t0 + dtheta
                    pixels.append((px_x0, px_x1, px_t0, px_t1))
        cx += (COLS + CGAP) * PIX

    return pixels

def is_engraved(x_mid, t_mid, engraved_pixels):
    for (x0, x1, t0, t1) in engraved_pixels:
        if x0 <= x_mid <= x1:
            # Check normal, and +/- 2PI wraps to catch angles centered around zero
            if (t0 <= t_mid <= t1) or (t0 <= t_mid - 2*np.pi <= t1) or (t0 <= t_mid + 2*np.pi <= t1):
                return True
    return False

# ============================================================
# CONTINUOUS MODEL
# ============================================================
def build_continuous_engraved_outer(engraved_pixels):
    tris = []
    x_arr = np.linspace(X_MIN, X_MAX, NX)
    t_arr = np.linspace(0, 2*np.pi, NT, endpoint=False)

    # Calculate radius PER VERTEX instead of per face to prevent mesh tearing
    R_outer = np.zeros((NT, NX))
    for i in range(NT):
        for j in range(NX):
            eng = is_engraved(x_arr[j], t_arr[i], engraved_pixels)
            R_outer[i, j] = f(x_arr[j]) - (ENGRAVE_D if eng else 0.0)

    for i in range(NT):
        i1 = (i + 1) % NT
        for j in range(NX - 1):
            r00 = R_outer[i, j]
            r01 = R_outer[i, j+1]
            r10 = R_outer[i1, j]
            r11 = R_outer[i1, j+1]

            P00 = [x_arr[j],   r00*np.cos(t_arr[i]),  r00*np.sin(t_arr[i])]
            P01 = [x_arr[j+1], r01*np.cos(t_arr[i]),  r01*np.sin(t_arr[i])]
            P10 = [x_arr[j],   r10*np.cos(t_arr[i1]), r10*np.sin(t_arr[i1])]
            P11 = [x_arr[j+1], r11*np.cos(t_arr[i1]), r11*np.sin(t_arr[i1])]

            # Correct winding order (points OUTWARDS)
            tris.append([P00, P10, P01])
            tris.append([P10, P11, P01])

    return tris

# ============================================================
# INNER SURFACE + CAPS
# ============================================================
def build_continuous_inner_and_caps():
    tris = []
    x_arr = np.linspace(X_MIN, X_MAX, NX)
    t_arr = np.linspace(0, 2*np.pi, NT, endpoint=False)

    # INNER SURFACE
    for i in range(NT):
        i1 = (i + 1) % NT
        for j in range(NX - 1):
            ri0 = inner_r(x_arr[j])
            ri1 = inner_r(x_arr[j+1])

            P00 = [x_arr[j],   ri0*np.cos(t_arr[i]),  ri0*np.sin(t_arr[i])]
            P01 = [x_arr[j+1], ri1*np.cos(t_arr[i]),  ri1*np.sin(t_arr[i])]
            P10 = [x_arr[j],   ri0*np.cos(t_arr[i1]), ri0*np.sin(t_arr[i1])]
            P11 = [x_arr[j+1], ri1*np.cos(t_arr[i1]), ri1*np.sin(t_arr[i1])]

            # Correct winding order (points INWARDS)
            tris.append([P00, P01, P10])
            tris.append([P10, P01, P11])

    # CAPS
    t_cap = np.linspace(0, 2*np.pi, NT, endpoint=False)
    for x_val, sign in [(X_MIN, -1), (X_MAX, 1)]:
        ro = f(x_val)
        ri = inner_r(x_val)
        n_rings = 8

        for ring in range(n_rings):
            r0 = ri + (ro-ri) * ring / n_rings
            r1 = ri + (ro-ri) * (ring+1) / n_rings

            for i in range(NT):
                i1 = (i + 1) % NT

                A = [x_val, r0*np.cos(t_cap[i]),  r0*np.sin(t_cap[i])]
                B = [x_val, r1*np.cos(t_cap[i]),  r1*np.sin(t_cap[i])]
                C = [x_val, r1*np.cos(t_cap[i1]), r1*np.sin(t_cap[i1])]
                D = [x_val, r0*np.cos(t_cap[i1]), r0*np.sin(t_cap[i1])]

                if sign == -1:
                    tris.extend([[A,C,B],[A,D,C]])
                else:
                    tris.extend([[A,B,C],[A,C,D]])

    return tris

# ============================================================
# DISCRETE MODEL
# ============================================================
def build_discrete_engraved(engraved_pixels):
    tris = []
    x_vals = np.linspace(X_MIN, X_MAX, N_WASHERS + 1)
    t_arr = np.linspace(0, 2*np.pi, NT, endpoint=False)

    for k in range(N_WASHERS):
        x0 = x_vals[k]
        x1 = x_vals[k+1]
        Ro_base = f(x0)
        Ri = max(Ro_base * 0.3, Ro_base - THICKNESS)
        x_mid = (x0 + x1) / 2.0

        # Calculate depth per theta-vertex to close gaps
        Ro_arr = np.zeros(NT)
        for i in range(NT):
            eng = is_engraved(x_mid, t_arr[i], engraved_pixels)
            Ro_arr[i] = Ro_base - (ENGRAVE_D if eng else 0.0)

        for i in range(NT):
            i1 = (i + 1) % NT
            co = np.cos(t_arr[i]);  si = np.sin(t_arr[i])
            co1= np.cos(t_arr[i1]); si1= np.sin(t_arr[i1])

            Ro  = Ro_arr[i]
            Ro1 = Ro_arr[i1]

            P00 = [x0, Ro*co, Ro*si]
            P10 = [x0, Ro1*co1, Ro1*si1]
            P01 = [x1, Ro*co, Ro*si]
            P11 = [x1, Ro1*co1, Ro1*si1]

            P00_i = [x0, Ri*co, Ri*si]
            P10_i = [x0, Ri*co1, Ri*si1]
            P01_i = [x1, Ri*co, Ri*si]
            P11_i = [x1, Ri*co1, Ri*si1]

            # OUTER (outwards winding)
            tris += [[P00, P10, P01], [P10, P11, P01]]
            
            # INNER (inwards winding)
            tris += [[P00_i, P01_i, P10_i], [P10_i, P01_i, P11_i]]
            
            # LEFT (x0) cap (-X normal)
            tris += [[P00, P00_i, P10], [P10, P00_i, P10_i]]
            
            # RIGHT (x1) cap (+X normal)
            tris += [[P01, P11, P01_i], [P11, P11_i, P01_i]]

    return tris

# ============================================================
# SAVE STL
# ============================================================
def save_stl(tris, filename):
    arr = np.array(tris)
    m = mesh.Mesh(np.zeros(len(arr), dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(arr):
        m.vectors[i] = tri
    m.save(filename)
    print(f"Saved: {filename}")
    print(f"Triangles: {len(arr)}")

# ============================================================
# GENERATE
# ============================================================
engraved_pixels = get_engraved_pixels()

print("\n================================================")
print("ENGRAVING INFO")
print("================================================")
print(f"Text              : {TEXT}")
print(f"Pixel size        : {PIX:.4f}")
print(f"Text width        : {TEXT_W:.4f}")
print(f"Text height       : {TEXT_H:.4f}")
print(f"Engraving depth   : {ENGRAVE_D}")
print(f"Text x-range      : {TX0:.4f} -> {TX0 + TEXT_W:.4f}")
print(f"Text angular span : {np.degrees(DTHETA_TOT):.2f} degrees")
print(f"Engraved pixels   : {len(engraved_pixels)}")
print("================================================\n")

# CONTINUOUS
cont_tris = build_continuous_engraved_outer(engraved_pixels)
cont_tris += build_continuous_inner_and_caps()

save_stl(cont_tris, "stl_output/continuous_engraving.stl")

# DISCRETE
disc_tris = build_discrete_engraved(engraved_pixels)

save_stl(disc_tris, "stl_output/discrete_engraving.stl")

print("\nDONE.")