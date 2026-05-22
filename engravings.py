import numpy as np
from stl import mesh
import os

os.makedirs("stl_output", exist_ok=True)

# ============================================================
# SETTINGS — match your main test.py
# ============================================================
SCALE     = 1
THICKNESS = 0.75 * SCALE

def f(x):
    return (np.sin(x) + 1.5) * SCALE

X_MIN, X_MAX = 0, 2 * np.pi * SCALE

# Plate dimensions (mm — size to match the models)
PLATE_W   = 80.0    # width  (along x)
PLATE_H   = 30.0    # height (along y)
PLATE_D   = 4.0     # thickness (z)
ENGRAVE_D = 1.2     # how deep the engraved letters go (must be < PLATE_D)
MARGIN    = 4.0     # blank border around text

# ============================================================
# 5x9 PIXEL FONT  (1 = solid, 0 = engraved/hole)
# Tall grid for maximum clarity at this size
# ============================================================
FONT = {
    's': [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,0],
        [0,1,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ],
    'i': [
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ],
    'n': [
        [1,0,0,0,1],
        [1,1,0,0,1],
        [1,1,0,0,1],
        [1,0,1,0,1],
        [1,0,1,0,1],
        [1,0,0,1,1],
        [1,0,0,1,1],
        [1,0,0,0,1],
        [0,0,0,0,0],
    ],
    '(': [
        [0,0,1,1,0],
        [0,1,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,1,0],
        [0,0,0,0,0],
    ],
    'x': [
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,0,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,0,0,0,0],
    ],
    ')': [
        [0,1,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
    ],
    '+': [
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    '1': [
        [0,0,1,0,0],
        [0,1,1,0,0],
        [1,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,0,0,0],
    ],
    '.': [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
    ],
    '5': [
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ],
}

TEXT   = "sin(x)+1.5"
COLS   = 5
ROWS   = 9
CGAP   = 2    # pixel gap between chars

# pixel physical size — scale to fill plate height minus margins
PIX    = (PLATE_H - 2 * MARGIN) / ROWS

# total text width
TEXT_W = (len(TEXT) * (COLS + CGAP) - CGAP) * PIX

# auto-widen plate if text is wider than default
if TEXT_W + 2 * MARGIN > PLATE_W:
    PLATE_W = TEXT_W + 2 * MARGIN

# horizontal centering offset
TEXT_X0 = (PLATE_W - TEXT_W) / 2

# ============================================================
# GEOMETRY HELPERS
# ============================================================
triangles = []

def tri(a, b, c):
    triangles.append([list(a), list(b), list(c)])

def quad_face(x0, y0, x1, y1, z):
    """Single flat quad at height z (two triangles)."""
    tri([x0,y0,z], [x1,y0,z], [x1,y1,z])
    tri([x0,y0,z], [x1,y1,z], [x0,y1,z])

def quad_face_rev(x0, y0, x1, y1, z):
    """Reversed winding — for bottom face."""
    tri([x0,y0,z], [x1,y1,z], [x1,y0,z])
    tri([x0,y0,z], [x0,y1,z], [x1,y1,z])

def wall_xconst(x, y0, y1, z0, z1):
    tri([x,y0,z0], [x,y1,z1], [x,y1,z0])
    tri([x,y0,z0], [x,y0,z1], [x,y1,z1])

def wall_yconst(y, x0, x1, z0, z1):
    tri([x0,y,z0], [x1,y,z0], [x1,y,z1])
    tri([x0,y,z0], [x1,y,z1], [x0,y,z1])

def wall_xconst_rev(x, y0, y1, z0, z1):
    tri([x,y0,z0], [x,y1,z0], [x,y1,z1])
    tri([x,y0,z0], [x,y1,z1], [x,y0,z1])

def wall_yconst_rev(y, x0, x1, z0, z1):
    tri([x0,y,z0], [x1,y,z1], [x1,y,z0])
    tri([x0,y,z0], [x0,y,z1], [x1,y,z1])

# ============================================================
# BUILD PLATE + ENGRAVING
# ============================================================
def build_engraved_plate():
    global triangles
    triangles = []

    Z_BOT  = 0.0
    Z_TOP  = PLATE_D
    Z_ENGR = PLATE_D - ENGRAVE_D   # bottom of engraved pocket

    # --- bottom face ---
    quad_face_rev(0, 0, PLATE_W, PLATE_H, Z_BOT)

    # --- outer walls ---
    wall_yconst(0,       0,       PLATE_W, Z_BOT, Z_TOP)
    wall_yconst_rev(PLATE_H, 0,   PLATE_W, Z_BOT, Z_TOP)
    wall_xconst(0,       0,       PLATE_H, Z_BOT, Z_TOP)
    wall_xconst_rev(PLATE_W, 0,   PLATE_H, Z_BOT, Z_TOP)

    # --- collect engraved pixel rects ---
    engraved = []   # pixels that are engraved (grid value == 0)
    solid_px  = []  # pixels that are raised (grid value == 1)

    cx = TEXT_X0
    cy = MARGIN

    for ch in TEXT:
        if ch not in FONT:
            cx += (COLS + CGAP) * PIX
            continue
        grid = FONT[ch]
        for row in range(ROWS):
            for col in range(COLS):
                px0 = cx + col * PIX
                py0 = cy + (ROWS - 1 - row) * PIX
                px1 = px0 + PIX
                py1 = py0 + PIX
                if grid[row][col] == 0:
                    engraved.append((px0, py0, px1, py1))
                else:
                    solid_px.append((px0, py0, px1, py1))
        cx += (COLS + CGAP) * PIX

    # --- top surface: full plate top MINUS engraved pockets ---
    # We tile the top as solid, but for engraved pixels we drop to Z_ENGR
    # Easiest watertight approach:
    # 1. Top face solid everywhere that ISN'T engraved
    # 2. Engraved pockets: floor at Z_ENGR + 4 pocket walls

    # Build a set of solid top rects by splitting the top face around pockets.
    # Simpler: just add the solid pixel tops + the background regions separately.

    # background left of text
    quad_face(0, 0, TEXT_X0, PLATE_H, Z_TOP)
    # background right of text
    quad_face(TEXT_X0 + TEXT_W, 0, PLATE_W, PLATE_H, Z_TOP)
    # background below text
    quad_face(TEXT_X0, 0, TEXT_X0 + TEXT_W, MARGIN, Z_TOP)
    # background above text
    quad_face(TEXT_X0, MARGIN + ROWS * PIX, TEXT_X0 + TEXT_W, PLATE_H, Z_TOP)

    # gaps between characters
    gx = TEXT_X0
    for ch in TEXT:
        char_end = gx + COLS * PIX
        gap_end  = gx + (COLS + CGAP) * PIX
        if gap_end <= TEXT_X0 + TEXT_W:
            quad_face(char_end, MARGIN, gap_end, MARGIN + ROWS * PIX, Z_TOP)
        gx += (COLS + CGAP) * PIX

    # solid pixels — top face at Z_TOP
    for (x0, y0, x1, y1) in solid_px:
        quad_face(x0, y0, x1, y1, Z_TOP)

    # engraved pixels — floor at Z_ENGR + pocket walls
    for (x0, y0, x1, y1) in engraved:
        # pocket floor
        quad_face_rev(x0, y0, x1, y1, Z_ENGR)
        # pocket walls (going down from Z_TOP to Z_ENGR)
        wall_yconst_rev(y0, x0, x1, Z_ENGR, Z_TOP)
        wall_yconst(y1,     x0, x1, Z_ENGR, Z_TOP)
        wall_xconst_rev(x0, y0, y1, Z_ENGR, Z_TOP)
        wall_xconst(x1,     y0, y1, Z_ENGR, Z_TOP)

    return list(triangles)

# ============================================================
# SAVE HELPER
# ============================================================
def save_stl(tris, filename):
    arr = np.array(tris)
    m = mesh.Mesh(np.zeros(len(arr), dtype=mesh.Mesh.dtype))
    for i, t in enumerate(arr):
        m.vectors[i] = t
    m.save(filename)
    print(f"Saved: {filename}  ({len(arr)} triangles)")

# ============================================================
# GENERATE BOTH FILES
# ============================================================
tris = build_engraved_plate()
save_stl(tris, "stl_output/continuous_engraving.stl")

tris = build_engraved_plate()   # identical label for both
save_stl(tris, "stl_output/discrete_engraving.stl")

print(f"\nPlate dimensions: {PLATE_W:.1f} x {PLATE_H:.1f} x {PLATE_D:.1f} mm")
print(f"Engraving depth : {ENGRAVE_D:.1f} mm")
print(f"Pixel size      : {PIX:.2f} mm")
print(f"Text            : {TEXT}")