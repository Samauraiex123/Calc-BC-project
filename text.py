import numpy as np
from stl import mesh
import os

os.makedirs("stl_output", exist_ok=True)

# ============================================
# SETTINGS
# ============================================

TEXT = "sin(x)+1.5"

PLATE_D = 3.0
PIXEL = 2.0
MARGIN = 4.0

# ============================================
# 5x7 FONT
# ============================================

FONT = {
    's': [
        "01110",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "01110",
    ],
    'i': [
        "00100",
        "00000",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    'n': [
        "00000",
        "11010",
        "10101",
        "10001",
        "10001",
        "10001",
        "10001",
    ],
    '(': [
        "00100",
        "01000",
        "10000",
        "10000",
        "10000",
        "01000",
        "00100",
    ],
    'x': [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "01010",
        "10001",
    ],
    ')': [
        "00100",
        "00010",
        "00001",
        "00001",
        "00001",
        "00010",
        "00100",
    ],
    '+': [
        "00000",
        "00100",
        "00100",
        "11111",
        "00100",
        "00100",
        "00000",
    ],
    '1': [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    '.': [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00100",
        "00100",
    ],
    '5': [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
}

ROWS = 7
COLS = 5
CHAR_SPACING = 2

# ============================================
# BUILD PIXEL MAP
# ============================================

pixels = []

cursor_x = 0

for ch in TEXT:

    grid = FONT[ch]

    for row in range(ROWS):
        for col in range(COLS):

            if grid[row][col] == "0":
                continue

            x = cursor_x + col
            y = ROWS - 1 - row

            pixels.append((x, y))

    cursor_x += COLS + CHAR_SPACING

TEXT_W = cursor_x - CHAR_SPACING
TEXT_H = ROWS

PLATE_W = (TEXT_W * PIXEL) + 2 * MARGIN
PLATE_H = (TEXT_H * PIXEL) + 2 * MARGIN

# ============================================
# TRIANGLE STORAGE
# ============================================

triangles = []

def tri(a, b, c):
    triangles.append([a, b, c])

def quad(v1, v2, v3, v4):
    tri(v1, v2, v3)
    tri(v1, v3, v4)

# ============================================
# FULL SOLID PLATE
# ============================================

# corners
A  = [0, 0, 0]
B  = [PLATE_W, 0, 0]
C  = [PLATE_W, PLATE_H, 0]
D  = [0, PLATE_H, 0]

A2 = [0, 0, PLATE_D]
B2 = [PLATE_W, 0, PLATE_D]
C2 = [PLATE_W, PLATE_H, PLATE_D]
D2 = [0, PLATE_H, PLATE_D]

# bottom
quad(A, B, C, D)

# top
quad(A2, D2, C2, B2)

# outer walls
quad(A, B, B2, A2)
quad(B, C, C2, B2)
quad(C, D, D2, C2)
quad(D, A, A2, D2)

# ============================================
# CUT LETTER HOLES
# ============================================

for (gx, gy) in pixels:

    x0 = MARGIN + gx * PIXEL
    y0 = MARGIN + gy * PIXEL

    x1 = x0 + PIXEL
    y1 = y0 + PIXEL

    # remove top face by surrounding it with walls

    # left wall
    quad(
        [x0, y0, 0],
        [x0, y1, 0],
        [x0, y1, PLATE_D],
        [x0, y0, PLATE_D]
    )

    # right wall
    quad(
        [x1, y1, 0],
        [x1, y0, 0],
        [x1, y0, PLATE_D],
        [x1, y1, PLATE_D]
    )

    # front wall
    quad(
        [x0, y0, 0],
        [x1, y0, 0],
        [x1, y0, PLATE_D],
        [x0, y0, PLATE_D]
    )

    # back wall
    quad(
        [x1, y1, 0],
        [x0, y1, 0],
        [x0, y1, PLATE_D],
        [x1, y1, PLATE_D]
    )

# ============================================
# SAVE STL
# ============================================

tri_data = np.array(triangles)

solid = mesh.Mesh(np.zeros(len(tri_data), dtype=mesh.Mesh.dtype))

for i, t in enumerate(tri_data):
    solid.vectors[i] = t

filename = "stl_output/text.stl"
solid.save(filename)

print("Saved:", filename)
print("Plate:", PLATE_W, "x", PLATE_H, "x", PLATE_D)
print("Triangles:", len(triangles))