# from warping import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# https://stackoverflow.com/questions/5071063/is-there-a-library-for-image-warping-image-morphing-for-python-with-controlled
def quad_as_rect(quad):
    if quad[0] != quad[2]:
        return False
    if quad[1] != quad[7]:
        return False
    if quad[4] != quad[6]:
        return False
    if quad[3] != quad[5]:
        return False
    return True


def quad_to_rect(quad):
    assert len(quad) == 8
    assert quad_as_rect(quad)
    return (quad[0], quad[1], quad[4], quad[3])


def rect_to_quad(rect):
    assert len(rect) == 4
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])


def shape_to_rect(shape):
    assert len(shape) == 2
    return (0, 0, shape[0], shape[1])


def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid


def grid_to_mesh(src_grid, dst_grid):
    assert src_grid.shape == dst_grid.shape
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [
                src_grid[i, j, 0],
                src_grid[i, j, 1],
                src_grid[i + 1, j, 0],
                src_grid[i + 1, j, 1],
                src_grid[i + 1, j + 1, 0],
                src_grid[i + 1, j + 1, 1],
                src_grid[i, j + 1, 0],
                src_grid[i, j + 1, 1],
            ]
            dst_quad = [
                dst_grid[i, j, 0],
                dst_grid[i, j, 1],
                dst_grid[i + 1, j, 0],
                dst_grid[i + 1, j, 1],
                dst_grid[i + 1, j + 1, 0],
                dst_grid[i + 1, j + 1, 1],
                dst_grid[i, j + 1, 0],
                dst_grid[i, j + 1, 1],
            ]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh


im_undistored = Image.open("images/undistorted.png")
im = Image.open("images/barrel_distorted.png")  # pincushion_distorted

[ydim, xdim] = im.size
mid = round(max(xdim, ydim) / 2)

n = 100
dst = griddify(shape_to_rect(im.size), n, n)
src = dst

k = -0.0000022  # barrel_distorted
# k =  0.000008 # pincushion_distorted
dst = dst - mid

for i in range(n + 1):
    for j in range(n + 1):
        x = dst[i, j, 0]
        y = dst[i, j, 1]
        r = np.sqrt(x**2 + y**2)
        src[i, j, 0] = x * (1 + k * r**2)
        src[i, j, 1] = y * (1 + k * r**2)

dst = dst + mid
dst = dst.astype(int)
src = src + mid
src = src.astype(int)

mesh = grid_to_mesh(src, dst)
imt = im.transform(im.size, Image.MESH, mesh)

plt.figure()
plt.subplot(1, 3, 1)
plt.title("undistorted")
plt.imshow(im_undistored)
plt.subplot(1, 3, 2)
plt.imshow(im)
plt.title("distorted")
plt.subplot(1, 3, 3)
plt.imshow(imt)
plt.title("distorted-corrected")
plt.show()
