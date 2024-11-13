import cupy as cp
import math

def generate_random_points_in_circle(center_x, center_y, radius, count):
    angles = cp.random.uniform(0, 2 * math.pi, count)
    radii = cp.sqrt(cp.random.uniform(0, 1, count)) * radius
    x_coords = center_x + radii * cp.cos(angles)
    y_coords = center_y + radii * cp.sin(angles)
    return cp.stack((x_coords, y_coords), axis=1).astype(cp.int32)