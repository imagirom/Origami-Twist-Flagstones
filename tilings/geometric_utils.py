import numpy as np


def total_polygon_curvature(poly):
    if not(isinstance(poly, np.ndarray)):
        poly = np.array(poly, dtype=np.float32)
    assert poly.shape[1] == 2, f'{poly} is not a polygon'
    r_poly = np.concatenate([poly[1:], poly[[0]]], axis=0)
    return -np.sum((r_poly[:, 0] - poly[:, 0]) * (r_poly[:, 1] + poly[:, 1]))


if __name__ == '__main__':
    poly = [[0, 0], [1, 0], [.5, 2]]
    print(total_polygon_curvature(poly))
    print(total_polygon_curvature(list(reversed(poly))))
