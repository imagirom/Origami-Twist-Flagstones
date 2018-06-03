import numpy as np

pi = np.pi


def U(alpha):
    """
    :param alpha: angle in radians
    :return: unit vector in direction alpha
    """
    return np.array([np.cos(alpha), np.sin(alpha)])


def rot_mat(alpha):
    """
    rotation matrix by angle alpha counter-clockwise
    :param alpha: angle in radians
    :return: 2x2 rotation matrix
    """
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]], dtype=np.float32)


def tiled_triangle(i=0, j=0, a=0):
    """
    
    :param i: first grid coordinate of first triangle corner
    :param j: second grid coordinate of first triangle corner
    :param a: orientation of the triangle in [0, 5]
    :return: corners of a equilateral triangle on the standard grid, as a numpy array of shape (3, 2)
    """
    assert 0 <= a < 6
    corners = np.array([[0, 0], U(0), U(np.pi / 3)], dtype=np.float32)
    corners = corners @ rot_mat(a * np.pi / 3)
    corners += i * U(0)
    corners += j * U(pi / 3)
    return corners


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from plotting_utils import plot_polygon
    plot_polygon(tiled_triangle(2, 3, 0))
    plot_polygon(tiled_triangle(4, 5, 0))
    plot_polygon(tiled_triangle(2, 3, 3))
    plt.show()
