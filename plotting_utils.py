import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def plot_polygon(poly, ax=None, autoscale=True, show=False):
    poly = np.array(poly, dtype=np.float32)
    lines = np.stack([poly, np.concatenate([poly[1:], poly[[0]]])], axis=1)
    lc = mc.LineCollection(lines, linewidths=1)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    if autoscale:
        ax.autoscale()
        ax.margins(0.1)
    if show:
        plt.show()


def show_cp(polygons, connections=None, centers=None, center_connections=None, ax=None, show=True):
    [plot_polygon(poly, ax=ax) for poly in polygons]
    lines = []
    if connections is not None:
        for con in connections:
            lines.append(np.array([polygons[con[0, 0]][con[0, 1]], polygons[con[1, 0]][con[1, 1]]]))

    if centers is not None:
        for i, center in enumerate(centers):
            for con in center_connections[i]:
                lines.append(np.stack([polygons[con[0]][con[1]], center]))
                lines.append(np.stack([polygons[con[0]][(con[1] - 1) % len(polygons[con[0]])], center]))
    lc = mc.LineCollection(lines, linewidths=1)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.05)
    if show:
        plt.show()
