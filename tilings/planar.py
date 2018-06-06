from .base import *
from .geometric_utils import total_polygon_curvature
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import cmath


class PlanarNode(Node):
    def __init__(self, pos):
        assert isinstance(pos, tuple) and len(pos) == 2
        super(PlanarNode, self).__init__(pos)


class ComplexNode(Node):
    def __init__(self, z, epsilon=1e-5):
        super(ComplexNode, self).__init__(z)
        self.epsilon = epsilon

    def __hash__(self):
        return self.data.__hash__()

    def __eq__(self, other):
        if isinstance(other, ComplexNode):
            return abs(self.data - other.data) < min(self.epsilon, other.epsilon)
        else:
            return False

    def coords(self):
        return self.data.real, self.data.imag

    def to_array(self):
        return np.array([self.data.real, self.data.imag], dtype=np.float32)


class ComplexSquareGridNode(ComplexNode):
    def __init__(self, i, j):
        z = i + np.exp(1j * np.pi / 3) * j
        z = z ** 2
        super(ComplexSquareGridNode, self).__init__(z)


class PlanarNodeCollection(NodeCollection):
    def __init__(self):
        super(PlanarNodeCollection, self).__init__()

    def is_valid_node(self, node):
        return isinstance(node, ComplexNode)

    def __contains__(self, item):
        return item in self._node_ids.keys()


class PlanarEdgeCollection(EdgeCollection):
    pass


class PlanarFaceCollection(FaceCollection):

    def _normalize(self, face):
        nodes = [node.to_array() for node in self.get_face_nodes(face)]
        if total_polygon_curvature(nodes) < 0:
            face = tuple(reversed(face))
        return super(PlanarFaceCollection, self)._normalize(face)


class PlanarTiling(Tiling):
    def __init__(self):
        super(PlanarTiling, self).__init__()
        self._nodes = PlanarNodeCollection()
        self._edges = PlanarEdgeCollection(self._nodes)
        self._faces = PlanarFaceCollection(self._edges)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        lines = []
        for edge in self.edges:
            lines.append([self.nodes[edge[0]].coords(), self.nodes[edge[1]].coords()])
        lines = np.array(lines)
        lc = mc.LineCollection(lines, linewidths=1)

        if ax is None:
            ax = plt.gca()
        ax.add_collection(lc)
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.05)
        plt.show()


