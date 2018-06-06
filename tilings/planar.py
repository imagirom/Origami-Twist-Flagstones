from .base import *
from .geometric_utils import total_polygon_curvature
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
import cmath


class PlanarNode(Node):
    def to_array(self):
        return np.array(self.data)

    def coords(self):
        return self.data[0], self.data[1]


class ComplexNode(PlanarNode):
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


class FixedFaceNode(PlanarNode):
    def __init__(self, node, face):
        super(FixedFaceNode, self).__init__((face, node))
        self._face = face
        self._node = node

    @property
    def node(self):
        return self._node

    @property
    def face(self):
        return self._face

    def to_array(self):
        return self.node.to_array()

    def coords(self):
        return self.node.coords()


class PlanarNodeCollection(NodeCollection):
    def __init__(self):
        super(PlanarNodeCollection, self).__init__()

    def is_valid_node(self, node):
        return isinstance(node, PlanarNode)

    def __contains__(self, item):
        return item in self._node_ids.keys()


class PlanarEdgeCollection(EdgeCollection):
    pass


class PlanarFaceCollection(FaceCollection):

    def _normalize(self, face):
        nodes = [self.edges.nodes[id].to_array() for id in self._compute_face_nodes(face)]
        if total_polygon_curvature(nodes) < 0:
            face = tuple(reversed(face))
        return super(PlanarFaceCollection, self)._normalize(face)


class PlanarTiling(Tiling):
    def __init__(self, nodes=None, edges=None, faces=None):
        if nodes is None:
            nodes = PlanarNodeCollection()
        if edges is None:
            edges = PlanarEdgeCollection(nodes)
        if faces is None:
            faces = PlanarFaceCollection(edges)
        super(PlanarTiling, self).__init__(nodes, edges, faces)

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

    def split_face_tiliing(self):
        result = PlanarTiling()

        for face in self.faces:
            new_face = []
            for node in self.faces.get_face_nodes(face):
                new_face.append(FixedFaceNode(node, face))
            result.add_face(new_face)

        return result

    def get_node_array(self):
        return np.stack([node.to_array() for node in self.nodes], axis=0)

    def get_numpy_polys(self):
        node_array = self.get_node_array()
        return [node_array[np.array(self.faces.get_face_node_ids(face))] for face in self.faces]

    def get_circle_polys(self):
        face_node_ids = [np.array(self.faces.get_face_node_ids(i)) for i in range(len(self.faces))]
        result = []
        for node_id in range(len(self.nodes)):
            adjacent_face_ids = [i for i, face in enumerate(face_node_ids) if node_id in face]
            if len(adjacent_face_ids) < 3:
                continue
            circle_poly = []
            for i in adjacent_face_ids:
                pos = np.argwhere(face_node_ids[i] - node_id == 0)[0, 0]
                circle_poly.append([i, pos])
            result.append(np.array(circle_poly))
        return result

    def get_flagstone_connections(self):
        face_node_ids = [np.array(self.faces.get_face_node_ids(i)) for i in range(len(self.faces))]
        result = []
        for edge_id, edge in enumerate(self.edges):
            adjacent_face_ids = [face_id for face_id, face in enumerate(self.faces) if edge_id in face]
            if len(adjacent_face_ids) != 2:
                continue
            face_node_ids_0 = face_node_ids[adjacent_face_ids[0]]
            face_node_ids_1 = face_node_ids[adjacent_face_ids[1]]
            offset = np.argwhere(face_node_ids_0 - edge[0] == 0)[0, 0] - \
                     np.argwhere(face_node_ids_0 - edge[1] == 0)[0, 0]
            if offset % len(face_node_ids_0) != 1:
                result.append([
                    [adjacent_face_ids[0], np.argwhere(face_node_ids_0 - edge[0] == 0)[0, 0]],
                    [adjacent_face_ids[1], np.argwhere(face_node_ids_1 - edge[1] == 0)[0, 0]]
                ])
            else:
                result.append([
                    [adjacent_face_ids[0], np.argwhere(face_node_ids_0 - edge[1] == 0)[0, 0]],
                    [adjacent_face_ids[1], np.argwhere(face_node_ids_1 - edge[0] == 0)[0, 0]]
                ])
        return np.array(result)
