import numpy as np
import collections


class Node:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.data})'

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return self.data.__hash__()


class FloatNode(Node):

    def __init__(self, f):
        assert isinstance(f, float)
        super(FloatNode, self).__init__(f)


class GridNode(Node):

    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], collections.Iterable) and len(args[0]) == 2
            super(GridNode, self).__init__(tuple(args[0]))
        elif len(args) == 2:
            super(GridNode, self).__init__(tuple(args))
        else:
            assert False


class NodeCollection:
    def __init__(self):
        self._nodes = []  # list of objects that can be connected (e.g. trigrid_node(i, j))
        self._node_ids = {}

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes

    def __str__(self):
        return f'Node Collection: {self._nodes}'

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, idx):
        return self._nodes[idx]

    def is_valid_node(self, node):
        return isinstance(node, Node)

    def add_node_direct(self, node):
        self._nodes.append(node)

    def add_node(self, *args):
        if len(args) == 1 and self.is_valid_node(args[0]):
            node = args[0]
            if node not in self:
                self.add_node_direct(node)
                self._node_ids[node] = len(self) - 1
                return len(self) - 1
            else:
                return self._node_ids[node]

        elif len(args) == 1 and isinstance(args[0], collections.Iterable):
            result = []
            for node in args[0]:
                result.append(self.add_node(node))
            return result

        elif len(args) > 1:
            result = []
            for node in args:
                result.append(self.add_node(node))
            return result

        else:
            assert False, f'Invalid arguments: {args}'


class EdgeCollection:
    def __init__(self, node_collection=None, oriented=False):
        if node_collection is None:
            node_collection = NodeCollection()
        self._nodes = node_collection
        self._edges = []
        self._edge_ids = {}

        self.oriented = oriented

    def __len__(self):
        return len(self._edges)

    def __contains__(self, edge):
        return edge in self._edges

    def __str__(self):
        return f'Edge Collection: {self._edges}'

    def __iter__(self):
        return iter(self._edges)

    def __getitem__(self, idx):
        return self._edges[idx]

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def add_edge(self, *args, **kwargs):
        if len(args) == 2 and isinstance(args[0], int):
            assert isinstance(args[1], int), \
                f'{args[1]} is no integer. If the first argument is an integer the second one must be, too'
            edge = args
            if not self.oriented:
                edge = tuple(sorted(edge))
            assert max(edge) < len(self._nodes), \
                f'{max(edge)} out of range. Only {len(self._nodes)} nodes in the graph'
            if edge not in self._edges:
                self._edges.append(edge)
                self._edge_ids[edge] = len(self) - 1
                return len(self) - 1
            else:
                if kwargs.get('soft', True):
                    return self._edge_ids[edge]
                else:
                    assert False, f'Edge {edge} already present'

        elif len(args) == 2 and self._nodes.is_valid_node(args[0]):
            assert self._nodes.is_valid_node(args[0]), \
                f'{args[1]} is not a Node. If the first argument is a Node the second one must be, too'
            nodes = args
            ids = self._nodes.add_node(nodes)
            return self.add_edge(ids, **kwargs)

        elif len(args) == 1:
            return self.add_edge(*(args[0]), **kwargs)

        elif len(args) > 1 and isinstance(args[0], collections.Sized) and len(args[0]) > 1:
            result = []
            for edge in args:
                result.append(self.add_edge(edge))
            return result

        elif len(args) > 1 and self._nodes.is_valid_node(args[0]) or isinstance(args[0], int):
            if not kwargs.get('closed', False):
                return self.add_edge([[args[i], args[i+1]] for i in range(len(args) - 1)], **kwargs)
            else:
                nodes = list(args)
                nodes.append(nodes[0])
                kwargs['closed'] = False
                return self.add_edge(nodes, **kwargs)
        else:
            assert False, f'Invalid inputs: {args}'

    def edge_id(self, edge):
        assert isinstance(edge, collections.Iterable)
        return self._edge_ids[tuple(sorted(edge))]

    def get_edge_nodes(self, edge):
        return self.nodes[edge[0]], self.nodes[edge[1]]

    def connection_matrix(self, oriented=None):
        if oriented is None:
            oriented = self.oriented
        result = np.zeros((len(self._nodes), len(self._nodes)), dtype=np.bool)
        if oriented:
            for edge in self:
                result[edge[0], edge[1]] = True
        else:
            for edge in self:
                result[edge[0], edge[1]] = True
                result[edge[1], edge[0]] = True
        return result

    def extract_triangles(self):
        connection_mat = self.connection_matrix()
        connected_to = [np.argwhere(connection_mat[i][:i])[:, 0] for i in range(len(self._nodes))]
        tris = []
        for i in range(len(self._nodes)):
            for j in connected_to[i]:
                for k in connected_to[j]:
                    if connection_mat[k, i]:
                        tris.append([self.edge_id(edge) for edge in [(i, j), (j, k), (k, i)]])
        result = FaceCollection(self)
        result.add_face(tris)
        return result


class FaceCollection:
    def __init__(self, edge_collection=None):
        if edge_collection is None:
            edge_collection = EdgeCollection()
        self._edges = edge_collection
        self._faces = []
        self._face_ids = {}

    def __len__(self):
        return len(self._faces)

    def __contains__(self, face):
        return face in self._faces

    def __str__(self):
        return f'Face Collection: {self._faces}'

    def __add__(self, other):  # TODO: make copy so add does not work inplace
        assert isinstance(other, FaceCollection)
        assert self.edges.nodes == other.edges.nodes
        if self.edges == other.edges:
            self.add_face(other.faces)
            return self
        else:
            self.add_face([other.get_face_nodes(face) for face in other.faces])
            return self

    def _normalize(self, face):
        i = np.argmin(face)
        return face[i:] + face[0:i]

    @property
    def edges(self):
        return self._edges

    @property
    def faces(self):
        return self._faces

    def add_face(self, *args, **kwargs):
        if len(args) > 2 and all(isinstance(a, int) for a in args):
            for a in args:
                assert isinstance(a, int), \
                    f'{a} is no integer. If the first argument is an integer, all the other ones must be, too'
            face = self._normalize(args)
            assert max(face) < len(self._edges), \
                f'{max(face)} out of range. Only {len(self._edges)} edges in the graph'
            if face not in self._faces:
                self._faces.append(face)
                self._face_ids[face] = len(self) - 1
                return len(self) - 1
            else:
                if kwargs.get('soft', True):
                    return self._face_ids[face]
                else:
                    assert False, f'Face {face} already present'

        elif len(args) > 2 and all(isinstance(a, collections.Sized) and len(a) == 2 for a in args):
            edges = args
            ids = self._edges.add_edge(edges)
            return self.add_face(ids, **kwargs)

        elif len(args) > 2 and all(isinstance(a, int) or self._edges.nodes.is_valid_node(args[0]) for a in args):
            edges = []
            for i in range(len(args)):
                edges.append([args[i], args[(i+1) % len(args)]])
            return self.add_face(edges, **kwargs)

        elif len(args) == 1:
            return self.add_face(*(args[0]), **kwargs)

        elif len(args) > 1 and all(isinstance(a, collections.Sized) and len(a) > 2 for a in args):
            result = []
            for face in args:
                result.append(self.add_face(face))
            return result

        else:
            assert False, f'Invalid inputs: {args}'

    def get_face_nodes(self, face):
        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3
        node_ids = []
        for i in range(len(face)):
            node_ids.append(intersection(self.edges[face[i]], self.edges[face[(i-1) % len(face)]])[0])
        return [self.edges.nodes[i] for i in node_ids]


class Tiling:
    def __init__(self):
        self._nodes = NodeCollection()  # list of objects that can be connected (e.g. trigrid_node(i, j))
        self._edges = EdgeCollection(self._nodes)
        self._faces = FaceCollection(self._edges)

    def __str__(self):
        return f'Tiling Object\n{self.nodes}\n{self.edges}\n{self.faces}'

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def faces(self):
        return self._faces

    def add_node(self, *args):
        return self._nodes.add_node(*args)

    def add_edge(self, *args, **kwargs):
        return self._edges.add_edge(*args, **kwargs)

    def add_face(self, *args):
        return self._faces.add_face(*args)

    def add_edge_triangles(self):
        self._faces += self._edges.extract_triangles()


if __name__ == "__main__":
    tiling = Tiling()
    n = 20
    tiling.add_face([GridNode(i, j+1), GridNode(i, j), GridNode(i+1, j), GridNode(i+1, j+1)]
                    for i in range(n-1) for j in range(n-1))
    print(tiling)

