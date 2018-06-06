from tilings import *


def max_coord(i, j):
    coords = (i, j, -i-j)
    return max(abs(x) for x in coords)

def min_coord(i, j):
    coords = (i, j, -i-j)
    return min(abs(x) for x in coords)


if __name__ == '__main__':

    print(np.argwhere([[1,0,1,0,0,1], [1,0,1,0,0,1]])[:, 0])

    tiling = PlanarTiling()
    edges = []
    lower = 2
    upper = 5
    tiling.add_edge([ComplexSquareGridNode(i, j), ComplexSquareGridNode(i+1, j)]
                    for i in range(-upper, upper) for j in range(-upper, upper)
                    if lower <= max_coord(i, j) < upper
                    and lower <= max_coord(i+1, j) < upper
                    )

    tiling.add_edge([ComplexSquareGridNode(i, j), ComplexSquareGridNode(i, j+1)]
                    for i in range(-upper, upper) for j in range(-upper, upper)
                    if lower <= max_coord(i, j) < upper
                    and lower <= max_coord(i, j+1) < upper
                    )

    tiling.add_edge([ComplexSquareGridNode(i, j), ComplexSquareGridNode(i-1, j+1)]
                    for i in range(-upper, upper) for j in range(-upper, upper)
                    if lower <= max_coord(i, j) < upper
                    and lower <= max_coord(i-1, j+1) < upper
                    )

    tiling.add_edge([
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(2, 0)],
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(1, 1)],
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(0, 2)],
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(-1, 2)],
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(-2, 2)],
        [ComplexSquareGridNode(0, 0), ComplexSquareGridNode(-2, 1)]
    ])

    tiling.add_edge_triangles()
    print(tiling)
    tiling.plot()