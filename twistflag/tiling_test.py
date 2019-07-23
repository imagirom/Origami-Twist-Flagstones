from tilings import *
from plotting_utils import *
import torch
from torch.autograd import Variable
from torch.optim import SGD
from cp_optimization import CPLoss
import saving


def max_coord(i, j):
    coords = (i, j, -i-j)
    return max(abs(x) for x in coords)


def min_coord(i, j):
    coords = (i, j, -i-j)
    return min(abs(x) for x in coords)


if __name__ == '__main__':

    device = torch.device('cpu')
    n_iters = 50000
    filename = 'test1.svg'

    tiling = PlanarTiling()
    edges = []
    lower = 2
    upper = 7

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

    # init polys and cp object
    polys = [torch.FloatTensor(poly) for poly in tiling.get_numpy_polys()]
    n_polys = len(polys)
    # print(polys)
    coms = torch.stack([poly.mean(dim=0) for poly in polys])
    connections = tiling.get_flagstone_connections()
    circle_polys = tiling.get_circle_polys()
    cp = CPLoss(polys, connections=connections, circle_polys=circle_polys, device=device)
    # init variables to be optimized
    positions = Variable(1.41 * coms, requires_grad=True)
    angles = Variable(torch.FloatTensor([np.pi / 12 for i in range(n_polys)]).to(device), requires_grad=True)
    print(angles.device)
    circle_centers = Variable(cp.initial_circle_centers(positions, angles), requires_grad=True)

    loss_curve = []
    optim = SGD([positions, angles, circle_centers], lr=.1, momentum=.0)
    for i in range(n_iters):
        optim.zero_grad()
        loss = cp(positions, angles, circle_centers)
        loss.backward()
        optim.step()

        loss_curve.append(loss.data.cpu().numpy())

        if i % 1000 == 0:
            print(f'step {i}, loss = {loss.data.cpu().numpy()}')
            tris = cp.mapped_points(positions, angles)
        if i == 0:
            plt.figure(figsize=(8, 8))
            show_cp(tris, connections, centers=circle_centers.data.cpu().numpy(), center_connections=circle_polys)

        if loss < 1e-10:
            break

    tris = cp.mapped_points(positions, angles)
    plt.figure()
    show_cp(tris, connections, centers=circle_centers.data.numpy(), center_connections=circle_polys, show=False)
    plt.figure()
    plt.semilogy(loss_curve)
    plt.show()
    saving.save_cp(filename, polygons=tris, connections=connections, centers=circle_centers.data.cpu().numpy(),
                   center_connections=circle_polys)
