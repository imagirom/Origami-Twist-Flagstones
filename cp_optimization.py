import torch
import torch.nn as nn
from torch_utils import rotate_tensor


class CPLoss(nn.Module):
    def __init__(self, base_polys, connections=None, connection_lengths=None, circle_polys=None):
        super(CPLoss, self).__init__()

        base_poly_ids = []
        base_points = []
        base_offsets = []
        reverse_poly_ids = []
        for i, poly in enumerate(base_polys):
            current_id = len(base_poly_ids)
            reverse_poly_ids.append(torch.range(current_id, current_id + len(poly) - 1).long())
            base_poly_ids.extend((i,) * len(poly))
            poly = torch.FloatTensor(poly)
            com = torch.mean(poly, dim=0)
            base_points.extend(poly - com)
            base_offsets.append(com)
        self.base_points = torch.stack(base_points, dim=0)
        self.poly_ids = torch.LongTensor(base_poly_ids)
        self.reverse_poly_ids = reverse_poly_ids
        self.base_offsets = torch.stack(base_offsets, dim=0)

        if connections is None or len(connections) == 0:
            self.has_connections = False
        else:
            self.has_connections = True
            connection_ids = []
            for con in connections:
                ind1 = self.reverse_poly_ids[con[0, 0]][0] + float(con[0, 1])
                ind2 = self.reverse_poly_ids[con[1, 0]][0] + float(con[1, 1])
                connection_ids.append(torch.LongTensor([ind1, ind2]))
            self.connection_ids = torch.stack(connection_ids)
            self.connected_polys = torch.LongTensor(connections[:, :, 0])

            self.min_poly_dist = torch.FloatTensor([1])  # TODO

            if connection_lengths is not None:
                self.connection_lengths = torch.FloatTensor(connection_lengths)
            else:
                self.connection_lengths = self.get_connection_lengths()

        if circle_polys is None or len(circle_polys) == 0:
            self.has_circle_polys = False
        else:
            self.has_circle_polys = True
            circle_poly_ids = []
            circle_poly_grouping = []
            reverse_circle_poly_grouping = []
            current_id = 0
            for i, poly in enumerate(circle_polys):
                reverse_circle_poly_grouping.append(torch.range(current_id, current_id + len(poly) - 1).long())
                circle_poly_grouping.extend((i,) * len(poly))
                ind = torch.stack([self.reverse_poly_ids[poly[i, 0]][0] + float(poly[i, 1]) for i in range(len(poly))])
                circle_poly_ids.append(torch.LongTensor(ind))
                current_id += len(ind)

            self.circle_poly_ids = torch.cat(circle_poly_ids, dim=0)
            self.reverse_circle_poly_grouping = reverse_circle_poly_grouping
            self.circle_poly_grouping = torch.LongTensor(circle_poly_grouping)

    def get_connection_lengths(self):
        points = self.base_points + self.base_offsets[self.poly_ids]
        connections = points[self.connection_ids]
        return torch.sqrt(torch.sum((connections[:, 0] - connections[:, 1]) ** 2, dim=-1))

    def mapped_points(self, positions, angles):
        points = self.base_points
        points = rotate_tensor(points, angles[self.poly_ids])
        coms = (self.base_offsets + positions)[self.poly_ids]
        points = points + coms
        return [points[ind].data.numpy() for ind in self.reverse_poly_ids]

    def initial_circle_centers(self, positions, angles):
        points = self.base_points
        points = rotate_tensor(points, angles[self.poly_ids])
        coms = (self.base_offsets + positions)[self.poly_ids]
        points = points + coms
        circle_poly_points = points[self.circle_poly_ids]
        return torch.stack([circle_poly_points[group].mean(dim=0) for group in self.reverse_circle_poly_grouping])

    def forward(self, positions, angles, circle_centers=None):
        loss = 0
        points = self.base_points
        points = rotate_tensor(points, angles[self.poly_ids])
        coms = (self.base_offsets + positions)
        points = points + coms[self.poly_ids]

        if self.has_connections:
            connections = points[self.connection_ids]
            dists = torch.sqrt(torch.sum((connections[:, 0] - connections[:, 1]) ** 2, dim=-1))
            connection_loss = ((dists - self.connection_lengths) ** 2).sum()
            loss += connection_loss

            dists = torch.sqrt(
                torch.sum((coms[self.connected_polys[:, 0]] - coms[self.connected_polys[:, 1]]) ** 2, dim=-1))
            poly_dist_loss = (torch.max(self.min_poly_dist.expand_as(dists) - dists, torch.zeros_like(dists)) ** 2).sum()
            loss += poly_dist_loss

        if self.has_circle_polys:
            circle_poly_points = points[self.circle_poly_ids]
            dists = torch.sqrt(torch.sum((circle_poly_points - circle_centers[self.circle_poly_grouping]) ** 2, dim=-1))
            avg_dists = []
            for group in self.reverse_circle_poly_grouping:
                avg_dists.append(dists[group].mean())
            avg_dists = torch.Tensor(avg_dists)[self.circle_poly_grouping]
            circle_loss = (((dists - avg_dists) / avg_dists) ** 2).mean()
            loss += 50 * circle_loss

        return loss
