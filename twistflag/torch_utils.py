import torch


def rotate_tensor(tensor, angles):
    s = torch.sin(angles)
    c = torch.cos(angles)
    rot_mat = torch.stack([c, s, -s, c], dim=-1).contiguous().view((-1, 2, 2))
    result = (tensor[:, :, None] * rot_mat).sum(dim=-2)
    return result


def one_hot(l, i):
    result = torch.zeros(l)
    result[i] = 1
    return result


if __name__ == '__main__':
    print(one_hot(10, 3))
