import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_curvature_pca_ball(queries, refs, num_neighbors=10, radius=0.1, eps=1e-8):
    # batch_size, num_points, num_features = points.size()
    if radius <= 0:
        dis = torch.cdist(queries, queries)
        radius = torch.topk(dis, k=2, largest=False)[0][:, 1].mean(dim=-1, keepdim=True)
        
    idx = query_ball_point(radius, num_neighbors, refs, queries)  # [B, K, n_sampling]
    mean_node = torch.mean(queries, dim=-2, keepdim=True)
    cat_points = torch.cat([refs, mean_node], dim=1)
    os = torch.ones((refs.shape[0], refs.shape[1])).to(refs)
    neighbor_points = index_gather(cat_points, idx)  # [B, n_point, n_sample, 3]
    cat_os = torch.cat([os, torch.zeros_like(os[:, :1])], dim=-1).unsqueeze(-1)
    neighbor_os = index_gather(cat_os, idx).squeeze(-1)
    # Calculate covariance matrices
    inners = torch.sum(neighbor_os, dim=-1, keepdim=True)
    # w_neighbor_points = torch.einsum('bnkd,bnk->bnkd', neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    centered_neighbor_points = neighbor_points - queries.unsqueeze(2)
    w_centered_neighbor_points = torch.einsum(
        'bnkd,bnk->bnkd', centered_neighbor_points, neighbor_os) / inners.unsqueeze(-1)
    cov_matrices = torch.matmul(centered_neighbor_points.transpose(-2, -1), w_centered_neighbor_points)
    # Calculate eigenvalues and curvatures
    eigenvalues = torch.linalg.eigvalsh(cov_matrices + eps)
    lmd = [eigenvalues[:, :, 2].clip(min=10*eps), eigenvalues[:, :, 1], eigenvalues[:, :, 0]]
    features = [(lmd[0] - lmd[1]) / lmd[0], (lmd[1] - lmd[2]) / lmd[0],  lmd[2] / lmd[0]]

    return torch.stack(features, dim=-1)


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Inputs:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = torch.cdist(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx.clip(min=0, max=N)


def index_gather(points, idx):
    """
    Input:
        points: input feats semdata, [B, N, C]
        idx: sample index semdata, [B, S, K]
    Return:
        new_points:, indexed feats semdata, [B, S, K, C]
    """
    dim = points.size(-1)
    n_clu = idx.size(1)
    # device = points.device
    view_list = list(idx.shape)
    view_len = len(view_list)
    # feats_shape = view_list
    xyz_shape = [-1] * (view_len + 1)
    xyz_shape[-1] = dim
    feats_shape = [-1] * (view_len + 1)
    feats_shape[1] = n_clu
    batch_indices = idx.unsqueeze(-1).expand(xyz_shape)
    points = points.unsqueeze(1).expand(feats_shape)
    new_points = torch.gather(points, dim=-2, index=batch_indices)
    return new_points

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)  # idx = knn(x[:, :3], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

# B, C, H, W = feat.size()
# feat = feat.view(B, C, H * W)
# u, s, v = torch.linalg.svd(feat, full_matrices=False)

# # Asssume feats [:, 128]
# feat = feat - s[:, 118:].unsqueeze(2) * u[:, :, 118:].bmm(v[:, 118:, :])
# feat = feat.view(B, C, H, W)