import copy
import torch
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from torch import nn
import numpy as np
import torch.nn.functional as F


def evaluate_svm(train_features, train_labels, test_features, test_labels, md='svc'):
    if md == 'sgd':
        clf = SGDClassifier(max_iter=1000, tol=1e-3)
    elif md == 'lsvc':
        clf = LinearSVC(C=0.1)
    else:
        clf = SVC(C=0.042, kernel='linear')
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return metrics.accuracy_score(test_labels, pred), metrics.balanced_accuracy_score(test_labels, pred)


def svm_data(loader, encoder):
    encoder.eval()
    features = list()
    label = list()
    for _, data in enumerate(loader, 0):
        points, target = data[0], data[-1]
        points, target = points.cuda(), target.cuda()
        feature = encoder(points, is_eval=True)
        target = target.view(-1)
        features.append(feature.data)
        label.append(target.data)
    features = torch.cat(features, dim=0)
    label = torch.cat(label, dim=0)

    return features, label


def validate(train_loader, test_loader, encoder, best_acc, best_avg_acc, logger, md='svc'):
    # feature extraction
    with torch.no_grad():
        train_features, train_label = svm_data(train_loader, encoder)
        test_features, test_label = svm_data(test_loader, encoder)
    # train svm
    svm_acc, svm_avg_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                                        test_features.data.cpu().numpy(), test_label.data.cpu().numpy(), md)

    if svm_acc > best_acc or svm_avg_acc > best_avg_acc:
        best_acc = svm_acc
        best_avg_acc = svm_avg_acc

    encoder.train()
    logger.info('SVM classification results: Overall Acc=%f,\t Mean Class Acc=%f,\t best svm acc=%f,'
                '\t best avg svm acc=%f' % (svm_acc, svm_avg_acc, best_acc, best_avg_acc))
    print('SVM classification results: Overall Acc=%f,\t Mean Class Acc=%f,\t Best Overall Acc=%f,'
          '\t Best Mean Class Acc=%f' % (svm_acc, svm_avg_acc, best_acc, best_avg_acc))
    return svm_acc, svm_avg_acc


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)


class MeanShiftTorch(nn.Module):
    def __init__(self, bandwidth=0.05, max_iter=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, feats):
        """
        params: A: [N, 3]
        """
        N, c = feats.size()
        it = 0
        new_feats = feats.clone()
        while True:
            it += 1
            Ar = feats.view(1, N, c).repeat(N, 1, 1)
            Cr = new_feats.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            shift_feats = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            delta = torch.norm(shift_feats - new_feats, dim=1)
            # print(C, new_C)
            new_feats = shift_feats
            if torch.max(delta) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = feats.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return new_feats[max_idx, :], labels

    def fit_batch_npts(self, points):
        """
        params: A: [bs, pts, 3]
        """
        bs, N, cn = points.size()
        it = 0
        C = points.clone()
        while True:
            it += 1
            Ar = points.view(bs, 1, N, cn).repeat(1, N, 1, 1)
            Cr = C.view(bs, N, 1, cn).repeat(1, 1, N, 1)
            dis = torch.norm(Cr - Ar, dim=4)
            w = gaussian_kernel(dis, self.bandwidth).view(bs, N, N, 1)
            new_C = torch.sum(w * Ar, dim=-1) / torch.sum(w, dim=-1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=-1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = points.view(N, 1, cn).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=4)
        num_in = torch.sum(dis < self.bandwidth, dim=3)
        # print(num_in.size())
        max_num, max_idx = torch.max(num_in, 2)
        dis = torch.gather(dis, 2, max_idx.reshape(bs, 1))
        labels = dis < self.bandwidth
        ctrs = torch.gather(
            C, 2, max_idx.reshape(bs, 1, 1).repeat(1, 1, cn))
        return ctrs, labels


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, n_point, is_center=False):
    """
    Input:
        pts: point cloud data, [B, N, 3]
        n_point: number of samples
    Return:
        sub_xyz: sampled point cloud index, [B, n_point]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_point, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(n_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


class KNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, src, dst):
        pairwise_distance = torch.cdist(src, dst)
        values, idx = pairwise_distance.topk(k=self.k, dim=-1, largest=False)
        return values, idx


def knn(x, y, k):
    pairwise_distance = torch.cdist(x, y)
    idx = pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1]
    return idx


def fps(xyz, npoint, is_center=True):
    centroids = farthest_point_sample(xyz, npoint, is_center)
    return index_points(xyz, centroids)


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = False, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)
            log_alpha_padded = torch.nan_to_num(log_alpha_padded, nan=0.0)
            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)
            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
            log_alpha = torch.nan_to_num(log_alpha, nan=0.0)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return torch.exp(log_alpha)


def ot_assign(x, y, tau=0.1, thresh=1e-3, max_iter=30, is_unin=False):
    with torch.no_grad():
        # Calculate pairwise cost between elements in x and y
        cost = torch.cdist(x, y)

        # Calculate the second smallest distances in y to y to set a threshold for masking
        second_smallest = torch.topk(torch.cdist(y, y), k=2, largest=False).values[:, :, 1]
        mask_thresh = second_smallest.mean(dim=1, keepdim=True).unsqueeze(-1)

        # Apply threshold mask to the cost
        mask = (cost < mask_thresh).to(cost.dtype)
        # Logarithm of alpha values, scaled by the maximum across the cost matrix for numerical stability
        max_cost = torch.max(cost, dim=-1, keepdim=True).values
        log_alpha = (max_cost - cost) * mask

        if is_unin:
            device = log_alpha.device
            batch_size, num_x, num_y = log_alpha.shape
            # Initialize uniform distributions for both marginals
            p = torch.full((batch_size, num_x), 1.0 / num_x, dtype=torch.float, device=device)
            q = torch.full((batch_size, num_y), 1.0 / num_y, dtype=torch.float, device=device)

            # Call Sinkhorn with uniform distributions
            gamma = sinkhorn(log_alpha / tau, p, q, thresh=thresh, max_iter=max_iter)
        else:
            # Call RPM version of Sinkhorn (assuming parameters are adapted for this function)
            gamma = sinkhorn_rpm(log_alpha / tau, n_iters=max_iter, eps=thresh)

    return gamma


def gmm_params(gamma, pts, return_sigma=False):
    """
    gamma: B feats N feats J
    pts: B feats N feats D
    """
    # pi: B feats J
    D = pts.size(-1)
    pi = gamma.mean(dim=1)
    npi = pi * gamma.shape[1] + 1e-5
    # p: B feats J feats D
    mu = gamma.transpose(1, 2) @ pts / npi.unsqueeze(2)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu


def log_boltzmann_kernel(log_alpha, u, v, epsilon):
    kernel = (log_alpha + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(cost, p=None, q=None, epsilon=1e-2, thresh=1e-2, max_iter=100):
    # Initialise approximation vectors in log domain
    if p is None or q is None:
        batch_size, num_x, num_y = cost.shape
        device = cost.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    return gamma


def online_clustering(log_alpha, p=None, q=None, epsilon=1e-2, thresh=1e-5, max_iter=20):
    if p is None or q is None:
        num_x, num_y = log_alpha.shape
        device = log_alpha.device
        if p is None:
            p = torch.empty(num_x, dtype=torch.float, requires_grad=False, device=device).fill_(1.0 / num_x)
        if q is None:
            q = torch.empty(num_y, dtype=torch.float, requires_grad=False, device=device).fill_(1.0 / num_y)
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, gmmlib iterations
    for _ in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(log_alpha, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff).detach()
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(log_alpha, u, v, epsilon)
    gamma = torch.exp(K)
    return torch.argmax(gamma, dim=-1, keepdim=False).long()  # size N


def wkeans(x, num_clusters, dst='feats', iters=10, tau=0.1):
    bs, num, _ = x.shape
    # ids = torch.randperm(num)[:num_clusters]
    # centroids = x[:, ids, :]
    ids = farthest_point_sample(F.normalize(copy.deepcopy(x), dim=-1), num_clusters, is_center=True)
    centroids = index_points(x, ids)
    gamma = torch.zeros((bs, num, num_clusters), requires_grad=False).to(x)
    for i in range(iters):
        if dst == 'eu':
            log_alpha = -torch.cdist(x, centroids)
        else:
            log_alpha = torch.einsum('bnd,bmd->bnm', x, centroids)
        gamma = sinkhorn(log_alpha / tau, max_iter=10)
        gamma = gamma / torch.sum(gamma, dim=1, keepdim=True).clip(min=1e-4)
        centroids = torch.einsum('bmk,bmd->bkd', gamma, x)
    return gamma, centroids


def assignment(x, centroids, iters=10, tau=0.1, is_norm=True):
    bs, num, _ = x.shape
    num_clusters = centroids.size(1)
    gamma = torch.zeros((bs, num, num_clusters), requires_grad=False).to(x)
    for i in range(iters):
        log_alpha = torch.einsum('bnd,md->bnm', x, centroids)
        gamma = sinkhorn(log_alpha / tau, max_iter=10)
        gamma = gamma / torch.sum(gamma, dim=1, keepdim=True).clip(min=1e-4)
        if is_norm:
            centroids = F.normalize(centroids, dim=-1)
    return gamma
