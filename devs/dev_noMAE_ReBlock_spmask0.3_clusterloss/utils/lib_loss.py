import torch
from torch import nn
from torch.nn import functional as F
# from torch_scatter import scatter_mean

from utils.lib_utilis import online_clustering, ot_assign


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss.mean()

def contrastive_loss_3d(k, q, tau=0.1, is_norm=False):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    b, n, d = k.shape  # Assuming k and q have the same shape [b, n, d]

    # Normalize k and q along the feature dimension
    if is_norm:
        k = F.normalize(k, dim=2)
        q = F.normalize(q, dim=2)
    # Compute cosine similarity as dot product of k and q across all pairs
    logits = torch.einsum('bnd,bmd->bnm', [k, q]) / tau

    # Create labels for positive pairs: each sample matches with its corresponding one in the other set
    labels = torch.arange(n).repeat(b).to(logits.device)
    labels = labels.view(b, n)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

def align_loss_3d(pts, logits, tau=0.1, sink_tau=0.1, is_unin=False):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    
    k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    torch.Tensor: Scalar tensor representing the loss.
    """
    with torch.no_grad():# Compute cosine similarity as dot product of k and q across all pairs
        scores = torch.softmax(logits / tau, dim=1)
        protos = torch.einsum('bnd,bnk->bkd', [pts, scores]) / torch.sum(
            scores, dim=1).clip(min=1e-3).unsqueeze(-1)
        gamma = ot_assign(pts, protos, tau=sink_tau, thresh=1e-3, max_iter=10, is_unin=is_unin)
        gamma = gamma / torch.sum(gamma, dim=-1, keepdim=True).clip(min=1e-3)# Use log_softmax for numerical stability and compute the cross-entropy loss
    loss = -torch.sum(gamma.detach() * torch.log_softmax(logits / tau, dim=-1), dim=-1).mean()

    return loss



def contrastive_loss_2d(k, q, tau=0.1, is_norm=False):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    b = k.size(0)
    # Normalize k and q along the feature dimension
    if is_norm:
        k = F.normalize(k, dim=1)
        q = F.normalize(q, dim=1)
    # Compute cosine similarity as dot product of k and q across all pairs
    logits = torch.einsum('md,nd->mn', [k, q]) / tau

    # Create labels for positive pairs: each sample matches with its corresponding one in the other set
    labels = torch.arange(b).to(logits.device)
    labels = labels.view(-1)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


class ClusterContrastLoss(nn.Module):
    def __init__(self, dim, k=64, mu=0.9999, temperature=0.1, base_temperature=2, lamb=25, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # Sinkhorn-Knopp stuff
        self.num_classes = k  # number of clusters
        self.k_ban = 10
        # initiate labels as shuffled.
        self.dev = device
        self.dtype = torch.float32
        self.lamb = lamb  # the parameter lambda in the SK algorithm
        # MEM stuff
        self.dim = dim
        self.mu = mu

        # cluster_center
        self.cluster_center = torch.randn((self.num_classes, self.dim), requires_grad=False).to(self.dev)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=-1)
        self.new_cluster_center = torch.zeros((self.num_classes, self.dim)).to(self.dev).detach()

        self.pixel_update_freq = 10  # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 5

        self.point_queue = torch.randn((self.num_classes, self.pixel_size, self.dim), requires_grad=False).to(device)
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=-1)
        self.point_queue_ptr = torch.zeros(self.num_classes, dtype=torch.long, requires_grad=False).to(device)

    def _update_operations(self):
        self.cluster_center = self.cluster_center * self.mu + self.new_cluster_center * (1 - self.mu)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=-1).detach_()

    def _queue_operations(self, feats, labels):
        this_feat = feats.contiguous().view(-1, self.dim)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)
            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            updata_cnt = min(num_pixel, self.pixel_update_freq)
            feat = this_feat[perm[:updata_cnt], :]
            ptr = int(self.point_queue_ptr[lb])

            if ptr + updata_cnt > self.pixel_size:
                self.point_queue[lb, -updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=-1).detach_()
                self.point_queue_ptr[lb] = 0
            else:
                self.point_queue[lb, ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=-1).detach_()
                self.point_queue_ptr[lb] = (self.point_queue_ptr[lb] + updata_cnt) % self.pixel_size

    def _assigning_class_labels(self, X, y_hat):
        batch_size = X.shape[0]

        # Initializing lists for storing results
        X_, y_, X_contrast, y_contrast = [], [], [], []

        # Iterate over each sample in the batch
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_x = X[ii]
            # Identify unique classes predicted in this sample
            unique_classes = torch.unique(this_y_hat)
            # Process each class
            for cls_id in unique_classes:
                # Find indices for hard and easy examples
                indices = (this_y_hat == cls_id)
                # Only process if there are any relevant samples
                if indices.any():
                    # Gather the points and labels for contrastive learning
                    xc = this_x[indices]
                    yc = this_y_hat[indices]

                    # Collect data for the positive and contrastive parts
                    X_.append(xc)
                    y_.append(yc)
                    X_contrast.append(self.cluster_center[cls_id])
                    y_contrast.append(cls_id)

        # Concatenate all results to form the final output tensors
        X_ = torch.cat(X_, dim=0).float()
        y_ = torch.cat(y_, dim=0).float()
        X_contrast = torch.stack(X_contrast, dim=0).float()
        y_contrast = torch.stack(y_contrast, dim=0).float()

        return X_, y_, X_contrast, y_contrast

    def _batch_centers(self, feats, labels):
        c_labels = torch.arange(self.num_classes).long().to(self.dev)
        new_feats = torch.cat((feats, self.cluster_center), dim=0)
        new_labels = torch.cat((labels, c_labels), dim=0).long()
        new_cluster_center = scatter_mean(new_feats, new_labels, dim=0, dim_size=self.num_classes).detach()
        return new_cluster_center

    def _sample_negative(self, k):
        class_num, cache_size, feat_size = self.point_queue.shape
        # Check if k is not greater than cache_size
        if k > cache_size:
            raise ValueError("k cannot be greater than the number of samples per class (cache_size)")

        # Generate random indices for all classes at once
        indices = torch.randperm(cache_size)[:k].repeat(class_num, 1)  # Repeat for each class
        class_indices = torch.arange(class_num, device=self.dev).view(-1, 1)  # Expand class indices

        # Gather the samples using the generated indices
        X_ = self.point_queue[class_indices, indices].view(-1, feat_size)
        y_ = class_indices.repeat(1, k).view(-1, 1).float()  # Expand class indices for labels

        return X_, y_

    def _ppc2_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor

        X_contrast, y_contrast = self._sample_negative(self.pixel_size - self.k_ban)
        y_contrast = y_contrast.contiguous().view(-1, 1)

        contrast_feature = X_contrast
        contrast_label = y_contrast
        mask = torch.eq(y_anchor, contrast_label.T).float().to(self.dev)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        # neg_logits denotes the sum of logits of all negative pairs of one anchor
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)  # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _ppc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor
        anchor_num = X_anchor.shape[0]

        contrast_feature = X_anchor
        contrast_label = y_anchor

        mask = torch.eq(y_anchor, contrast_label.T).float().to(self.dev)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num).view(
            -1, 1).to(self.dev), 0)
        mask = mask * logits_mask
        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        # neg_logits denotes the sum of logits of all negative pairs of one anchor
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)  # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)
        o_i = torch.where(mask.sum(1) != 0)[0]

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _pcc_contrastive(self, X_anchor, y_anchor, X_contrast, y_contrast):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        y_contrast = y_contrast.contiguous().view(-1, 1)

        anchor_feature = X_anchor
        anchor_label = y_anchor

        contrast_feature = X_contrast
        contrast_label = y_contrast

        mask = torch.eq(anchor_label, contrast_label.T).float().to(self.dev)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # it is to avoid the numerical overflow

        neg_mask = 1 - mask
        neg_logits = torch.exp(logits) * neg_mask
        # neg_logits denotes the sum of logits of all negative pairs of one anchor
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)  # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, off_feats):
        bs, n, dim = off_feats.size()
        n_off_feats = torch.nn.functional.normalize(off_feats.reshape(-1, dim).contiguous(), p=2, dim=-1)
        cluster_feats = torch.cat([n_off_feats, self.point_queue.view(-1, dim)], dim=0)
        log_alpha = torch.einsum('nd,kd->nk', cluster_feats, self.cluster_center)
        labels = online_clustering(log_alpha * self.lamb)[:bs * n]

        feats = nn.functional.normalize(feats, p=2, dim=-1)
        # logits = torch.einsum('bnd,kd->bk', feats, self.cluster_center)

        labels = labels.contiguous().view(bs, -1)
        # predict = torch.argmax(logits * self.lamb, dim=-1, keepdim=False).long()
        # predict = predict.contiguous().view(bs, -1)

        # feats_, labels_, feats_contrast, labels_contrast = self._assigning_class_labels(feats, labels)

        feats_, labels_ = feats.reshape(-1, dim).contiguous(), labels.reshape(-1).contiguous()

        loss = self._ppc2_contrastive(feats_, labels_)
        loss += self._ppc_contrastive(feats_, labels_)
        # loss += self._pcc_contrastive(feats_, labels_, feats_contrast, labels_contrast)

        self.new_cluster_center = self._batch_centers(feats_, labels_)
        self._queue_operations(feats_, labels_.long())
        self._update_operations()
        return loss
