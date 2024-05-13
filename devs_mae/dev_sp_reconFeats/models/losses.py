import torch
from torch import nn
from torch.nn import functional as F
from models.helper import online_clustering, ot_assign, wkeans


def align_loss_3d(q_features, k_features, num_clusters=16, tau=0.2, sink_tau=0.1):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    with torch.no_grad():
        # Compute cosine similarity as dot product of k and q across all pairs
        gamma, centers = wkeans(k_features, num_clusters, dst='feats', iters=10, tau=sink_tau)
        gamma = gamma / torch.sum(gamma, dim=-1, keepdim=True).clip(min=1e-3)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    logits = torch.einsum('bnd,bkd->bnk', q_features, centers.detach())
    loss = -torch.sum(gamma.detach() * torch.log_softmax(logits / tau, dim=-1), dim=-1).mean()

    return loss


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