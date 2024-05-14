import torch
from torch import nn
from torch.nn import functional as F
from models.helper import online_clustering, ot_assign, wkeans, assignment


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


def align_loss_3d(q_features, k_features, centers, tau=0.2, sink_tau=0.1):
    """
    Compute contrastive loss between k and q using NT-Xent loss.
    Args:
    - k (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - q (torch.Tensor): Tensor of shape [b, n, d] containing a batch of embeddings.
    - tau (float): A temperature scaling factor to control the separation of the distributions.
    Returns:
    - torch.Tensor: Scalar tensor representing the loss.
    """
    k_features = F.normalize(k_features, dim=-1)
    centers = F.normalize(centers, dim=-1)
    q_features = F.normalize(q_features, dim=-1)
    with torch.no_grad():
        # Compute cosine similarity as dot product of k and q across all pairs
        gamma = assignment(k_features, centers, iters=10, tau=sink_tau)
        gamma = gamma / torch.sum(gamma, dim=-1, keepdim=True).clip(min=1e-3)
    # Use log_softmax for numerical stability and compute the cross-entropy loss
    logits = torch.einsum('bnd,kd->bnk', q_features, centers)
    loss = -torch.sum(gamma.detach() * torch.log_softmax(logits / tau, dim=-1), dim=-1).mean()
    n_matrix = torch.matmul(centers, centers.transpose(1, 0))
    device = n_matrix.device
    n_loss = F.mse_loss(torch.eye(n_matrix.size(0)).to(device), n_matrix)

    return loss + 0.01*n_loss


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