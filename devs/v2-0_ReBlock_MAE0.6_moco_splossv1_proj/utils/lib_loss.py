import torch
import torch.nn.functional as F


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


# Example usage:
b, n, d = 2, 5, 3  # Example dimensions
k = torch.randn(b, n, d)  # Random embeddings for k
q = k  # Random embeddings for q

# Calculate contrastive loss
loss = contrastive_loss_3d(k, q, is_norm=True)
print("Contrastive Loss:", loss.item())
