import torch

def uniformity_loss(features):
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T 
    loss = sim.pow(2).mean()
    return loss