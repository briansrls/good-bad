"""Baseline scalar net + G/B variant in one file."""
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# Public helpers -----------------------------------------------------

def make_scalar():
    return MLP(2) # Changed to MLP(2) for Dual Scalar Model

def make_gb():
    return MLP(2)

# Paraconsistency‑aware loss ----------------------------------------

def gb_loss(pred, targ, lam=0.2):
    pG, pB = pred[:,0], pred[:,1]
    tG, tB = targ[:,0], targ[:,1]
    
    # Add a small epsilon to pG.log() and pB.log() to prevent log(0) if predictions are exactly 0.
    epsilon = 1e-7
    
    kl_g = nn.functional.kl_div(torch.log(pG + epsilon), tG, reduction='batchmean')
    kl_b = nn.functional.kl_div(torch.log(pB + epsilon), tB, reduction='batchmean')
    
    margin = lam * torch.mean((pG - pB) - (tG - tB))
    return kl_g + kl_b + margin 