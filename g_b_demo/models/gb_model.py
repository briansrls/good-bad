"""Baseline scalar net + G/B variant in one file."""
import torch, torch.nn as nn
import torch.nn.functional as F # For F.binary_cross_entropy

class MLP(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        # Using the single linear layer for this diagnostic
        self.final_layer = nn.Linear(10, out_dim) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.final_layer(x)
        output = self.sigmoid(logits)
        return output

# Public helpers -----------------------------------------------------

def make_scalar():
    return MLP(2) # Remains 2 outputs, expected to fail

def make_gb():
    # DIAGNOSTIC: gb_model will now have only 1 output neuron
    return MLP(1) 

# Paraconsistency‑aware loss ----------------------------------------

def gb_loss(pred, targ, lam_margin=0.0, lam_confidence=0.0):
    # pred is [batch_size, 1] from the modified make_gb
    # targ is still [batch_size, 2]
    
    pG_as_Nx1 = pred           # pred is already [batch_size, 1]
    tG_as_Nx1 = targ[:,0].unsqueeze(1) # Reshape target G to [batch_size, 1]
    
    # DIAGNOSTIC: Use Mean Squared Error loss for the G component
    loss_g_mse = F.mse_loss(pG_as_Nx1, tG_as_Nx1, reduction='mean')
    
    return loss_g_mse
