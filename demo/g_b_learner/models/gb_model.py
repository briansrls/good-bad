"""Baseline scalar net + G/B variant in one file."""
import torch, torch.nn as nn
import torch.nn.functional as F # For F.binary_cross_entropy

# Default hidden dimension for the model
DEFAULT_HIDDEN_DIM = 32 # Changed from 64 to 32

class GBModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=DEFAULT_HIDDEN_DIM): # Use the new default
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer now produces 2 scores: one for goodness, one for badness
        self.final_layer = nn.Linear(hidden_dim, 2) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Outputs raw logits for each of the two scores
        # Sigmoid will be applied in the loss function (BCEWithLogitsLoss is preferred)
        # or if gb_loss expects probabilities, apply sigmoid here.
        # For now, let's assume gb_loss will handle logits with BCEWithLogitsLoss.
        return self.final_layer(x)

# Public helpers -----------------------------------------------------

# Assuming N_ATOMS is defined elsewhere, e.g., in data.generator
# If not, you might need to pass input_dim to make_gb or define N_ATOMS here.
# For now, let's hardcode a common value or make it an argument.
DEFAULT_INPUT_DIM = 10 # Example: Replace with N_ATOMS if available

def make_gb(input_dim=DEFAULT_INPUT_DIM, hidden_dim_override=None):
    # Allow overriding hidden_dim if needed, otherwise use the class default
    actual_hidden_dim = hidden_dim_override if hidden_dim_override is not None else DEFAULT_HIDDEN_DIM
    return GBModel(input_dim=input_dim, hidden_dim=actual_hidden_dim)

def make_scalar(input_dim=DEFAULT_INPUT_DIM, hidden_dim_override=None): # Kept for compatibility if other parts of repo use it
    # This model is not suitable for the bivalent logic directly but might be used for comparison
    actual_hidden_dim = hidden_dim_override if hidden_dim_override is not None else DEFAULT_HIDDEN_DIM
    model = nn.Sequential(
        nn.Linear(input_dim, actual_hidden_dim),
        nn.ReLU(),
        nn.Linear(actual_hidden_dim, actual_hidden_dim),
        nn.ReLU(),
        nn.Linear(actual_hidden_dim, 1) # Original scalar output
    )
    return model

# Paraconsistency‑aware loss ----------------------------------------

def gb_loss(outputs, targets, good_weight=1.0, bad_weight=1.0):
    """
    Calculates bivalent loss.
    Assumes outputs are raw logits from the model, shape [N, 2].
    outputs[:, 0] are logits for "goodness".
    outputs[:, 1] are logits for "badness".
    Assumes targets are labels {0, 1}, shape [N, 2].
    targets[:, 0] is the target for "goodness".
    targets[:, 1] is the target for "badness".
    """
    if outputs.shape[1] != 2 or targets.shape[1] != 2:
        raise ValueError(f"Outputs and targets must have 2 columns for bivalent loss. Got shapes: {outputs.shape}, {targets.shape}")

    # BCEWithLogitsLoss is numerically more stable than Sigmoid + BCE
    # It expects raw logits as input.
    loss_goodness = F.binary_cross_entropy_with_logits(outputs[:, 0], targets[:, 0], reduction='mean')
    loss_badness = F.binary_cross_entropy_with_logits(outputs[:, 1], targets[:, 1], reduction='mean')
    
    total_loss = (good_weight * loss_goodness) + (bad_weight * loss_badness)
    return total_loss

# --- Old gb_loss for reference, to be removed or commented out ---
# def gb_loss_old(output, target, lam_margin=0.0, lam_confidence=1.0):
#     ... original implementation ...
