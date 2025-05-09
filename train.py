"""CLI: python train.py --epochs 3 --save runs/demo"""
import argparse
import torch
import yaml
import os
from rich.console import Console
from data.generator import build_split, N_ATOMS
from models.gb_model import make_scalar, make_gb, gb_loss

# Load configuration
try:
    with open('experiments.cfg') as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: experiments.cfg not found. Please create it.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing experiments.cfg: {e}")
    exit(1)

# Argument parser
parser = argparse.ArgumentParser(description="Train baseline and G/B models.")
parser.add_argument('--epochs', type=int, default=cfg.get('epochs', 3), 
                    help='Number of training epochs.')
parser.add_argument('--save', type=str, default='runs/demo', 
                    help='Directory to save trained models.')
args = parser.parse_args()

# Prepare data
train_x, train_y = build_split(cfg['n_train'])

# --- DIAGNOSTIC PRINTS FOR train_x ---
print(f"Shape of train_x: {train_x.shape}")
if train_x.numel() > 0:
    unique_train_x_rows = torch.unique(train_x, dim=0)
    print(f"Number of unique train_x rows: {unique_train_x_rows.shape[0]}")
    print(f"Sample train_x (first 5 rows):\n{train_x[:5]}")
    if unique_train_x_rows.shape[0] < min(N_ATOMS, cfg.get('n_train', 10)):
        print("WARNING: Very few unique input samples! Check data generation.")
else:
    print("train_x is empty!")
# --- END DIAGNOSTIC PRINTS ---

# val_x, val_y = build_split(cfg['n_val']) # Validation data not used in this loop, can be added later

# Initialize models and optimizers
scalar_model = make_scalar() # Still initialize for consistency, but won't train
gb_model = make_gb()

opt_s = torch.optim.AdamW(scalar_model.parameters(), lr=cfg['lr']) # Still init for consistency
# DIAGNOSTIC: Revert opt_g to AdamW to match minimal_pytorch_test.py
opt_g = torch.optim.AdamW(gb_model.parameters(), lr=cfg['lr'])

# --- DIAGNOSTIC: Print initial weights of gb_model final layer ---
print("Initial gb_model.final_layer weights:\n", gb_model.final_layer.weight.data)
if gb_model.final_layer.bias is not None:
    print("Initial gb_model.final_layer bias:\n", gb_model.final_layer.bias.data)
# --- END DIAGNOSTIC ---

# Rich console for logging
console = Console()

console.log(f"Starting training for {args.epochs} epochs. Models will be saved to '{args.save}'")

for ep in range(args.epochs):
    # # --- Training loop (Dual Scalar Model) --- # COMMENTED OUT FOR DIAGNOSTIC
    # scalar_model.train()
    # opt_s.zero_grad()
    # scalar_outputs = scalar_model(train_x)
    # loss_s_g = torch.nn.functional.binary_cross_entropy(scalar_outputs[:,0], train_y[:,0])
    # loss_s_b = torch.nn.functional.binary_cross_entropy(scalar_outputs[:,1], train_y[:,1])
    # loss_s = loss_s_g + loss_s_b
    # loss_s.backward()
    # opt_s.step()
    
    # --- Training loop (G/B Model) -------------------------
    gb_model.train()
    gb_outputs = gb_model(train_x) # Should be [N,1]

    # --- DIAGNOSTIC: Print outputs and targets for gb_model's loss ---
    if ep < 2: # Print for first 2 epochs only to see initial state and after 1 step
        print(f"--- Epoch {ep+1} Diagnostic for gb_model loss calculation ---")
        print(f"  gb_outputs (predictions, first 5): {gb_outputs.detach()[:5].squeeze().tolist()}")
        print(f"  train_y[:,0] (targets for G, first 5): {train_y[:5,0].tolist()}")
    # --- END DIAGNOSTIC ---

    loss_g = gb_loss(gb_outputs, train_y, 
                       lam_margin=cfg.get('gb_loss_lambda_margin', 0.0), 
                       lam_confidence=cfg.get('gb_loss_lambda_confidence', 0.0))
    
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    
    # --- DIAGNOSTIC: Print weights of gb_model final layer after step ---
    if ep < 5: # Print for first 5 epochs
        print(f"Epoch {ep+1} gb_model.final_layer weights:\n", gb_model.final_layer.weight.data)
        if gb_model.final_layer.bias is not None:
            print(f"Epoch {ep+1} gb_model.final_layer bias:\n", gb_model.final_layer.bias.data)
    # --- END DIAGNOSTIC ---
    
    # console.log(f"Epoch {ep+1}/{args.epochs}: L_scalar={loss_s:.3f} | L_gb={loss_g:.3f}") # loss_s undefined
    console.log(f"Epoch {ep+1}/{args.epochs}: L_gb={loss_g:.3f}") # Log only L_gb

# Save models
os.makedirs(args.save, exist_ok=True)
torch.save(scalar_model.state_dict(), os.path.join(args.save, 'scalar_model.pth')) # Save scalar anyway
torch.save(gb_model.state_dict(), os.path.join(args.save, 'gb_model.pth'))
console.log(f"Models saved to {args.save}") 