"""CLI: python train.py --epochs 3 --save runs/demo"""
import argparse
import torch
import yaml
import os
from rich.console import Console
from data.generator import build_split
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
# val_x, val_y = build_split(cfg['n_val']) # Validation data not used in this loop, can be added later

# Initialize models and optimizers
scalar_model = make_scalar()
gb_model = make_gb()

opt_s = torch.optim.AdamW(scalar_model.parameters(), lr=cfg['lr'])
opt_g = torch.optim.AdamW(gb_model.parameters(), lr=cfg['lr'])

# Rich console for logging
console = Console()

console.log(f"Starting training for {args.epochs} epochs. Models will be saved to '{args.save}'")

for ep in range(args.epochs):
    # --- Training loop (Dual Scalar Model) -----------------
    scalar_model.train()
    opt_s.zero_grad()
    scalar_outputs = scalar_model(train_x)
    loss_s_g = torch.nn.functional.binary_cross_entropy(scalar_outputs[:,0], train_y[:,0])
    loss_s_b = torch.nn.functional.binary_cross_entropy(scalar_outputs[:,1], train_y[:,1])
    loss_s = loss_s_g + loss_s_b
    loss_s.backward()
    opt_s.step()
    
    # --- Training loop (G/B Model) -------------------------
    gb_model.train()
    opt_g.zero_grad()
    gb_outputs = gb_model(train_x)
    loss_g = gb_loss(gb_outputs, train_y, lam=cfg.get('gb_loss_lambda', 0.2)) # Allow lam to be in cfg
    loss_g.backward()
    opt_g.step()
    
    console.log(f"Epoch {ep+1}/{args.epochs}: L_scalar={loss_s:.3f} | L_gb={loss_g:.3f}")

# Save models
os.makedirs(args.save, exist_ok=True)
torch.save(scalar_model.state_dict(), os.path.join(args.save, 'scalar_model.pth'))
torch.save(gb_model.state_dict(), os.path.join(args.save, 'gb_model.pth'))
console.log(f"Models saved to {args.save}") 