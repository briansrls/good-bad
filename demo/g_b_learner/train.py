"""CLI: python train.py --epochs 3 --save runs/demo"""
import argparse
import torch
import yaml
import os
from rich.console import Console
from data.generator import build_split, N_ATOMS
from models.gb_model import make_gb, gb_loss

# Load configuration
CONFIG_PATH = 'experiments.cfg' # experiments.cfg is in the same directory as this script
try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_PATH}' not found in the current directory.")
    print(f"Please ensure '{CONFIG_PATH}' exists alongside {__file__}")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing {CONFIG_PATH}: {e}")
    exit(1)

# Argument parser
parser = argparse.ArgumentParser(description="Train G/B models with bivalent output.")
parser.add_argument('--epochs', type=int, default=cfg.get('epochs', 3), 
                    help='Number of training epochs.')
parser.add_argument('--save', type=str, default='runs/bivalent_demo',
                    help='Directory to save trained models.')
args = parser.parse_args()

# Ensure the save directory exists
os.makedirs(args.save, exist_ok=True)

# Initialize Rich console for logging
console = Console()

# Prepare data: generate a new KB for this training run and get it back
train_x, train_y, fixed_kb_train = build_split(cfg['n_train'], cfg=cfg, existing_kb=None) 

# --- DIAGNOSTIC PRINTS FOR train_x ---
print(f"Shape of train_x: {train_x.shape}")
if train_x.numel() > 0:
    unique_train_x_rows = torch.unique(train_x, dim=0)
    print(f"Number of unique train_x rows: {unique_train_x_rows.shape[0]}")
    # Only print sample train_x if N_ATOMS is small to avoid large output
    if N_ATOMS <= 20: 
        print(f"Sample train_x (first 5 rows):\n{train_x[:5]}")
    else:
        print(f"Sample train_x (first 5 rows, first 20 features):\n{train_x[:5, :20]}")

    if unique_train_x_rows.shape[0] < min(N_ATOMS, cfg.get('n_train', 10)): # This check might need adjustment if N_ATOMS is very large
        print("WARNING: Very few unique input samples relative to N_ATOMS! Check data generation or n_train.")
else:
    print("train_x is empty!")
# --- END DIAGNOSTIC PRINTS ---

# Save the fixed_kb used for this training run
kb_save_path = os.path.join(args.save, 'training_kb.pt')
try:
    torch.save(fixed_kb_train, kb_save_path)
    console.log(f"Saved training knowledge base ({len(fixed_kb_train)} items) to {kb_save_path}")
except Exception as e:
    console.log(f"[bold red]Error saving training KB to {kb_save_path}: {e}[/bold red]")

# Initialize models and optimizers
gb_model = make_gb(input_dim=N_ATOMS)

opt_g = torch.optim.AdamW(gb_model.parameters(), lr=cfg['lr'])

# --- DIAGNOSTIC: Print initial weights of gb_model final layer ---
# Only print if N_ATOMS (and thus hidden_dim related weights) are not excessively large
if N_ATOMS <= 20: # Or a different threshold for hidden_dim size
    print("Initial gb_model.final_layer weights:\n", gb_model.final_layer.weight.data)
    if gb_model.final_layer.bias is not None:
        print("Initial gb_model.final_layer bias:\n", gb_model.final_layer.bias.data)
else:
    print(f"Initial gb_model.final_layer weights shape: {gb_model.final_layer.weight.data.shape}")
    if gb_model.final_layer.bias is not None:
        print(f"Initial gb_model.final_layer bias shape: {gb_model.final_layer.bias.data.shape}")
# --- END DIAGNOSTIC ---

console.log(f"Starting bivalent training for {args.epochs} epochs. Models will be saved to '{args.save}'")
# Log only a summary of the fixed_kb_train
console.log(f"Using fixed training KB with {len(fixed_kb_train)} items. First 3: {{k: v for i, (k, v) in enumerate(fixed_kb_train.items()) if i < 3}}")

for ep in range(args.epochs):
    gb_model.train()
    gb_outputs = gb_model(train_x) # Should be [N,2]

    good_loss_weight = cfg.get('gb_loss_good_weight', 1.0)
    bad_loss_weight = cfg.get('gb_loss_bad_weight', 1.0)

    loss_g = gb_loss(gb_outputs, train_y, 
                       good_weight=good_loss_weight,
                       bad_weight=bad_loss_weight)
    
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    
    log_interval = args.epochs // 10 if args.epochs > 10 else 1
    if ep < 5 or ep % log_interval == 0 or ep == args.epochs -1 : # Log first few, last, and interval
        console.log(f"Epoch {ep+1}/{args.epochs}: L_gb={loss_g.item():.4f}")
        if ep < 2 and N_ATOMS <=20: # Only log detailed outputs for small N_ATOMS and early epochs
            console.print(f"  gb_outputs (logits, first 3):\n{gb_outputs.detach()[:3].tolist()}")
            console.print(f"  gb_outputs (probs, first 3):\n{torch.sigmoid(gb_outputs.detach()[:3]).tolist()}")
            console.print(f"  train_y (targets, first 3):\n{train_y[:3].tolist()}")

model_save_path = os.path.join(args.save, 'gb_bivalent_model.pth')
torch.save(gb_model.state_dict(), model_save_path)
console.log(f"Bivalent model saved to {model_save_path}") 
