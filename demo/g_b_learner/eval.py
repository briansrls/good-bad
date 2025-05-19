import argparse
import torch
import torch.nn.functional as F # For MSE and MAE
import yaml
import os
from rich.console import Console
from data.generator import build_split, N_ATOMS # Assuming N_ATOMS is exposed
from models.gb_model import make_gb, gb_loss # make_gb and gb_loss are needed
import matplotlib.pyplot as plt
import numpy as np

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
parser = argparse.ArgumentParser(description="Evaluate G/B bivalent model with fuzzy targets.")
parser.add_argument('--model_path', type=str, default='runs/bivalent_demo/gb_bivalent_model.pth',
                    help='Path to the trained bivalent G/B model.')
parser.add_argument('--n_eval', type=int, default=cfg.get('n_val', 1000), # Use n_val from cfg or default
                    help='Number of samples for evaluation.')
args = parser.parse_args()

# Rich console for logging
console = Console()

# Determine paths
model_dir = os.path.dirname(args.model_path)
kb_path = os.path.join(model_dir, 'training_kb.pt')

# Load the fixed_kb used during training
if not os.path.exists(kb_path):
    console.log(f"[bold red]Error: Training KB not found at {kb_path}[/bold red]")
    console.log(f"Ensure '{os.path.basename(kb_path)}' was saved in '{model_dir}' during training.")
    exit(1)

try:
    loaded_fixed_kb = torch.load(kb_path)
    console.log(f"Successfully loaded training knowledge base from {kb_path}")
    console.log(f"Content of loaded training KB (first few items): {{k: v for i, (k, v) in enumerate(loaded_fixed_kb.items()) if i < 3}}")
except Exception as e:
    console.log(f"[bold red]Error loading training KB from {kb_path}: {e}[/bold red]")
    exit(1)

# Prepare evaluation data using the loaded KB
eval_x, eval_y, _ = build_split(args.n_eval, cfg=cfg, existing_kb=loaded_fixed_kb) # The third return (kb itself) can be ignored

# Load model
if not os.path.exists(args.model_path):
    console.log(f"[bold red]Error: Model not found at {args.model_path}[/bold red]")
    exit(1)

# Instantiate the model with the correct input dimension
model = make_gb(input_dim=N_ATOMS)
model.load_state_dict(torch.load(args.model_path))
model.eval()

console.log(f"Evaluating model: {args.model_path} on {args.n_eval} samples, using the KB it was trained on.")

with torch.no_grad():
    outputs = model(eval_x) # Raw logits, shape [N, 2]
    
    # Calculate overall loss on evaluation set
    # Get loss weights from config, default to 1.0 if not present
    good_loss_weight = cfg.get('gb_loss_good_weight', 1.0)
    bad_loss_weight = cfg.get('gb_loss_bad_weight', 1.0)
    eval_loss_bce = gb_loss(outputs, eval_y, 
                            good_weight=good_loss_weight, 
                            bad_weight=bad_loss_weight).item()
    
    console.log(f"Evaluation BCE Loss (Training Objective): {eval_loss_bce:.4f}")

    # Get probabilities by applying sigmoid to logits
    probs = torch.sigmoid(outputs) # Shape [N, 2]
    preds_goodness = (probs[:, 0] > 0.5).float()
    preds_badness = (probs[:, 1] > 0.5).float()

    targets_goodness = eval_y[:, 0]
    targets_badness = eval_y[:, 1]
    probs_goodness = probs[:, 0]
    probs_badness = probs[:, 1]

    # --- MSE and MAE for fuzzy targets ---
    mse_goodness = F.mse_loss(probs_goodness, targets_goodness).item()
    mae_goodness = F.l1_loss(probs_goodness, targets_goodness).item() # L1 loss is MAE
    console.log(f"Goodness Prediction MSE: {mse_goodness:.4f}, MAE: {mae_goodness:.4f}")

    mse_badness = F.mse_loss(probs_badness, targets_badness).item()
    mae_badness = F.l1_loss(probs_badness, targets_badness).item()
    console.log(f"Badness Prediction MSE: {mse_badness:.4f}, MAE: {mae_badness:.4f}")

    # --- Threshold-based "Accuracy" (for a rough guide) ---
    # Binarize fuzzy targets and predictions at 0.5 threshold
    preds_goodness_binary = (probs_goodness > 0.5).float()
    targets_goodness_binary = (targets_goodness > 0.5).float()
    acc_goodness_binary = (preds_goodness_binary == targets_goodness_binary).float().mean().item()
    console.log(f"Accuracy for Binarized Goodness (@0.5 thr): {acc_goodness_binary:.4f}")

    preds_badness_binary = (probs_badness > 0.5).float()
    targets_badness_binary = (targets_badness > 0.5).float()
    acc_badness_binary = (preds_badness_binary == targets_badness_binary).float().mean().item()
    console.log(f"Accuracy for Binarized Badness (@0.5 thr): {acc_badness_binary:.4f}")
    
    correct_bivalent_binary = ((preds_goodness_binary == targets_goodness_binary) & 
                               (preds_badness_binary == targets_badness_binary)).float().mean().item()
    console.log(f"Overall Bivalent Binarized Accuracy (@0.5 thr): {correct_bivalent_binary:.4f}")

    console.log("\n--- Detailed Metrics ---")

    # Metrics for "Goodness" aspect
    # Samples that ARE truly good (target_goodness == 1)
    true_good_samples_mask = (targets_goodness == 1.0)
    if true_good_samples_mask.any():
        avg_prob_good_when_good = probs[true_good_samples_mask, 0].mean().item()
        console.log(f"Avg. predicted 'goodness' prob for truly Good targets: {avg_prob_good_when_good:.4f}")
    
    # Samples that are NOT truly good (target_goodness == 0)
    true_not_good_samples_mask = (targets_goodness == 0.0)
    if true_not_good_samples_mask.any():
        avg_prob_good_when_not_good = probs[true_not_good_samples_mask, 0].mean().item()
        console.log(f"Avg. predicted 'goodness' prob for truly Not-Good targets: {avg_prob_good_when_not_good:.4f}")

    # Metrics for "Badness" aspect
    # Samples that ARE truly bad (target_badness == 1)
    true_bad_samples_mask = (targets_badness == 1.0)
    if true_bad_samples_mask.any():
        avg_prob_bad_when_bad = probs[true_bad_samples_mask, 1].mean().item()
        console.log(f"Avg. predicted 'badness' prob for truly Bad targets: {avg_prob_bad_when_bad:.4f}")

    # Samples that are NOT truly bad (target_badness == 0)
    true_not_bad_samples_mask = (targets_badness == 0.0)
    if true_not_bad_samples_mask.any():
        avg_prob_bad_when_not_bad = probs[true_not_bad_samples_mask, 1].mean().item()
        console.log(f"Avg. predicted 'badness' prob for truly Not-Bad targets: {avg_prob_bad_when_not_bad:.4f}")

    # --- Plotting ---
    console.log("\n--- Displaying Plots (close plot window to continue) ---")

    # Detach tensors and move to CPU for plotting
    targets_goodness_np = targets_goodness.cpu().numpy()
    targets_badness_np = targets_badness.cpu().numpy()
    probs_goodness_np = probs_goodness.cpu().numpy()
    probs_badness_np = probs_badness.cpu().numpy()

    # Create a single figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12)) # Adjust figsize as needed
    fig.suptitle('Bivalent Model Evaluation (Fuzzy Targets)', fontsize=16)

    # Plot 1: Target Goodness vs. Predicted Goodness Probability
    ax = axs[0, 0]
    # Jitter is less critical now targets are continuous, but can still help if many targets cluster.
    # For continuous targets, direct scatter is often clear.
    ax.scatter(targets_goodness_np, probs_goodness_np, alpha=0.2, s=10) 
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Ideal') # Add y=x line
    ax.set_xlabel("Target Goodness (Fuzzy)")
    ax.set_ylabel("Predicted Goodness Probability")
    ax.set_title("Goodness: Target vs. Predicted Probability")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Target Badness vs. Predicted Badness Probability
    ax = axs[0, 1]
    ax.scatter(targets_badness_np, probs_badness_np, alpha=0.2, s=10)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Ideal') # Add y=x line
    ax.set_xlabel("Target Badness (Fuzzy)")
    ax.set_ylabel("Predicted Badness Probability")
    ax.set_title("Badness: Target vs. Predicted Probability")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Histograms of Target vs. Predicted Goodness
    ax = axs[1, 0]
    ax.hist(targets_goodness_np, bins=np.linspace(0,1,51), alpha=0.7, label='Target Goodness Scores', density=True)
    ax.hist(probs_goodness_np, bins=np.linspace(0,1,51), alpha=0.7, label='Predicted Goodness Probs', density=True)
    ax.set_xlabel("Score / Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution: Target Goodness vs. Prediction")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Histograms of Target vs. Predicted Badness
    ax = axs[1, 1]
    ax.hist(targets_badness_np, bins=np.linspace(0,1,51), alpha=0.7, label='Target Badness Scores', density=True)
    ax.hist(probs_badness_np, bins=np.linspace(0,1,51), alpha=0.7, label='Predicted Badness Probs', density=True)
    ax.set_xlabel("Score / Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution: Target Badness vs. Prediction")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show() # Display the single figure with all subplots
    console.log(f"Displayed all plots in a single window.")

console.log("\nEvaluation complete.") 
