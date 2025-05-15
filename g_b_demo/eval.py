import argparse
import torch
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
parser = argparse.ArgumentParser(description="Evaluate G/B bivalent model.")
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
    console.log(f"Content of loaded training KB: {loaded_fixed_kb}")
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
    eval_loss = gb_loss(outputs, eval_y, 
                        good_weight=good_loss_weight, 
                        bad_weight=bad_loss_weight).item()
    
    console.log(f"Evaluation Loss: {eval_loss:.4f}")

    # Get probabilities by applying sigmoid to logits
    probs = torch.sigmoid(outputs) # Shape [N, 2]
    preds_goodness = (probs[:, 0] > 0.5).float()
    preds_badness = (probs[:, 1] > 0.5).float()

    targets_goodness = eval_y[:, 0]
    targets_badness = eval_y[:, 1]

    # Accuracy for goodness prediction
    acc_goodness = (preds_goodness == targets_goodness).float().mean().item()
    console.log(f"Accuracy for Goodness Prediction: {acc_goodness:.4f}")

    # Accuracy for badness prediction
    acc_badness = (preds_badness == targets_badness).float().mean().item()
    console.log(f"Accuracy for Badness Prediction: {acc_badness:.4f}")
    
    # Overall Bivalent Accuracy (both goodness and badness must be correct)
    correct_bivalent = ((preds_goodness == targets_goodness) & (preds_badness == targets_badness)).float().mean().item()
    console.log(f"Overall Bivalent Accuracy (Goodness AND Badness correct): {correct_bivalent:.4f}")

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
    probs_goodness_np = probs[:, 0].cpu().numpy()
    probs_badness_np = probs[:, 1].cpu().numpy()

    # Create a single figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12)) # Adjust figsize as needed
    fig.suptitle('Bivalent Model Evaluation Plots', fontsize=16)

    # Plot 1: Target Goodness vs. Predicted Goodness Probability (Top-Left)
    ax = axs[0, 0]
    jittered_targets_good = targets_goodness_np + np.random.normal(0, 0.02, size=targets_goodness_np.shape)
    ax.scatter(jittered_targets_good, probs_goodness_np, alpha=0.2, s=10)
    ax.set_xlabel("Target Goodness (0 or 1, jittered)")
    ax.set_ylabel("Predicted Goodness Probability")
    ax.set_title("Goodness: Target vs. Predicted Probability")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Target Badness vs. Predicted Badness Probability (Top-Right)
    ax = axs[0, 1]
    jittered_targets_bad = targets_badness_np + np.random.normal(0, 0.02, size=targets_badness_np.shape)
    ax.scatter(jittered_targets_bad, probs_badness_np, alpha=0.2, s=10)
    ax.set_xlabel("Target Badness (0 or 1, jittered)")
    ax.set_ylabel("Predicted Badness Probability")
    ax.set_title("Badness: Target vs. Predicted Probability")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Histograms of Predicted Goodness Probabilities (Bottom-Left)
    ax = axs[1, 0]
    ax.hist(probs_goodness_np[targets_goodness_np == 0], bins=np.linspace(0,1,51), alpha=0.7, label='Target Goodness = 0', density=True)
    ax.hist(probs_goodness_np[targets_goodness_np == 1], bins=np.linspace(0,1,51), alpha=0.7, label='Target Goodness = 1', density=True)
    ax.set_xlabel("Predicted Goodness Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predicted Goodness Probabilities")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Histograms of Predicted Badness Probabilities (Bottom-Right)
    ax = axs[1, 1]
    ax.hist(probs_badness_np[targets_badness_np == 0], bins=np.linspace(0,1,51), alpha=0.7, label='Target Badness = 0', density=True)
    ax.hist(probs_badness_np[targets_badness_np == 1], bins=np.linspace(0,1,51), alpha=0.7, label='Target Badness = 1', density=True)
    ax.set_xlabel("Predicted Badness Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predicted Badness Probabilities")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show() # Display the single figure with all subplots
    console.log(f"Displayed all plots in a single window.")

console.log("\nEvaluation complete.") 
