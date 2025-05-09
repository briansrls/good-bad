import matplotlib.pyplot as plt
from data.generator import build_split, N_ATOMS, CONTR_RATE, LABEL_NOISE # Import params for title
import torch # Ensure torch is imported for .numpy()

# Number of samples for visualization
N_SAMPLES_VIZ = 5000 # Increased for a better view of the distribution

print(f"Generating {N_SAMPLES_VIZ} target samples for visualization...")
print(f"Current settings from data/generator.py: N_ATOMS={N_ATOMS}, CONTR_RATE={CONTR_RATE}, LABEL_NOISE={LABEL_NOISE}")

# We only need the y values (targets)
# Ensure build_split returns tensors that can be converted to numpy
# If it already returns torch tensors, .numpy() is fine.
try:
    _, train_y_viz = build_split(N_SAMPLES_VIZ)
    if not isinstance(train_y_viz, torch.Tensor):
        train_y_viz = torch.tensor(train_y_viz) # Convert if not already a tensor
except Exception as e:
    print(f"Error during build_split: {e}")
    print("Please ensure data.generator.py is functioning correctly.")
    exit()

# Extract G and B components
try:
    g_values = train_y_viz[:, 0].numpy()
    b_values = train_y_viz[:, 1].numpy()
except AttributeError:
    print("Error: train_y_viz does not have .numpy(). It might not be a PyTorch tensor.")
    print(f"Type of train_y_viz: {type(train_y_viz)}")
    exit()
except IndexError:
    print("Error: train_y_viz does not have expected shape. Cannot extract G and B columns.")
    print(f"Shape of train_y_viz: {train_y_viz.shape if hasattr(train_y_viz, 'shape') else 'N/A'}")
    exit()


print(f"Generated {len(g_values)} samples.")
# print(f"Sample G values (first 10): {g_values[:10]}") # Optional: for debugging
# print(f"Sample B values (first 10): {b_values[:10]}") # Optional: for debugging


# Create plots
plt.figure(figsize=(19, 6)) # Adjusted figure size

# Histogram for G values
plt.subplot(1, 3, 1)
plt.hist(g_values, bins=50, range=(-0.05,1.05), density=True, alpha=0.7, label='G Targets')
plt.title(f'Distribution of G Target Values\n(Noise: {LABEL_NOISE}, Contr: {CONTR_RATE})')
plt.xlabel('G Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Histogram for B values
plt.subplot(1, 3, 2)
plt.hist(b_values, bins=50, range=(-0.05,1.05), density=True, alpha=0.7, label='B Targets', color='orange')
plt.title(f'Distribution of B Target Values\n(Noise: {LABEL_NOISE}, Contr: {CONTR_RATE})')
plt.xlabel('B Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Scatter plot for (G,B) pairs
plt.subplot(1, 3, 3)
# Downsample for scatter plot if too many points, for performance
num_scatter_points = min(N_SAMPLES_VIZ, 2000)
indices = torch.randperm(N_SAMPLES_VIZ)[:num_scatter_points]
plt.scatter(g_values[indices.numpy()], b_values[indices.numpy()], alpha=0.2, s=10) # s for marker size, increased alpha slightly
plt.title(f'(G,B) Target Distribution ({num_scatter_points} points)\n(Noise: {LABEL_NOISE}, Contr: {CONTR_RATE})')
plt.xlabel('G Value')
plt.ylabel('B Value')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axhline(0.5, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0.5, color='grey', linestyle='--', linewidth=0.5)
plt.plot([0,1],[1,0], 'k--', alpha=0.5, label='G+B=1 line') # G+B=1 line
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')


plt.suptitle(f'Target Label Distribution (LABEL_NOISE={LABEL_NOISE}, CONTR_RATE={CONTR_RATE})', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()

print("\nCheck the displayed plot window.")
print("Observations to make:")
print("1. G/B Histograms: Are values sharply peaked near 0 and 1, or more spread out/centered around 0.5?")
print("   How significant is the spread introduced by the noise?")
print("2. (G,B) Scatter: Where do the points primarily cluster? ")
print("   - Near (1,0) and (0,1) (sharp, distinct states)?")
print("   - Near (0,0) (if unknowns were sampled, which they are not currently)?")
print("   - Or are they more spread out towards the center, forming a 'cross' or 'cloud' shape?")
print("   - How many points fall into the 'contradictory' quadrants (e.g., G high AND B high, or G low AND B low)?") 