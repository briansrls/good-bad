import torch
import matplotlib.pyplot as plt
import os
from data.generator import build_split, N_ATOMS # Import N_ATOMS if needed by models, though MLP is fixed size
from models.gb_model import make_scalar, make_gb

TEST_SAMPLES = 1000
MODEL_DIR = "runs/demo"  # Directory where models are saved

# Prepare data
# Note: build_split uses global N_ATOMS from data.generator
x_test, y_test = build_split(TEST_SAMPLES)

# Initialize models
scalar_model = make_scalar()
gb_model = make_gb()

# Load trained model weights
scalar_model_path = os.path.join(MODEL_DIR, 'scalar_model.pth')
gb_model_path = os.path.join(MODEL_DIR, 'gb_model.pth')

models_loaded = False
if os.path.exists(scalar_model_path) and os.path.exists(gb_model_path):
    try:
        scalar_model.load_state_dict(torch.load(scalar_model_path))
        gb_model.load_state_dict(torch.load(gb_model_path))
        models_loaded = True
        print(f"Loaded trained models from {MODEL_DIR}")
    except Exception as e:
        print(f"Error loading models: {e}. Using randomly initialized models.")
else:
    print(f"Warning: Model files not found in {MODEL_DIR}. Using randomly initialized models.")

scalar_model.eval() # Set to evaluation mode
gb_model.eval()     # Set to evaluation mode

# Perform inference
with torch.no_grad():
    ps_dual = scalar_model(x_test)  # Dual scalar model outputs 2 values
    pg_gb = gb_model(x_test)        # G/B model outputs 2 values

# Calculate RMSE
# For scalar model (dual output), compare directly with (G,B) targets
rmse_scalar = torch.mean((ps_dual - y_test)**2).sqrt().item()
# For G/B model, compare directly with (G,B) targets
rmse_gb = torch.mean((pg_gb - y_test)**2).sqrt().item()

print(f"RMSE Dual Scalar Model: {rmse_scalar:.4f}")
print(f"RMSE G/B Model:         {rmse_gb:.4f}")

# Plotting results for the G/B model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test[:,0].numpy(), pg_gb[:,0].numpy(), alpha=0.5, label='Good (G)')
plt.scatter(y_test[:,1].numpy(), pg_gb[:,1].numpy(), alpha=0.5, label='Bad (B)', marker='x')
plt.plot([0,1],[0,1], 'k--', alpha=0.75) # Diagonal line
plt.xlabel('Target G/B Values')
plt.ylabel('Predicted G/B Values (from G/B Model)')
plt.title('G/B Model Predictions vs Targets')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# Plotting G vs B for targets and predictions of the G/B model
plt.scatter(y_test[:,0].numpy(), y_test[:,1].numpy(), alpha=0.3, label='Target (G,B) distribution')
plt.scatter(pg_gb[:,0].numpy(), pg_gb[:,1].numpy(), alpha=0.3, label='G/B Model (G,B) distribution', marker='x')
plt.xlabel('Good (G) Value')
plt.ylabel('Bad (B) Value')
plt.xlim(0,1); plt.ylim(0,1)
plt.plot([0,1],[1,0], 'k--', alpha=0.75) # G+B=1 line for reference
plt.axhline(0.5, color='grey', linestyle='--', linewidth=0.5)
plt.axvline(0.5, color='grey', linestyle='--', linewidth=0.5)
plt.title('(G,B) Space: Targets and G/B Model Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

if not models_loaded:
    print("Note: Evaluation was run on randomly initialized models.") 