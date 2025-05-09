import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data.generator import build_split, N_ATOMS # N_ATOMS needed by build_split indirectly via _random_kb
from models.gb_model import MLP # Import the MLP class directly

# --- Configuration for this diagnostic script ---
N_TRAIN_SAMPLES = 8000 # Back to larger dataset
LEARNING_RATE = 0.1 # Keep high LR that worked in minimal test
EPOCHS = 50 # Can adjust; 50 should show clear trend with mini-batching
BATCH_SIZE = 64
# Ensure data/generator.py is set to CLEAN data (LABEL_NOISE=0.0, CONTR_RATE=0.0)
# Ensure models/gb_model.py MLP is the single Linear(10,1)+Sigmoid version
print(f"--- train_diagnostic_v2.py ---")
print(f"Data: {N_TRAIN_SAMPLES} samples, clean (expected LABEL_NOISE=0.0, CONTR_RATE=0.0)")
print(f"Model: MLP(out_dim=1) from models.gb_model (expected Linear(10,1)+Sigmoid)")
print(f"Optimizer: AdamW, LR={LEARNING_RATE}")
print(f"Loss: BCELoss on single output vs G-target")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print("-------------------------------------")

# 1. Data
print(f"Generating data ({N_TRAIN_SAMPLES} samples)...")
# NOTE: build_split uses global N_ATOMS, CONTR_RATE, LABEL_NOISE from data.generator module
# Ensure these are set to 0.0 in data/generator.py for this clean diagnostic test.
X_data, y_data_full = build_split(N_TRAIN_SAMPLES)
y_data = y_data_full[:,0].unsqueeze(1) # Targets for G component, shape [N,1]
print(f"Data generated: X_data shape {X_data.shape}, y_data shape {y_data.shape}")
print(f"Actual X_data:\n{X_data}") # Print the actual small X_data
print(f"Actual y_data:\n{y_data}") # Print the actual small y_data
unique_targets = torch.unique(y_data)
print(f"Unique y_data values: {unique_targets.tolist()}")
if not (len(unique_targets) <= 2 and all(val in [0.0, 1.0] for val in unique_targets.tolist())):
    print("WARNING: Targets y_data are not clean 0.0/1.0 values! Check data.generator.py settings.")

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_data, y_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Created DataLoader with batch size {BATCH_SIZE}.")

# 2. Model
model = MLP(out_dim=1) 

# --- DIAGNOSTIC: Explicitly set requires_grad for parameters ---
# This should be True by default for nn.Module parameters
for param in model.parameters():
    param.requires_grad = True
print(f"Model parameters require_grad check (e.g., final_layer.weight): {model.final_layer.weight.requires_grad}")
# --- END DIAGNOSTIC ---

# --- DIAGNOSTIC: Custom Initialization for the final layer ---
with torch.no_grad():
    if hasattr(model, 'final_layer'):
        # Initialize weights to be a bit larger/more varied
        model.final_layer.weight.data.uniform_(-0.5, 0.5) 
        if model.final_layer.bias is not None:
            # Initialize bias to be potentially non-zero and varied
            model.final_layer.bias.data.uniform_(-0.5, 0.5)
        print("Applied custom initialization to model.final_layer")
    else:
        print("Could not apply custom init: model.final_layer not found")
# --- END DIAGNOSTIC ---

# 3. Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print(f"\nInitial model.final_layer.weight (if direct attribute): {model.final_layer.weight.data if hasattr(model, 'final_layer') else 'N/A - check MLP structure'}")
print(f"Initial model.final_layer.bias (if direct attribute): {model.final_layer.bias.data if hasattr(model, 'final_layer') and model.final_layer.bias is not None else 'N/A'}")

# 4. Training Loop
print("\nStarting training with mini-batches...")
for epoch in range(EPOCHS):
    model.train() 
    epoch_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        # ... (Gradient printing can be added here for a few batches if needed, but remove for now for cleaner output) ...
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    if (epoch + 1) % 5 == 0: # Print every 5 epochs for this run
        print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.6f}')

print("\nTraining finished.")
if hasattr(model, 'final_layer'):
    print(f"Final model.final_layer.weight: {model.final_layer.weight.data}")
    print(f"Final model.final_layer.bias: {model.final_layer.bias.data}")

final_outputs = model(X_data)
print(f"Sample final_outputs (first 5): {final_outputs.data[:5].squeeze().tolist()}")
print(f"Sample targets y_data (first 5): {y_data[:5].squeeze().tolist()}")

predicted_classes = (final_outputs > 0.5).float()
correct_predictions = (predicted_classes == y_data).sum().item()
accuracy = correct_predictions / y_data.size(0)
print(f"Accuracy: {accuracy*100:.2f}%")

if avg_epoch_loss > 0.1: 
    print("WARNING: Minimal diagnostic FAILED. Loss is still high. ({avg_epoch_loss:.6f})")
else:
    print("Minimal diagnostic PASSED: Loss is low. ({avg_epoch_loss:.6f})") 