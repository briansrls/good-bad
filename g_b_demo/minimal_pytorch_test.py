import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. Tiny, simple, perfectly separable data
# Two one-hot inputs (like atom0, atom1) and two classes (0 or 1)
X_data = torch.tensor([    # Represents 4 samples
    [1.0, 0.0],         # Sample 0 (atom 0)
    [0.0, 1.0],         # Sample 1 (atom 1)
    [1.0, 0.0],         # Sample 2 (atom 0)
    [0.0, 1.0]          # Sample 3 (atom 1)
])
y_data = torch.tensor([    # Target for a single output neuron
    [0.0],              # Sample 0 -> class 0
    [1.0],              # Sample 1 -> class 1
    [0.0],              # Sample 2 -> class 0
    [1.0]               # Sample 3 -> class 1
])

# 2. Simplest Model: Logistic Regression Unit (Linear + Sigmoid)
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        return self.sigmoid(logits)

model = LogisticRegression(input_dim=2, output_dim=1)

# 3. Optimizer and Loss
learning_rate = 0.1 # Use a higher LR for simple problem
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Can also try SGD
criterion = nn.BCELoss() # Standard Binary Cross-Entropy

print(f"Initial model weights: {model.linear.weight.data}")
print(f"Initial model bias: {model.linear.bias.data}")

# 4. Training Loop
epochs = 200
for epoch in range(epochs):
    model.train() # Set model to training mode

    # Forward pass
    outputs = model(X_data)
    loss = criterion(outputs, y_data)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        # print(f'  Outputs: {outputs.data.squeeze().tolist()}')
        # print(f'  Weights: {model.linear.weight.data}')
        # print(f'  Bias: {model.linear.bias.data}')

print("Training finished.")
print(f"Final model weights: {model.linear.weight.data}")
print(f"Final model bias: {model.linear.bias.data}")
final_outputs = model(X_data)
print(f"Final outputs: {final_outputs.data.squeeze().tolist()}")
print(f"Targets: {y_data.squeeze().tolist()}")

# Check if predictions are correct
predicted_classes = (final_outputs > 0.5).float()
correct_predictions = (predicted_classes == y_data).sum().item()
accuracy = correct_predictions / y_data.size(0)
print(f"Accuracy: {accuracy*100:.2f}%")

if loss.item() > 0.1: # BCE loss should be very low for this task
    print("WARNING: Loss is still high. Model did not learn effectively.")
else:
    print("Minimal test PASSED: Loss is low, model likely learned.") 