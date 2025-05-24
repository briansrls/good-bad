import torch
import torch.nn as nn

class GBOutputLayer(nn.Module):
    """
    A layer that takes activations and produces G (Goodness) and B (Badness) values
    for a set of categories.
    """
    def __init__(self, input_features, num_categories):
        super(GBOutputLayer, self).__init__()
        self.num_categories = num_categories
        # Each category will have a G and a B value, so 2 * num_categories outputs
        self.fc = nn.Linear(input_features, 2 * num_categories)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, input_features)
        gb_outputs = self.fc(x) # shape: (batch_size, 2 * num_categories)
        gb_outputs_sig = self.sigmoid(gb_outputs)

        # Reshape to separate G and B values
        # G values will be in the first half, B values in the second half for each category
        # Or, more conveniently, reshape to (batch_size, num_categories, 2)
        # where the last dimension is [G, B]
        g_values = gb_outputs_sig[:, :self.num_categories]
        b_values = gb_outputs_sig[:, self.num_categories:]
        
        # Stack them along a new dimension to get (batch_size, num_categories, 2)
        # where output[..., 0] is G and output[..., 1] is B
        output = torch.stack((g_values, b_values), dim=2)
        
        return output # shape: (batch_size, num_categories, 2)

# Example usage (for testing the layer)
if __name__ == '__main__':
    batch_size = 4
    input_features = 128
    num_categories = 5

    gb_layer = GBOutputLayer(input_features, num_categories)
    dummy_input = torch.randn(batch_size, input_features)
    gb_values = gb_layer(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output G-B values shape:", gb_values.shape)
    print("Example G-B output for first item in batch, first category:", gb_values[0, 0, :])
    assert gb_values.shape == (batch_size, num_categories, 2)
    assert torch.all(gb_values >= 0) and torch.all(gb_values <= 1)
    print("GBOutputLayer test passed.")

class L1Network(nn.Module):
    """
    L1 Foundational Network. A small CNN for initial input processing.
    Outputs G-B values for a set of categories.
    """
    def __init__(self, input_channels, image_size, num_categories, fc_hidden_features=128):
        super(L1Network, self).__init__()
        self.num_categories = num_categories

        # Convolutional layers
        # Assuming image_size is a single int for square images (e.g., 32 for 32x32)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size after pool1: image_size / 2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size after pool2: image_size / 4

        # Calculate the flattened size after convolutions and pooling
        conv_output_size = image_size // 4
        flattened_size = 32 * conv_output_size * conv_output_size

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, fc_hidden_features)
        self.relu3 = nn.ReLU()
        
        # G-B Output Layer
        self.gb_output_layer = GBOutputLayer(fc_hidden_features, num_categories)

    def forward(self, x):
        # x shape: (batch_size, input_channels, image_height, image_width)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the output for the FC layers
        x = x.view(x.size(0), -1) 
        
        x = self.relu3(self.fc1(x))
        gb_values = self.gb_output_layer(x)
        
        return gb_values # shape: (batch_size, num_categories, 2)

# Example usage (for testing the L1Network)
if __name__ == '__main__':
    # GBOutputLayer test (already present)
    batch_size_gb = 4
    input_features_gb = 128
    num_categories_gb = 5
    gb_layer = GBOutputLayer(input_features_gb, num_categories_gb)
    dummy_input_gb = torch.randn(batch_size_gb, input_features_gb)
    gb_values_test = gb_layer(dummy_input_gb)
    print("\n--- GBOutputLayer Test ---")
    print("Input shape:", dummy_input_gb.shape)
    print("Output G-B values shape:", gb_values_test.shape)
    print("Example G-B output for first item in batch, first category:", gb_values_test[0, 0, :])
    assert gb_values_test.shape == (batch_size_gb, num_categories_gb, 2)
    assert torch.all(gb_values_test >= 0) and torch.all(gb_values_test <= 1)
    print("GBOutputLayer test passed.")

    print("\n--- L1Network Test ---")
    batch_size_l1 = 2
    input_channels_l1 = 1 # e.g., grayscale image
    image_size_l1 = 32 # e.g., 32x32 image
    num_categories_l1 = 3 # e.g., "circle", "square", "noise"

    l1_network = L1Network(input_channels_l1, image_size_l1, num_categories_l1)
    dummy_image_input = torch.randn(batch_size_l1, input_channels_l1, image_size_l1, image_size_l1)
    l1_gb_output = l1_network(dummy_image_input)

    print("Input image shape:", dummy_image_input.shape)
    print("L1 Network G-B output shape:", l1_gb_output.shape)
    print("Example L1 G-B output for first item, first category:", l1_gb_output[0, 0, :])
    assert l1_gb_output.shape == (batch_size_l1, num_categories_l1, 2)
    assert torch.all(l1_gb_output >= 0) and torch.all(l1_gb_output <= 1)
    print("L1Network test passed.")

class L1ENetwork(nn.Module):
    """
    L1-Expansion (L1-E) Network.
    Similar to L1Network, dynamically recruited to handle ambiguities or novelties.
    Outputs G-B values for its set of specialized categories.
    """
    def __init__(self, input_channels, image_size, num_categories, fc_hidden_features=128, l1_context_features=0):
        super(L1ENetwork, self).__init__()
        self.num_categories = num_categories

        # Convolutional layers (same as L1 for now)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_output_size = image_size // 4
        flattened_image_size = 32 * conv_output_size * conv_output_size
        
        # Fully connected layers
        # Input to FC1 now includes flattened image features + L1 context features
        self.fc1_input_size = flattened_image_size + l1_context_features
        self.fc1 = nn.Linear(self.fc1_input_size, fc_hidden_features)
        self.relu3 = nn.ReLU()
        
        self.gb_output_layer = GBOutputLayer(fc_hidden_features, num_categories)

    def forward(self, image_input, l1_context_input=None):
        # image_input shape: (batch_size, input_channels, image_height, image_width)
        # l1_context_input shape: (batch_size, l1_context_features) or None

        x_img = self.pool1(self.relu1(self.conv1(image_input)))
        x_img = self.pool2(self.relu2(self.conv2(x_img)))
        x_img = x_img.view(x_img.size(0), -1) # Flatten image features

        if l1_context_input is not None:
            # Concatenate image features with L1 context features
            x = torch.cat((x_img, l1_context_input), dim=1)
        else:
            x = x_img
            # If l1_context_features was > 0 in init but no input given, this might error in self.fc1
            # This assumes l1_context_input is provided if l1_context_features > 0
            assert self.fc1_input_size == x.shape[1], \
                f"FC1 input size mismatch. Expected {self.fc1_input_size}, got {x.shape[1]}. " \
                f"Ensure l1_context_input is provided if l1_context_features > 0."

        x = self.relu3(self.fc1(x))
        gb_values = self.gb_output_layer(x)
        
        return gb_values # shape: (batch_size, num_categories, 2)


# Example usage (for testing the L1ENetwork)
if __name__ == '__main__':
    # ... (previous tests for GBOutputLayer and L1Network) ...
    print("\n--- GBOutputLayer Test (repeated for clarity in full script) ---")
    batch_size_gb = 4
    input_features_gb = 128
    num_categories_gb = 5
    gb_layer = GBOutputLayer(input_features_gb, num_categories_gb)
    dummy_input_gb = torch.randn(batch_size_gb, input_features_gb)
    gb_values_test = gb_layer(dummy_input_gb)
    print("Input shape:", dummy_input_gb.shape)
    print("Output G-B values shape:", gb_values_test.shape)
    assert gb_values_test.shape == (batch_size_gb, num_categories_gb, 2)
    print("GBOutputLayer test passed.")

    print("\n--- L1Network Test (repeated for clarity in full script) ---")
    batch_size_l1 = 2
    input_channels_l1 = 1
    image_size_l1 = 32
    num_categories_l1 = 3
    l1_network = L1Network(input_channels_l1, image_size_l1, num_categories_l1)
    dummy_image_input_l1 = torch.randn(batch_size_l1, input_channels_l1, image_size_l1, image_size_l1)
    l1_gb_output = l1_network(dummy_image_input_l1)
    print("Input image shape:", dummy_image_input_l1.shape)
    print("L1 Network G-B output shape:", l1_gb_output.shape)
    assert l1_gb_output.shape == (batch_size_l1, num_categories_l1, 2)
    print("L1Network test passed.")

    print("\n--- L1ENetwork Test ---")
    batch_size_l1e = 2
    input_channels_l1e = 1
    image_size_l1e = 32
    num_categories_l1e = 2 # L1-E might handle fewer, specialized categories
    l1_context_features_l1e = 10 # Example size for L1 context vector

    l1e_network = L1ENetwork(input_channels_l1e, image_size_l1e, num_categories_l1e, l1_context_features=l1_context_features_l1e)
    
    dummy_image_input_l1e = torch.randn(batch_size_l1e, input_channels_l1e, image_size_l1e, image_size_l1e)
    dummy_context_input_l1e = torch.randn(batch_size_l1e, l1_context_features_l1e)
    l1e_gb_output = l1e_network(dummy_image_input_l1e, dummy_context_input_l1e)

    print("Input image shape:", dummy_image_input_l1e.shape)
    print("Input context shape:", dummy_context_input_l1e.shape)
    print("L1-E Network G-B output shape:", l1e_gb_output.shape)
    assert l1e_gb_output.shape == (batch_size_l1e, num_categories_l1e, 2)
    assert torch.all(l1e_gb_output >= 0) and torch.all(l1e_gb_output <= 1)
    print("L1ENetwork test (with context) passed.")

    # Test L1-E without context (l1_context_features = 0)
    l1e_network_no_context = L1ENetwork(input_channels_l1e, image_size_l1e, num_categories_l1e, l1_context_features=0)
    l1e_gb_output_no_context = l1e_network_no_context(dummy_image_input_l1e, None)
    print("L1-E Network G-B output shape (no context):", l1e_gb_output_no_context.shape)
    assert l1e_gb_output_no_context.shape == (batch_size_l1e, num_categories_l1e, 2)
    print("L1ENetwork test (no context) passed.") 