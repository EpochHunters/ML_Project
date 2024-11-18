import torch
import torch.nn as nn

# Define TemporalBlock
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super().__init__()
        layers = []
        for i, dilation in enumerate(dilation_rates):
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1 // 2) * dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels  # Update for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define SpatialBlock
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels  # Update for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


# Define Encoder
class EEGEncoder(nn.Module):
    def __init__(self, num_channels=128, temp_channels=64, spatial_channels=32, kernel_size=3, dilation_rates=[1,2,4,8], num_residual_blocks=2, num_classes=40, latent_dim=128):
        super().__init__()

        # Temporal Block to capture temporal dependencies
        self.temporal_block = TemporalBlock(num_channels, temp_channels, kernel_size, dilation_rates)

        # Spatial Block to capture spatial dependencies among channels
        self.spatial_block = SpatialBlock(1, spatial_channels, num_layers=2)  # Conv2d expects (batch, channel, height, width)

        # Residual Blocks for added depth
        self.res_blocks = nn.ModuleList([ResidualBlock(spatial_channels) for _ in range(num_residual_blocks)])

        # Fully connected layers for latent vector and classification
        # Here we need to compute the correct input size to the fully connected layer.
        # The size after residual block and pooling is (batch_size, spatial_channels, 1, 1).
        # We need to flatten it into (batch_size, spatial_channels) before feeding it to fc_latent.
        self.fc_latent = nn.Linear(spatial_channels, latent_dim)  # Create latent vector of size 128
        self.fc_classification = nn.Linear(latent_dim, num_classes)  # Final classification layer (num_classes = 40)

    def forward(self, x):
        x=x.float()
        # Pass through Temporal Block
        x = self.temporal_block(x)  # Shape: (batch_size, temp_channels, time_steps)

        # Reshape for Spatial Block (treat time as "width" in Conv2d)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, temp_channels, time_steps)
        x = self.spatial_block(x)  # Shape: (batch_size, spatial_channels, temp_channels, time_steps)

        # Pass through Residual Blocks
        for res_block in self.res_blocks:
            x = res_block(x)  # Shape maintained after residual block (batch_size, spatial_channels, temp_channels, time_steps)

        # Apply Adaptive Average Pooling to reduce spatial dimensions (time_steps)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)  # Shape: (batch_size, spatial_channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten the output (batch_size, spatial_channels)

        # Create the latent vector (128)
        latent_vector = self.fc_latent(x)  # Shape: (batch_size, latent_dim)

        # Classification Output (num_classes)
        class_output = self.fc_classification(latent_vector)  # Shape: (batch_size, num_classes)

        return class_output, latent_vector


# Testing the model with a dummy input
if __name__ == "__main__":
    # Example input: (batch_size=16, input_channels=128, num_timestamps=440)
    model = EEGEncoder()
    
    # Create dummy input tensor with the expected size
    dummy_input = torch.randn(16, 128, 440)  # (batch_size, channels, num_timestamps)
    
    # Ensure the input tensor is float32
    dummy_input = dummy_input.float()  # Explicitly cast to float32
    
    # Forward pass
    class_output, latent_vector = model(dummy_input)
    print("Class Output Shape:", class_output.shape)  # Should be (16, num_classes)
    print("Latent Vector Shape:", latent_vector.shape)  # Should be (16, latent_dim)
