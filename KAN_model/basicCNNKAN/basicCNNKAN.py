import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt
import math

from version.kan.kan import KANLinear

class BasicCNNKAN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNNKAN, self).__init__()
        
        # Convolutional layers - 3 layers with 3x3 kernels
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 224x224 -> 56x56
            nn.Dropout2d(0.25),
            
            # Second conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 56x56 -> 14x14
            nn.Dropout2d(0.25),
            
            # Third conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 14x14 -> 3x3 (with some rounding)
            nn.Dropout2d(0.25),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # KAN layers for classification
        self.kan1 = KANLinear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.kan2 = KANLinear(128, num_classes)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 256]
        
        # KAN classification layers
        x = self.kan1(x)
        x = self.dropout(x)
        x = self.kan2(x)
        
        return x

def print_parameter_details(model):
    total_params = 0
    trainable_params = 0
    
    print("Layer-wise parameter count:")
    print("-" * 50)
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        
        if parameter.requires_grad:
            trainable_params += params
            print(f"{name}: {params:,} (trainable)")
        else:
            print(f"{name}: {params:,} (frozen)")
    
    print("-" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def count_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BasicCNNKAN(num_classes=10).to(device)

print("Model Architecture:")
print("=" * 60)
print(model)
print("\n")

print("Parameter Details:")
print("=" * 60)
print_parameter_details(model)
print(f"Model size: {count_model_size(model):.2f} MB")
print("\n")

print("Model Summary:")
print("=" * 60)
try:
    summary(model, input_size=(3, 224, 224))
except Exception as e:
    print(f"Summary error: {e}")
    print("Manual forward pass test:")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")

# Optional: Test with different input sizes
print("\nTesting different input sizes:")
print("-" * 30)
test_sizes = [(224, 224), (128, 128), (96, 96)]

for h, w in test_sizes:
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, h, w).to(device)
            output = model(test_input)
            print(f"Input {h}x{w}: Output shape {output.shape}")
    except Exception as e:
        print(f"Input {h}x{w}: Error - {e}")

# Clean up
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()