import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt
from torchvision.models import densenet121

class DenseNetMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetMLP, self).__init__()
        
        # Use DenseNet121 backbone from torchvision
        densenet = densenet121(pretrained=False)
        self.features = densenet.features
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP layers for classification
        self.fc1 = nn.Linear(1024, 128)  # DenseNet121 outputs 1024 features
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 1024]
        
        # MLP classification layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
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

model = DenseNetMLP(num_classes=10).to(device)

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

# Test with different input sizes
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